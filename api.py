import io, os, json, time, logging, warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
import librosa
import torch

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

SR = 16000
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = "microsoft/wavlm-base-plus-sv"

# متغيرات عامة
_wavlm_model, _wavlm_ext = None, None

# --- دوال تحميل النموذج ---
def _load_wavlm():
    global _wavlm_model, _wavlm_ext
    try:
        t0 = time.time()
        log.info("⏳ Loading WavLM model (this may take a while on first run)...")
        _wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _wavlm_model = WavLMForXVector.from_pretrained(MODEL_NAME).eval()
        log.info(f"✅ WavLM model loaded in {int((time.time()-t0)*1000)}ms")
        return True
    except Exception as e:
        log.error(f"❌ Failed to load WavLM: {e}")
        return False

# --- دوال المعالجة ---
def get_embedding(audio: np.ndarray):
    if _wavlm_model is None or _wavlm_ext is None:
        raise HTTPException(503, "Model is not ready yet.")
    with torch.no_grad():
        inp = _wavlm_ext(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = _wavlm_model(**inp).embeddings.squeeze().cpu().numpy().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)

def load_wav(raw: bytes):
    if not raw: raise ValueError("empty audio")
    arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    # نقطع أول 4 ثواني بس عشان السرعة
    return arr.astype(np.float32)[:SR * 4]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# --- بدء التطبيق وتحميل النموذج ---
app = FastAPI(title="VoxGuard API - Fast Edition")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# حدث تحميل النموذج عند بدء التشغيل
@app.on_event("startup")
async def startup_event():
    log.info("Application startup - loading model...")
    if not _load_wavlm():
        log.error("Could not load model at startup. Will try on first request.")
    else:
        log.info("Model loaded and ready.")

# --- الـ API Endpoints ---
@app.get("/health")
async def health():
    return {"ok": True, "wavlm_loaded": _wavlm_model is not None}

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    if _wavlm_model is None: # لو النموذج مش جاهز، جرب تحميله
        _load_wavlm()
    if _wavlm_model is None:
        raise HTTPException(503, "Model failed to load")

    t0 = time.time()
    from concurrent.futures import ThreadPoolExecutor
    
    async def process_audio(file):
        raw = await file.read()
        wav = load_wav(raw)
        return get_embedding(wav)
    
    # معالجة الملفات الصوتية بالتوازي
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(process_audio, f) for f in [audio_1, audio_2, audio_3]]
        embeddings = [future.result() for future in futures]
    
    final = np.mean(embeddings, axis=0)
    final = final / np.linalg.norm(final)
    
    return {
        "ok": True,
        "user_id": user_id,
        "embedding": final.tolist(),
        "ms": int((time.time()-t0)*1000)
    }

@app.post("/verify")
async def verify(
    audio: UploadFile = File(...),
    embedding: str = Form(...),
    threshold: float = Form(THRESHOLD),
):
    if _wavlm_model is None:
        _load_wavlm()
    if _wavlm_model is None:
        raise HTTPException(503, "Model failed to load")
    try:
        saved_emb = np.array(json.loads(embedding), dtype=np.float32)
    except:
        raise HTTPException(400, "Invalid embedding format. Please provide a JSON array.")
    
    t0 = time.time()
    raw = await audio.read()
    wav = load_wav(raw)
    new_emb = get_embedding(wav)
    score = cosine_sim(new_emb, saved_emb)
    
    return {
        "ok": True,
        "match": bool(score >= threshold),
        "score": round(float(score), 4),
        "ms": int((time.time()-t0)*1000)
    }

# للتعامل مع أي مسارات غير معروفة
@app.api_route("/{path:path}")
async def catch_all(path: str):
    return JSONResponse({"ok": False, "error": f"/{path} not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
