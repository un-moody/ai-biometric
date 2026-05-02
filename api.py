import io, os, json, time, logging, warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
import torch

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

SR = 16000
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
PORT = int(os.environ.get("PORT", "8080"))

# مجلد تخزين النموذج (هيتعمل تلقائياً)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "voxguard")
MODEL_NAME = "microsoft/wavlm-base-plus-sv"

_wavlm_model, _wavlm_ext, _wavlm_ok = None, None, False

def _load_wavlm():
    global _wavlm_model, _wavlm_ext, _wavlm_ok
    try:
        t0 = time.time()
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        # تحقق من وجود النموذج في الكاش
        model_path = os.path.join(CACHE_DIR, "models--microsoft--wavlm-base-plus-sv")
        
        if os.path.exists(model_path):
            log.info("📦 Loading model from cache...")
            _wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
            )
            _wavlm_model = WavLMForXVector.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
            ).eval()
        else:
            log.info("📥 First run - downloading model (one-time only)...")
            _wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR
            )
            _wavlm_model = WavLMForXVector.from_pretrained(
                MODEL_NAME, cache_dir=CACHE_DIR
            ).eval()
            log.info("✅ Model downloaded and cached")
        
        _wavlm_ok = True
        log.info(f"✅ WavLM ready in {int((time.time()-t0)*1000)}ms")
        return True
    except Exception as e:
        log.error(f"❌ WavLM failed: {e}")
        _wavlm_ok = False
        return False

def get_embedding(audio):
    with torch.no_grad():
        inp = _wavlm_ext(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = _wavlm_model(**inp).embeddings.squeeze().cpu().numpy().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-8)

def load_wav(raw):
    if not raw: raise ValueError("empty audio")
    import librosa
    arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
    return arr.astype(np.float32)[:SR*3]

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# تشغيل التحميل
_load_wavlm()

app = FastAPI(title="VoxGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"ok": _wavlm_ok, "cached": os.path.exists(os.path.join(CACHE_DIR, "models--microsoft--wavlm-base-plus-sv"))}

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    if not _wavlm_ok: raise HTTPException(503, "Model not ready yet")
    t0 = time.time()
    
    from concurrent.futures import ThreadPoolExecutor
    
    async def process_audio(file):
        raw = await file.read()
        return get_embedding(load_wav(raw))
    
    # معالجة الملفات بالتوازي
    embeddings = []
    for f in [audio_1, audio_2, audio_3]:
        emb = await process_audio(f)
        embeddings.append(emb)
    
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
    threshold: float = Form(0.72),
):
    if not _wavlm_ok: raise HTTPException(503, "Model not ready yet")
    try:
        saved_emb = np.array(json.loads(embedding), dtype=np.float32)
    except:
        raise HTTPException(400, "Invalid embedding")
    
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

@app.api_route("/{path:path}")
async def catch_all(path: str):
    return JSONResponse({"ok": False, "error": f"/{path} not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
