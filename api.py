import io, os, json, time, logging, warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
import librosa
import torch

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# منع الاتصال بالإنترنت لتحميل النماذج
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

SR = 16000
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
PORT = int(os.environ.get("PORT", "8000"))

# المسار المحلي للنموذج
MODEL_PATH = os.environ.get("VG_MODEL_PATH", "./wavlm_local")

_wavlm_model, _wavlm_ext, _wavlm_ok, _load_err = None, None, False, ""

def _load_wavlm():
    global _wavlm_model, _wavlm_ext, _load_err
    try:
        t0 = time.time()
        log.info(f"Loading WavLM from local path: {MODEL_PATH}")
        _wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained(
            MODEL_PATH, 
            local_files_only=True
        )
        _wavlm_model = WavLMForXVector.from_pretrained(
            MODEL_PATH, 
            local_files_only=True
        ).eval()
        log.info(f"✅ WavLM loaded in {int((time.time()-t0)*1000)}ms")
        return True
    except Exception as e:
        _load_err = str(e)
        return False

def get_embedding(audio):
    with torch.no_grad():
        inp = _wavlm_ext(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = _wavlm_model(**inp).embeddings.squeeze().cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 1e-8 else emb

def load_wav(raw):
    if not raw: raise ValueError("empty audio")
    try:
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if arr.ndim > 1: arr = arr.mean(axis=1)
        if sr != SR:
            g = gcd(sr, SR)
            arr = resample_poly(arr, SR // g, sr // g).astype(np.float32)
        return arr[:SR*4]
    except:
        arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        return arr.astype(np.float32)[:SR*4]

def cosine_sim(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na < 1e-8 or nb < 1e-8 else float(np.dot(a, b) / (na * nb))

_wavlm_ok = _load_wavlm()

app = FastAPI(title="VoxGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"ok": True, "wavlm": _wavlm_ok, "model_path": MODEL_PATH}

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    if not _wavlm_ok: raise HTTPException(503, f"Model not loaded: {_load_err}")
    t0 = time.time()
    embeddings = []
    for f in [audio_1, audio_2, audio_3]:
        raw = await f.read()
        wav = load_wav(raw)
        emb = get_embedding(wav)
        embeddings.append(emb)
    final = np.mean(embeddings, axis=0)
    n = float(np.linalg.norm(final))
    if n > 1e-8: final = final / n
    return {
        "ok": True, "user_id": user_id, "dim": int(len(final)),
        "embedding": final.tolist(), "ms": int((time.time()-t0)*1000)
    }

@app.post("/verify_embedding")
async def verify_embedding(
    audio: UploadFile = File(...),
    embedding: str = Form(...),
    threshold: float = Form(0.72),
):
    if not _wavlm_ok: raise HTTPException(503, f"Model not loaded: {_load_err}")
    try:
        saved_emb = np.array(json.loads(embedding), dtype=np.float32)
    except:
        raise HTTPException(400, "Invalid embedding")
    t0 = time.time()
    raw = await audio.read()
    wav = load_wav(raw)
    new_emb = get_embedding(wav)
    score = cosine_sim(new_emb, saved_emb)
    match = bool(score >= threshold)
    return {
        "ok": True, "match": match, "score": round(float(score), 4),
        "sos_trigger": match, "ms": int((time.time()-t0)*1000)
    }

@app.api_route("/{path:path}")
async def catch_all(path: str):
    return JSONResponse({"ok": False, "error": f"/{path} not found"}, status_code=404)

if __name__ == "__main__":
    import uvicorn
    log.info(f"🚀 Starting VoxGuard on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
