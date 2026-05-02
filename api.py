"""
VoxGuard Voice Biometric API
Optimized for Railway deployment
"""
import io, os, json, time, logging
import warnings
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor

# إخفاء التحذيرات
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

SR = 16000
PORT = int(os.environ.get("PORT", "8000"))

# ─── Model ───
_wlm_mdl, _wlm_fe = None, None

log.warning("Loading WavLM...")
import torch
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

_wlm_fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
_wlm_mdl = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
log.warning("WavLM ready")

executor = ThreadPoolExecutor(max_workers=2)

def load_wav(raw: bytes) -> np.ndarray:
    import soundfile as sf
    arr, sr = sf.read(io.BytesIO(raw), dtype="float32")
    if arr.ndim > 1: arr = arr.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, SR)
        arr = resample_poly(arr, SR//g, sr//g)
    return arr.astype(np.float32)[:SR*5]

def get_embedding(wav: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        inp = _wlm_fe(wav, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = _wlm_mdl(**inp).embeddings.squeeze().cpu().numpy()
    n = float(np.linalg.norm(emb))
    return (emb / n).astype(np.float32) if n > 1e-8 else emb.astype(np.float32)

app = FastAPI(title="VoxGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"ok": True}

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    # قراءة الملفات بالتوازي
    raws = await asyncio_gather(
        audio_1.read(), audio_2.read(), audio_3.read()
    )
    
    # تحويل WAV بالتوازي
    wavs = list(executor.map(load_wav, raws))
    
    # استخراج embeddings بالتوازي
    embs = list(executor.map(get_embedding, wavs))
    
    final = np.mean(embs, axis=0)
    final /= float(np.linalg.norm(final)) + 1e-8
    
    return {
        "ok": True,
        "user_id": user_id,
        "dim": len(final),
        "embedding": final.tolist()
    }

@app.post("/verify")
async def verify(
    audio: UploadFile = File(...),
    embedding: str = Form(...),
    threshold: float = Form(0.72),
):
    try:
        saved = np.array(json.loads(embedding), dtype=np.float32)
    except:
        raise HTTPException(400, "Invalid embedding")
    
    raw = await audio.read()
    wav = executor.submit(load_wav, raw).result()
    new_emb = executor.submit(get_embedding, wav).result()
    
    score = float(np.dot(new_emb, saved)) / (float(np.linalg.norm(new_emb)) * float(np.linalg.norm(saved)) + 1e-8)
    match = bool(score >= threshold)
    
    return {
        "ok": True,
        "match": match,
        "score": round(score, 4),
        "sos_trigger": match
    }

# Helper for async parallel reads
async def asyncio_gather(*coros):
    import asyncio
    return await asyncio.gather(*coros)
