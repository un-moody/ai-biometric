import io, os, json, time, wave, logging, threading
from pathlib import Path
from typing import Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

SR = 16000
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
PORT = int(os.environ.get("PORT", "8000"))

# ─── WavLM Model ───
_wavlm_model, _wavlm_ext, _wavlm_ok, _load_err = None, None, False, ""

def _load_wavlm():
    global _wavlm_model, _wavlm_ext, _load_err
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        _wavlm_ext = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        _wavlm_model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        return True
    except Exception as e:
        _load_err = str(e)
        return False

def get_embedding(audio):
    import torch
    with torch.no_grad():
        inp = _wavlm_ext(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = _wavlm_model(**inp).embeddings.squeeze().cpu().numpy().astype(np.float32)
    n = float(np.linalg.norm(emb))
    return emb / n if n > 1e-8 else emb

def load_wav(raw):
    if not raw: raise ValueError("empty audio")
    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if arr.ndim > 1: arr = arr.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SR)
            arr = resample_poly(arr, SR // g, sr // g).astype(np.float32)
        return arr
    except:
        import librosa
        arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        return arr.astype(np.float32)

def cosine_sim(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return 0.0 if na < 1e-8 or nb < 1e-8 else float(np.dot(a, b) / (na * nb))

log.info("Loading WavLM...")
_wavlm_ok = _load_wavlm()
log.info(f"WavLM: {'OK' if _wavlm_ok else 'FAILED'}")

# ─── FastAPI App ───
app = FastAPI(title="VoxGuard API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"ok": True, "wavlm": _wavlm_ok}

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    if not _wavlm_ok: raise HTTPException(503, f"Model: {_load_err}")
    embeddings = []
    for f in [audio_1, audio_2, audio_3]:
        wav = load_wav(await f.read())
        embeddings.append(get_embedding(wav))
    final = np.mean(embeddings, axis=0)
    n = float(np.linalg.norm(final))
    final = final / n if n > 1e-8 else final
    return {"ok": True, "user_id": user_id, "dim": len(final), "embedding": final.tolist()}

@app.post("/verify_embedding")
async def verify_embedding(
    audio: UploadFile = File(...),
    embedding: str = Form(...),
    threshold: float = Form(0.72),
):
    if not _wavlm_ok: raise HTTPException(503, f"Model: {_load_err}")
    saved_emb = np.array(json.loads(embedding), dtype=np.float32)
    wav = load_wav(await audio.read())
    new_emb = get_embedding(wav)
    score = cosine_sim(new_emb, saved_emb)
    match = bool(score >= threshold)
    return {"ok": True, "match": match, "score": round(score, 4), "sos_trigger": match}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
