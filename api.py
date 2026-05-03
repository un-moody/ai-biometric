#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoxGuard Voice Biometric API - Stateless for Laravel Integration
"""

import os
import json
import time
import asyncio
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# Configuration
# ============================================================================

SR = 16000
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
PORT = int(os.environ.get("PORT", "9000"))
MODEL_PATH = os.environ.get("WAVLM_PATH", "wavlm_model")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("voice_api")

_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# No database! Python is stateless.

# ============================================================================
# Audio Processing
# ============================================================================

def _load_wav_pcm(raw: bytes) -> np.ndarray:
    import wave
    import io
    
    with wave.open(io.BytesIO(raw), "rb") as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    
    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 3:
        bytes_arr = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        audio = np.zeros(len(bytes_arr), dtype=np.int32)
        for i in range(len(bytes_arr)):
            audio[i] = (bytes_arr[i, 0] | (bytes_arr[i, 1] << 8) | (bytes_arr[i, 2] << 16))
        audio = audio.astype(np.float32) / 8388608.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    
    if nchan > 1:
        audio = audio.reshape(-1, nchan).mean(axis=1)
    
    if framerate != SR:
        ratio = SR / framerate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return audio.astype(np.float32)


def _load_wav_scipy(raw: bytes) -> np.ndarray:
    import io
    from scipy.io import wavfile
    
    sr, audio = wavfile.read(io.BytesIO(raw))
    
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)
    
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    if sr != SR:
        ratio = SR / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return audio.astype(np.float32)


def load_audio(raw: bytes) -> np.ndarray:
    if not raw:
        raise ValueError("Empty audio data")
    
    try:
        return _load_wav_pcm(raw)
    except Exception:
        pass
    
    try:
        return _load_wav_scipy(raw)
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {e}")


# ============================================================================
# WavLM Model
# ============================================================================

_model = None
_extractor = None
_model_ready = False
_model_error = ""


def _load_model():
    global _model, _extractor, _model_ready, _model_error
    
    try:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except ImportError as e:
        _model_error = f"Missing dependencies: {e}"
        log.error(_model_error)
        return False
    
    if Path(MODEL_PATH).exists():
        try:
            _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
            _model = WavLMForXVector.from_pretrained(MODEL_PATH, local_files_only=True).eval()
            log.info(f"Model loaded from {MODEL_PATH}")
            _model_ready = True
            return True
        except Exception as e:
            log.warning(f"Local load failed: {e}")
    
    try:
        log.info("Downloading WavLM model (first run only)...")
        _extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        _model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()
        log.info("Model ready")
        _model_ready = True
        return True
    except Exception as e:
        _model_error = f"Download failed: {e}"
        log.error(_model_error)
        return False


log.info("Loading WavLM model...")
_load_model()
log.info(f"Model status: {'READY' if _model_ready else 'FAILED'}")


def get_embedding(audio: np.ndarray) -> np.ndarray:
    if not _model_ready:
        raise RuntimeError(f"Model not ready: {_model_error}")
    
    import torch
    with torch.no_grad():
        inputs = _extractor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        embedding = _model(**inputs).embeddings.squeeze().cpu().numpy()
    
    norm = np.linalg.norm(embedding)
    return (embedding / norm).tolist() if norm > 1e-8 else embedding.tolist()


def cosine_similarity(a: list, b: list) -> float:
    a_np = np.array(a)
    b_np = np.array(b)
    return float(np.dot(a_np, b_np) / (np.linalg.norm(a_np) * np.linalg.norm(b_np) + 1e-8))


# ============================================================================
# FastAPI App - Stateless
# ============================================================================

app = FastAPI(title="VoxGuard Voice API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Laravel to call
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    user_name: str = Form(""),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    """Returns embedding for Laravel to store in MySQL"""
    if not _model_ready:
        raise HTTPException(503, f"Model not ready: {_model_error}")
    
    start_time = time.time()
    
    async def load_sample(file: UploadFile, idx: int):
        raw = await file.read()
        audio = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, load_audio, raw)
        
        duration = len(audio) / SR
        if duration < 1.0:
            raise HTTPException(400, f"Sample {idx} too short ({duration:.2f}s)")
        
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.006:
            raise HTTPException(400, f"Sample {idx} too silent")
        
        return audio
    
    audios = await asyncio.gather(
        load_sample(audio_1, 1),
        load_sample(audio_2, 2),
        load_sample(audio_3, 3)
    )
    
    embeddings = []
    for audio in audios:
        emb = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, get_embedding, audio)
        embeddings.append(np.array(emb))
    
    final_embedding = np.mean(embeddings, axis=0).tolist()
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(f"Enroll {user_id}: {elapsed_ms}ms")
    
    return {
        "success": True,
        "user_id": user_id,
        "name": user_name or user_id,
        "embedding": final_embedding,
        "dimension": 512,
        "processing_ms": elapsed_ms
    }


@app.post("/verify")
async def verify(
    embedding: str = Form(..., description="JSON embedding from Laravel MySQL"),
    audio: UploadFile = File(...),
    threshold: float = Form(THRESHOLD),
    user_id: str = Form(""),
):
    """Compare audio with stored embedding (sent from Laravel)"""
    if not _model_ready:
        raise HTTPException(503, f"Model not ready: {_model_error}")
    
    start_time = time.time()
    
    # Parse embedding from Laravel
    try:
        stored_emb = json.loads(embedding)
        if len(stored_emb) != 512:
            raise ValueError(f"Expected 512 dims, got {len(stored_emb)}")
    except Exception as e:
        raise HTTPException(400, f"Invalid embedding: {e}")
    
    # Load audio
    raw = await audio.read()
    wav = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, load_audio, raw)
    
    duration = len(wav) / SR
    if duration < 0.5:
        raise HTTPException(400, f"Audio too short ({duration:.2f}s)")
    
    # Get new embedding
    new_emb = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, get_embedding, wav)
    
    # Compare
    score = cosine_similarity(new_emb, stored_emb)
    match = score >= threshold
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(f"Verify {user_id}: score={score:.4f}, match={match}, ms={elapsed_ms}")
    
    return {
        "success": True,
        "match": match,
        "score": round(score, 4),
        "score_percent": f"{score * 100:.1f}%",
        "threshold": threshold,
        "processing_ms": elapsed_ms
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if _model_ready else "degraded",
        "model_ready": _model_ready,
        "sample_rate": SR
    }


if __name__ == "__main__":
    log.info("=" * 50)
    log.info("VoxGuard Voice API - Stateless")
    log.info(f"Port: {PORT}")
    log.info(f"Model: {'READY' if _model_ready else 'FAILED'}")
    log.info("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
