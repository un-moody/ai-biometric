#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  VoxGuard Voice Biometric API - Production Ready
================================================================================

Fast, clean, production-ready voice biometric API with WavLM embeddings.

Endpoints:
    POST   /enroll   - Register user with 3 WAV samples
    POST   /verify   - Verify user identity with 1 WAV sample
    GET    /health   - Service health check

================================================================================
"""

import os
import json
import time
import asyncio
import sqlite3
import logging
import threading
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
DB_PATH = os.environ.get("VG_DB_PATH", "voxguard.db")
MODEL_PATH = os.environ.get("WAVLM_PATH", "wavlm_model")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("voxguard")

_EXECUTOR = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# Database Layer (Compatible with existing databases)
# ============================================================================

_db: dict = {}
_db_lock = threading.Lock()


def _init_db():
    """Initialize SQLite database - handles existing schemas gracefully"""
    global _db
    
    conn = sqlite3.connect(DB_PATH)
    
    # Check if users table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = cursor.fetchone() is not None
    
    if not table_exists:
        # Create new table
        conn.execute("""
            CREATE TABLE users (
                user_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                name TEXT,
                enrolled_at TEXT,
                dim INTEGER DEFAULT 512
            )
        """)
        log.info("Created new users table")
    else:
        # Check existing columns
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Add missing columns if needed
        if "user_id" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN user_id TEXT")
        if "embedding" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN embedding TEXT")
        if "name" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
        if "enrolled_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN enrolled_at TEXT")
        if "dim" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN dim INTEGER DEFAULT 512")
        
        log.info("Verified table schema")
    
    # Load existing users
    try:
        cursor = conn.execute("SELECT user_id, embedding, name, enrolled_at FROM users WHERE user_id IS NOT NULL AND embedding IS NOT NULL")
        for row in cursor.fetchall():
            uid, emb_json, name, ts = row
            if uid and emb_json:
                try:
                    _db[uid] = {
                        "embedding": json.loads(emb_json),
                        "name": name or uid,
                        "enrolled_at": ts or ""
                    }
                except json.JSONDecodeError:
                    log.warning(f"Invalid embedding for user {uid}, skipping")
    except sqlite3.OperationalError as e:
        log.warning(f"Could not load existing users: {e}")
    
    conn.close()
    log.info(f"Database ready: {len(_db)} user(s) loaded")


def _save_user(user_id: str, data: dict):
    """Save or update user in database"""
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT OR REPLACE INTO users (user_id, embedding, name, enrolled_at, dim) VALUES (?, ?, ?, ?, ?)",
            (user_id, json.dumps(data["embedding"]), data["name"], data["enrolled_at"], 512)
        )
        conn.commit()
        conn.close()


def _delete_user(user_id: str):
    """Delete user from database"""
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()


# Initialize database with error handling
try:
    _init_db()
except Exception as e:
    log.error(f"Database init failed: {e}")
    log.info("Creating fresh database...")
    # Backup old if exists
    if Path(DB_PATH).exists():
        backup_path = f"{DB_PATH}.backup"
        Path(DB_PATH).rename(backup_path)
        log.info(f"Backed up old database to {backup_path}")
    _init_db()  # Try again with fresh DB

# ============================================================================
# Audio Processing
# ============================================================================

def _load_wav_pcm(raw: bytes) -> np.ndarray:
    """Load standard PCM WAV (fast path)"""
    import wave
    import io
    
    with wave.open(io.BytesIO(raw), "rb") as wf:
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    
    # Convert to float32
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
    
    # Convert to mono
    if nchan > 1:
        audio = audio.reshape(-1, nchan).mean(axis=1)
    
    # Resample if needed
    if framerate != SR:
        ratio = SR / framerate
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len)
        audio = np.interp(indices, np.arange(len(audio)), audio)
    
    return audio.astype(np.float32)


def _load_wav_scipy(raw: bytes) -> np.ndarray:
    """Fallback using scipy"""
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
    """Load and convert audio to 16kHz mono float32"""
    if not raw:
        raise ValueError("Empty audio data")
    
    # Try pure Python first (fastest)
    try:
        return _load_wav_pcm(raw)
    except Exception:
        pass
    
    # Fallback to scipy
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
    """Load WavLM model (called once on startup)"""
    global _model, _extractor, _model_ready, _model_error
    
    try:
        import torch
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except ImportError as e:
        _model_error = f"Missing dependencies: {e}"
        log.error(_model_error)
        return False
    
    # Try local path first
    if Path(MODEL_PATH).exists():
        try:
            _extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_PATH, local_files_only=True)
            _model = WavLMForXVector.from_pretrained(MODEL_PATH, local_files_only=True).eval()
            log.info(f"Model loaded from {MODEL_PATH}")
            _model_ready = True
            return True
        except Exception as e:
            log.warning(f"Local load failed: {e}")
    
    # Download from HuggingFace
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
    """Extract 512-dim speaker embedding"""
    if not _model_ready:
        raise RuntimeError(f"Model not ready: {_model_error}")
    
    import torch
    with torch.no_grad():
        inputs = _extractor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        embedding = _model(**inputs).embeddings.squeeze().cpu().numpy()
    
    # L2 normalize
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 1e-8 else embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="VoxGuard Voice Biometric API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    user_name: str = Form(""),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    """Enroll a new user with 3 voice samples"""
    if not _model_ready:
        raise HTTPException(503, f"Model not ready: {_model_error}")
    
    start_time = time.time()
    
    async def load_sample(file: UploadFile, idx: int):
        raw = await file.read()
        audio = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, load_audio, raw)
        
        duration = len(audio) / SR
        if duration < 1.0:
            raise HTTPException(400, f"Sample {idx} too short ({duration:.2f}s), need ≥1.0s")
        
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 0.006:
            raise HTTPException(400, f"Sample {idx} too silent (RMS={rms:.4f})")
        
        return audio, duration, rms
    
    samples = await asyncio.gather(
        load_sample(audio_1, 1),
        load_sample(audio_2, 2),
        load_sample(audio_3, 3)
    )
    
    audios = [s[0] for s in samples]
    
    # Extract embeddings
    embeddings = []
    for audio in audios:
        emb = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, get_embedding, audio)
        embeddings.append(emb)
    
    # Average embeddings
    final_embedding = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(final_embedding)
    final_embedding = (final_embedding / norm).tolist() if norm > 0 else final_embedding.tolist()
    
    # Save to database
    user_data = {
        "embedding": final_embedding,
        "name": user_name or user_id,
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    _db[user_id] = user_data
    await asyncio.get_event_loop().run_in_executor(_EXECUTOR, _save_user, user_id, user_data)
    
    elapsed_ms = int((time.time() - start_time) * 1000)
    log.info(f"Enrolled {user_id} in {elapsed_ms}ms")
    
    return {
        "success": True,
        "user_id": user_id,
        "name": user_data["name"],
        "dimension": 512,
        "enrolled_at": user_data["enrolled_at"],
        "processing_ms": elapsed_ms
    }


@app.post("/verify")
async def verify(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    threshold: float = Form(THRESHOLD),
):
    """Verify a user's identity against their enrolled voice print"""
    if not _model_ready:
        raise HTTPException(503, f"Model not ready: {_model_error}")
    
    if user_id not in _db:
        raise HTTPException(404, f"User '{user_id}' not found. Please enroll first.")
    
    start_time = time.time()
    
    raw = await audio.read()
    try:
        wav = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, load_audio, raw)
    except Exception as e:
        raise HTTPException(400, f"Invalid audio: {str(e)}")
    
    duration = len(wav) / SR
    if duration < 0.5:
        raise HTTPException(400, f"Audio too short ({duration:.2f}s), need ≥0.5s")
    
    new_embedding = await asyncio.get_event_loop().run_in_executor(_EXECUTOR, get_embedding, wav)
    stored_embedding = np.array(_db[user_id]["embedding"])
    
    score = await asyncio.get_event_loop().run_in_executor(
        _EXECUTOR, cosine_similarity, new_embedding, stored_embedding
    )
    
    is_match = score >= threshold
    elapsed_ms = int((time.time() - start_time) * 1000)
    
    log.info(f"Verified {user_id}: score={score:.4f}, match={is_match}, ms={elapsed_ms}")
    
    return {
        "success": True,
        "match": is_match,
        "score": round(score, 4),
        "score_percent": f"{score * 100:.1f}%",
        "threshold": threshold,
        "user_id": user_id,
        "duration_sec": round(duration, 2),
        "processing_ms": elapsed_ms
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy" if _model_ready else "degraded",
        "model_ready": _model_ready,
        "enrolled_users": len(_db),
        "sample_rate": SR,
        "default_threshold": THRESHOLD
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    log.info("=" * 50)
    log.info("VoxGuard Voice Biometric API")
    log.info(f"Port: {PORT}")
    log.info(f"Model: {'READY' if _model_ready else 'FAILED'}")
    log.info(f"Enrolled users: {len(_db)}")
    log.info(f"Health: http://localhost:{PORT}/health")
    log.info("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
