#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoxGuard API - WavLM-SV Only
Fast, Clean, Single-Model Voice Biometric
"""

import io
import os
import json
import time
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SR = 16000
PORT = int(os.environ.get("PORT", "9000"))
THRESHOLD = float(os.environ.get("VG_THRESHOLD", "0.72"))
DEVICE = "cpu"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("VG")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL - Load once at startup
# ═══════════════════════════════════════════════════════════════════════════════

model = None
extractor = None
model_ready = False

def load_model():
    """تحميل WavLM مرة واحدة عند بداية التشغيل"""
    global model, extractor, model_ready
    
    t0 = time.time()
    log.info("Loading WavLM-SV...")
    
    try:
        extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        model = WavLMForXVector.from_pretrained(
            "microsoft/wavlm-base-plus-sv",
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).eval()
        
        model_ready = True
        log.info(f"✅ WavLM ready in {int((time.time()-t0)*1000)}ms")
        
        # تسخين - أول طلب بعد كده هيبقى سريع
        log.info("🔥 Warming up...")
        dummy = np.zeros(SR * 2, dtype=np.float32)
        _ = extract_embedding(dummy)
        log.info("✅ Warmup done")
        
    except Exception as e:
        log.error(f"❌ Failed to load WavLM: {e}")
        model_ready = False


def extract_embedding(audio: np.ndarray) -> np.ndarray:
    """استخراج بصمة 512-dim من الصوت"""
    audio = audio[:SR * 8]  # max 8 seconds
    
    with torch.inference_mode():
        inputs = extractor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        emb = model(**inputs).embeddings
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    
    return emb.squeeze().cpu().numpy().astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_audio(raw: bytes) -> np.ndarray:
    """تحميل الصوت - WAV raw bytes → float32 numpy array @ 16kHz"""
    if not raw:
        raise ValueError("Empty audio")
    
    # نحاول soundfile الأول (أسرع)
    try:
        import soundfile as sf
        arr, orig_sr = sf.read(io.BytesIO(raw), dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if orig_sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, SR)
            arr = resample_poly(arr, SR // g, orig_sr // g).astype(np.float32)
        return arr.astype(np.float32)
    except:
        pass
    
    # fallback: librosa
    try:
        import librosa
        arr, _ = librosa.load(io.BytesIO(raw), sr=SR, mono=True)
        return arr.astype(np.float32)
    except:
        pass
    
    raise ValueError("Cannot decode audio - use WAV format")


# ═══════════════════════════════════════════════════════════════════════════════
# IN-MEMORY DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

db: dict = {}  # {user_id: {"name": str, "embedding": list, "enrolled_at": str}}


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="VoxGuard API", version="3.0.0", docs_url="/docs")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

executor = ThreadPoolExecutor(max_workers=4)


@app.on_event("startup")
async def startup():
    """تحميل النموذج عند بداية التشغيل"""
    await asyncio.get_event_loop().run_in_executor(executor, load_model)


# ═══════════════════════════════════════════════════════════════════════════════
# ENROLL
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    user_name: str = Form(""),
    audio_1: UploadFile = File(...),
    audio_2: UploadFile = File(...),
    audio_3: UploadFile = File(...),
):
    """
    تسجيل بصمة من 3 عينات
    
    - يتحقق إن العينات الـ 3 لنفس الشخص (cross-check)
    - ياخد المتوسط ويخزنه
    """
    if not model_ready:
        raise HTTPException(503, "Model not ready")
    
    t0 = time.time()
    
    # تحميل 3 عينات بالتوازي
    raws = await asyncio.gather(audio_1.read(), audio_2.read(), audio_3.read())
    
    loop = asyncio.get_event_loop()
    
    # تحويل الصوت بالتوازي
    wavs = await asyncio.gather(
        loop.run_in_executor(executor, load_audio, raws[0]),
        loop.run_in_executor(executor, load_audio, raws[1]),
        loop.run_in_executor(executor, load_audio, raws[2]),
    )
    
    # التحقق من مدة الصوت
    for i, w in enumerate(wavs, 1):
        dur = len(w) / SR
        if dur < 1.0:
            raise HTTPException(400, f"Sample {i}: too short ({dur:.1f}s, need ≥1s)")
    
    # استخراج البصمات
    embeddings = [extract_embedding(w) for w in wavs]
    
    # Cross-check: التأكد إن العينات لنفس الشخص
    s12 = float(np.dot(embeddings[0], embeddings[1]))
    s13 = float(np.dot(embeddings[0], embeddings[2]))
    s23 = float(np.dot(embeddings[1], embeddings[2]))
    
    min_score = min(s12, s13, s23)
    
    if min_score < 0.65:
        raise HTTPException(
            400,
            f"Samples don't match! Scores: s12={s12:.3f} s13={s13:.3f} s23={s23:.3f}. "
            "Re-record with the same person."
        )
    
    # متوسط البصمات
    final = np.mean(embeddings, axis=0)
    final = final / (np.linalg.norm(final) + 1e-8)
    
    # تخزين
    db[user_id] = {
        "name": user_name or user_id,
        "embedding": final.tolist(),
        "enrolled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    ms = int((time.time() - t0) * 1000)
    log.info(f"✅ ENROLL {user_id} | cross={min_score:.3f} | {ms}ms")
    
    return {
        "ok": True,
        "user_id": user_id,
        "name": user_name or user_id,
        "cross_check": {"s12": round(s12, 4), "s13": round(s13, 4), "s23": round(s23, 4)},
        "embedding_dim": len(final),
        "ms": ms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/verify")
async def verify(
    user_id: str = Form(...),
    audio: UploadFile = File(...),
    threshold: float = Form(THRESHOLD),
):
    """
    التحقق من الهوية
    
    - يقارن العينة بالبصمة المخزنة لـ user_id ده فقط
    - مبيعملش بحث في كل المستخدمين
    """
    if not model_ready:
        raise HTTPException(503, "Model not ready")
    
    if user_id not in db:
        raise HTTPException(404, f"User '{user_id}' not enrolled. Use /enroll first.")
    
    t0 = time.time()
    
    raw = await audio.read()
    
    loop = asyncio.get_event_loop()
    wav = await loop.run_in_executor(executor, load_audio, raw)
    
    dur = len(wav) / SR
    if dur < 0.5:
        raise HTTPException(400, f"Audio too short ({dur:.1f}s)")
    
    new_emb = extract_embedding(wav)
    stored_emb = np.array(db[user_id]["embedding"], dtype=np.float32)
    
    score = float(np.dot(new_emb, stored_emb))
    match = score >= threshold
    
    ms = int((time.time() - t0) * 1000)
    log.info(f"{'✅' if match else '❌'} VERIFY {user_id} | score={score:.4f} | {ms}ms")
    
    return {
        "ok": True,
        "match": match,
        "score": round(score, 4),
        "score_pct": f"{round(score * 100, 1)}%",
        "threshold": threshold,
        "user_id": user_id,
        "name": db[user_id]["name"],
        "ms": ms,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# USERS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/users")
async def list_users():
    """قائمة المستخدمين المسجلين"""
    return {
        "ok": True,
        "count": len(db),
        "users": [
            {"user_id": uid, "name": info["name"], "enrolled_at": info["enrolled_at"]}
            for uid, info in db.items()
        ],
    }


@app.get("/users/{user_id}/embedding")
async def get_embedding(user_id: str):
    """جلب البصمة المخزنة (للحفظ في قاعدة بيانات خارجية)"""
    if user_id not in db:
        raise HTTPException(404, f"User '{user_id}' not found")
    return {
        "ok": True,
        "user_id": user_id,
        "embedding": db[user_id]["embedding"],
    }


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """حذف مستخدم"""
    if user_id not in db:
        raise HTTPException(404, f"User '{user_id}' not found")
    del db[user_id]
    return {"ok": True, "deleted": user_id}


@app.get("/health")
async def health():
    return {"ok": model_ready, "users": len(db)}


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting VoxGuard on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
