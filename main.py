from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import random
import time
import asyncio

app = FastAPI(title="AI Video Detector API")

# Konfigurasi CORS agar frontend dapat melakukan fetch ke backend ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Dalam produksi, ganti dengan domain frontend Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/analyze")
async def analyze_video(video: UploadFile = File(...)):
    # 1. Simpan file video secara lokal (Chain of Custody simulasi)
    file_path = os.path.join(UPLOAD_DIR, video.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
    
    # 2. Simulasi waktu pemrosesan model AI (ekstraksi frame, EfficientNet, RawNet)
    await asyncio.sleep(3) 
    
    # 3. Kalkulasi probabilitas (Mock up AI Logic)
    # Dalam implementasi nyata, di sinilah model ML Anda melakukan inferensi
    is_fake = random.choice([True, False])
    base_score = random.uniform(85.0, 98.5) if is_fake else random.uniform(2.0, 15.0)
    
    # Generate heatmap / frame data
    frame_data = []
    for i in range(10):
        frame_data.append({
            "timestamp": f"00:0{i}",
            "artifact_score": random.uniform(70, 100) if is_fake else random.uniform(0, 20)
        })

    # 4. Hapus file temporary setelah analisis selesai (opsional)
    if os.path.exists(file_path):
        os.remove(file_path)

    # 5. Kembalikan response JSON ke frontend
    return {
        "status": "success",
        "filename": video.filename,
        "classification": "AI GENERATED" if is_fake else "AUTHENTIC",
        "confidence_score": round(base_score, 1),
        "details": {
            "visual_artifacts": round(random.uniform(70, 95) if is_fake else random.uniform(5, 15), 1),
            "audio_inconsistencies": round(random.uniform(60, 90) if is_fake else random.uniform(2, 10), 1),
            "temporal_glitches": round(random.uniform(65, 98) if is_fake else random.uniform(1, 12), 1)
        },
        "frame_analysis": frame_data
    }

# Jalankan server dengan perintah: uvicorn main:app --reload --port 8000