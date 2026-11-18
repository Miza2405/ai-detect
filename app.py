from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import os
import shutil
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

app = FastAPI(title="VeriSight AI Forensic Engine v3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ENSEMBLE MODEL AI GENERATIF MODERN (DIFFUSION / SYNTHETIC DETECTORS) ---
CANDIDATE_MODELS = [
    "Organika/sdxl-detector",
    "umm-maybe/AI-image-detector",
]

loaded_detectors = []

for model_id in CANDIDATE_MODELS:
    try:
        print(f"[INFO] Memuat detector AI: {model_id} ...")
        proc = AutoImageProcessor.from_pretrained(model_id)
        mdl = AutoModelForImageClassification.from_pretrained(model_id)
        mdl.eval()

        fake_indices = []
        for idx, label in mdl.config.id2label.items():
            if any(k in label.lower() for k in ["fake", "artificial", "ai", "generated", "synthetic"]):
                fake_indices.append(int(idx))
        if not fake_indices:
            fake_indices = [0]

        loaded_detectors.append({
            "name": model_id,
            "processor": proc,
            "model": mdl,
            "fake_indices": fake_indices,
        })
        print(f"[INFO] Berhasil memuat {model_id} | Mapping: {mdl.config.id2label}")
    except Exception as e:
        print(f"[WARNING] Gagal memuat {model_id}: {e}")


def extract_aspect_preserving_crops(rgb_frame: np.ndarray) -> list:
    """
    Membuat 3 potongan persegi beresolusi asli tanpa membuat frame 9:16 / 16:9 gepeng:
    1. Area Atas (kepala, wajah, bahu/pangkat)
    2. Area Tengah (tangan & interaksi objek)
    3. Zoom Subjek Utama (65% area fokus tengah-atas)
    """
    h, w, _ = rgb_frame.shape
    crops = []

    if h >= w:
        # Video vertikal (misal 9:16 TikTok / Reels / Shorts)
        side = w
        top_crop = rgb_frame[0:side, 0:side]
        mid_y = max(0, (h - side) // 2)
        mid_crop = rgb_frame[mid_y:mid_y + side, 0:side]
        crops.extend([Image.fromarray(top_crop), Image.fromarray(mid_crop)])
    else:
        # Video horizontal (misal 16:9 YouTube)
        side = h
        left_x = max(0, (w - side) // 2)
        center_crop = rgb_frame[0:side, left_x:left_x + side]
        crops.append(Image.fromarray(center_crop))

    # Zoom fokus subjek utama untuk membaca detail tekstur & artefak difusi
    y1, y2 = int(h * 0.12), int(h * 0.72)
    x1, x2 = int(w * 0.15), int(w * 0.85)
    zoom_crop = rgb_frame[y1:y2, x1:x2]
    if zoom_crop.size > 0:
        crops.append(Image.fromarray(zoom_crop))

    return crops


def predict_generative_ai_score(rgb_frame: np.ndarray) -> float:
    """
    Menjalankan inferensi ensemble Swin Transformer pada seluruh potongan frame.
    """
    if not loaded_detectors:
        return -1.0

    crops = extract_aspect_preserving_crops(rgb_frame)
    detector_scores = []

    for det in loaded_detectors:
        inputs = det["processor"](images=crops, return_tensors="pt")
        with torch.no_grad():
            outputs = det["model"](**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        crop_probs = []
        for i in range(probs.shape[0]):
            p_ai = sum(float(probs[i, idx].item()) for idx in det["fake_indices"])
            crop_probs.append(p_ai * 100.0)

        # Kombinasikan skor tertinggi antar-crop (area paling mencurigakan) dan rata-rata
        frame_score = 0.65 * max(crop_probs) + 0.35 * float(np.mean(crop_probs))
        detector_scores.append(frame_score)

    return float(np.mean(detector_scores))


def compute_optical_flow_morphing(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """
    Mendeteksi inkonsistensi struktur (morphing) pada objek bergerak antar-frame
    menggunakan Dense Optical Flow (Farneback).
    """
    h, w = curr_gray.shape
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Fokus hanya pada piksel objek yang sedang bergerak aktif
    motion_mask = (mag > 1.5) & (mag < 28.0)
    if np.sum(motion_mask) < 350:
        return 0.0

    grid_x, grid_y = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = grid_x - flow[..., 0]
    map_y = grid_y - flow[..., 1]

    warped_prev = cv2.remap(
        prev_gray, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    residual = np.abs(curr_gray.astype(np.float32) - warped_prev.astype(np.float32))
    normalized_err = residual[motion_mask] / (mag[motion_mask] + 2.0)

    score = float(np.clip((np.mean(normalized_err) - 1.65) * 32.0, 0.0, 100.0))
    return score


def compute_spectral_anomaly(gray_frame: np.ndarray) -> float:
    """
    Mengukur distribusi spektrum frekuensi 2D FFT yang tahan terhadap kompresi ringan.
    """
    resized = cv2.resize(gray_frame, (256, 256))
    f_shift = np.fft.fftshift(np.fft.fft2(resized.astype(np.float32)))
    magnitude = np.log1p(np.abs(f_shift))

    cy, cx = 128, 128
    y, x = np.ogrid[:256, :256]
    radius = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    low_e = np.mean(magnitude[radius <= 24]) + 1e-6
    mid_high_e = np.mean(magnitude[(radius > 45) & (radius <= 95)])
    ratio = mid_high_e / low_e

    return float(np.clip((ratio - 0.48) * 280.0, 0.0, 100.0))


@app.post("/api/scan-video")
async def scan_video(file: UploadFile = File(...)):
    temp_filename = f"temp_{os.getpid()}_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        cap = cv2.VideoCapture(temp_filename)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_samples = 20
        step = max(1, total_frames // target_samples) if total_frames > 0 else 5

        frame_idx = 0
        sampled_count = 0

        dl_scores = []
        morph_scores = []
        fft_scores = []
        laplacian_vals = []
        noise_vals = []

        prev_small_gray = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0 and sampled_count < target_samples:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 1. Telemetri dasar
                laplacian_vals.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                noise_vals.append(float(np.mean(cv2.absdiff(gray, blurred))))

                # 2. Analisis Spektrum Frekuensi
                fft_scores.append(compute_spectral_anomaly(gray))

                # 3. Analisis Optical Flow Morphing antar-frame
                small_gray = cv2.resize(gray, (256, 256))
                if prev_small_gray is not None:
                    m_score = compute_optical_flow_morphing(prev_small_gray, small_gray)
                    if m_score > 0:
                        morph_scores.append(m_score)
                prev_small_gray = small_gray

                # 4. Inferensi Swin Transformer (jalankan pada 10 frame merata agar cepat & akurat)
                if loaded_detectors and sampled_count % 2 == 0:
                    ai_prob = predict_generative_ai_score(rgb)
                    if ai_prob >= 0:
                        dl_scores.append(ai_prob)

                sampled_count += 1

            frame_idx += 1

        cap.release()
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

        if sampled_count == 0:
            return JSONResponse(status_code=400, content={"error": "Gagal membaca frame video."})

        avg_laplacian = float(np.mean(laplacian_vals))
        avg_noise = float(np.mean(noise_vals))
        avg_fft = float(np.mean(fft_scores))
        avg_morph = float(np.percentile(morph_scores, 75)) if morph_scores else 15.0

        if dl_scores:
            # Gunakan persentil ke-80 karena artefak AI difusi muncul paling jelas saat gerakan kompleks
            dl_final = float(np.percentile(dl_scores, 80))
            fake_probability = (0.72 * dl_final) + (0.18 * avg_morph) + (0.10 * avg_fft)
            active_engines = ", ".join(d["name"] for d in loaded_detectors)
        else:
            dl_final = 0.0
            fake_probability = (0.60 * avg_morph) + (0.40 * avg_fft)
            active_engines = "Optical Flow Morphing + FFT Fallback"

        fake_probability = float(np.clip(fake_probability, 1.0, 99.4))
        is_fake = fake_probability >= 55.0

        return {
            "status": "success",
            "fake_probability": round(fake_probability, 1),
            "is_fake": is_fake,
            "metrics": {
                "blur_index": round(avg_laplacian, 2),
                "noise_inconsistency": round(avg_noise, 2),
                "frames_analyzed": sampled_count,
                "dl_vit_score": round(dl_final, 1),
                "morphing_score": round(avg_morph, 1),
                "fft_spectral_score": round(avg_fft, 1),
                "engine": active_engines,
            },
        }

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return JSONResponse(status_code=500, content={"error": str(e)})