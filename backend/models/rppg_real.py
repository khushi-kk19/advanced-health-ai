# backend/models/rppg_real.py
"""
Realistic rPPG estimator (video -> heart rate).
- Reads video file path (or frames array)
- Extracts forehead ROI via Haar cascade (fallback) — optionally replace with MediaPipe
- Computes mean green-channel signal, detrends, bandpass filters, FFT/Welch peak picking
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
from scipy.signal import butter, filtfilt, detrend, welch
import os
import logging

logger = logging.getLogger(__name__)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def _bandpass_filter(sig: np.ndarray, fs: float, low: float = 0.7, high: float = 4.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    low_n, high_n = low / nyq, high / nyq
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, sig)

def _dominant_freq(sig: np.ndarray, fs: float, fmin=0.7, fmax=4.0):
    # Welch PSD for robustness
    nperseg = min(256, len(sig))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return None, freqs, psd
    idx = np.argmax(psd[mask])
    dom_freq = freqs[mask][idx]
    return float(dom_freq), freqs, psd

class RPpgReal:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def _extract_face_roi(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60,60))
        if len(faces) == 0:
            # fallback center crop (safe)
            h, w = frame.shape[:2]
            s = int(min(w,h) * 0.35)
            cx, cy = w//2, h//2
            y0 = max(0, cy - s//2); x0 = max(0, cx - s//2)
            return frame[y0:y0+s, x0:x0+s]
        x,y,wf,hf = faces[0]
        # forehead ROI: top 25% of face box
        fh = max(10, int(hf * 0.25))
        roi = frame[y:y+fh, x:x+wf]
        return roi

    def estimate_from_video(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        green_means = []
        frames = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1
            try:
                roi = self._extract_face_roi(frame)
            except Exception:
                continue
            if roi is None or roi.size == 0:
                continue
            gmean = np.mean(roi[:, :, 1])
            green_means.append(gmean)
        cap.release()

        if len(green_means) < int(0.5 * fps):  # less than 0.5s of data -> unreliable
            return {"heart_rate": None, "confidence": 0.0, "frames": frames}

        sig = np.array(green_means, dtype=float)
        sig = detrend(sig)
        std = np.std(sig) + 1e-8
        sig = (sig - np.mean(sig)) / std
        filtered = _bandpass_filter(sig, fs=fps, low=0.7, high=4.0)
        dom_freq, freqs, psd = _dominant_freq(filtered, fs=fps, fmin=0.7, fmax=4.0)
        if dom_freq is None:
            return {"heart_rate": None, "confidence": 0.0, "frames": frames}
        hr_bpm = dom_freq * 60.0
        # confidence based on PSD peak prominence
        mask = (freqs >= 0.7) & (freqs <= 4.0)
        peak_power = float(np.max(psd[mask]))
        total_power = float(np.sum(psd[mask]) + 1e-8)
        confidence = float(peak_power / total_power)
        return {"heart_rate": float(hr_bpm), "confidence": confidence, "frames": frames}
