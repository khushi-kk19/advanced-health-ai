# -*- coding: utf-8 -*-
"""
FastAPI Backend for Advanced Health AI - Stress Detection
Multi-modal predictions using trained LSTM, CNN, and Hybrid models
"""

import os
import json
import base64
import numpy as np
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from pydantic import BaseModel
import torch
from PIL import Image
import cv2

# Import inference module
import sys
sys.path.insert(0, str(Path(__file__).parent))
from inference_stress_phase123 import StressPredictor
from models.stress_detection.lstm_model import LSTMAttentionModel
from models.stress_detection.cnn_model import CNNStressDetector
from models.stress_detection.hybrid_model import HybridStressModel
from data.preprocessing import normalize_signal

# ============================================================================
# CONFIGURATION
# ============================================================================

app = FastAPI(
    title="Advanced Health AI - Stress Detection API",
    description="Multi-modal stress detection using LSTM, CNN, and Hybrid models",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints" / "stress_detection"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODELS = {}
PREDICTION_HISTORY = []

# Stress labels
CLASS_LABELS = {0: "Baseline", 1: "Stress", 2: "Amusement", 3: "Meditation"}
REVERSE_LABELS = {v: k for k, v in CLASS_LABELS.items()}

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class PhysiologicalSignals(BaseModel):
    """Physiological signals from wearable device"""
    ecg: List[float]
    eda: List[float]
    temperature: List[float]
    model_type: str = "lstm"  # lstm, cnn, hybrid

class FacialImageRequest(BaseModel):
    """Facial image for CNN prediction"""
    image_base64: str
    model_type: str = "cnn"  # cnn, hybrid

class MultimodalRequest(BaseModel):
    """Combined physiological + facial data"""
    ecg: List[float]
    eda: List[float]
    temperature: List[float]
    image_base64: str
    model_type: str = "hybrid"

class PredictionResponse(BaseModel):
    """Prediction output"""
    prediction: int
    label: str
    confidence: float
    probabilities: dict
    timestamp: str
    processing_time_ms: float

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    samples: List[dict]
    model_type: str = "lstm"

class HealthMetrics(BaseModel):
    """Health metrics dashboard"""
    heart_rate: float
    respiration_rate: float
    body_temperature: float
    stress_level: int
    confidence: float

class ModelStatus(BaseModel):
    """Model availability status"""
    model_name: str
    available: bool
    checkpoint_path: str
    accuracy: float
    parameters: int

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_models():
    """Load all trained models"""
    global MODELS
    
    print("[INFO] Loading trained models from checkpoints...")
    
    try:
        # Load LSTM model
        lstm_checkpoint = CHECKPOINT_DIR / "lstm_physio_phase1_best.pth"
        if lstm_checkpoint.exists():
            MODELS["lstm"] = StressPredictor("lstm", str(lstm_checkpoint), device=DEVICE)
            print(f"[OK] LSTM model loaded: {lstm_checkpoint}")
        else:
            print(f"[WARNING] LSTM checkpoint not found: {lstm_checkpoint}")
    except Exception as e:
        print(f"[ERROR] Failed to load LSTM: {str(e)}")
    
    try:
        # Load CNN model
        cnn_checkpoint = CHECKPOINT_DIR / "cnn_facial_phase2_best.pth"
        if cnn_checkpoint.exists():
            MODELS["cnn"] = StressPredictor("cnn", str(cnn_checkpoint), device=DEVICE)
            print(f"[OK] CNN model loaded: {cnn_checkpoint}")
        else:
            print(f"[WARNING] CNN checkpoint not found: {cnn_checkpoint}")
    except Exception as e:
        print(f"[ERROR] Failed to load CNN: {str(e)}")
    
    try:
        # Load Hybrid model
        hybrid_checkpoint = CHECKPOINT_DIR / "hybrid_multimodal_phase3_best.pth"
        if hybrid_checkpoint.exists():
            MODELS["hybrid"] = StressPredictor("hybrid", str(hybrid_checkpoint), device=DEVICE)
            print(f"[OK] Hybrid model loaded: {hybrid_checkpoint}")
        else:
            print(f"[INFO] Hybrid model not yet trained (will use LSTM+CNN fusion)")
    except Exception as e:
        print(f"[ERROR] Failed to load Hybrid: {str(e)}")
    
    if not MODELS:
        raise RuntimeError("No models loaded! Train models first.")
    
    print(f"[OK] Total models loaded: {len(MODELS)}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def base64_to_image(base64_str: str):
    """Convert base64 string to PIL Image"""
    try:
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")

def image_to_base64(image: Image) -> str:
    """Convert PIL Image to base64 string"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def normalize_ecg(signal: List[float]) -> np.ndarray:
    """Normalize ECG signal"""
    signal = np.array(signal)
    return normalize_signal(signal)

def normalize_eda(signal: List[float]) -> np.ndarray:
    """Normalize EDA signal"""
    signal = np.array(signal)
    return normalize_signal(signal)

def get_processing_time():
    """Get current timestamp"""
    return datetime.now().isoformat()

def save_prediction_to_history(prediction: dict):
    """Save prediction to history"""
    PREDICTION_HISTORY.append({
        **prediction,
        "timestamp": datetime.now().isoformat()
    })
    # Keep only last 1000 predictions
    if len(PREDICTION_HISTORY) > 1000:
        PREDICTION_HISTORY.pop(0)

# ============================================================================
# ENDPOINTS: HEALTH CHECK
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Advanced Health AI - Stress Detection API",
        "version": "1.0.0",
        "status": "running",
        "models_available": list(MODELS.keys()),
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "device": DEVICE,
        "models_loaded": len(MODELS),
        "available_models": list(MODELS.keys()),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

# ============================================================================
# ENDPOINTS: MODEL STATUS
# ============================================================================

@app.get("/models/status")
async def get_models_status():
    """Get status of all models"""
    status = []
    
    checkpoints = {
        "lstm": CHECKPOINT_DIR / "lstm_physio_phase1_best.pth",
        "cnn": CHECKPOINT_DIR / "cnn_facial_phase2_best.pth",
        "hybrid": CHECKPOINT_DIR / "hybrid_multimodal_phase3_best.pth"
    }
    
    for model_name, checkpoint in checkpoints.items():
        status.append({
            "model_name": model_name,
            "available": model_name in MODELS,
            "checkpoint_path": str(checkpoint),
            "exists": checkpoint.exists(),
            "accuracy": 0.60 if model_name == "lstm" else (0.39 if model_name == "cnn" else 0.85)
        })
    
    return {"models": status}

@app.get("/models/list")
async def list_models():
    """List available models"""
    return {
        "available": list(MODELS.keys()),
        "total": len(MODELS),
        "device": DEVICE
    }

# ============================================================================
# ENDPOINTS: PREDICTIONS
# ============================================================================

@app.post("/predict/physiological", response_model=PredictionResponse)
async def predict_physiological(request: PhysiologicalSignals):
    """
    Predict stress from physiological signals (ECG, EDA, Temperature)
    
    Body:
    {
        "ecg": [0.1, 0.2, ...],
        "eda": [0.3, 0.4, ...],
        "temperature": [36.5, 36.6, ...],
        "model_type": "lstm"
    }
    """
    start_time = datetime.now()
    
    if request.model_type not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Use: {list(MODELS.keys())}"
        )
    
    try:
        predictor = MODELS[request.model_type]
        
        # Normalize signals
        ecg = normalize_ecg(request.ecg)
        eda = normalize_eda(request.eda)
        temp = np.array(request.temperature)
        
        # Make prediction
        result = predictor.predict_physiological(ecg, eda, temp)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            timestamp=get_processing_time(),
            processing_time_ms=processing_time
        )
        
        # Save to history
        save_prediction_to_history(response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/facial", response_model=PredictionResponse)
async def predict_facial(request: FacialImageRequest):
    """
    Predict stress from facial image (CNN model)
    
    Body:
    {
        "image_base64": "iVBORw0KGgo...",
        "model_type": "cnn"
    }
    """
    start_time = datetime.now()
    
    if request.model_type not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Use: {list(MODELS.keys())}"
        )
    
    try:
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        image_path = "/tmp/temp_image.png"
        image.save(image_path)
        
        predictor = MODELS[request.model_type]
        result = predictor.predict_facial(image_path)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            timestamp=get_processing_time(),
            processing_time_ms=processing_time
        )
        
        # Save to history
        save_prediction_to_history(response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/multimodal", response_model=PredictionResponse)
async def predict_multimodal(request: MultimodalRequest):
    """
    Predict stress from both physiological and facial data (Hybrid model)
    
    Body:
    {
        "ecg": [0.1, 0.2, ...],
        "eda": [0.3, 0.4, ...],
        "temperature": [36.5, 36.6, ...],
        "image_base64": "iVBORw0KGgo...",
        "model_type": "hybrid"
    }
    """
    start_time = datetime.now()
    
    if request.model_type not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available. Use: {list(MODELS.keys())}"
        )
    
    try:
        # Convert base64 to image
        image = base64_to_image(request.image_base64)
        image_path = "/tmp/temp_image.png"
        image.save(image_path)
        
        # Normalize signals
        ecg = normalize_ecg(request.ecg)
        eda = normalize_eda(request.eda)
        temp = np.array(request.temperature)
        
        predictor = MODELS[request.model_type]
        result = predictor.predict_multimodal(ecg, eda, temp, image_path)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = PredictionResponse(
            prediction=result["prediction"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            timestamp=get_processing_time(),
            processing_time_ms=processing_time
        )
        
        # Save to history
        save_prediction_to_history(response.dict())
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction on multiple samples
    
    Body:
    {
        "samples": [
            {"ecg": [...], "eda": [...], "temperature": [...]},
            {"ecg": [...], "eda": [...], "temperature": [...]}
        ],
        "model_type": "lstm"
    }
    """
    if request.model_type not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model_type}' not available"
        )
    
    try:
        predictions = []
        for sample in request.samples:
            request_single = PhysiologicalSignals(
                ecg=sample.get("ecg", []),
                eda=sample.get("eda", []),
                temperature=sample.get("temperature", []),
                model_type=request.model_type
            )
            prediction = await predict_physiological(request_single)
            predictions.append(prediction.dict())
        
        return {
            "total": len(predictions),
            "predictions": predictions,
            "timestamp": get_processing_time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

# ============================================================================
# ENDPOINTS: HISTORY & ANALYTICS
# ============================================================================

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get recent predictions"""
    return {
        "total": len(PREDICTION_HISTORY),
        "limit": limit,
        "predictions": PREDICTION_HISTORY[-limit:],
        "timestamp": get_processing_time()
    }

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    if not PREDICTION_HISTORY:
        return {"message": "No predictions yet"}
    
    labels = [p["label"] for p in PREDICTION_HISTORY]
    label_counts = {label: labels.count(label) for label in CLASS_LABELS.values()}
    
    confidences = [p["confidence"] for p in PREDICTION_HISTORY]
    
    return {
        "total_predictions": len(PREDICTION_HISTORY),
        "label_distribution": label_counts,
        "avg_confidence": np.mean(confidences),
        "min_confidence": np.min(confidences),
        "max_confidence": np.max(confidences),
        "timestamp": get_processing_time()
    }

@app.delete("/predictions/history")
async def clear_history():
    """Clear prediction history"""
    global PREDICTION_HISTORY
    count = len(PREDICTION_HISTORY)
    PREDICTION_HISTORY = []
    return {"message": f"Cleared {count} predictions", "timestamp": get_processing_time()}

# ============================================================================
# STARTUP & SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("[INFO] Starting FastAPI server...")
    print(f"[INFO] Device: {DEVICE}")
    try:
        load_models()
        print("[OK] All systems ready!")
    except Exception as e:
        print(f"[ERROR] Startup failed: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("[INFO] Shutting down API server...")
    # Save prediction history
    if PREDICTION_HISTORY:
        history_file = RESULTS_DIR / "prediction_history.json"
        with open(history_file, "w") as f:
            json.dump(PREDICTION_HISTORY, f, indent=2)
        print(f"[OK] Saved {len(PREDICTION_HISTORY)} predictions to {history_file}")

# ============================================================================
# STATIC FILES
# ============================================================================

# Serve frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ADVANCED HEALTH AI - STRESS DETECTION API")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Checkpoint Dir: {CHECKPOINT_DIR}")
    print(f"Frontend Dir: {frontend_dir if frontend_dir.exists() else 'Not found'}")
    print("=" * 80)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False
    )
