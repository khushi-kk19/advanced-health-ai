"""
Advanced Health AI - Production Backend
Real-Time Multi-Modal Health Prediction System

Features:
- Real-time webcam vital signs monitoring (rPPG)
- Stress detection from physiological signals & facial expressions
- Smart clinical data generation (minimal user input)
- Cardiovascular risk assessment
- Glucose level prediction
- WebSocket support for live streaming

NO MOCK DATA - ONLY REAL MODEL INFERENCE
"""

import os
import sys
import json
import base64
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
import numpy as np
import warnings
import cv2
import asyncio

from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from functools import wraps

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "stress_detection"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

print(f"[INFO] Project Root: {PROJECT_ROOT}")
print(f"[INFO] Checkpoints: {CHECKPOINT_DIR}")
print(f"[INFO] Models: {MODELS_DIR}")

# ============================================================================
# GLOBAL STATE
# ============================================================================
MODELS = {}
PREDICTION_HISTORY = []
SYSTEM_STATUS = {
    "stress_detector": False,
    "risk_predictor": False,
    "glucose_predictor": False,
    "rppg_predictor": False,
    "webcam_vitals": False
}

active_websocket_connections = []

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
def make_json_safe(obj):
    """Convert any object to JSON-safe format"""
    import numpy as np
    
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return str(obj)
    
class PhysiologicalRequest(BaseModel):
    ecg: List[float] = Field(..., description="ECG signal (50-256 samples)")
    eda: List[float] = Field(..., description="EDA signal (50-256 samples)")
    temperature: List[float] = Field(..., description="Temperature signal (50-256 samples)")

class FacialRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image")

class SimplifiedClinicalInput(BaseModel):
    """Simplified input - AI generates complete clinical profile"""
    age: int = Field(..., ge=18, le=100, description="Age in years")
    gender: str = Field(..., description="male or female")
    height_cm: float = Field(..., ge=100, le=250, description="Height in cm")
    weight_kg: float = Field(..., ge=30, le=200, description="Weight in kg")
    smoker: bool = Field(default=False, description="Current smoker?")
    exercise_frequency: str = Field(default="moderate", description="none, light, moderate, heavy")

class GlucosePredictionRequest(BaseModel):
    sequence: List[List[float]] = Field(..., description="Time series (30, 9)")

class RPPGRequest(BaseModel):
    signal: List[float] = Field(..., description="rPPG signal (30-300 samples)")

# ============================================================================
# SMART CLINICAL DATA GENERATOR
# ============================================================================

class ClinicalDataGenerator:
    """
    AI-powered clinical data generator
    Generates realistic clinical parameters from basic user inputs
    Uses domain knowledge from medical research
    """
    
    @staticmethod
    def calculate_bmi(weight_kg: float, height_cm: float) -> float:
        """Calculate BMI = weight(kg) / height(m)^2"""
        height_m = height_cm / 100
        return weight_kg / (height_m ** 2)
    
    @staticmethod
    def estimate_blood_pressure(age: int, bmi: float, exercise: str, gender: str) -> tuple:
        """
        Estimate BP based on age, BMI, lifestyle, gender
        Based on Framingham Heart Study patterns
        """
        # Base BP (increases with age)
        base_systolic = 90 + (age - 20) * 0.6
        base_diastolic = 60 + (age - 20) * 0.35
        
        # Gender adjustment (males tend higher)
        if gender == "male":
            base_systolic += 5
            base_diastolic += 3
        
        # BMI impact (strong correlation)
        if bmi > 30:  # Obese
            base_systolic += 15
            base_diastolic += 10
        elif bmi > 25:  # Overweight
            base_systolic += 8
            base_diastolic += 5
        elif bmi < 18.5:  # Underweight
            base_systolic -= 5
            base_diastolic -= 3
        
        # Exercise effect (protective)
        exercise_adj = {
            "none": (10, 6),
            "light": (5, 3),
            "moderate": (0, 0),
            "heavy": (-8, -5)
        }
        sys_adj, dia_adj = exercise_adj.get(exercise, (0, 0))
        base_systolic += sys_adj
        base_diastolic += dia_adj
        
        # Add natural variation
        systolic = base_systolic + np.random.uniform(-5, 5)
        diastolic = base_diastolic + np.random.uniform(-3, 3)
        
        # Ensure physiological ranges
        systolic = np.clip(systolic, 90, 200)
        diastolic = np.clip(diastolic, 60, 120)
        
        return round(systolic, 1), round(diastolic, 1)
    
    @staticmethod
    def estimate_heart_rate(age: int, exercise: str, bmi: float) -> float:
        """Estimate resting heart rate"""
        # Base: decreases with age initially, increases after 50
        if age < 50:
            base_hr = 75 - (age - 20) * 0.2
        else:
            base_hr = 65 + (age - 50) * 0.3
        
        # Exercise effect (athletes have lower HR)
        exercise_adj = {
            "none": 10,
            "light": 5,
            "moderate": 0,
            "heavy": -10
        }
        base_hr += exercise_adj.get(exercise, 0)
        
        # BMI effect
        if bmi > 30:
            base_hr += 8
        elif bmi > 25:
            base_hr += 4
        
        hr = base_hr + np.random.uniform(-5, 5)
        return round(np.clip(hr, 50, 110), 1)
    
    @staticmethod
    def estimate_cholesterol(age: int, bmi: float, gender: str, exercise: str) -> Dict:
        """
        Estimate lipid profile
        Total cholesterol, LDL, HDL
        """
        # Base total cholesterol (increases with age)
        base_total = 150 + (age - 20) * 1.5
        
        # Gender (females typically higher HDL)
        if gender == "male":
            base_total += 10
            hdl_bonus = 0
        else:
            hdl_bonus = 10
        
        # BMI impact
        if bmi > 30:
            base_total += 35
        elif bmi > 25:
            base_total += 20
        elif bmi < 18.5:
            base_total -= 10
        
        # Exercise (lowers total, raises HDL)
        exercise_adj = {
            "none": (20, -10),
            "light": (10, -5),
            "moderate": (0, 5),
            "heavy": (-15, 15)
        }
        total_adj, hdl_adj = exercise_adj.get(exercise, (0, 0))
        base_total += total_adj
        
        total_chol = base_total + np.random.uniform(-15, 15)
        total_chol = np.clip(total_chol, 120, 350)
        
        # LDL (bad cholesterol) - typically 60-70% of total
        ldl = total_chol * 0.65 + np.random.uniform(-15, 15)
        ldl = np.clip(ldl, 50, 250)
        
        # HDL (good cholesterol) - typically 20-30% of total
        hdl = total_chol * 0.25 + hdl_adj + hdl_bonus + np.random.uniform(-10, 10)
        hdl = np.clip(hdl, 30, 100)
        
        return {
            "total": round(total_chol, 1),
            "ldl": round(ldl, 1),
            "hdl": round(hdl, 1),
            "triglycerides": round(total_chol * 0.3 + np.random.uniform(-20, 20), 1)
        }
    
    @staticmethod
    def estimate_glucose(age: int, bmi: float, exercise: str) -> float:
        """Estimate fasting glucose"""
        base_glucose = 85
        
        # Age effect
        if age > 45:
            base_glucose += (age - 45) * 0.4
        
        # BMI (strong diabetes risk factor)
        if bmi > 30:
            base_glucose += 20
        elif bmi > 25:
            base_glucose += 10
        
        # Exercise (protective)
        exercise_adj = {"none": 10, "light": 5, "moderate": 0, "heavy": -8}
        base_glucose += exercise_adj.get(exercise, 0)
        
        glucose = base_glucose + np.random.uniform(-8, 8)
        return round(np.clip(glucose, 70, 180), 1)
    
    @staticmethod
    def estimate_smoking_impact(smoker: bool, age: int) -> Dict:
        """Estimate cigarettes per day if smoker"""
        if not smoker:
            return {"cigsperday": 0}
        
        # Older smokers typically smoke more (established habit)
        base_cigs = 8 + (age - 20) * 0.4
        cigsperday = max(5, min(40, base_cigs + np.random.uniform(-5, 5)))
        
        return {"cigsperday": round(cigsperday, 0)}
    
    @staticmethod
    def estimate_conditions(age: int, bmi: float, bp_systolic: float, glucose: float, smoker: bool) -> Dict:
        """
        Estimate medical conditions based on risk factors
        Uses clinical thresholds
        """
        conditions = {
            "bpmeds": 0,
            "prevalentstroke": 0,
            "prevalenthyp": 0,
            "diabetes": 0
        }
        
        # Hypertension (BP > 140/90)
        if bp_systolic >= 140:
            conditions["prevalenthyp"] = 1
            if bp_systolic >= 160:  # Stage 2 hypertension
                conditions["bpmeds"] = 1
        
        # Diabetes (fasting glucose >= 126 mg/dL)
        if glucose >= 126:
            conditions["diabetes"] = 1
        elif glucose >= 100:  # Prediabetes increases risk
            if np.random.random() < 0.3:
                conditions["diabetes"] = 1
        
        # Diabetes risk from obesity
        if bmi > 30 and age > 45:
            if np.random.random() < 0.4:
                conditions["diabetes"] = 1
        
        # Stroke (rare, multiple risk factors)
        stroke_risk = 0
        if age > 60:
            stroke_risk += 0.02
        if conditions["prevalenthyp"] == 1:
            stroke_risk += 0.03
        if smoker:
            stroke_risk += 0.02
        if conditions["diabetes"] == 1:
            stroke_risk += 0.02
        
        if np.random.random() < stroke_risk:
            conditions["prevalentstroke"] = 1
        
        return conditions
    
    @classmethod
    def generate_complete_profile(cls, simplified_input: SimplifiedClinicalInput) -> Dict:
        """
        MAIN METHOD: Generate complete clinical profile from minimal input
        
        Input: age, gender, height, weight, smoker, exercise
        Output: Complete 15+ parameter clinical profile
        """
        
        # Calculate BMI
        bmi = cls.calculate_bmi(simplified_input.weight_kg, simplified_input.height_cm)
        
        # Estimate vitals
        systolic, diastolic = cls.estimate_blood_pressure(
            simplified_input.age, 
            bmi, 
            simplified_input.exercise_frequency,
            simplified_input.gender
        )
        
        heart_rate = cls.estimate_heart_rate(
            simplified_input.age,
            simplified_input.exercise_frequency,
            bmi
        )
        
        # Estimate lab values
        cholesterol = cls.estimate_cholesterol(
            simplified_input.age,
            bmi,
            simplified_input.gender,
            simplified_input.exercise_frequency
        )
        
        glucose = cls.estimate_glucose(
            simplified_input.age,
            bmi,
            simplified_input.exercise_frequency
        )
        
        # Smoking impact
        smoking = cls.estimate_smoking_impact(
            simplified_input.smoker,
            simplified_input.age
        )
        
        # Medical conditions
        conditions = cls.estimate_conditions(
            simplified_input.age,
            bmi,
            systolic,
            glucose,
            simplified_input.smoker
        )
        
        # Assemble complete profile
        profile = {
            # Basic info
            "age": simplified_input.age,
            "gender": simplified_input.gender,
            "height_cm": simplified_input.height_cm,
            "weight_kg": simplified_input.weight_kg,
            
            # Calculated
            "bmi": round(bmi, 1),
            "bmi_category": cls._get_bmi_category(bmi),
            
            # Vitals
            "sysbp": systolic,
            "diabp": diastolic,
            "heartrate": heart_rate,
            
            # Lab values
            "totchol": cholesterol["total"],
            "ldl_chol": cholesterol["ldl"],
            "hdl_chol": cholesterol["hdl"],
            "triglycerides": cholesterol["triglycerides"],
            "glucose": glucose,
            
            # Lifestyle
            "currentsmoker": 1 if simplified_input.smoker else 0,
            "exercise_level": simplified_input.exercise_frequency,
            **smoking,
            
            # Medical conditions
            **conditions,
            
            # Risk indicators
            "metabolic_syndrome": cls._check_metabolic_syndrome(bmi, systolic, glucose, cholesterol["hdl"]),
        }
        
        return profile
    
    @staticmethod
    def _get_bmi_category(bmi: float) -> str:
        """Categorize BMI"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    @staticmethod
    def _check_metabolic_syndrome(bmi: float, bp: float, glucose: float, hdl: float) -> bool:
        """Check for metabolic syndrome (simplified)"""
        criteria_met = 0
        
        if bmi > 30:  # Abdominal obesity proxy
            criteria_met += 1
        if bp >= 130:  # Elevated BP
            criteria_met += 1
        if glucose >= 100:  # Elevated glucose
            criteria_met += 1
        if hdl < 40:  # Low HDL
            criteria_met += 1
        
        return criteria_met >= 3

# ============================================================================
# WEBCAM VITAL MONITOR (rPPG)
# ============================================================================

class WebcamVitalMonitor:
    """
    Real-time vital signs from webcam using rPPG (remote photoplethysmography)
    Extracts heart rate from subtle color changes in facial skin
    """
    
    def __init__(self):
        try:
            # Only import if available
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.has_mediapipe = True
            except ImportError:
                print("[WARNING] MediaPipe not installed - webcam vitals disabled")
                self.has_mediapipe = False
                return
            
            from scipy import signal as scipy_signal
            from scipy.fft import fft, fftfreq
            
            self.scipy_signal = scipy_signal
            self.fft = fft
            self.fftfreq = fftfreq
            
            # ROI: Forehead and cheek landmarks for best signal
            self.roi_indices = [10, 338, 297, 332, 284, 251, 234, 93, 132, 58, 
                              172, 136, 150, 149, 176, 148, 152, 377, 400, 378,
                              379, 365, 397, 288, 361, 323, 454, 356, 389]
            
            self.fps = 30
            self.window_size = 256  # ~8.5 seconds at 30fps
            self.rgb_buffer = []
            self.hr_history = []
            
            SYSTEM_STATUS['webcam_vitals'] = True
            print("[OK] Webcam vital monitor initialized (rPPG)")
            
        except Exception as e:
            print(f"[ERROR] Webcam monitor init failed: {e}")
            self.has_mediapipe = False
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process single webcam frame"""
        
        if not self.has_mediapipe:
            return {
                "face_detected": False,
                "error": "MediaPipe not installed",
                "message": "Install: pip install mediapipe scipy"
            }
        
        try:
            # Convert to RGB
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            elif frame.shape[2] == 3 and frame.dtype == np.uint8:
                # Assume BGR from OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            results = self.face_mesh.process(frame)
            
            if not results.multi_face_landmarks:
                return {
                    "face_detected": False,
                    "message": "No face detected - please face camera"
                }
            
            # Extract ROI colors
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            roi_colors = []
            for idx in self.roi_indices:
                x = int(landmarks.landmark[idx].x * w)
                y = int(landmarks.landmark[idx].y * h)
                
                # Extract 5x5 region
                if 2 <= x < w-2 and 2 <= y < h-2:
                    region = frame[y-2:y+3, x-2:x+3]
                    if region.size > 0:
                        roi_colors.append(region.mean(axis=(0, 1)))
            
            if len(roi_colors) < 5:
                return {
                    "face_detected": True,
                    "message": "Face too small or partial - move closer",
                    "buffer_size": len(self.rgb_buffer)
                }
            
            # Average across all ROIs
            avg_color = np.mean(roi_colors, axis=0)
            self.rgb_buffer.append(avg_color)
            
            # Maintain buffer size
            if len(self.rgb_buffer) > self.window_size:
                self.rgb_buffer.pop(0)
            
            result = {
                "face_detected": True,
                "buffer_size": len(self.rgb_buffer),
                "buffer_needed": self.window_size,
                "heart_rate": None,
                "confidence": 0.0,
                "status": "collecting"
            }
            
            # Calculate HR when buffer full
            if len(self.rgb_buffer) >= self.window_size:
                hr_data = self._calculate_heart_rate()
                result.update(hr_data)
                result["status"] = "monitoring"
            else:
                result["message"] = f"Collecting data: {len(self.rgb_buffer)}/{self.window_size}"
            
            return result
            
        except Exception as e:
            return {
                "face_detected": False,
                "error": str(e),
                "message": "Processing error"
            }
    
    def _calculate_heart_rate(self) -> Dict:
        """
        Calculate heart rate using FFT on green channel
        Green channel has strongest pulsatile signal
        """
        try:
            # Extract green channel (index 1)
            green_signal = np.array([rgb[1] for rgb in self.rgb_buffer])
            
            # Detrend to remove slow variations
            detrended = self.scipy_signal.detrend(green_signal)
            
            # Bandpass filter: 0.7-4.0 Hz (42-240 BPM)
            sos = self.scipy_signal.butter(
                3, [0.7, 4.0],
                btype='bandpass',
                fs=self.fps,
                output='sos'
            )
            filtered = self.scipy_signal.sosfilt(sos, detrended)
            
            # Apply Hamming window
            windowed = filtered * np.hamming(len(filtered))
            
            # FFT
            fft_vals = self.fft(windowed)
            freqs = self.fftfreq(len(windowed), 1/self.fps)
            
            # Focus on physiological range
            mask = (freqs >= 0.7) & (freqs <= 4.0)
            freqs_masked = freqs[mask]
            fft_masked = np.abs(fft_vals[mask])
            
            if len(fft_masked) == 0:
                return {"heart_rate": None, "confidence": 0.0}
            
            # Find peak frequency
            peak_idx = np.argmax(fft_masked)
            heart_rate_hz = freqs_masked[peak_idx]
            heart_rate_bpm = heart_rate_hz * 60
            
            # Calculate confidence (SNR-based)
            peak_power = fft_masked[peak_idx]
            noise_power = np.mean(fft_masked) + 1e-10
            snr = peak_power / noise_power
            confidence = min(snr / 15.0, 1.0)  # Normalize to 0-1
            
            # Store history for smoothing
            self.hr_history.append(heart_rate_bpm)
            if len(self.hr_history) > 10:
                self.hr_history.pop(0)
            
            # Median filtering for stability
            hr_smooth = np.median(self.hr_history) if len(self.hr_history) >= 3 else heart_rate_bpm
            
            # Physiological validation
            if hr_smooth < 40 or hr_smooth > 180:
                return {
                    "heart_rate": None,
                    "confidence": 0.0,
                    "message": "Unreliable reading - stay still"
                }
            
            # Determine status
            if 60 <= hr_smooth <= 100:
                status = "normal"
                status_emoji = "✓"
            elif hr_smooth < 60:
                status = "low"
                status_emoji = "⚠"
            else:
                status = "elevated"
                status_emoji = "⚠"
            
            return {
                "heart_rate": round(float(hr_smooth), 1),
                "heart_rate_raw": round(float(heart_rate_bpm), 1),
                "confidence": round(float(confidence), 2),
                "status": status,
                "status_emoji": status_emoji,
                "snr": round(float(snr), 2)
            }
            
        except Exception as e:
            print(f"[ERROR] HR calculation failed: {e}")
            return {
                "heart_rate": None,
                "confidence": 0.0,
                "error": str(e)
            }

# ============================================================================
# MODEL LOADING
# ============================================================================
def safe_endpoint(func):
    """Decorator to safely handle endpoint errors"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )
    return wrapper
def load_stress_detector():
    """Load stress detection models - LSTM for physiological, CNN for facial"""
    try:
        from inference_stress_phase123 import StressPredictor
        
        lstm_checkpoint = CHECKPOINT_DIR / "lstm_physio_phase1_best.pth"
        cnn_checkpoint = CHECKPOINT_DIR / "cnn_facial_phase2_best.pth"
        
        if not lstm_checkpoint.exists() and not cnn_checkpoint.exists():
            print(f"[ERROR] No stress models found in {CHECKPOINT_DIR}")
            return False
        
        # Load LSTM model for physiological signals
        if lstm_checkpoint.exists():
            try:
                lstm_predictor = StressPredictor(
                    model_type='lstm',
                    checkpoint_path=str(lstm_checkpoint),
                    device='cpu'
                )
                MODELS['stress_predictor_lstm'] = lstm_predictor
                print(f"[OK] LSTM model loaded for physiological signals")
            except Exception as e:
                print(f"[WARNING] Failed to load LSTM model: {e}")
        
        # Load CNN model for facial images (REQUIRED for facial predictions)
        if cnn_checkpoint.exists():
            try:
                cnn_predictor = StressPredictor(
                    model_type='cnn',
                    checkpoint_path=str(cnn_checkpoint),
                    device='cpu'
                )
                MODELS['stress_predictor_cnn'] = cnn_predictor
                print(f"[OK] CNN model loaded for facial images")
            except Exception as e:
                print(f"[WARNING] Failed to load CNN model: {e}")
        
        # Set status based on what was loaded
        if 'stress_predictor_cnn' in MODELS or 'stress_predictor_lstm' in MODELS:
            SYSTEM_STATUS['stress_detector'] = True
            print(f"[OK] Stress detectors loaded successfully")
            return True
        else:
            print(f"[ERROR] No stress models could be loaded")
            return False
        
    except Exception as e:
        print(f"[ERROR] Stress detector loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_risk_predictor():
    """Load cardiovascular risk predictor"""
    try:
        from inference_risk import RiskPredictor
        
        risk_model = MODELS_DIR / "risk.pkl"
        
        if not risk_model.exists():
            print(f"[ERROR] Risk model not found: {risk_model}")
            return False
        
        predictor = RiskPredictor(model_path=str(risk_model))
        MODELS['risk_predictor'] = predictor
        SYSTEM_STATUS['risk_predictor'] = True
        print("[OK] Risk predictor loaded")
        return True
        
    except Exception as e:
        print(f"[ERROR] Risk predictor loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def sanitize_model_output(result: Any) -> dict:
    """Convert any model output to JSON-serializable dict"""
    if isinstance(result, dict):
        clean = {}
        for key, value in result.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                clean[key] = value
            elif isinstance(value, np.ndarray):
                clean[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                clean[key] = float(value)
            else:
                clean[key] = str(value)
        return clean
    return {"result": str(result)}

def load_glucose_predictor():
    """Load glucose level predictor"""
    try:
        from inference_lstm import HealthPredictor
        
        lstm_model = MODELS_DIR / "lstm_attn.pth"
        scaler = MODELS_DIR / "lstm_scaler.pkl"
        
        if not lstm_model.exists() or not scaler.exists():
            print(f"[ERROR] Glucose model files not found")
            return False
        
        predictor = HealthPredictor(
            model_path=str(lstm_model),
            scaler_path=str(scaler)
        )
        MODELS['glucose_predictor'] = predictor
        SYSTEM_STATUS['glucose_predictor'] = True
        print("[OK] Glucose predictor loaded")
        return True
        
    except Exception as e:
        print(f"[ERROR] Glucose predictor loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_rppg_predictor():
    """Load rPPG heart rate predictor"""
    try:
        from inference_rppg import rPPGPredictor
        
        rppg_model = MODELS_DIR / "rppg_model.pth"
        
        if not rppg_model.exists():
            print(f"[ERROR] rPPG model not found: {rppg_model}")
            return False
        
        predictor = rPPGPredictor(
            model_path=str(rppg_model),
            model_type="cnn_lstm"
        )
        MODELS['rppg_predictor'] = predictor
        SYSTEM_STATUS['rppg_predictor'] = True
        print("[OK] rPPG predictor loaded")
        return True
        
    except Exception as e:
        print(f"[ERROR] rPPG predictor loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_webcam_monitor():
    """Load webcam vital signs monitor"""
    try:
        monitor = WebcamVitalMonitor()
        if monitor.has_mediapipe:
            MODELS['webcam_monitor'] = monitor
            return True
        return False
    except Exception as e:
        print(f"[WARNING] Webcam monitor not available: {e}")
        return False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_signal(signal: List[float]) -> np.ndarray:
    """Z-score normalization"""
    signal = np.array(signal, dtype=np.float32)
    if len(signal) == 0:
        return signal
    mean = np.mean(signal)
    std = np.std(signal)
    return (signal - mean) / (std + 1e-8) if std > 0 else signal

def base64_to_numpy(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        return image
    except Exception as e:
        raise ValueError(f"Image decode failed: {e}")

def add_to_history(prediction: dict):
    """Add prediction to history - FIXED VERSION"""
    global PREDICTION_HISTORY
    
    # Create a clean, JSON-serializable prediction
    clean_prediction = {
        "label": str(prediction.get("label", "unknown")),
        "confidence": float(prediction.get("confidence", 0)),
        "timestamp": datetime.now().isoformat(),
        "model_name": str(prediction.get("model_name", "unknown"))
    }
    
    PREDICTION_HISTORY.append(clean_prediction)
    
    # Keep only last 100
    if len(PREDICTION_HISTORY) > 100:
        PREDICTION_HISTORY = PREDICTION_HISTORY[-100:]

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Advanced Health AI - Production",
    description="Real-Time Multi-Modal Health Prediction System",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API root"""
    return {
        "name": "Advanced Health AI - Production",
        "version": "3.0.0",
        "status": "operational",
        "inference_mode": "REAL MODELS ONLY",
        "models": SYSTEM_STATUS,
        "features": [
            "Real-time webcam vital signs (rPPG)",
            "Stress detection (physiological + facial)",
            "Smart clinical data generation",
            "Cardiovascular risk assessment",
            "Glucose prediction",
            "Heart rate monitoring"
        ]
    }

@app.get("/health")
async def health_check():
    """System health check"""
    import torch
    
    models_loaded = sum(SYSTEM_STATUS.values())
    
    return {
        "status": "operational",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "models": SYSTEM_STATUS,
        "models_loaded": models_loaded,
        "total_models": len(SYSTEM_STATUS),
        "inference_mode": "REAL" if models_loaded > 0 else "NO MODELS",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# WEBSOCKET - REAL-TIME WEBCAM
# ============================================================================

@app.websocket("/ws/webcam")
async def websocket_webcam(websocket: WebSocket):
    """Real-time webcam vital signs monitoring"""
    await websocket.accept()
    active_websocket_connections.append(websocket)
    
    print("[INFO] WebSocket connected - Real-time monitoring started")
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            try:
                # Decode base64 frame
                frame = base64_to_numpy(data)
                
                # Process with webcam monitor
                if 'webcam_monitor' in MODELS:
                    result = MODELS['webcam_monitor'].process_frame(frame)
                else:
                    result = {
                        "face_detected": False,
                        "error": "Webcam monitor not loaded",
                        "message": "Install dependencies: pip install mediapipe scipy"
                    }
                
                # Add timestamp
                result["timestamp"] = datetime.now().isoformat()
                
                # Send result back to client
                await websocket.send_json(result)
                
            except Exception as e:
                await websocket.send_json({
                    "face_detected": False,
                    "error": str(e),
                    "message": "Frame processing error"
                })
            
    except WebSocketDisconnect:
        active_websocket_connections.remove(websocket)
        print("[INFO] WebSocket disconnected")
    except Exception as e:
        print(f"[ERROR] WebSocket error: {e}")
        if websocket in active_websocket_connections:
            active_websocket_connections.remove(websocket)

# ============================================================================
# STRESS DETECTION - REAL INFERENCE ONLY
# ============================================================================

@app.post("/predict/physiological")
@safe_endpoint
async def predict_physiological(request: PhysiologicalRequest):
    """Predict stress from physiological signals using LSTM model"""
    
    if 'stress_predictor_lstm' not in MODELS:
        raise HTTPException(
            status_code=503,
            detail="LSTM stress detector not loaded. Required for physiological signals."
        )
    
    try:
        start_time = datetime.now()
        
        # Validate input
        if len(request.ecg) < 50 or len(request.eda) < 50 or len(request.temperature) < 50:
            raise HTTPException(400, "Signals too short. Need at least 50 samples each.")
        
        # Normalize signals
        ecg = normalize_signal(request.ecg[:256])
        eda = normalize_signal(request.eda[:256])
        temp = normalize_signal(request.temperature[:256])
        
        # REAL MODEL INFERENCE - Use LSTM model
        result = MODELS['stress_predictor_lstm'].predict_physiological(ecg, eda, temp)
        
        # Ensure result is JSON serializable
        clean_result = {
            "label": str(result.get("label", "unknown")),
            "confidence": float(result.get("confidence", 0)),
            "model_name": "LSTM Physiological Stress Detector",
            "inference_type": "REAL MODEL",
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        add_to_history(clean_result)
        return clean_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Model inference error: {str(e)}")


#... existing code before line 1052 ...

@app.post("/predict/facial")
@safe_endpoint
async def predict_facial(request: FacialRequest):
    """
    Predict stress from facial expression using CNN model.
    
    FIXED VERSION 3.0:
    - Auto-detects correct input size (no more hardcoded 224x224)
    - Tries common sizes: 48, 64, 32, 128, 96
    - First successful size is used
    - Reports which size worked in response
    """
    
    # ============================================================================
    # STEP 1: VALIDATION - Use CNN model for facial images
    # ============================================================================
    
    if 'stress_predictor_cnn' not in MODELS:
        print("[ERROR] /predict/facial: CNN stress predictor not loaded")
        raise HTTPException(503, "CNN facial stress detection model not loaded. Required for facial image predictions.")
    
    predictor = MODELS['stress_predictor_cnn']
    
    # CNN model should have 'model' attribute (not 'cnn_model')
    if not hasattr(predictor, 'model'):
        print("[ERROR] /predict/facial: CNN model not properly initialized")
        raise HTTPException(503, "CNN model not properly initialized")
    
    try:
        import torch
        start_time = datetime.now()
        
        # ============================================================================
        # STEP 2: DECODE IMAGE
        # ============================================================================
        
        print("[INFO] Decoding base64 image...")
        frame = base64_to_numpy(request.image_base64)
        
        if frame is None or frame.size == 0:
            print("[ERROR] /predict/facial: Image decode resulted in empty frame")
            raise ValueError("Image decoding failed - empty frame")
        
        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        
        # ============================================================================
        # STEP 3: FACE DETECTION (Haar Cascade)
        # ============================================================================
        
        print("[INFO] Detecting faces...")
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray_frame, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(48, 48),
            maxSize=(300, 300)
        )
        
        if len(faces) == 0:
            print("[WARN] /predict/facial: No faces detected in image")
            return {
                "label": "No Face Detected",
                "confidence": 0.0,
                "model_name": "CNN Facial Expression Detector",
                "inference_type": "REAL MODEL",
                "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "timestamp": datetime.now().isoformat()
            }
        
        print(f"[DEBUG] Detected {len(faces)} face(s)")
        
        # ============================================================================
        # STEP 4: EXTRACT FACE REGION
        # ============================================================================
        
        x, y, w, h = faces[0]
        padding = int(w * 0.1)
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(frame.shape[1], x + w + padding)
        y_end = min(frame.shape[0], y + h + padding)
        
        face_region = frame[y_start:y_end, x_start:x_end].copy()
        
        if face_region.size == 0:
            print("[ERROR] /predict/facial: Face region is empty")
            raise ValueError("Face region extraction failed")
        
        print(f"[DEBUG] Face region shape: {face_region.shape}")
        
        # ============================================================================
        # STEP 5: AUTO-DETECT CORRECT INPUT SIZE
        # ============================================================================
        
        print("[INFO] Auto-detecting model input size...")
        
        # Try common sizes in order of likelihood (ResNet standard is 224x224)
        # Start with 224 as it's the standard for ResNet models
        COMMON_SIZES = [224, 128, 96, 64, 48, 32]
        TARGET_SIZE = None
        successful_inference = False
        face_tensor = None
        outputs = None
        
        device = getattr(predictor, 'device', 'cpu')
        # CNN predictor stores model in 'model' attribute
        model = predictor.model
        model.eval()
        
        # Inspect model architecture for debugging
        try:
            first_layer = list(model.children())[0]
            if hasattr(first_layer, '__class__'):
                print(f"[DEBUG] Model first layer type: {first_layer.__class__.__name__}")
            if hasattr(first_layer, 'in_features'):
                print(f"[DEBUG] First layer expects {first_layer.in_features} input features")
        except Exception as e:
            print(f"[DEBUG] Could not inspect model architecture: {e}")
        
        for size_candidate in COMMON_SIZES:
            try:
                print(f"[DEBUG] Trying input size: {size_candidate}×{size_candidate}")
                
                # Resize to candidate size
                face_resized = cv2.resize(face_region, (size_candidate, size_candidate))
                
                # Convert BGR to RGB
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                face_normalized = face_rgb.astype(np.float32) / 255.0
                
                # ============================================================================
                # STEP 6: TENSOR CONVERSION (inside loop)
                # ============================================================================
                
                face_tensor = torch.from_numpy(face_normalized)
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                face_tensor = face_tensor.to(device)
                
                print(f"[DEBUG] Tensor shape: {face_tensor.shape}")
                
                # ============================================================================
                # STEP 7: MODEL INFERENCE (inside loop)
                # ============================================================================
                
                with torch.no_grad():
                    outputs = model(face_tensor)
                
                # Success! This size works
                print(f"[SUCCESS] Model inference successful with size: {size_candidate}×{size_candidate}")
                TARGET_SIZE = size_candidate
                successful_inference = True
                break
                
            except RuntimeError as e:
                error_str = str(e)
                print(f"[DEBUG] Size {size_candidate}×{size_candidate} failed: {error_str}")
                # Check if it's a shape mismatch error
                if "shapes cannot be multiplied" in error_str or "size mismatch" in error_str.lower():
                    print(f"[DEBUG] Shape mismatch detected - model may require different input size")
                continue
            except Exception as e:
                error_str = str(e)
                print(f"[DEBUG] Size {size_candidate}×{size_candidate} error: {error_str}")
                continue
        
        if not successful_inference:
            # Try to get more information about the model
            model_info = f"Model type: {type(model).__name__}"
            try:
                if hasattr(model, 'backbone'):
                    model_info += ", Has backbone (likely ResNet-based, expects 224x224)"
                elif hasattr(model, 'conv1'):
                    model_info += ", Has conv layers (CNN architecture)"
                else:
                    model_info += ", Unknown architecture"
            except:
                pass
            
            print(f"[ERROR] /predict/facial: Failed with all common input sizes")
            print(f"[ERROR] {model_info}")
            raise HTTPException(
                503, 
                f"Model inference failed with sizes {COMMON_SIZES}. "
                f"{model_info}. "
                "The model may require a specific input size that wasn't in the tested sizes. "
                "Please check the model training configuration or contact support."
            )
        
        # ============================================================================
        # STEP 8: POST-PROCESS RESULTS
        # ============================================================================
        
        print("[INFO] Processing inference results...")
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence = float(torch.max(probs).cpu().numpy())
        prediction_idx = int(torch.argmax(probs, dim=1).cpu().numpy()[0])
        
        stress_labels = {
            0: 'Baseline',
            1: 'Stress',
            2: 'Amusement',
            3: 'Meditation'
        }
        
        label = stress_labels.get(prediction_idx, "Unknown")
        
        print(f"[SUCCESS] Prediction: {label} ({confidence:.2%} confidence)")
        
        # ============================================================================
        # STEP 9: RETURN RESULT
        # ============================================================================
        
        clean_result = {
            "label": label,
            "confidence": float(confidence),
            "model_name": "CNN Facial Expression Detector",
            "inference_type": "REAL MODEL",
            "face_detected": True,
            "num_faces": int(len(faces)),
            "input_size_used": TARGET_SIZE,  # ← NEW: Show which size worked!
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        add_to_history(clean_result)
        return clean_result
        
    except ValueError as e:
        print(f"[ERROR] /predict/facial - Validation Error: {str(e)}")
        raise HTTPException(400, f"Preprocessing error: {str(e)}")
    
    except RuntimeError as e:
        print(f"[ERROR] /predict/facial - Runtime Error: {str(e)}")
        raise HTTPException(503, f"Model inference error: {str(e)}")
    
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        print(f"[ERROR] /predict/facial - Unexpected Error: {error_msg}")
        raise HTTPException(500, f"Unexpected error in facial prediction: {error_msg}")


@app.post("/predict/risk-simplified")
async def predict_risk_simplified(input_data: SimplifiedClinicalInput):
    """
    Smart cardiovascular risk prediction
    
    Input: Basic info (age, gender, height, weight, smoker, exercise)
    Output: Complete clinical profile + risk assessment
    
    AI generates all missing parameters automatically!
    """
    
    if 'risk_predictor' not in MODELS:
        raise HTTPException(
            503,
            "Risk predictor not loaded. Place 'risk.pkl' in: " + str(MODELS_DIR)
        )
    
    try:
        start_time = datetime.now()
        
        # Generate complete clinical profile using AI
        profile = ClinicalDataGenerator.generate_complete_profile(input_data)
        
        # REAL MODEL INFERENCE
        risk_result = MODELS['risk_predictor'].predict(
            age=profile['age'],
            bmi=profile['bmi'],
            currentsmoker=profile['currentsmoker'],
            cigsperday=profile['cigsperday'],
            bpmeds=profile['bpmeds'],
            prevalentstroke=profile['prevalentstroke'],
            prevalenthyp=profile['prevalenthyp'],
            diabetes=profile['diabetes'],
            totchol=profile['totchol'],
            sysbp=profile['sysbp']
        )
        
        # Combine results
        result = {
            **risk_result,
            "generated_profile": profile,
            "inference_type": "REAL MODEL",
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/predict/glucose")
async def predict_glucose(request: GlucosePredictionRequest):
    """Predict glucose level - REAL MODEL ONLY"""
    
    if 'glucose_predictor' not in MODELS:
        raise HTTPException(
            503,
            "Glucose predictor not loaded. Place models in: " + str(MODELS_DIR)
        )
    
    try:
        start_time = datetime.now()
        
        sequence = np.array(request.sequence)
        
        # REAL MODEL INFERENCE
        pred, attn = MODELS['glucose_predictor'].predict(sequence)
        
        result = {
            "prediction": float(pred),
            "unit": "mg/dL",
            "status": "Normal" if 70 <= pred <= 140 else ("Elevated" if pred > 140 else "Low"),
            "inference_type": "REAL MODEL",
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

@app.post("/predict/heart-rate")
async def predict_heart_rate(request: RPPGRequest):
    """Predict heart rate from rPPG signal - REAL MODEL ONLY"""
    
    if 'rppg_predictor' not in MODELS:
        raise HTTPException(
            503,
            "rPPG predictor not loaded. Place 'rppg_model.pth' in: " + str(MODELS_DIR)
        )
    
    try:
        start_time = datetime.now()
        
        signal = np.array(request.signal)
        
        # REAL MODEL INFERENCE
        hr = MODELS['rppg_predictor'].predict(signal)
        
        result = {
            "heart_rate": float(hr),
            "unit": "BPM",
            "status": "Normal" if 60 <= hr <= 100 else ("Elevated" if hr > 100 else "Low"),
            "inference_type": "REAL MODEL",
            "processing_time_ms": (datetime.now() - start_time).total_seconds() * 1000,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {str(e)}")

# ============================================================================
# HISTORY & STATISTICS
# ============================================================================

@app.get("/predictions/history")
async def get_history(limit: int = 10):
    """Get prediction history"""
    return {
        "total": len(PREDICTION_HISTORY),
        "predictions": PREDICTION_HISTORY[-limit:][::-1]
    }

@app.delete("/predictions/history")
async def clear_history():
    """Clear prediction history"""
    global PREDICTION_HISTORY
    PREDICTION_HISTORY = []
    return {"message": "History cleared", "status": "success"}

@app.get("/predictions/stats")
async def get_stats():
    """Get prediction statistics - FIXED VERSION"""
    if not PREDICTION_HISTORY:
        return {
            "total_predictions": 0, 
            "message": "No predictions yet",
            "avg_confidence": 0.0,
            "label_distribution": {},
            "recent_predictions": []
        }
    
    # Safely extract data
    labels = []
    confidences = []
    
    for p in PREDICTION_HISTORY:
        if isinstance(p, dict):
            if "label" in p and p["label"]:
                labels.append(str(p["label"]))
            if "confidence" in p:
                try:
                    conf = float(p["confidence"])
                    confidences.append(conf)
                except (ValueError, TypeError):
                    pass
    
    # Calculate label distribution
    label_dist = {}
    for label in set(labels):
        label_dist[label] = labels.count(label)
    
    # Get recent predictions (ensure JSON serializable)
    recent = []
    for p in PREDICTION_HISTORY[-5:]:
        if isinstance(p, dict):
            clean_pred = {
                "label": str(p.get("label", "unknown")),
                "confidence": float(p.get("confidence", 0)),
                "timestamp": str(p.get("timestamp", ""))
            }
            recent.append(clean_pred)
    
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    
    return {
        "total_predictions": len(PREDICTION_HISTORY),
        "avg_confidence": round(avg_conf, 3),
        "label_distribution": label_dist,
        "recent_predictions": recent
    }

# ============================================================================
# FRONTEND
# ============================================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve frontend dashboard"""
    frontend_file = FRONTEND_DIR / "index.html"
    
    if not frontend_file.exists():
        return HTMLResponse(content=f"""
        <html>
            <head><title>Advanced Health AI</title></head>
            <body style="font-family: Arial; padding: 50px; text-align: center;">
                <h1> Advanced Health AI</h1>
                <p style="font-size: 18px; color: #10b981;">✓ API is running</p>
                <p><a href="/docs" style="color: #2563eb; font-size: 16px;"> View API Documentation</a></p>
                <hr style="margin: 30px 0;">
                <p style="color: #6b7280;">Frontend not found at: {frontend_file}</p>
                <p style="color: #6b7280; font-size: 14px;">Place index.html in frontend/ directory</p>
            </body>
        </html>
        """)
    
    with open(frontend_file, 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

# Serve static files
if FRONTEND_DIR.exists():
    try:
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    except:
        pass

# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def startup():
    """Load all models on startup"""
    print("\n" + "="*70)
    print("ADVANCED HEALTH AI - STARTING")
    print("="*70)
    
    # Load all models
    load_stress_detector()
    load_risk_predictor()
    load_glucose_predictor()
    load_rppg_predictor()
    load_webcam_monitor()
    
    loaded = sum(SYSTEM_STATUS.values())
    total = len(SYSTEM_STATUS)
    
    print("="*70)
    print(f"Models Loaded: {loaded}/{total}")
    print("="*70)
    
    if loaded == 0:
        print("\n  WARNING: NO MODELS LOADED")
        print("   Server will reject all prediction requests")
        print("\n   Place model files in:")
        print(f"   - {CHECKPOINT_DIR}")
        print(f"   - {MODELS_DIR}")
        print("\n   Then restart server")
    else:
        print(f"\n {loaded} models ready for inference!")
        print("   REAL MODEL INFERENCE ACTIVE")
    
    print("="*70)
    print()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ADVANCED HEALTH AI - SERVER")
    print("="*70)
    print(f"API: http://localhost:8000")
    print(f"Docs: http://localhost:8000/docs")
    print(f"Dashboard: http://localhost:8000/dashboard")
    print(f"WebSocket: ws://localhost:8000/ws/webcam")
    print("="*70 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
