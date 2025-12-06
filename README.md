# 🏥 Advanced Health AI - Multi-Modal Prediction System

A production-ready, real-time multi-modal health prediction system powered by deep learning. Features live webcam vital signs monitoring (rPPG), stress detection from physiological signals and facial expressions, cardiovascular risk assessment, and glucose level prediction.


##  Key Features

### 1.  Real-Time Webcam Vital Signs (rPPG)
- **Non-contact heart rate monitoring** using facial video
- Remote Photoplethysmography (rPPG) algorithm
- MediaPipe face mesh detection
- Real-time FFT-based HR extraction
- WebSocket streaming for live updates

### 2. Stress Detection
- **Physiological Signals**: LSTM model analyzing ECG, EDA, and temperature
- **Facial Expression**: CNN model detecting stress from facial images
- 4 states: Baseline, Stress, Amusement, Meditation
- Real-time inference with confidence scores

### 3. Smart Health Risk Assessment
- **AI-powered clinical data generation** from minimal user input
- Cardiovascular disease (CVD) risk prediction
- Framingham Risk Score implementation
- Automatically estimates: BP, cholesterol, glucose, BMI, and 15+ parameters

### 4. Glucose Level Prediction
- LSTM with attention mechanism
- Time-series analysis of health metrics
- 9 features: glucose, insulin, carbs, activity, HR, sleep, stress, weight, medication


##  Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (HTML/CSS/JS)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Webcam   │ │  Stress  │ │  Risk    │ │ Glucose  │      │
│  │ Monitor  │ │ Detection│ │Assessment│ │Prediction│      │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘      │
└────────┼────────────┼────────────┼────────────┼────────────┘
         │            │            │            │
    WebSocket       REST API    REST API    REST API
         │            │            │            │
┌────────▼────────────▼────────────▼────────────▼────────────┐
│              FastAPI Backend (app.py)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Model Inference Layer                    │  │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐      │  │
│  │  │rPPG  │ │LSTM  │ │ CNN  │ │Risk  │ │LSTM  │      │  │
│  │  │Model │ │Stress│ │Facial│ │Model │ │Glucose│     │  │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘      │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Requirements

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Webcam (for real-time monitoring)
- Modern web browser (Chrome/Firefox/Edge)


## 🚀 Installation

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/advanced-health-ai.git
cd advanced-health-ai
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 4. Download Model Checkpoints

1. Go to [Releases](https://github.com/YOUR_USERNAME/advanced-health-ai/releases)
2. Download `models-v1.0.zip`
3. Extract to project root:
   ```
   checkpoints/stress_detection/
   ├── lstm_physio_phase1_best.pth
   └── cnn_facial_phase2_best.pth
   
   data/models/
   ├── lstm_attn.pth
   ├── lstm_scaler.pkl
   ├── risk.pkl
   └── rppg_model.pth
   ```


##  Quick Start

### Start the Server
```bash
cd backend
python app.py
```

The server will start at:
- 🌐 **API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs
- 🎨 **Dashboard**: http://localhost:8000/dashboard
- 🔌 **WebSocket**: ws://localhost:8000/ws/webcam

### Access the Dashboard
1. Open browser to http://localhost:8000/dashboard
2. Try each feature:
   - **Webcam**: Click "Start Monitoring" (allow camera access)
   - **Stress Detection**: Generate signals or upload face image
   - **Health Risk**: Enter basic info, AI generates full profile
   - **Glucose**: Generate time-series data for prediction


##  Training Your Own Models

### 1. Prepare Dataset
```bash
# Place your data in:
data/raw/WESAD/  # For stress detection
data/raw/clinical/  # For risk models
```

### 2. Train Models
```bash
# Stress detection (Phase 1-3)
python backend/train_stress_phase123.py

# Risk predictor
python backend/train_risk.py

# Glucose predictor
python backend/train_lstm.py

# rPPG model
python backend/train_rppg.py
```

### 3. Evaluate
```bash
python backend/evaluate_stress_models.py
```

## Project Structure

```
advanced-health-ai/
│
├── backend/
│   ├── app.py                          # Main FastAPI server
│   ├── requirements.txt                # Python dependencies
│   │
│   ├── models/                         # Model architectures
│   │   ├── stress_detection/
│   │   │   ├── lstm_model.py
│   │   │   ├── cnn_model.py
│   │   │   └── hybrid_model.py
│   │   ├── lstm_attention.py           # Glucose LSTM
│   │   ├── risk_models.py              # CVD risk
│   │   └── rppg_real.py                # rPPG model
│   │
│   ├── data/                           # Data processing
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   └── unified_loader.py
│   │
│   ├── inference_*.py                  # Inference scripts
│   ├── train_*.py                      # Training scripts
│   └── test_pipline.py                 # Testing
│
├── checkpoints/                        # Model weights
│   └── stress_detection/
│       ├── lstm_physio_phase1_best.pth
│       └── cnn_facial_phase2_best.pth
│
├── data/
│   └── models/                         # Trained models
│       ├── lstm_attn.pth
│       ├── lstm_scaler.pkl
│       ├── risk.pkl
│       └── rppg_model.pth
│
├── frontend/
│   └── index.html                      # Web dashboard
│
├── docs/                               # Documentation
│   ├── API.md
│   ├── MODELS.md
│   └── TRAINING.md
│
├── .gitignore
├── README.md
├── LICENSE
└── download_models.sh                  # Model download script
```
