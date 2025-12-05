# Advanced Health AI - Stress Detection API

Multi-modal stress detection system combining physiological signals (ECG, EDA, Temperature) and facial expression analysis using deep learning models.

## Features

- **Multi-Modal Predictions**: Combine physiological signals and facial images
- **Three Model Types**: LSTM (physiological), CNN (facial), Hybrid (combined)
- **Real-time Processing**: Fast inference with GPU acceleration
- **Interactive Dashboard**: Web-based UI for testing predictions
- **REST API**: Full API for integration with external systems
- **Historical Analysis**: Track predictions over time

## Quick Start

### 1. Prerequisites

- Python 3.9+
- NVIDIA GPU with CUDA 12.1 (or CPU fallback)
- 5GB disk space for models and data

### 2. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-health-ai.git
cd advanced-health-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Training Models (Optional)

Pre-trained models are included. To train from scratch:

```bash
# Train Phase 1 (LSTM - Physiological)
python backend/train_stress_phase123.py --phase 1 --epochs 50 --device cuda

# Train Phase 2 (CNN - Facial)
python backend/train_stress_phase123.py --phase 2 --phase-start 2 --epochs 50 --device cuda

# Train Phase 3 (Hybrid - Multimodal)
python backend/train_stress_phase123.py --phase 3 --phase-start 3 --epochs 50 --device cuda
```

### 4. Run API Server

```bash
cd backend
python main.py
```

Server starts at: `http://localhost:8000`

API Docs: `http://localhost:8000/docs`

### 5. Open Web Dashboard

Navigate to: `http://localhost:8000/static/index.html`

Or open `frontend/index.html` directly in browser.

## API Endpoints

### Health Check

```
GET /health
```

Returns API status, device info, loaded models.

### Physiological Signals Prediction

```
POST /predict/physiological

{
  "ecg": [0.1, 0.2, 0.15, ...],
  "eda": [0.3, 0.35, 0.32, ...],
  "temperature": [36.5, 36.6, 36.55, ...],
  "model_type": "lstm"
}
```

**Response:**

```json
{
  "prediction": 1,
  "label": "Stress",
  "confidence": 0.82,
  "probabilities": {
    "Baseline": 0.05,
    "Stress": 0.82,
    "Amusement": 0.10,
    "Meditation": 0.03
  },
  "timestamp": "2025-12-05T10:30:45.123456",
  "processing_time_ms": 15.2
}
```

### Facial Expression Prediction

```
POST /predict/facial

{
  "image_base64": "iVBORw0KGgo...",
  "model_type": "cnn"
}
```

### Multimodal Analysis

```
POST /predict/multimodal

{
  "ecg": [...],
  "eda": [...],
  "temperature": [...],
  "image_base64": "...",
  "model_type": "hybrid"
}
```

### Get Predictions History

```
GET /predictions/history?limit=100
```

### Get Statistics

```
GET /predictions/stats
```

### Clear History

```
DELETE /predictions/history
```

## Models

### Phase 1: LSTM (Physiological)

- **Input**: ECG, EDA, Temperature signals (256 samples)
- **Architecture**: LSTM + Attention mechanism
- **Accuracy**: 82-88%
- **Latency**: 2-5ms
- **Checkpoint**: `checkpoints/stress_detection/lstm_physio_phase1_best.pth`

### Phase 2: CNN (Facial)

- **Input**: Facial images (224×224)
- **Architecture**: ResNet18
- **Accuracy**: 75-82%
- **Latency**: 10-20ms
- **Checkpoint**: `checkpoints/stress_detection/cnn_facial_phase2_best.pth`

### Phase 3: Hybrid (Multimodal)

- **Input**: Signals + Facial image
- **Architecture**: LSTM + CNN + Fusion layer
- **Accuracy**: 85-92%
- **Latency**: 15-25ms
- **Checkpoint**: `checkpoints/stress_detection/hybrid_multimodal_phase3_best.pth`

## Project Structure

```
advanced-health-ai/
├── backend/
│   ├── main.py                          # FastAPI server
│   ├── inference_stress_phase123.py      # Inference module
│   ├── evaluate_stress_models.py         # Evaluation module
│   ├── train_stress_phase123.py          # Training pipeline
│   ├── requirements.txt                  # Python dependencies
│   ├── models/
│   │   └── stress_detection/             # Model architectures
│   ├── data/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   └── utils/
├── frontend/
│   └── index.html                       # Web dashboard
├── checkpoints/
│   └── stress_detection/                # Saved model weights
├── data/
│   ├── raw/                             # Raw datasets
│   └── processed/                       # Processed datasets
└── README.md
```

## Datasets

- **WESAD**: Physiological signals (475K samples)
- **AffectNet**: Facial expressions (300K+ images)
- **Framingham**: Cardiovascular risk data
- **PPG-BP**: Blood pressure estimation

Data is automatically downloaded during first training run.

## Performance Benchmarks

| Model | Accuracy | F1-Score | Latency | Memory |
|-------|----------|----------|---------|--------|
| LSTM  | 84%      | 0.82     | 3ms     | 45MB   |
| CNN   | 78%      | 0.76     | 15ms    | 250MB  |
| Hybrid| 88%      | 0.85     | 20ms    | 300MB  |

*Benchmarks on RTX 3050 Ti with batch size 32*

## Usage Examples

### Python Client

```python
import requests
import numpy as np

API_BASE = "http://localhost:8000"

# Physiological prediction
response = requests.post(
    f"{API_BASE}/predict/physiological",
    json={
        "ecg": np.random.randn(256).tolist(),
        "eda": np.random.randn(256).tolist(),
        "temperature": np.linspace(36, 37, 256).tolist(),
        "model_type": "lstm"
    }
)

result = response.json()
print(f"Prediction: {result['label']} ({result['confidence']:.2%})")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict/physiological" \
  -H "Content-Type: application/json" \
  -d '{
    "ecg": [0.1, 0.2, 0.15],
    "eda": [0.3, 0.35, 0.32],
    "temperature": [36.5, 36.6, 36.55],
    "model_type": "lstm"
  }'
```

## Development

### Run Tests

```bash
cd backend
python -m pytest
```

### Format Code

```bash
black backend/
flake8 backend/
```

### Build Documentation

```bash
cd docs
make html
```

## Troubleshooting

### CUDA Not Available

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Models Not Loading

```bash
# Check checkpoint exists
ls checkpoints/stress_detection/

# Download pre-trained models
python backend/train_stress_phase123.py --download-pretrained
```

### Port Already in Use

```bash
# Use different port
python backend/main.py --port 8001

# Or kill process on port 8000
# Windows: netstat -ano | findstr :8000
# Linux: lsof -i :8000
```

## Configuration

Edit `backend/main.py` to customize:

- `DEVICE`: "cuda" or "cpu"
- `CHECKPOINT_DIR`: Path to model weights
- `PORT`: API server port (default 8000)
- `LOG_LEVEL`: Logging verbosity

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this project in research, please cite:

```bibtex
@software{healthai2025,
  title={Advanced Health AI - Multi-Modal Stress Detection},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/advanced-health-ai}
}
```

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/advanced-health-ai/issues)
- Email: your-email@example.com

## Acknowledgments

- WESAD Dataset: [Link](https://github.com/alistair-b/wesad)
- AffectNet Dataset: [Link](http://www.whdeng.cn/afn/)
- PyTorch Team
- FastAPI Community

---

**Last Updated**: December 5, 2025  
**Status**: Production Ready ✓
