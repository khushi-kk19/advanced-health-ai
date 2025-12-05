"""
FastAPI + WebSocket real-time health monitoring API.
Phase 4: Production-ready API with streaming.
"""
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import asyncio
import cv2
import numpy as np
import json
from datetime import datetime
import torch
import threading

from models.vision.rpPG import rPPGDetector
from models.vision.stress_detection import StressDetector
from models.multimodal_fusion import MultiModalHealthAssessment
from inference_rppg import rPPGPredictor
from inference_risk import RiskPredictor
from inference import HealthPredictor

app = FastAPI(title="Advanced Health AI - Phase 4 API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
print("Loading models...")
try:
    rppg_model = rPPGPredictor(model_path="data/models/rppg_model.pth")
    risk_model = RiskPredictor()
    glucose_model = HealthPredictor()
    print("✓ All models loaded")
except Exception as e:
    print(f"⚠ Warning: {e}")

# Initialize detectors
rpPG_detector = rPPGDetector()
stress_detector = StressDetector()
fusion = MultiModalHealthAssessment()

@app.get("/")
def root():
    """API info."""
    return {
        "name": "Advanced Health AI API",
        "version": "1.0.0",
        "phase": "4",
        "endpoints": {
            "ws": "/ws/health-monitor",
            "rest": {
                "health": "GET /health",
                "summary": "GET /summary",
                "risk": "POST /predict/risk",
                "glucose": "POST /predict/glucose"
            }
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/summary")
def get_summary():
    """Get current health summary."""
    summary = fusion.get_health_summary()
    return {
        "status": "success",
        "data": summary,
        "timestamp": datetime.now().isoformat()
    }

@app.websocket("/ws/health-monitor")
async def websocket_health_monitor(websocket: WebSocket):
    """
    WebSocket endpoint for real-time health monitoring.
    Streams heart rate, stress, and health scores.
    """
    await websocket.accept()
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        await websocket.send_json({
            "error": "Cannot open webcam",
            "status": "failed"
        })
        await websocket.close()
        return
    
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame_count += 1
            
            # rPPG detection
            hr, quality = rpPG_detector.process_frame(frame)
            
            # Stress detection
            stress, indicators = stress_detector.process_frame(frame)
            
            # Add to fusion
            if hr is not None:
                fusion.add_vision_data(hr, stress, quality)
            
            # Get comprehensive health summary every 30 frames
            if frame_count % 30 == 0:
                summary = fusion.get_health_summary()
                
                await websocket.send_json({
                    "type": "health_update",
                    "timestamp": datetime.now().isoformat(),
                    "vision": {
                        "heart_rate": float(hr) if hr is not None else None,
                        "stress_score": float(stress),
                        "signal_quality": float(quality),
                        "indicators": {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                      for k, v in indicators.items()}
                    },
                    "summary": summary,
                    "frame": frame_count
                })
            
            # Allow client to send control messages
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                if message == "stop":
                    break
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.033)  # ~30fps
    
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e), "status": "failed"})
    finally:
        cap.release()
        await websocket.close()

@app.post("/predict/risk")
async def predict_risk(data: dict):
    """
    Predict disease risk.
    
    Example:
    ```json
    {
      "age": 55,
      "bmi": 28,
      "currentsmoker": 1,
      ...
    }
    ```
    """
    try:
        result = risk_model.predict(**data)
        return {
            "status": "success",
            "prediction": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/glucose")
async def predict_glucose(data: dict):
    """
    Predict glucose level.
    
    Example:
    ```json
    {
      "sequence": [[1.6, 69.3, ...], [...], ...]
    }
    ```
    """
    try:
        sequence = np.array(data['sequence'])
        pred, attn = glucose_model.predict(sequence)
        
        return {
            "status": "success",
            "prediction": float(pred),
            "confidence": "high" if pred > 0 else "low",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/dashboard")
async def dashboard():
    """Serve interactive dashboard HTML."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced Health AI - Real-Time Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .metric {
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            }
            .label {
                font-size: 14px;
                opacity: 0.9;
            }
            .status {
                text-align: center;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
                font-weight: bold;
            }
            .status.good { background: #4caf50; }
            .status.warning { background: #ff9800; }
            .status.critical { background: #f44336; }
            .chart {
                background: white;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            button {
                background: #667eea;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 5px;
            }
            button:hover { background: #764ba2; }
            .controls {
                text-align: center;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Advanced Health AI - Real-Time Dashboard</h1>
            
            <div class="controls">
                <button onclick="startMonitoring()">▶ Start Monitoring</button>
                <button onclick="stopMonitoring()">⏹ Stop Monitoring</button>
            </div>
            
            <div class="grid">
                <div class="card">
                    <div class="label">Heart Rate</div>
                    <div class="metric" id="hr">-- BPM</div>
                    <div class="status good" id="hr-status">Normal</div>
                </div>
                <div class="card">
                    <div class="label">Stress Level</div>
                    <div class="metric" id="stress">-- %</div>
                    <div class="status warning" id="stress-status">Neutral</div>
                </div>
                <div class="card">
                    <div class="label">Health Score</div>
                    <div class="metric" id="health">-- /100</div>
                    <div class="status good" id="health-status">Good</div>
                </div>
                <div class="card">
                    <div class="label">Signal Quality</div>
                    <div class="metric" id="quality">-- %</div>
                    <div class="status good">Connected</div>
                </div>
            </div>
            
            <div class="chart">
                <div id="hrChart" style="width:100%;height:300px;"></div>
            </div>
            
            <div class="chart">
                <div id="stressChart" style="width:100%;height:300px;"></div>
            </div>
            
            <div id="debug" style="background: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 12px; max-height: 200px; overflow-y: auto;"></div>
        </div>
        
        <script>
            let ws = null;
            let hrData = [];
            let stressData = [];
            let timestamps = [];
            
            function startMonitoring() {
                if (ws) return;
                
                ws = new WebSocket("ws://localhost:8000/ws/health-monitor");
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    if (data.vision) {
                        // Update metrics
                        document.getElementById('hr').textContent = 
                            (data.vision.heart_rate || '--').toFixed(1) + ' BPM';
                        document.getElementById('stress').textContent = 
                            (data.vision.stress_score || '--').toFixed(1) + ' %';
                        document.getElementById('quality').textContent = 
                            (data.vision.signal_quality || '--').toFixed(1) + ' %';
                        
                        // Update status indicators
                        const hrStatus = data.vision.heart_rate < 100 ? 'good' : 'warning';
                        document.getElementById('hr-status').className = 'status ' + hrStatus;
                        
                        const stressStatus = data.vision.stress_score < 40 ? 'good' : 
                                           data.vision.stress_score < 70 ? 'warning' : 'critical';
                        document.getElementById('stress-status').className = 'status ' + stressStatus;
                        
                        // Add to chart data
                        hrData.push(data.vision.heart_rate);
                        stressData.push(data.vision.stress_score);
                        timestamps.push(new Date().toLocaleTimeString());
                        
                        // Keep only last 60 points
                        if (hrData.length > 60) {
                            hrData.shift();
                            stressData.shift();
                            timestamps.shift();
                        }
                        
                        // Update charts
                        updateCharts();
                    }
                    
                    if (data.summary) {
                        document.getElementById('health').textContent = 
                            data.summary.composite_health_score.toFixed(1);
                        document.getElementById('health-status').className = 
                            'status ' + (data.summary.health_status === 'Excellent' || data.summary.health_status === 'Good' ? 'good' : 'warning');
                    }
                    
                    logDebug(JSON.stringify(data, null, 2));
                };
                
                ws.onerror = function(error) {
                    logDebug('WebSocket error: ' + error);
                };
                
                ws.onclose = function() {
                    logDebug('WebSocket closed');
                    ws = null;
                };
            }
            
            function stopMonitoring() {
                if (ws) {
                    ws.close();
                    ws = null;
                }
            }
            
            function updateCharts() {
                // HR Chart
                Plotly.react('hrChart', [{
                    y: hrData,
                    x: timestamps,
                    name: 'Heart Rate',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {color: '#ff6b6b', width: 3}
                }], {
                    title: 'Heart Rate Over Time',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'BPM'},
                    margin: {l: 50, r: 20, t: 40, b: 50}
                });
                
                // Stress Chart
                Plotly.react('stressChart', [{
                    y: stressData,
                    x: timestamps,
                    name: 'Stress Level',
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: {color: '#ffa94d', width: 3}
                }], {
                    title: 'Stress Level Over Time',
                    xaxis: {title: 'Time'},
                    yaxis: {title: 'Stress Score (0-100)'},
                    margin: {l: 50, r: 20, t: 40, b: 50}
                });
            }
            
            function logDebug(message) {
                const debugDiv = document.getElementById('debug');
                const timestamp = new Date().toLocaleTimeString();
                debugDiv.innerHTML = `[${timestamp}] ${message}<br>` + debugDiv.innerHTML;
                if (debugDiv.children.length > 10) {
                    debugDiv.removeChild(debugDiv.lastChild);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("Starting Advanced Health AI API - Phase 4")
    print("="*60)
    print("🌐 API Docs: http://localhost:8000/docs")
    print("📊 Dashboard: http://localhost:8000/dashboard")
    print("📡 WebSocket: ws://localhost:8000/ws/health-monitor")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)