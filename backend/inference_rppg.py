"""
Load trained rPPG model and make heart rate predictions.
"""
import torch
import numpy as np
import joblib
from pathlib import Path

from models.vision.rpPG_model import rPPGCNNLSTM, rPPGResNet1D

class rPPGPredictor:
    """Load and use trained rPPG model."""
    
    def __init__(self, model_path="data/models/rppg_model.pth", 
                 model_type="cnn_lstm", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        if model_type == "cnn_lstm":
            self.model = rPPGCNNLSTM(input_size=1, hidden_size=64)
        else:
            self.model = rPPGResNet1D(input_size=1, hidden_size=64)
        
        # Load weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded rPPG model from {model_path}")
    
    def predict(self, signal):
        """
        Predict heart rate from rPPG signal.
        
        Args:
            signal: numpy array shape (seq_len,) or (batch, seq_len)
        
        Returns:
            heart_rate: Predicted heart rate in BPM
        """
        # Ensure 3D tensor (batch, 1, seq_len)
        if signal.ndim == 1:
            signal = signal[np.newaxis, np.newaxis, :]
        elif signal.ndim == 2:
            signal = signal[:, np.newaxis, :]
        
        signal_tensor = torch.FloatTensor(signal).to(self.device)
        
        with torch.no_grad():
            pred, attn = self.model(signal_tensor)
        
        hr = pred.cpu().numpy().squeeze()
        
        return hr
    
    def predict_batch(self, signals_list):
        """Predict on multiple signals."""
        predictions = []
        
        for signal in signals_list:
            hr = self.predict(signal)
            predictions.append(hr)
        
        return np.array(predictions)

if __name__ == "__main__":
    # Test inference
    predictor = rPPGPredictor(model_type="cnn_lstm")
    
    # Create dummy signal
    dummy_signal = np.random.randn(30)
    
    pred = predictor.predict(dummy_signal)
    print(f"Predicted heart rate: {pred:.1f} BPM")