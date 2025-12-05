"""
Load trained model and make predictions on new data.
"""
import torch
import numpy as np
import joblib
from pathlib import Path
from models.lstm_attention import LSTMAttnModel

class HealthPredictor:
    def __init__(self, model_path="data/models/lstm_attn.pth",
                 scaler_path="data/models/lstm_scaler.pkl",
                 n_features=9, hidden_size=128, num_layers=2, attn_heads=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load model
        self.model = LSTMAttnModel(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            attn_heads=attn_heads
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"✓ Loaded scaler from {scaler_path}")
    
    def predict(self, X_seq):
        """
        Predict on a single sequence.
        
        Args:
            X_seq: numpy array shape (seq_len, n_features)
        
        Returns:
            prediction: scalar float value
            attention_weights: (seq_len, attn_heads)
        """
        # Scale
        X_scaled = self.scaler.transform(X_seq)
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            pred, attn_weights = self.model(X_tensor)
        
        return pred.item(), attn_weights.squeeze().cpu().numpy()
    
    def predict_batch(self, X_batch):
        """
        Predict on multiple sequences.
        
        Args:
            X_batch: numpy array shape (batch_size, seq_len, n_features)
        
        Returns:
            predictions: array of shape (batch_size,)
        """
        X_scaled = self.scaler.transform(X_batch.reshape(-1, X_batch.shape[-1]))
        X_scaled = X_scaled.reshape(X_batch.shape)
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            preds, _ = self.model(X_tensor)
        
        return preds.squeeze().cpu().numpy()

if __name__ == "__main__":
    # Example usage
    predictor = HealthPredictor()
    
    # Create dummy sequence (30 timesteps × 9 features)
    X_dummy = np.random.randn(30, 9)
    
    pred, attn = predictor.predict(X_dummy)
    print(f"Prediction: {pred:.2f}")
    print(f"Attention shape: {attn.shape}")