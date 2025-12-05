# -*- coding: utf-8 -*-
"""
INFERENCE PIPELINE: Stress Detection Predictions
- Load trained models
- Perform inference on new data
- Real-time prediction from signals/images
"""

import os
import sys
from pathlib import Path
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import warnings
warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.stress_detection import (
    create_lstm_classifier,
    create_resnet_detector,
    create_hybrid_detector
)
from backend.data.feature_engineering import FeatureExtractor


class StressPredictor:
    """Unified predictor for stress detection models."""
    
    def __init__(self, model_type='lstm', checkpoint_path=None, device='cuda'):
        """
        Args:
            model_type: 'lstm', 'cnn', or 'hybrid'
            checkpoint_path: Path to saved model
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.device = device
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        
        # Map classes to labels
        self.class_labels = {
            0: 'Baseline',
            1: 'Stress',
            2: 'Amusement',
            3: 'Meditation'
        }
        
        # Load model
        self.model = self._load_model(model_type)
        
        if self.checkpoint_path and self.checkpoint_path.exists():
            self._load_checkpoint()
        
        self.model.eval()
        self.feature_extractor = FeatureExtractor()
    
    def _load_model(self, model_type):
        """Load model architecture."""
        if model_type == 'lstm':
            model = create_lstm_classifier(num_classes=4, use_attention=True)
        elif model_type == 'cnn':
            model = create_resnet_detector(num_classes=4, model_name='resnet18', pretrained=False)
        elif model_type == 'hybrid':
            model = create_hybrid_detector(num_classes=4, pretrained_cnn=False)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device)
    
    def _load_checkpoint(self):
        """Load trained weights from checkpoint."""
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Loaded checkpoint from: {self.checkpoint_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load checkpoint: {e}")
    
    def predict_physiological(self, ecg, eda, temperature, return_confidence=True):
        """
        Predict stress from physiological signals.
        
        Args:
            ecg: ECG signal (array or list, length=256)
            eda: EDA signal (array or list, length=256)
            temperature: Temperature signal (array or list, length=256)
            return_confidence: Return confidence scores
            
        Returns:
            {
                'prediction': int (0-3),
                'label': str,
                'confidence': float,
                'probabilities': dict,
                'features': dict
            }
        """
        if self.model_type == 'cnn':
            print("[ERROR] CNN model requires facial images, not physiological signals")
            return None
        
        # Stack signals: (256, 3)
        signal = np.stack([ecg, eda, temperature], axis=1).astype(np.float32)
        
        # Convert to tensor: (1, 256, 3)
        signal_tensor = torch.from_numpy(signal).unsqueeze(0).to(self.device)
        
        # Prediction
        with torch.no_grad():
            logits, features = self.model(signal_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Extract results
        prediction = logits.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
        
        result = {
            'prediction': prediction,
            'label': self.class_labels[prediction],
            'confidence': float(confidence),
            'probabilities': {
                self.class_labels[i]: float(probs[0, i].item())
                for i in range(4)
            }
        }
        
        if return_confidence:
            result['all_scores'] = logits[0].cpu().numpy()
        
        return result
    
    def predict_facial(self, image_path, return_confidence=True):
        """
        Predict stress from facial image.
        
        Args:
            image_path: Path to image file or image array (224, 224, 3)
            return_confidence: Return confidence scores
            
        Returns:
            {
                'prediction': int (0-3),
                'label': str,
                'confidence': float,
                'probabilities': dict
            }
        """
        if self.model_type == 'lstm':
            print("[ERROR] LSTM model requires physiological signals, not images")
            return None
        
        # Load or convert image
        if isinstance(image_path, str):
            import cv2
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
        else:
            image = image_path
        
        # Convert to tensor: (1, 3, 224, 224)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        image_tensor = image_tensor / 255.0  # Normalize
        
        # Prediction
        with torch.no_grad():
            logits = self.model(image_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Extract results
        prediction = logits.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
        
        result = {
            'prediction': prediction,
            'label': self.class_labels[prediction],
            'confidence': float(confidence),
            'probabilities': {
                self.class_labels[i]: float(probs[0, i].item())
                for i in range(4)
            }
        }
        
        if return_confidence:
            result['all_scores'] = logits[0].cpu().numpy()
        
        return result
    
    def predict_multimodal(self, ecg, eda, temperature, image_path, return_confidence=True):
        """
        Predict stress from both physiological signals and facial image.
        
        Args:
            ecg, eda, temperature: Physiological signals
            image_path: Path to facial image
            return_confidence: Return confidence scores
            
        Returns:
            {
                'prediction': int (0-3),
                'label': str,
                'confidence': float,
                'probabilities': dict,
                'fusion_method': str
            }
        """
        if self.model_type != 'hybrid':
            print("[ERROR] Hybrid model required for multimodal prediction")
            return None
        
        # Prepare physiological data
        signal = np.stack([ecg, eda, temperature], axis=1).astype(np.float32)
        signal_tensor = torch.from_numpy(signal).unsqueeze(0).to(self.device)
        
        # Prepare facial data
        import cv2
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))
        else:
            image = image_path
        
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        image_tensor = image_tensor / 255.0
        
        # Prediction
        with torch.no_grad():
            logits = self.model(signal_tensor, image_tensor)
            probs = F.softmax(logits, dim=1)
        
        # Extract results
        prediction = logits.argmax(dim=1).item()
        confidence = probs[0, prediction].item()
        
        result = {
            'prediction': prediction,
            'label': self.class_labels[prediction],
            'confidence': float(confidence),
            'probabilities': {
                self.class_labels[i]: float(probs[0, i].item())
                for i in range(4)
            },
            'fusion_method': 'multimodal_hybrid'
        }
        
        if return_confidence:
            result['all_scores'] = logits[0].cpu().numpy()
        
        return result
    
    def predict_batch(self, signals=None, images=None):
        """
        Batch prediction on multiple samples.
        
        Args:
            signals: List of (ecg, eda, temp) tuples
            images: List of image paths or arrays
            
        Returns:
            List of predictions
        """
        results = []
        
        if signals:
            for ecg, eda, temp in signals:
                result = self.predict_physiological(ecg, eda, temp)
                results.append(result)
        
        if images:
            for image_path in images:
                result = self.predict_facial(image_path)
                results.append(result)
        
        return results


def main():
    """Demo inference pipeline."""
    parser = argparse.ArgumentParser(description="Stress Detection Inference")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'cnn', 'hybrid'],
                       help='Model type')
    parser.add_argument('--checkpoint', type=str,
                       help='Path to checkpoint file')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device')
    parser.add_argument('--input-type', type=str, default='physiological',
                       choices=['physiological', 'facial', 'multimodal'],
                       help='Input type')
    
    args = parser.parse_args()
    
    # Default checkpoint paths
    if not args.checkpoint:
        checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'stress_detection'
        if args.model == 'lstm':
            args.checkpoint = checkpoint_dir / 'lstm_physio_phase1_best.pth'
        elif args.model == 'cnn':
            args.checkpoint = checkpoint_dir / 'cnn_facial_phase2_best.pth'
    
    # Initialize predictor
    print("\n" + "="*70)
    print("STRESS DETECTION INFERENCE PIPELINE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Input type: {args.input_type}")
    print("="*70)
    
    predictor = StressPredictor(
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Demo prediction
    print("\n[DEMO] Sample Predictions:")
    print("="*70)
    
    # Generate dummy data
    dummy_ecg = np.random.randn(256).astype(np.float32)
    dummy_eda = np.random.randn(256).astype(np.float32)
    dummy_temp = np.random.randn(256).astype(np.float32)
    
    if args.input_type in ['physiological', 'multimodal'] and args.model != 'cnn':
        print("\n[1] Physiological Signal Prediction:")
        result = predictor.predict_physiological(dummy_ecg, dummy_eda, dummy_temp)
        if result:
            print(f"    Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
            print(f"    Probabilities: {result['probabilities']}")
    
    if args.input_type in ['facial', 'multimodal'] and args.model != 'lstm':
        print("\n[2] Facial Image Prediction:")
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = predictor.predict_facial(dummy_image)
        if result:
            print(f"    Prediction: {result['label']} (confidence: {result['confidence']:.2%})")
            print(f"    Probabilities: {result['probabilities']}")
    
    print("\n" + "="*70)
    print("Ready for production inference!")
    print("="*70)
    print("\nUsage Examples:")
    print("  predictor = StressPredictor('lstm', 'checkpoint.pth')")
    print("  result = predictor.predict_physiological(ecg, eda, temp)")
    print("  result = predictor.predict_facial('image.jpg')")
    print("  result = predictor.predict_multimodal(ecg, eda, temp, 'image.jpg')")
    print()


if __name__ == "__main__":
    main()
