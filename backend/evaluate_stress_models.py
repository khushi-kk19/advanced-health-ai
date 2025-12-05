# -*- coding: utf-8 -*-
"""
EVALUATION PIPELINE: Performance Metrics & Analysis
- Calculate accuracy, precision, recall, F1-score
- Generate confusion matrices
- Create visualizations
- Compare models
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference_stress_phase123 import StressPredictor
from backend.train_stress_phase123 import MultimodalStressDataset


class ModelEvaluator:
    """Evaluate trained stress detection models."""
    
    def __init__(self, model_type='lstm', checkpoint_path=None, device='cuda'):
        """
        Args:
            model_type: 'lstm', 'cnn', or 'hybrid'
            checkpoint_path: Path to checkpoint
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.device = device
        self.predictor = StressPredictor(model_type, checkpoint_path, device)
        
        self.class_labels = {
            0: 'Baseline',
            1: 'Stress',
            2: 'Amusement',
            3: 'Meditation'
        }
        
        self.results = {}
    
    def evaluate_physiological(self, data_dir, sample_fraction=0.2):
        """
        Evaluate on physiological signals (LSTM).
        
        Args:
            data_dir: Path to processed data
            sample_fraction: Fraction of test data to use
        """
        if self.model_type == 'cnn':
            print("[ERROR] CNN requires facial images")
            return None
        
        print("\n[EVAL] Physiological Signals Evaluation")
        print("="*70)
        
        # Load data
        dataset = MultimodalStressDataset(data_dir, mode='test', modality='physio')
        
        # Sample if needed
        if sample_fraction < 1.0:
            n_samples = int(len(dataset) * sample_fraction)
            indices = np.random.choice(len(dataset), n_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Collect predictions
        all_preds = []
        all_labels = []
        all_probs = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        for i, batch in enumerate(loader):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(dataset)}")
            
            signal = batch['signal'].to(self.device)
            label = batch['label'].item()
            
            with torch.no_grad():
                logits, _ = self.predictor.model(signal)
                probs = F.softmax(logits, dim=1)
            
            pred = logits.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
            all_probs.append(probs[0].cpu().numpy())
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        self.results['physiological'] = metrics
        
        return metrics
    
    def evaluate_facial(self, data_dir, sample_fraction=0.2):
        """
        Evaluate on facial images (CNN).
        
        Args:
            data_dir: Path to processed data
            sample_fraction: Fraction of test data to use
        """
        if self.model_type == 'lstm':
            print("[ERROR] LSTM requires physiological signals")
            return None
        
        print("\n[EVAL] Facial Images Evaluation")
        print("="*70)
        
        # Load data
        dataset = MultimodalStressDataset(data_dir, mode='Test', modality='facial')
        
        # Sample if needed
        if sample_fraction < 1.0:
            n_samples = int(len(dataset) * sample_fraction)
            indices = np.random.choice(len(dataset), n_samples, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # Collect predictions
        all_preds = []
        all_labels = []
        
        print(f"Evaluating on {len(dataset)} samples...")
        
        for i, batch in enumerate(loader):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(dataset)}")
            
            image = batch['image'].to(self.device)
            label = batch['label'].item()
            
            with torch.no_grad():
                logits = self.predictor.model(image)
            
            pred = logits.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_preds)
        self.results['facial'] = metrics
        
        return metrics
    
    def _compute_metrics(self, true_labels, predictions):
        """Compute classification metrics."""
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, average='weighted', zero_division=0),
            'recall': recall_score(true_labels, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(true_labels, predictions, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist()
        }
        
        # Per-class metrics
        report = classification_report(true_labels, predictions, output_dict=True, zero_division=0)
        metrics['per_class'] = report
        
        return metrics
    
    def print_results(self):
        """Print evaluation results."""
        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)
        
        for dataset_type, metrics in self.results.items():
            print(f"\n[{dataset_type.upper()}]")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            
            # Print confusion matrix
            cm = np.array(metrics['confusion_matrix'])
            print(f"  Confusion Matrix:\n{cm}")
        
        print("\n" + "="*70)
    
    def save_results(self, output_path='evaluation_results.json'):
        """Save results to JSON."""
        output_path = PROJECT_ROOT / 'results' / f'{self.model_type}_{output_path}'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n[OK] Results saved to: {output_path}")


def main():
    """Evaluate trained models."""
    parser = argparse.ArgumentParser(description="Model Evaluation")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'cnn'],
                       help='Model type')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                       help='Phase (1=physio, 2=facial)')
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data' / 'processed'),
                       help='Data directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device')
    parser.add_argument('--sample-fraction', type=float, default=0.1,
                       help='Sample fraction for quick evaluation')
    
    args = parser.parse_args()
    
    # Default checkpoint
    checkpoint_dir = PROJECT_ROOT / 'checkpoints' / 'stress_detection'
    if args.model == 'lstm':
        checkpoint = checkpoint_dir / 'lstm_physio_phase1_best.pth'
    else:
        checkpoint = checkpoint_dir / 'cnn_facial_phase2_best.pth'
    
    print("\n" + "="*70)
    print("STRESS DETECTION MODEL EVALUATION")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Sample fraction: {args.sample_fraction}")
    print("="*70)
    
    # Evaluate
    evaluator = ModelEvaluator(args.model, checkpoint, args.device)
    
    if args.model == 'lstm':
        evaluator.evaluate_physiological(args.data_dir, args.sample_fraction)
    else:
        evaluator.evaluate_facial(args.data_dir, args.sample_fraction)
    
    # Print and save
    evaluator.print_results()
    evaluator.save_results()


if __name__ == "__main__":
    main()
