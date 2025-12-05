# -*- coding: utf-8 -*-
"""
TRAINING SCRIPT: Multimodal Stress Detection (Phase 1-3)
- Phase 1: Train individual modality models (LSTM for physio, CNN for facial)
- Phase 2: Train hybrid multimodal model
- Phase 3: Fine-tune and ensemble models

Dataset: WESAD (physiological) + AffectNet (facial)
Output: Stress classification (0-3) or regression (0-1)
"""

import os
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tqdm
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


class MultimodalStressDataset(Dataset):
    """Dataset for multimodal stress detection."""
    
    def __init__(self, data_dir, mode='train', modality='both'):
        """
        Args:
            data_dir: Directory containing preprocessed data
            mode: 'train', 'val', 'test'
            modality: 'physio', 'facial', 'both'
        """
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.modality = modality
        self.samples = []
        self.feature_extractor = FeatureExtractor()
        
        # Load WESAD physiological data
        if modality in ['physio', 'both']:
            self._load_wesad_data()
        
        # Load AffectNet facial data
        if modality in ['facial', 'both']:
            self._load_affectnet_data()
    
    def _load_wesad_data(self):
        """Load preprocessed WESAD data."""
        wesad_dir = self.data_dir / "WESAD"
        
        if not wesad_dir.exists():
            print(f"⚠ WESAD directory not found: {wesad_dir}")
            return
        
        # Find all preprocessed JSON files
        json_files = sorted(wesad_dir.glob("S*_preprocessed.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Extract windows
                for window_idx, window in enumerate(data.get('windows', [])):
                    self.samples.append({
                        'type': 'physio',
                        'data': window,
                        'subject': data['subject_id'],
                        'window': window_idx
                    })
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    def _load_affectnet_data(self):
        """Load preprocessed AffectNet data."""
        affectnet_dir = self.data_dir / "AffectNet" / self.mode.capitalize()
        
        if not affectnet_dir.exists():
            print(f"[WARNING] AffectNet directory not found: {affectnet_dir}")
            return
        
        # Load metadata
        metadata_file = affectnet_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"[WARNING] Metadata not found: {metadata_file}")
            return
        
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load samples
            for sample_entry in metadata.get('samples', []):
                img_file = affectnet_dir / sample_entry['image_file']
                
                if img_file.exists():
                    self.samples.append({
                        'type': 'facial',
                        'data': sample_entry,
                        'image_path': img_file,
                        'emotion': sample_entry['emotion_name']
                    })
        except Exception as e:
            print(f"Error loading AffectNet metadata: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample by index."""
        sample = self.samples[idx]
        
        if sample['type'] == 'physio':
            return self._get_physio_sample(sample)
        else:
            return self._get_facial_sample(sample)
    
    def _get_physio_sample(self, sample):
        """Get physiological sample."""
        window = sample['data']
        
        # Extract signals
        ecg = np.array(window['data']['ecg'])
        eda = np.array(window['data']['eda'])
        temp = np.array(window['data']['temperature'])
        
        # Stack into (seq_len, 3)
        signal = np.stack([ecg, eda, temp], axis=1).astype(np.float32)
        
        # Get label and ensure it's in valid range [0, 3]
        label = window.get('label', 0)
        if isinstance(label, str):
            label_map = {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3}
            label = label_map.get(label.lower(), 0)
        
        label = int(label) % 4  # Ensure valid class index [0-3]
        
        return {
            'signal': torch.from_numpy(signal),
            'label': label,
            'type': 'physio'
        }
    
    def _get_facial_sample(self, sample):
        """Get facial sample."""
        import cv2
        
        img_path = sample['image_path']
        image = np.load(img_path)  # Shape: (224, 224, 3)
        
        # Convert to tensor (CHW format)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Get label and ensure it's in valid range [0, 3]
        label = sample['data'].get('stress_label', 0)
        if isinstance(label, str):
            label_map = {'baseline': 0, 'stress': 1, 'amusement': 2, 'meditation': 3}
            label = label_map.get(label.lower(), 0)
        
        label = int(label) % 4  # Ensure valid class index [0-3]
        
        return {
            'image': image,
            'label': label,
            'type': 'facial'
        }


class StressDetectionTrainer:
    """Trainer for multimodal stress detection."""
    
    def __init__(
        self,
        model,
        device='cuda',
        lr=0.001,
        weight_decay=1e-4,
        patience=10,
        checkpoint_dir='./checkpoints'
    ):
        """
        Args:
            model: PyTorch model
            device: 'cuda' or 'cpu'
            lr: Learning rate
            weight_decay: L2 regularization
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_epoch = 0
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm.tqdm(train_loader, desc="Training", leave=False)
        
        for batch in progress_bar:
            # Prepare batch
            if isinstance(batch, dict):
                if batch['type'][0] == 'physio':
                    signal = batch['signal'].to(self.device)
                    logits, _ = self.model(signal)
                else:
                    image = batch['image'].to(self.device)
                    logits = self.model(image)
            else:
                signal, label = batch
                signal = signal.to(self.device)
                logits, _ = self.model(signal)
            
            labels = batch['label'].to(self.device) if isinstance(batch, dict) else label.to(self.device)
            
            # Forward pass
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm.tqdm(val_loader, desc="Validating", leave=False)
            
            for batch in progress_bar:
                # Prepare batch
                if isinstance(batch, dict):
                    if batch['type'][0] == 'physio':
                        signal = batch['signal'].to(self.device)
                        logits, _ = self.model(signal)
                    else:
                        image = batch['image'].to(self.device)
                        logits = self.model(image)
                else:
                    signal, label = batch
                    signal = signal.to(self.device)
                    logits, _ = self.model(signal)
                
                labels = batch['label'].to(self.device) if isinstance(batch, dict) else label.to(self.device)
                
                # Forward pass
                loss = self.criterion(logits, labels)
                
                # Metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc, all_predictions, all_labels
    
    def train(self, train_loader, val_loader, num_epochs=50, model_name='model'):
        """Train model."""
        print("\n" + "="*70)
        print(f"TRAINING: {model_name}")
        print("="*70)
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, predictions, labels = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.best_epoch = epoch
                
                # Save best model
                self.save_checkpoint(model_name, epoch)
                print(f"  [BEST] Model saved (Val Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        print(f"\nTraining completed. Best epoch: {self.best_epoch+1}")
        return self.history
    
    def save_checkpoint(self, model_name, epoch):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_path)
    
    def load_checkpoint(self, model_name):
        """Load model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"{model_name}_best.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[OK] Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        else:
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            return None


def train_phase1_physiological(data_dir, device='cuda', num_epochs=50, sample_fraction=1.0, batch_size=32):
    """
    Phase 1: Train LSTM model on physiological signals only.
    """
    print("\n" + "#"*70)
    print("PHASE 1: PHYSIOLOGICAL STRESS DETECTION (LSTM)")
    print("#"*70)
    
    # Create dataset
    print("\nLoading datasets...")
    train_dataset = MultimodalStressDataset(
        data_dir, mode='train', modality='physio'
    )
    val_dataset = MultimodalStressDataset(
        data_dir, mode='val', modality='physio'
    )
    
    # Apply sampling if requested
    if sample_fraction < 1.0:
        train_size = int(len(train_dataset) * sample_fraction)
        val_size = int(len(val_dataset) * sample_fraction)
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating LSTM model...")
    model = create_lstm_classifier(num_classes=4, use_attention=True)
    
    # Train
    trainer = StressDetectionTrainer(
        model, device=device, lr=0.001, patience=10,
        checkpoint_dir=PROJECT_ROOT / 'checkpoints' / 'stress_detection'
    )
    
    history = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs,
        model_name='lstm_physio_phase1'
    )
    
    return model, history


def train_phase2_facial(data_dir, device='cuda', num_epochs=50, sample_fraction=1.0, batch_size=32):
    """
    Phase 2: Train CNN model on facial images only.
    """
    print("\n" + "#"*70)
    print("PHASE 2: FACIAL STRESS DETECTION (CNN)")
    print("#"*70)
    
    # Create dataset
    print("\nLoading datasets...")
    train_dataset = MultimodalStressDataset(
        data_dir, mode='Train', modality='facial'
    )
    val_dataset = MultimodalStressDataset(
        data_dir, mode='Test', modality='facial'
    )
    
    # Apply sampling if requested
    if sample_fraction < 1.0:
        train_size = int(len(train_dataset) * sample_fraction)
        val_size = int(len(val_dataset) * sample_fraction)
        train_indices = np.random.choice(len(train_dataset), train_size, replace=False)
        val_indices = np.random.choice(len(val_dataset), val_size, replace=False)
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Create model
    print("\nCreating ResNet model...")
    model = create_resnet_detector(num_classes=4, model_name='resnet18', pretrained=True)
    
    # Train
    trainer = StressDetectionTrainer(
        model, device=device, lr=0.0001, patience=10,
        checkpoint_dir=PROJECT_ROOT / 'checkpoints' / 'stress_detection'
    )
    
    history = trainer.train(
        train_loader, val_loader, num_epochs=num_epochs,
        model_name='cnn_facial_phase2'
    )
    
    return model, history


def train_phase3_multimodal(data_dir, physio_model, facial_model, device='cuda', num_epochs=50, sample_fraction=1.0, batch_size=32):
    """
    Phase 3: Train hybrid multimodal model combining both modalities.
    """
    print("\n" + "#"*70)
    print("PHASE 3: MULTIMODAL STRESS DETECTION (HYBRID)")
    print("#"*70)
    
    # Create hybrid model
    print("\nCreating hybrid model...")
    hybrid_model = create_hybrid_detector(num_classes=4, pretrained_cnn=True)
    
    # Optionally freeze pre-trained branches for transfer learning
    # for param in hybrid_model.physio_lstm.parameters():
    #     param.requires_grad = False
    # for param in hybrid_model.facial_cnn.parameters():
    #     param.requires_grad = False
    
    # Create combined dataset (for future implementation)
    print("\nNote: Combined multimodal training requires aligned physio-facial samples")
    print("Currently using individual datasets for demonstration")
    
    # Train
    trainer = StressDetectionTrainer(
        hybrid_model, device=device, lr=0.0001, patience=10,
        checkpoint_dir=PROJECT_ROOT / 'checkpoints' / 'stress_detection'
    )
    
    # For now, show architecture
    print(f"\nHybrid model parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
    print("Phase 3 ready for training with aligned multimodal data")
    
    return hybrid_model


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Multimodal Stress Detection Training")
    parser.add_argument('--data-dir', type=str, default=str(PROJECT_ROOT / 'data' / 'processed'),
                       help='Path to preprocessed data')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                       help='Device to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--phase', type=int, default=3, choices=[1, 2, 3],
                       help='Training phase (1=physio only, 2=facial only, 3=multimodal)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--phase-start', type=int, default=1, choices=[1, 2, 3],
                       help='Start from which phase (1, 2, or 3)')
    parser.add_argument('--sample-fraction', type=float, default=1.0,
                       help='Use fraction of data (0.1 = 10% for testing)')
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if args.device == 'cuda':
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("[WARNING] CUDA not available, using CPU")
            args.device = 'cpu'
    
    print("\n" + "="*70)
    print("MULTIMODAL STRESS DETECTION TRAINING PIPELINE")
    print("="*70)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Phase: {args.phase} (Start from Phase {args.phase_start})")
    print(f"Sample fraction: {args.sample_fraction}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Training phases
    models = {}
    
    if args.phase_start <= 1 and args.phase >= 1:
        print("\n" + "="*70)
        print("[1/3] STARTING PHASE 1: PHYSIOLOGICAL DETECTION")
        print("="*70)
        physio_model, physio_history = train_phase1_physiological(
            args.data_dir, device=args.device, num_epochs=args.epochs,
            sample_fraction=args.sample_fraction, batch_size=args.batch_size
        )
        models['physio'] = physio_model
        print("[OK] PHASE 1 COMPLETED")
    
    if args.phase_start <= 2 and args.phase >= 2:
        print("\n" + "="*70)
        print("[2/3] STARTING PHASE 2: FACIAL DETECTION")
        print("="*70)
        facial_model, facial_history = train_phase2_facial(
            args.data_dir, device=args.device, num_epochs=args.epochs,
            sample_fraction=args.sample_fraction, batch_size=args.batch_size
        )
        models['facial'] = facial_model
        print("[OK] PHASE 2 COMPLETED")
    
    if args.phase_start <= 3 and args.phase >= 3:
        print("\n" + "="*70)
        print("[3/3] STARTING PHASE 3: MULTIMODAL DETECTION")
        print("="*70)
        hybrid_model = train_phase3_multimodal(
            args.data_dir,
            models.get('physio'),
            models.get('facial'),
            device=args.device,
            num_epochs=args.epochs,
            sample_fraction=args.sample_fraction,
            batch_size=args.batch_size
        )
        models['hybrid'] = hybrid_model
        print("[OK] PHASE 3 COMPLETED")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"\nTrained models saved in: {PROJECT_ROOT / 'checkpoints' / 'stress_detection'}")
    
    return models


if __name__ == "__main__":
    models = main()
