"""
Train rPPG heart rate prediction model on UBFC dataset.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from data.ubfc_dataloader import UBFCDataset, create_train_val_test_split
from models.vision.rpPG_model import rPPGCNNLSTM, rPPGResNet1D

class rPPGTrainer:
    """Train rPPG model."""
    
    def __init__(self, model, device, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            signal = batch['signal'].to(self.device)  # (batch, 1, seq_len)
            label = batch['label'].to(self.device)     # (batch, 1)
            
            # Forward pass
            pred, _ = self.model(signal)
            loss = self.criterion(pred, label)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * signal.size(0)
            count += signal.size(0)
        
        return total_loss / count
    
    def evaluate(self, val_loader):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                signal = batch['signal'].to(self.device)
                label = batch['label'].to(self.device)
                
                pred, _ = self.model(signal)
                loss = self.criterion(pred, label)
                mae = torch.abs(pred - label).mean()
                
                total_loss += loss.item() * signal.size(0)
                total_mae += mae.item() * signal.size(0)
                count += signal.size(0)
        
        return total_loss / count, total_mae / count
    
    def train(self, train_loader, val_loader, epochs=50, save_path="data/models/rppg_model.pth"):
        """Train for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"rPPG MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Epochs: {epochs}\n")
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.evaluate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.6f} | "
                  f"val_loss={val_loss:.6f} | "
                  f"val_mae={val_mae:.2f}", end="")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(" ✓ SAVED")
            else:
                print()
        
        print(f"\n✓ Training complete!")
        print(f"✓ Best model saved to: {save_path}")
        print(f"✓ Best val_loss: {self.best_val_loss:.6f}")
        
        return self.history

def train(args):
    """Main training function."""
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading UBFC-rPPG dataset...")
    dataset = UBFCDataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        fps=args.fps
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_train_val_test_split(
        dataset, batch_size=args.batch_size
    )
    
    # Create model
    if args.model == "cnn_lstm":
        model = rPPGCNNLSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_filters=args.num_filters,
            dropout=args.dropout
        )
    else:
        model = rPPGResNet1D(
            input_size=1,
            hidden_size=args.hidden_size,
            num_blocks=args.num_blocks
        )
    
    # Train
    trainer = rPPGTrainer(model, device, learning_rate=args.lr)
    history = trainer.train(
        train_loader, val_loader, 
        epochs=args.epochs,
        save_path=args.output
    )
    
    # Save history
    history_path = Path(args.output).parent / "rppg_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    print(f"✓ History saved to: {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train rPPG model on UBFC dataset")
    parser.add_argument("--data_dir", default="data/processed/ubfc_rppg", 
                       help="Path to UBFC dataset")
    parser.add_argument("--model", choices=["cnn_lstm", "resnet"], 
                       default="cnn_lstm", help="Model architecture")
    parser.add_argument("--window_size", type=int, default=30, 
                       help="Sequence window size")
    parser.add_argument("--fps", type=int, default=30, 
                       help="Frames per second")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, 
                       help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, 
                       help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=64, 
                       help="Hidden layer size")
    parser.add_argument("--num_filters", type=int, default=32, 
                       help="Number of CNN filters")
    parser.add_argument("--num_blocks", type=int, default=4, 
                       help="Number of ResNet blocks")
    parser.add_argument("--dropout", type=float, default=0.2, 
                       help="Dropout rate")
    parser.add_argument("--output", default="data/models/rppg_model.pth", 
                       help="Output model path")
    
    args = parser.parse_args()
    train(args)