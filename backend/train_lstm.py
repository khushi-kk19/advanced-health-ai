# (Updated version with validation + logging)
"""
Robust LSTM+Attention training script with data validation.
"""
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import joblib
from models.lstm_attention import LSTMAttnModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
import json
from datetime import datetime


# Import validator
from validate_data import HealthDataValidator

class CleanArgumentParser(argparse.ArgumentParser):
    """Custom parser that cleans feature argument."""
    def parse_args(self, args=None, namespace=None):
        args_obj = super().parse_args(args, namespace)
        # Clean up features: remove spaces
        if hasattr(args_obj, 'features'):
            args_obj.features = ','.join([f.strip() for f in args_obj.features.split(',')])
        return args_obj
    
class SequenceDataset(Dataset):
    def __init__(self, X_seqs, y_seqs):
        self.X = X_seqs
        self.y = y_seqs

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def build_sequences_from_df(df, feature_cols, target_col, seq_len=30):
    """Build sliding-window sequences per patient."""
    required = feature_cols + [target_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    for c in required:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    if 'patient_id' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values(['patient_id', 'timestamp']).reset_index(drop=True)
    elif 'patient_id' in df.columns:
        df = df.sort_values(['patient_id']).reset_index(drop=True)

    X_seqs = []
    y_seqs = []

    if 'patient_id' in df.columns:
        groups = df.groupby('patient_id')
    else:
        groups = [('all', df)]

    skipped_patients = 0
    for pid, g in groups:
        g = g.reset_index(drop=True)
        g_feat = g[feature_cols].apply(pd.to_numeric, errors='coerce')
        g_target = pd.to_numeric(g[target_col], errors='coerce')
        valid_mask = (~g_feat.isna().any(axis=1)) & (~g_target.isna())
        g = g.loc[valid_mask].reset_index(drop=True)
        
        print(f"Patient {pid}: {len(g)} valid rows (need > {seq_len})")
        
        if len(g) <= seq_len:
            skipped_patients += 1
            continue
            
        arr_X = g[feature_cols].values.astype(float)
        arr_y = g[target_col].values.astype(float)
        
        for i in range(0, len(arr_X) - seq_len):
            X_seqs.append(arr_X[i:i+seq_len])
            y_seqs.append(arr_y[i+seq_len])
    
    if len(X_seqs) == 0:
        raise RuntimeError("No sequences constructed. Check seq_len and data quality.")
    
    print(f"\n✓ Built {len(X_seqs)} sequences (skipped {skipped_patients} patients)")
    X_seqs = np.stack(X_seqs, axis=0)
    y_seqs = np.array(y_seqs, dtype=float)
    return X_seqs, y_seqs

def prepare_data(path, feature_cols, target_col, seq_len=30, test_frac=0.1, random_state=42):
    """Load, validate, and prepare data."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    
    X, y = build_sequences_from_df(df, feature_cols, target_col, seq_len=seq_len)

    N, S, F = X.shape
    X_flat = X.reshape(-1, F)
    scaler = StandardScaler()
    scaler.fit(X_flat)
    
    os.makedirs("data/models", exist_ok=True)
    joblib.dump(scaler, "data/models/lstm_scaler.pkl")
    print("✓ Saved scaler to data/models/lstm_scaler.pkl")

    X_scaled = scaler.transform(X_flat).reshape(N, S, F)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=test_frac, random_state=random_state, shuffle=True
    )
    
    print(f"✓ Data split: {len(X_train)} train, {len(X_val)} val")
    return X_train, y_train, X_val, y_val

def train(args):
    feature_cols = [f.strip() for f in args.features.split(",")]
    target_col = args.target
    
    print("\n" + "="*60)
    print("LSTM+ATTENTION TRAINING")
    print("="*60)
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Hyperparameters: lr={args.lr}, batch_size={args.batch_size}, "
          f"hidden={args.hidden_size}, epochs={args.epochs}\n")
    
    # Step 1: Validate data
    if args.validate:
        print("Running data validation...")
        validator = HealthDataValidator(args.csv)
        if not validator.validate_all(feature_cols, target_col, seq_len=args.seq_len):
            print("✗ Data validation failed. Exiting.")
            return
    
    # Step 2: Prepare data
    print("\nPreparing data...")
    X_train, y_train, X_val, y_val = prepare_data(
        args.csv, feature_cols, target_col,
        seq_len=args.seq_len, test_frac=0.1
    )

    train_ds = SequenceDataset(X_train, y_train)
    val_ds = SequenceDataset(X_val, y_val) if X_val is not None else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds is not None else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LSTMAttnModel(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        attn_heads=args.attn_heads
    ).to(device)
    
    opt = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...\n")
    best_val = float("inf")
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.float().to(device)
            yb = yb.float().to(device).unsqueeze(1)
            preds, _ = model(xb)
            loss = F.mse_loss(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        train_loss = total / max(1, count)
        history['train_loss'].append(train_loss)

        if val_loader is not None:
            model.eval()
            vtotal = 0.0
            vcount = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.float().to(device)
                    yb = yb.float().to(device).unsqueeze(1)
                    preds, _ = model(xb)
                    l = F.mse_loss(preds, yb)
                    vtotal += l.item() * xb.size(0)
                    vcount += xb.size(0)
            val_loss = vtotal / max(1, vcount)
        else:
            val_loss = train_loss
        
        history['val_loss'].append(val_loss)
        print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}", end="")
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "data/models/lstm_attn.pth")
            print(" ✓ SAVED")
        else:
            print()
    
    # Save training history
    with open("data/models/training_history.json", 'w') as f:
        json.dump(history, f)
    
    print(f"\n✓ Training complete. Best val_loss: {best_val:.6f}")
    print("✓ Model saved to: data/models/lstm_attn.pth")
    print("✓ Scaler saved to: data/models/lstm_scaler.pkl")
    print("✓ History saved to: data/models/training_history.json")

if __name__ == "__main__":
    parser = CleanArgumentParser(description="Train LSTM+Attention model for health timeseries")
    parser = argparse.ArgumentParser(description="Train LSTM+Attention model for health timeseries")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--features", required=True, help="Comma-separated feature names")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--seq_len", type=int, default=30, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--attn_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--validate", action="store_true", default=True, help="Run data validation first")
    
    args = parser.parse_args()
    train(args)