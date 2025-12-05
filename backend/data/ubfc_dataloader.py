"""
Load and preprocess UBFC-rPPG dataset for training.
Converts rPPG signals to heart rate for model training.
"""
import os
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class UBFCDataset(Dataset):
    """
    UBFC-rPPG Dataset loader.
    Loads rPPG signals and derives heart rate from them.
    """
    
    def __init__(self, data_dir="data/processed/ubfc_rppg", subjects=None, 
                 window_size=30, fps=30, transform=None):
        """
        Args:
            data_dir: Path to UBFC dataset
            subjects: List of subject IDs to load (None = all)
            window_size: Number of frames per sequence
            fps: Frames per second
            transform: Optional data augmentation
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.fps = fps
        self.transform = transform
        
        # Find available subjects
        csv_files = list(self.data_dir.glob("subject*[0-9].csv"))
        available_subjects = [int(f.stem.replace("subject", "")) for f in csv_files]
        
        if subjects is None:
            self.subjects = sorted(available_subjects)
        else:
            self.subjects = [s for s in subjects if s in available_subjects]
        
        print(f"✓ Found {len(self.subjects)} subjects: {self.subjects}")
        
        # Load all data
        self.sequences = []
        self.ground_truth = []
        
        self._load_data()
    
    def _derive_heart_rate_from_rppg(self, rppg_signal, fps=30):
        """
        Derive heart rate (BPM) from rPPG signal using FFT.
        
        Args:
            rppg_signal: 1D array of rPPG values
            fps: Frames per second
        
        Returns:
            heart_rate: Dominant frequency converted to BPM
        """
        try:
            # Detrend
            rppg_detrended = scipy_signal.detrend(rppg_signal)
            
            # Bandpass filter (0.7-4 Hz = 42-240 BPM)
            sos = scipy_signal.butter(4, [0.7, 4.0], btype='band', fs=fps, output='sos')
            rppg_filtered = scipy_signal.sosfilt(sos, rppg_detrended)
            
            # Apply window
            windowed = rppg_filtered * scipy_signal.windows.hann(len(rppg_filtered))
            
            # FFT
            fft_vals = np.abs(fft(windowed))
            freq_axis = np.fft.fftfreq(len(windowed), d=1/fps)
            freq_axis = freq_axis[:len(freq_axis)//2]
            fft_vals = fft_vals[:len(fft_vals)//2]
            
            # Find peak in valid HR range
            mask = (freq_axis >= 0.7) & (freq_axis <= 4.0)
            if np.any(mask):
                peak_idx = np.argmax(fft_vals[mask])
                peak_freq = freq_axis[mask][peak_idx]
                heart_rate = peak_freq * 60  # Convert to BPM
                heart_rate = np.clip(heart_rate, 40, 200)  # Clamp to reasonable range
                return heart_rate
            else:
                return 70.0  # Default if no valid peak
        
        except Exception as e:
            print(f"⚠ Error deriving HR: {e}")
            return 70.0  # Default fallback
    
    def _load_data(self):
        """Load sequences from CSV files."""
        for subject_id in self.subjects:
            csv_path = self.data_dir / f"subject{subject_id}.csv"
            meta_path = self.data_dir / f"subject{subject_id}_meta.json"
            
            if not csv_path.exists():
                print(f"⚠ CSV not found: {csv_path}")
                continue
            
            # Load CSV
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"❌ Failed to load {csv_path}: {e}")
                continue
            
            print(f"\n📊 Subject {subject_id}:")
            print(f"   Rows: {len(df)}")
            print(f"   Columns: {df.columns.tolist()}")
            
            # Check for rppg column
            if 'rppg' not in df.columns:
                print(f"   ❌ No 'rppg' column found, skipping")
                continue
            
            # Convert rppg to numeric
            df['rppg'] = pd.to_numeric(df['rppg'], errors='coerce')
            df = df.dropna(subset=['rppg'])
            
            if len(df) < self.window_size:
                print(f"   ⚠ Too few samples ({len(df)} < {self.window_size}), skipping")
                continue
            
            rppg_values = df['rppg'].values
            
            # Create sliding windows
            n_sequences = len(rppg_values) - self.window_size
            
            for i in range(n_sequences):
                # Get window of rPPG signal
                window_rppg = rppg_values[i:i + self.window_size]
                
                # Derive heart rate from this window
                label = self._derive_heart_rate_from_rppg(window_rppg, fps=self.fps)
                
                # Store sequence
                self.sequences.append({
                    'subject': subject_id,
                    'start_idx': i,
                    'rppg_values': window_rppg.copy()
                })
                self.ground_truth.append(label)
            
            print(f"   ✓ Created {n_sequences} sequences")
        
        print(f"\n{'='*60}")
        print(f"✓ Total sequences: {len(self.sequences)}")
        if len(self.sequences) > 0:
            print(f"✓ Heart rate range: {np.min(self.ground_truth):.1f} - {np.max(self.ground_truth):.1f} BPM")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq_info = self.sequences[idx]
        
        # Get rPPG signal
        signal_data = seq_info['rppg_values'].astype(np.float32)
        
        # Normalize
        signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-6)
        
        # Get ground truth label
        label = self.ground_truth[idx]
        
        # Data augmentation (optional)
        if self.transform:
            signal_data = self.transform(signal_data)
        
        # ✅ FIX: Reshape to (1, seq_len) for Conv1D
        # Conv1D expects (batch, channels, length)
        # Since we have 1D signal, channels=1
        signal_data = signal_data.reshape(1, -1)
        
        return {
            'signal': torch.FloatTensor(signal_data),  # Shape: (1, 30)
            'label': torch.FloatTensor([label]),       # Shape: (1,)
            'subject': seq_info['subject']
        }

def create_train_val_test_split(dataset, train_frac=0.7, val_frac=0.15, 
                                random_state=42, batch_size=32):
    """
    Split dataset and create DataLoaders.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty!")
    
    n = len(dataset)
    indices = np.random.RandomState(random_state).permutation(n)
    
    train_idx = indices[:int(n * train_frac)]
    val_idx = indices[int(n * train_frac):int(n * (train_frac + val_frac))]
    test_idx = indices[int(n * (train_frac + val_frac)):]
    
    from torch.utils.data import Subset
    
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    print(f"\n{'='*60}")
    print(f"DATASET SPLIT")
    print(f"{'='*60}")
    print(f"Train: {len(train_set)} samples")
    print(f"Val:   {len(val_set)} samples")
    print(f"Test:  {len(test_set)} samples")
    print(f"Total: {len(dataset)} samples")
    print(f"{'='*60}\n")
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test dataset loader
    print("\nTesting UBFC Dataset Loader...")
    
    dataset = UBFCDataset(
        data_dir="data/processed/ubfc_rppg",
        window_size=30,
        fps=30
    )
    
    if len(dataset) > 0:
        train_loader, val_loader, test_loader = create_train_val_test_split(
            dataset, batch_size=32
        )
        
        # Show sample batch
        sample_batch = next(iter(train_loader))
        print(f"{'='*60}")
        print(f"SAMPLE BATCH")
        print(f"{'='*60}")
        print(f"Signal shape: {sample_batch['signal'].shape}")
        print(f"Label shape: {sample_batch['label'].shape}")
        print(f"Label values: {sample_batch['label'].squeeze().numpy()}")
        print(f"Subjects: {sample_batch['subject']}")
    else:
        print("❌ No sequences created!")