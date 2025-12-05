"""
Download and setup WESAD Dataset.
WESAD: Wearable Sensor-based Stress and Affect Detection
- 15 subjects
- Multimodal data: ECG, EDA, temperature, accelerometer
- Labeled stress/relax sessions
"""
import os
import urllib.request
import pickle
from pathlib import Path
import json

class WESADDownloader:
    """Download WESAD dataset."""
    
    def __init__(self, output_dir="data/raw/WESAD"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # WESAD metadata
        self.dataset_info = {
            "name": "WESAD: Wearable Sensor-based Stress and Affect Detection",
            "subjects": 15,
            "sessions_per_subject": 2,
            "duration_per_session": 20,  # minutes
            "sampling_rates": {
                "ECG": 700,
                "EDA": 4,
                "temperature": 4,
                "accelerometer": 64
            },
            "labels": {
                0: "baseline/relax",
                1: "stress (TSST)",
                2: "amusement",
                3: "meditation"
            },
            "source": "https://github.com/MariusBuchs/WESAD"
        }
    
    def download_wesad(self):
        """
        Download WESAD dataset.
        Note: You need to manually download from GitHub or request access.
        """
        print("\n" + "="*70)
        print("WESAD DATASET DOWNLOAD GUIDE")
        print("="*70)
        
        print("""
WESAD Dataset Information:
- Name: Wearable Sensor-based Stress and Affect Detection
- Subjects: 15
- Data types: ECG, EDA, Skin Temperature, Accelerometer
- Conditions: Baseline, Stress (TSST), Amusement, Meditation

HOW TO DOWNLOAD:
1. Go to: https://github.com/MariusBuchs/WESAD
2. Click "Releases" 
3. Download: WESAD.zip (large file ~2GB)
4. Extract to: data/raw/WESAD/

STRUCTURE AFTER EXTRACTION:
data/raw/WESAD/
├── S2/
│   ├── S2.pkl  (subject 2 data)
│   └── S2_E4_Data/  (raw wearable data)
├── S3/
├── S4/
...
├── S17/
└── WESAD_README.txt
        """)
        
        print("\nAlternatively, download via Python:")
        print("pip install gdown")
        print("gdown 'SHARE_LINK_FROM_README' -O data/raw/WESAD.zip")
        
        return self.dataset_info
    
    def load_wesad_subject(self, subject_id: int):
        """
        Load a single subject's data from WESAD.
        
        Args:
            subject_id: 2-17 (15 subjects)
        
        Returns:
            dict with keys: 'signal', 'label', 'subject'
        """
        pkl_file = self.output_dir / f"S{subject_id}" / f"S{subject_id}.pkl"
        
        if not pkl_file.exists():
            print(f"⚠ File not found: {pkl_file}")
            print(f"Please download WESAD dataset first")
            return None
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            print(f"✓ Loaded S{subject_id}: {len(data)} keys")
            return data
        
        except Exception as e:
            print(f"✗ Error loading {pkl_file}: {e}")
            return None
    
    def extract_features(self, data: dict) -> dict:
        """
        Extract stress-relevant features from WESAD data.
        
        WESAD structure:
        data['signal']['chest']: contains ECG, EDA, temperature
        data['label']: activity labels (0=baseline, 1=stress, 2=amusement, 3=meditation)
        """
        features = {
            'subject_id': None,
            'sessions': []
        }
        
        try:
            if 'signal' not in data or 'chest' not in data['signal']:
                return features
            
            chest_data = data['signal']['chest']
            labels = data['label']
            
            # WESAD format: [ECG, EDA, Temperature, Accelerometer_x, Accelerometer_y, Accelerometer_z]
            print(f"Signal shape: {chest_data.shape}")
            print(f"Label shape: {labels.shape}")
            
            # Extract channels
            ecg = chest_data[:, 0]
            eda = chest_data[:, 1]
            temperature = chest_data[:, 2]
            
            # Find unique sessions
            unique_labels = set(labels)
            
            for label in unique_labels:
                label_name = {
                    0: "baseline",
                    1: "stress",
                    2: "amusement",
                    3: "meditation"
                }.get(label, "unknown")
                
                indices = np.where(labels == label)[0]
                
                session = {
                    'label': label_name,
                    'samples': len(indices),
                    'ecg_mean': float(np.mean(ecg[indices])),
                    'ecg_std': float(np.std(ecg[indices])),
                    'eda_mean': float(np.mean(eda[indices])),
                    'eda_std': float(np.std(eda[indices])),
                    'temp_mean': float(np.mean(temperature[indices])),
                    'temp_std': float(np.std(temperature[indices])),
                }
                
                features['sessions'].append(session)
            
            return features
        
        except Exception as e:
            print(f"Error extracting features: {e}")
            return features
    
    def save_dataset_info(self):
        """Save dataset metadata."""
        info_file = self.output_dir / "dataset_info.json"
        with open(info_file, 'w') as f:
            json.dump(self.dataset_info, f, indent=2)
        print(f"✓ Saved dataset info to {info_file}")

if __name__ == "__main__":
    import numpy as np
    
    downloader = WESADDownloader()
    downloader.download_wesad()
    downloader.save_dataset_info()