"""
PREPROCESSING PIPELINE: Multimodal Stress Detection
- Handle missing values & outliers
- Normalize/standardize features
- Segment data into time windows
- Remove artifacts
- Quality assessment
"""
import numpy as np
import pandas as pd
import pickle
import cv2
from pathlib import Path
import os
from scipy import signal, stats
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import zscore
import json
import warnings
warnings.filterwarnings('ignore')


# FIX: Get absolute paths
def get_project_root():
    """Get project root directory."""
    current = Path(__file__).resolve().parent
    # Navigate up: data -> backend -> advanced-health-ai
    return current.parent.parent


PROJECT_ROOT = get_project_root()
WESAD_DEFAULT = PROJECT_ROOT / "data" / "raw" / "WESAD"
AFFECTNET_DEFAULT = PROJECT_ROOT / "data" / "raw" / "affectnet" / "archive (3)"
PROCESSED_DEFAULT = PROJECT_ROOT / "data" / "processed" / "WESAD"


class PhysiologicalPreprocessor:
    """Preprocess physiological signals (ECG, EDA, Temperature)."""
    
    def __init__(self, ecg_sampling_rate=700, eda_sampling_rate=4):
        """
        Args:
            ecg_sampling_rate: ECG sampling rate in Hz (700 for WESAD)
            eda_sampling_rate: EDA sampling rate in Hz (4 for WESAD)
        """
        self.ecg_sr = ecg_sampling_rate
        self.eda_sr = eda_sampling_rate
        self.preprocessing_log = {}
    
    # ===== MISSING VALUES HANDLING =====
    def handle_missing_values(self, signal_data, method='interpolate'):
        """
        Handle missing values in signal.
        
        Args:
            signal_data: 1D array
            method: 'interpolate', 'forward_fill', 'zero'
        
        Returns:
            signal with missing values handled
        """
        if np.isnan(signal_data).sum() == 0:
            return signal_data.copy()
        
        signal_out = signal_data.copy()
        nan_indices = np.isnan(signal_out)
        
        if method == 'interpolate':
            # Linear interpolation
            x = lambda z: z.nonzero()[0]
            if x(~nan_indices).size > 1:
                signal_out[nan_indices] = np.interp(
                    x(nan_indices), 
                    x(~nan_indices), 
                    signal_out[~nan_indices]
                )
            else:
                signal_out[nan_indices] = np.nanmean(signal_out)
        
        elif method == 'forward_fill':
            # Forward fill
            idx = np.where(~nan_indices)[0]
            for i in np.where(nan_indices)[0]:
                if idx.size > 0:
                    signal_out[i] = signal_out[idx[idx < i].max()] if idx[idx < i].size > 0 else 0
        
        elif method == 'zero':
            signal_out[nan_indices] = 0
        
        return signal_out
    
    # ===== OUTLIER DETECTION & REMOVAL =====
    def detect_outliers_zscore(self, signal_data, threshold=3):
        """Detect outliers using Z-score method."""
        z_scores = np.abs(zscore(signal_data))
        outlier_mask = z_scores > threshold
        return outlier_mask
    
    def detect_outliers_iqr(self, signal_data, multiplier=1.5):
        """Detect outliers using IQR method."""
        Q1 = np.percentile(signal_data, 25)
        Q3 = np.percentile(signal_data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outlier_mask = (signal_data < lower_bound) | (signal_data > upper_bound)
        return outlier_mask
    
    def remove_outliers(self, signal_data, method='zscore', threshold=3):
        """
        Remove outliers by replacing with median/interpolation.
        
        Args:
            signal_data: 1D array
            method: 'zscore' or 'iqr'
            threshold: Z-score threshold
        
        Returns:
            signal with outliers handled
        """
        signal_out = signal_data.copy()
        
        if method == 'zscore':
            outlier_mask = self.detect_outliers_zscore(signal_out, threshold)
        else:
            outlier_mask = self.detect_outliers_iqr(signal_out)
        
        if np.sum(outlier_mask) == 0:
            return signal_out
        
        # Replace outliers with median of surrounding values
        median_val = np.median(signal_out[~outlier_mask])
        signal_out[outlier_mask] = median_val
        
        return signal_out
    
    # ===== FILTERING =====
    def apply_bandpass_filter(self, signal_data, lowcut=0.5, highcut=50, order=4):
        """
        Apply bandpass Butterworth filter.
        
        Args:
            signal_data: 1D signal
            lowcut: Low cutoff frequency
            highcut: High cutoff frequency
            order: Filter order
        
        Returns:
            Filtered signal
        """
        nyquist = self.ecg_sr / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure valid range [0.001, 0.999]
        low = np.clip(low, 0.001, 0.999)
        high = np.clip(high, 0.001, 0.999)
        
        if low >= high:
            return signal_data
        
        try:
            b, a = butter(order, [low, high], btype='band')
            filtered = filtfilt(b, a, signal_data)
            return filtered
        except Exception as e:
            print(f"    Filter error: {e}")
            return signal_data
    
    def apply_lowpass_filter(self, signal_data, cutoff=10, order=4):
        """Apply lowpass Butterworth filter."""
        nyquist = self.ecg_sr / 2
        normalized_cutoff = cutoff / nyquist
        normalized_cutoff = np.clip(normalized_cutoff, 0.001, 0.999)
        
        try:
            b, a = butter(order, normalized_cutoff, btype='low')
            filtered = filtfilt(b, a, signal_data)
            return filtered
        except:
            return signal_data
    
    def apply_notch_filter(self, signal_data, freq=50, quality=30):
        """Apply notch filter to remove powerline interference (50/60 Hz)."""
        nyquist = self.ecg_sr / 2
        normalized_freq = freq / nyquist
        normalized_freq = np.clip(normalized_freq, 0.001, 0.999)
        
        try:
            b, a = signal.iirnotch(normalized_freq, quality)
            filtered = filtfilt(b, a, signal_data)
            return filtered
        except:
            return signal_data
    
    # ===== NORMALIZATION =====
    def normalize_zscore(self, signal_data):
        """Z-score normalization (standardization)."""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            std = 1e-6
        return (signal_data - mean) / std
    
    def normalize_minmax(self, signal_data, min_val=0, max_val=1):
        """Min-Max normalization."""
        data_min = np.min(signal_data)
        data_max = np.max(signal_data)
        
        if data_max - data_min == 0:
            return np.full_like(signal_data, (min_val + max_val) / 2, dtype=float)
        
        normalized = (signal_data - data_min) / (data_max - data_min)
        return normalized * (max_val - min_val) + min_val
    
    def normalize_robust(self, signal_data):
        """Robust normalization using median and IQR."""
        median = np.median(signal_data)
        Q1 = np.percentile(signal_data, 25)
        Q3 = np.percentile(signal_data, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            IQR = 1e-6
        
        return (signal_data - median) / IQR
    
    # ===== ECG PREPROCESSING =====
    def preprocess_ecg(self, ecg_signal, remove_dc=True, filter_type='bandpass'):
        """
        Complete ECG preprocessing pipeline.
        
        Args:
            ecg_signal: Raw ECG signal
            remove_dc: Remove DC component
            filter_type: 'bandpass', 'lowpass'
        
        Returns:
            Preprocessed ECG signal
        """
        # Step 1: Handle missing values
        ecg = self.handle_missing_values(ecg_signal.copy())
        
        # Step 2: Remove DC component
        if remove_dc:
            ecg = ecg - np.mean(ecg)
        
        # Step 3: Remove 50/60 Hz powerline interference
        ecg = self.apply_notch_filter(ecg, freq=50)
        
        # Step 4: Remove outliers
        ecg = self.remove_outliers(ecg, method='zscore', threshold=3)
        
        # Step 5: Apply bandpass filter (0.5-50 Hz typical for ECG)
        if filter_type == 'bandpass':
            ecg = self.apply_bandpass_filter(ecg, lowcut=0.5, highcut=50, order=4)
        else:
            ecg = self.apply_lowpass_filter(ecg, cutoff=50, order=4)
        
        # Step 6: Normalize
        ecg = self.normalize_zscore(ecg)
        
        return ecg
    
    # ===== EDA PREPROCESSING =====
    def preprocess_eda(self, eda_signal):
        """
        EDA preprocessing pipeline.
        EDA typically has slower dynamics (0.05-5 Hz).
        
        Args:
            eda_signal: Raw EDA signal
        
        Returns:
            Preprocessed EDA signal
        """
        # Step 1: Handle missing values
        eda = self.handle_missing_values(eda_signal.copy())
        
        # Step 2: Remove outliers (more conservative for EDA)
        eda = self.remove_outliers(eda, method='iqr', threshold=1.5)
        
        # Step 3: Apply lowpass filter (0.05-5 Hz for EDA)
        eda = self.apply_bandpass_filter(eda, lowcut=0.05, highcut=5, order=2)
        
        # Step 4: Normalize
        eda = self.normalize_zscore(eda)
        
        return eda
    
    # ===== TEMPERATURE PREPROCESSING =====
    def preprocess_temperature(self, temp_signal):
        """
        Temperature preprocessing pipeline.
        Temperature changes slowly (0.01-1 Hz).
        
        Args:
            temp_signal: Raw temperature signal
        
        Returns:
            Preprocessed temperature signal
        """
        # Step 1: Handle missing values
        temp = self.handle_missing_values(temp_signal.copy())
        
        # Step 2: Remove outliers (very conservative)
        temp = self.remove_outliers(temp, method='iqr', threshold=2)
        
        # Step 3: Apply light smoothing
        temp = self.apply_bandpass_filter(temp, lowcut=0.01, highcut=1, order=2)
        
        # Step 4: Normalize
        temp = self.normalize_zscore(temp)
        
        return temp
    
    # ===== SEGMENTATION =====
    def segment_into_windows(self, ecg, eda, temperature, labels, 
                            window_size=256, overlap=0.5):
        """
        Segment signals into fixed-length windows.
        
        Args:
            ecg, eda, temperature: Preprocessed signals
            labels: Activity labels
            window_size: Samples per window
            overlap: Overlap fraction (0-1)
        
        Returns:
            List of windows with metadata
        """
        step_size = int(window_size * (1 - overlap))
        windows = []
        
        num_samples = len(ecg)
        
        for i in range(0, num_samples - window_size, step_size):
            window_end = i + window_size
            
            # Extract window
            window = {
                'ecg': ecg[i:window_end],
                'eda': eda[i:window_end],
                'temperature': temperature[i:window_end]
            }
            
            # Get dominant label
            window_labels = labels[i:window_end]
            if len(window_labels) > 0:
                dominant_label = np.bincount(window_labels).argmax()
            else:
                continue
            
            # Metadata
            info = {
                'start_idx': i,
                'end_idx': window_end,
                'duration_sec': window_size / self.ecg_sr,
                'label_distribution': np.bincount(window_labels, minlength=4).tolist(),
                'quality_score': self._assess_quality(window)
            }
            
            windows.append({
                'data': window,
                'label': dominant_label,
                'info': info
            })
        
        return windows
    
    def _assess_quality(self, window, min_signal_var=0.01):
        """
        Assess signal quality.
        
        Returns:
            quality_score: 0-1 (1 = high quality)
        """
        quality = 1.0
        
        # Check signal variance
        for signal_name in ['ecg', 'eda', 'temperature']:
            signal_var = np.var(window[signal_name])
            if signal_var < min_signal_var:
                quality -= 0.2
        
        # Check for flat segments
        for signal_name in ['ecg', 'eda']:
            flat_count = np.sum(np.diff(window[signal_name]) == 0)
            if flat_count > len(window[signal_name]) * 0.3:
                quality -= 0.2
        
        return np.clip(quality, 0, 1)


class FacialPreprocessor:
    """Preprocess facial video frames."""
    
    def __init__(self, target_size=224):
        """
        Args:
            target_size: Target image size (square)
        """
        self.target_size = target_size
    
    # ===== FACE DETECTION =====
    def detect_face(self, frame, cascade_path=None):
        """
        Detect face in frame using Haar Cascade.
        
        Args:
            frame: Video frame (BGR)
            cascade_path: Path to cascade file
        
        Returns:
            Face region or None
        """
        if cascade_path is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        try:
            face_cascade = cv2.CascadeClassifier(cascade_path)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return None
            
            # Get largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            return frame[y:y+h, x:x+w]
        except:
            return None
    
    # ===== IMAGE NORMALIZATION =====
    def normalize_image(self, frame, method='zscore'):
        """
        Normalize image.
        
        Args:
            frame: Image (H, W, 3)
            method: 'zscore', 'minmax', 'imagenet'
        
        Returns:
            Normalized image
        """
        img = frame.astype(np.float32)
        
        if method == 'zscore':
            mean = np.mean(img)
            std = np.std(img)
            if std == 0:
                std = 1e-6
            img = (img - mean) / std
        
        elif method == 'minmax':
            img = img / 255.0
        
        elif method == 'imagenet':
            # ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]) * 255
            std = np.array([0.229, 0.224, 0.225]) * 255
            img = (img - mean) / std
        
        return img
    
    # ===== PREPROCESSING PIPELINE =====
    def preprocess_frame(self, frame, detect_face=False):
        """
        Complete frame preprocessing pipeline.
        
        Args:
            frame: Input video frame (BGR)
            detect_face: Whether to detect and crop face
        
        Returns:
            Preprocessed frame (RGB, normalized, resized)
        """
        # Step 1: Detect and extract face (optional)
        if detect_face:
            face_roi = self.detect_face(frame)
            if face_roi is None:
                return None
            frame = face_roi
        
        # Step 2: Resize
        frame = cv2.resize(frame, (self.target_size, self.target_size))
        
        # Step 3: Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Step 4: Normalize
        frame = self.normalize_image(frame, method='minmax')
        
        # Step 5: Check quality
        if self._assess_frame_quality(frame) < 0.3:
            return None
        
        return frame
    
    def _assess_frame_quality(self, frame):
        """
        Assess frame quality.
        
        Returns:
            quality_score: 0-1
        """
        try:
            # Convert to grayscale for quality assessment
            gray = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Laplacian variance (sharpness)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1
            quality = min(laplacian_var / 1000, 1.0)
            
            return quality
        except:
            return 0.5


class WESADPreprocessor:
    """Complete preprocessing for WESAD dataset."""
    
    def __init__(self, wesad_dir=None, output_dir=None):
        """
        Args:
            wesad_dir: WESAD dataset directory (uses default if None)
            output_dir: Output directory for preprocessed data
        """
        # FIX: Use absolute paths with defaults
        if wesad_dir is None:
            wesad_dir = WESAD_DEFAULT
        if output_dir is None:
            output_dir = PROCESSED_DEFAULT
        
        self.wesad_dir = Path(wesad_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.physio_preproc = PhysiologicalPreprocessor()
        
        # FIX: Verify paths exist
        if not self.wesad_dir.exists():
            print(f"⚠ WESAD directory not found: {self.wesad_dir}")
            print("Expected structure:")
            print("  data/raw/WESAD/")
            print("    ├── S2/")
            print("    ├── S3/")
            print("    └── S10/")
            print("      └── S10.pkl")
        else:
            print(f"✓ WESAD directory found: {self.wesad_dir}")
    
    def preprocess_subject(self, subject_id):
        """
        Preprocess single WESAD subject.
        
        FIX: Handles both dictionary and array formats for chest_data
        
        Args:
            subject_id: Subject ID (e.g., "S10")
        
        Returns:
            Preprocessed data dictionary
        """
        pkl_file = self.wesad_dir / subject_id / f"{subject_id}.pkl"
        
        if not pkl_file.exists():
            print(f"  ⚠ {subject_id}: File not found at {pkl_file}")
            return None
        
        try:
            # Load raw data
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            chest_data = data['signal']['chest']
            labels = data['label']
            
            # FIX: Handle WESAD data structure - can be dict or array
            # chest_data is usually a dictionary with keys: 'ACC', 'ECG', 'EDA', 'Temp'
            if isinstance(chest_data, dict):
                print(f"    Chest data is dict with keys: {list(chest_data.keys())}")
                
                # Extract signals from dictionary
                ecg_raw = np.array(chest_data.get('ECG', chest_data.get('ecg', np.array([])))) 
                eda_raw = np.array(chest_data.get('EDA', chest_data.get('eda', np.array([]))))
                temp_raw = np.array(chest_data.get('Temp', chest_data.get('temp', np.array([]))))
                
                # Handle 1D arrays (might be wrapped in lists or have extra dimensions)
                ecg_raw = np.array(ecg_raw).flatten()
                eda_raw = np.array(eda_raw).flatten()
                temp_raw = np.array(temp_raw).flatten()
            
            # Original code for array format
            else:
                print(f"    Chest data is array with shape: {chest_data.shape}")
                if chest_data.ndim == 2 and chest_data.shape[1] >= 3:
                    ecg_raw = chest_data[:, 0]
                    eda_raw = chest_data[:, 1]
                    temp_raw = chest_data[:, 2]
                else:
                    print(f"  ⚠ Unexpected chest_data array shape: {chest_data.shape}")
                    return None
            
            # Validate data
            if len(ecg_raw) == 0 or len(eda_raw) == 0 or len(temp_raw) == 0:
                print(f"  ⚠ {subject_id}: Missing signal data")
                print(f"    ECG: {len(ecg_raw)}, EDA: {len(eda_raw)}, Temp: {len(temp_raw)}")
                return None
            
            # Synchronize array lengths
            min_len = min(len(ecg_raw), len(eda_raw), len(temp_raw), len(labels))
            ecg_raw = ecg_raw[:min_len]
            eda_raw = eda_raw[:min_len]
            temp_raw = temp_raw[:min_len]
            labels = labels[:min_len]
            
            print(f"    Raw data shapes - ECG: {ecg_raw.shape}, EDA: {eda_raw.shape}, Temp: {temp_raw.shape}")
            
            # Preprocess
            ecg_proc = self.physio_preproc.preprocess_ecg(ecg_raw)
            eda_proc = self.physio_preproc.preprocess_eda(eda_raw)
            temp_proc = self.physio_preproc.preprocess_temperature(temp_raw)
            
            print(f"    Preprocessing complete")
            
            # Segment into windows
            windows = self.physio_preproc.segment_into_windows(
                ecg_proc, eda_proc, temp_proc, labels
            )
            
            print(f"    Created {len(windows)} windows")
            
            result = {
                'subject_id': subject_id,
                'windows': windows,
                'metadata': {
                    'total_samples': len(ecg_raw),
                    'total_windows': len(windows),
                    'raw_stats': {
                        'ecg': {
                            'mean': float(np.mean(ecg_raw)),
                            'std': float(np.std(ecg_raw)),
                            'min': float(np.min(ecg_raw)),
                            'max': float(np.max(ecg_raw))
                        },
                        'eda': {
                            'mean': float(np.mean(eda_raw)),
                            'std': float(np.std(eda_raw)),
                            'min': float(np.min(eda_raw)),
                            'max': float(np.max(eda_raw))
                        },
                        'temp': {
                            'mean': float(np.mean(temp_raw)),
                            'std': float(np.std(temp_raw)),
                            'min': float(np.min(temp_raw)),
                            'max': float(np.max(temp_raw))
                        }
                    }
                }
            }
            
            return result
        
        except Exception as e:
            print(f"  ✗ {subject_id}: Error - {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def preprocess_all(self):
        """Preprocess all WESAD subjects."""
        print("\n" + "="*70)
        print("PREPROCESSING WESAD DATASET")
        print("="*70)
        print(f"WESAD directory: {self.wesad_dir}\n")
        
        # FIX: Check if directory exists
        if not self.wesad_dir.exists():
            print(f"✗ WESAD directory does not exist!")
            print(f"Expected path: {self.wesad_dir}")
            print(f"Current working directory: {Path.cwd()}")
            print(f"Project root: {PROJECT_ROOT}")
            return None, None
        
        # FIX: Find available subjects
        try:
            subjects = sorted([
                d.name for d in self.wesad_dir.iterdir() 
                if d.is_dir() and d.name.startswith('S')
            ])
        except Exception as e:
            print(f"✗ Error reading directory: {e}")
            return None, None
        
        if not subjects:
            print("✗ No subjects found!")
            print(f"Contents of {self.wesad_dir}:")
            try:
                for item in self.wesad_dir.iterdir():
                    print(f"  - {item.name}")
            except:
                pass
            return None, None
        
        print(f"Found {len(subjects)} subjects: {subjects}\n")
        
        all_windows = []
        all_labels = []
        successful = 0
        
        for subject_id in subjects:
            print(f"Processing {subject_id}...")
            result = self.preprocess_subject(subject_id)
            
            if result:
                all_windows.extend(result['windows'])
                for window in result['windows']:
                    all_labels.append(window['label'])
                
                # Save individual subject
                output_file = self.output_dir / f"{subject_id}_preprocessed.json"
                with open(output_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON
                    data_to_save = {
                        'subject_id': subject_id,
                        'windows': [
                            {
                                'data': {
                                    'ecg': w['data']['ecg'].tolist(),
                                    'eda': w['data']['eda'].tolist(),
                                    'temperature': w['data']['temperature'].tolist()
                                },
                                'label': int(w['label']),
                                'info': w['info']
                            }
                            for w in result['windows']
                        ],
                        'metadata': result['metadata']
                    }
                    json.dump(data_to_save, f, indent=2)
                
                print(f"  ✓ {len(result['windows'])} windows extracted")
                print(f"  ✓ Saved to {output_file}")
                successful += 1
        
        # Summary statistics
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"✓ Successfully processed: {successful}/{len(subjects)} subjects")
        print(f"✓ Total windows: {len(all_windows)}")
        
        if len(all_labels) > 0:
            print(f"✓ Low stress (0): {sum(1 for l in all_labels if l == 0)}")
            print(f"✓ High stress (1): {sum(1 for l in all_labels if l == 1)}")
            
            # Save combined dataset
            combined_file = self.output_dir / "combined_dataset.json"
            with open(combined_file, 'w') as f:
                json.dump({
                    'total_windows': len(all_windows),
                    'label_distribution': {
                        '0': sum(1 for l in all_labels if l == 0),
                        '1': sum(1 for l in all_labels if l == 1)
                    }
                }, f, indent=2)
            
            print(f"✓ Saved combined metadata to {combined_file}")
        
        print("="*70)
        
        return all_windows, all_labels


class AffectNetPreprocessor:
    """Preprocess AffectNet facial dataset for stress detection."""
    
    # AffectNet emotion names to IDs and stress levels
    EMOTION_MAP = {
        'neutral': (0, 0.0),      # Neutral -> Low stress
        'happy': (1, 0.2),        # Happy -> Low stress (positive)
        'sad': (2, 0.9),          # Sad -> High stress
        'surprise': (3, 0.95),    # Surprise -> Very high stress
        'fear': (4, 0.85),        # Fear -> High stress
        'disgust': (5, 0.3),      # Disgust -> Medium stress
        'anger': (6, 0.7),        # Anger -> High stress
        'contempt': (7, 0.2),     # Contempt -> Low stress
    }
    
    # Map stress values to WESAD-like labels (0-3)
    # 0: Baseline, 1: Stress, 2: Positive/Amusement, 3: Meditation/Calm
    STRESS_TO_LABEL = {
        'neutral': 0,      # Baseline
        'happy': 2,        # Positive/Amusement
        'sad': 1,          # Stress
        'surprise': 1,     # Stress
        'fear': 1,         # Stress
        'disgust': 1,      # Stress
        'anger': 1,        # Stress
        'contempt': 0,     # Baseline
    }
    
    def __init__(self, affectnet_dir=None, output_dir=None, target_size=224):
        """
        Args:
            affectnet_dir: AffectNet dataset directory (uses default if None)
            output_dir: Output directory for preprocessed facial data
            target_size: Target image size (224x224 default)
        """
        if affectnet_dir is None:
            affectnet_dir = AFFECTNET_DEFAULT
        if output_dir is None:
            output_dir = PROJECT_ROOT / "data" / "processed" / "AffectNet"
        
        self.affectnet_dir = Path(affectnet_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.target_size = target_size
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ AffectNet directory: {self.affectnet_dir}")
        print(f"✓ Output directory: {self.output_dir}")
    
    def emotion_to_stress_label(self, emotion_name):
        """
        Convert AffectNet emotion to stress level.
        
        Args:
            emotion_name: Emotion name (e.g., 'neutral', 'happy', 'sad')
        
        Returns:
            stress_label: int [0-3] matching WESAD labels
            stress_value: float [0-1] stress intensity
        """
        emotion_lower = emotion_name.lower()
        
        if emotion_lower in self.STRESS_TO_LABEL:
            label = self.STRESS_TO_LABEL[emotion_lower]
            emotion_id, stress_value = self.EMOTION_MAP[emotion_lower]
            return label, stress_value
        
        # Default for unknown emotions
        return 0, 0.5
    
    def preprocess_image(self, image_path, detect_face=True):
        """
        Preprocess single facial image.
        
        Args:
            image_path: Path to image file
            detect_face: Whether to detect and crop face region
        
        Returns:
            Preprocessed image (H, W, 3) normalized [0, 1] or None if failed
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            # Detect and crop face if requested
            if detect_face:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                if len(faces) == 0:
                    # No face detected, use full image
                    pass
                else:
                    # Use largest face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    # Add padding
                    padding = 10
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2*padding)
                    h = min(image.shape[0] - y, h + 2*padding)
                    image = image[y:y+h, x:x+w]
            
            # Resize to target size
            image = cv2.resize(image, (self.target_size, self.target_size))
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Quality check (not too dark/bright)
            mean_intensity = np.mean(image)
            if mean_intensity < 0.1 or mean_intensity > 0.95:
                return None
            
            return image
        
        except Exception as e:
            print(f"    ⚠ Error preprocessing {image_path.name}: {e}")
            return None
    
    def parse_affectnet_structure(self):
        """
        Parse AffectNet directory structure.
        
        Expected structure:
        affectnet/archive (3)/
        ├── Train/
        │   ├── anger/
        │   ├── contempt/
        │   ├── disgust/
        │   ├── fear/
        │   ├── happy/
        │   ├── neutral/
        │   ├── sad/
        │   └── surprise/
        └── Test/
            └── (same emotion folders)
        
        Returns:
            Dictionary with dataset splits and image paths
        """
        dataset_structure = {'Train': {}, 'Test': {}}
        
        for split in ['Train', 'Test']:
            split_dir = self.affectnet_dir / split
            
            if not split_dir.exists():
                print(f"  ⚠ Split directory not found: {split_dir}")
                continue
            
            # Iterate through emotion directories
            for emotion_dir in split_dir.iterdir():
                if not emotion_dir.is_dir():
                    continue
                
                emotion_name = emotion_dir.name.lower()
                
                # Check if it's a valid emotion
                if emotion_name not in self.EMOTION_MAP:
                    print(f"  ⚠ Unknown emotion directory: {emotion_name}")
                    continue
                
                # Collect images (both .jpg and .png)
                image_files = sorted([
                    f for f in emotion_dir.glob('*.jpg') 
                    if f.is_file()
                ] + [
                    f for f in emotion_dir.glob('*.png') 
                    if f.is_file()
                ])
                
                dataset_structure[split][emotion_name] = {
                    'emotion_name': emotion_name,
                    'image_files': image_files,
                    'count': len(image_files)
                }
                
                print(f"  Found {len(image_files)} images in {split}/{emotion_name}")
        
        return dataset_structure
    
    def preprocess_split(self, split='Train', max_images_per_emotion=None):
        """
        Preprocess all images in a dataset split.
        
        Args:
            split: 'Train' or 'Test'
            max_images_per_emotion: Limit images per emotion (for testing)
        
        Returns:
            List of preprocessed samples with labels
        """
        dataset_structure = self.parse_affectnet_structure()
        
        if split not in dataset_structure:
            print(f"✗ Split '{split}' not found")
            return []
        
        print(f"\n{'='*70}")
        print(f"PREPROCESSING AFFECTNET - {split.upper()}")
        print(f"{'='*70}\n")
        
        processed_samples = []
        emotion_counts = {}
        
        for emotion_name in sorted(dataset_structure[split].keys()):
            emotion_data = dataset_structure[split][emotion_name]
            image_files = emotion_data['image_files']
            
            # Limit images if specified (for testing)
            if max_images_per_emotion:
                image_files = image_files[:max_images_per_emotion]
            
            print(f"Processing {emotion_name}: {len(image_files)} images...")
            
            processed_count = 0
            failed_count = 0
            
            for img_path in image_files:
                # Preprocess image
                processed_img = self.preprocess_image(img_path, detect_face=True)
                
                if processed_img is None:
                    failed_count += 1
                    continue
                
                # Convert emotion to stress label
                stress_label, stress_value = self.emotion_to_stress_label(emotion_name)
                emotion_id = self.EMOTION_MAP[emotion_name][0]
                
                # Create sample
                sample = {
                    'image': processed_img,  # Will be saved as numpy array
                    'emotion_id': int(emotion_id),
                    'emotion_name': emotion_name,
                    'stress_label': int(stress_label),
                    'stress_value': float(stress_value),
                    'filename': img_path.name
                }
                
                processed_samples.append(sample)
                processed_count += 1
            
            emotion_counts[emotion_name] = {
                'processed': processed_count,
                'failed': failed_count,
                'total': len(image_files)
            }
            
            print(f"  ✓ Processed: {processed_count}, Failed: {failed_count}\n")
        
        # Print summary
        print(f"{'='*70}")
        print(f"SPLIT SUMMARY: {split.upper()}")
        print(f"{'='*70}")
        total_processed = sum(v['processed'] for v in emotion_counts.values())
        total_failed = sum(v['failed'] for v in emotion_counts.values())
        
        for emotion_name, counts in emotion_counts.items():
            print(f"  {emotion_name}: {counts['processed']} processed, {counts['failed']} failed")
        
        print(f"\n  Total: {total_processed} processed, {total_failed} failed")
        print(f"{'='*70}\n")
        
        return processed_samples
    
    def save_processed_split(self, processed_samples, split='Train'):
        """
        Save preprocessed images to directory structure.
        
        Args:
            processed_samples: List of preprocessed samples
            split: 'Train' or 'Test'
        """
        split_output_dir = self.output_dir / split
        split_output_dir.mkdir(parents=True, exist_ok=True)
        
        metadata = {
            'split': split,
            'total_samples': len(processed_samples),
            'samples': []
        }
        
        stress_distribution = {0: 0, 1: 0, 2: 0, 3: 0}
        emotion_distribution = {}
        
        for idx, sample in enumerate(processed_samples):
            # Save image as numpy file
            img_filename = f"sample_{idx:06d}.npy"
            img_path = split_output_dir / img_filename
            
            np.save(img_path, sample['image'])
            
            # Track emotion distribution
            emotion_name = sample['emotion_name']
            if emotion_name not in emotion_distribution:
                emotion_distribution[emotion_name] = 0
            emotion_distribution[emotion_name] += 1
            
            # Update metadata (don't store image data, just reference)
            metadata_entry = {
                'index': idx,
                'image_file': img_filename,
                'original_filename': sample['filename'],
                'emotion_id': sample['emotion_id'],
                'emotion_name': sample['emotion_name'],
                'stress_label': sample['stress_label'],
                'stress_value': sample['stress_value']
            }
            metadata['samples'].append(metadata_entry)
            
            stress_distribution[sample['stress_label']] += 1
        
        # Save metadata
        metadata['stress_distribution'] = stress_distribution
        metadata['emotion_distribution'] = emotion_distribution
        metadata_file = split_output_dir / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved {len(processed_samples)} samples to {split_output_dir}")
        print(f"✓ Metadata saved to {metadata_file}")
        print(f"✓ Stress distribution: {stress_distribution}")
        print(f"✓ Emotion distribution: {emotion_distribution}")
    
    def preprocess_all(self):
        """Preprocess entire AffectNet dataset (Train and Test splits)."""
        print("\n" + "="*70)
        print("PREPROCESSING AFFECTNET DATASET")
        print("="*70)
        print(f"AffectNet directory: {self.affectnet_dir}\n")
        
        if not self.affectnet_dir.exists():
            print(f"✗ AffectNet directory not found: {self.affectnet_dir}")
            print(f"Expected structure: {self.affectnet_dir}/Train/emotion_name/")
            print(f"Expected emotions: {list(self.EMOTION_MAP.keys())}")
            return
        
        # Process both splits
        for split in ['Train', 'Test']:
            print(f"\n{'#'*70}")
            print(f"# Processing {split.upper()} split")
            print(f"{'#'*70}")
            
            # Limit images per emotion for testing (set to None for full dataset)
            processed_samples = self.preprocess_split(split=split, max_images_per_emotion=None)
            
            if processed_samples:
                self.save_processed_split(processed_samples, split=split)
            else:
                print(f"⚠ No samples processed for {split} split")


if __name__ == "__main__":
    print("\n🚀 MULTIMODAL PREPROCESSING PIPELINE\n")
    print("="*70)
    print("PATHS CONFIGURATION")
    print("="*70)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"WESAD path: {WESAD_DEFAULT}")
    print(f"AffectNet path: {AFFECTNET_DEFAULT}")
    print(f"Output path (WESAD): {PROCESSED_DEFAULT}")
    print(f"Output path (AffectNet): {PROJECT_ROOT / 'data' / 'processed' / 'AffectNet'}")
    print("="*70 + "\n")
    
    # Choice of what to preprocess
    print("Choose preprocessing option:")
    print("1. WESAD physiological signals only")
    print("2. AffectNet facial data only")
    print("3. Both WESAD and AffectNet (full multimodal)")
    
    choice = input("\nEnter choice (1-3) [default: 3]: ").strip() or "3"
    
    # ===== PREPROCESS WESAD =====
    if choice in ['1', '3']:
        print("\n" + "="*70)
        print("STEP 1: PREPROCESSING WESAD PHYSIOLOGICAL DATA")
        print("="*70)
        
        wesad_preprocessor = WESADPreprocessor()
        wesad_windows, wesad_labels = wesad_preprocessor.preprocess_all()
        
        if wesad_windows:
            print(f"\n✓ WESAD preprocessing successful!")
            print(f"  Total windows: {len(wesad_windows)}")
            print(f"  Baseline (0): {sum(1 for l in wesad_labels if l == 0)}")
            print(f"  Stress (1): {sum(1 for l in wesad_labels if l == 1)}")
            print(f"  Amusement (2): {sum(1 for l in wesad_labels if l == 2)}")
            print(f"  Meditation (3): {sum(1 for l in wesad_labels if l == 3)}")
        else:
            print("\n✗ WESAD preprocessing failed")
    
    # ===== PREPROCESS AFFECTNET =====
    if choice in ['2', '3']:
        print("\n" + "="*70)
        print("STEP 2: PREPROCESSING AFFECTNET FACIAL DATA")
        print("="*70)
        
        affectnet_preprocessor = AffectNetPreprocessor()
        affectnet_preprocessor.preprocess_all()
        
        print(f"\n✓ AffectNet preprocessing complete!")
    
    print("\n" + "="*70)
    print("MULTIMODAL PREPROCESSING PIPELINE COMPLETE")
    print("="*70)
    print(f"\n✓ Preprocessed data saved to:")
    print(f"  - WESAD: {PROCESSED_DEFAULT}")
    print(f"  - AffectNet: {PROJECT_ROOT / 'data' / 'processed' / 'AffectNet'}")
    print(f"\n✓ Ready for feature extraction and model training!")
