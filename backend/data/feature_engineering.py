"""
FEATURE ENGINEERING: Extract meaningful features from preprocessed signals
- Physiological features (HRV, EDA, Temperature)
- Temporal/statistical features
- Frequency domain features
- Facial features (landmarks, action units, temporal changes)
- Multimodal stress indicators
"""

import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks, welch, csd
from scipy.fft import fft, fftfreq
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# AffectNet emotion categories and stress mapping
AFFECTNET_EMOTIONS = {
    0: 'Neutral',
    1: 'Happy',
    2: 'Sad',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
    7: 'Contempt'
}

# Map AffectNet emotions to stress levels
# 0: Baseline (low stress)
# 1: Stress (high stress)
# 2: Amusement (positive valence, low stress)
# 3: Meditation (calm, very low stress)
EMOTION_TO_STRESS_MAP = {
    'Neutral': 0,      # Baseline
    'Happy': 2,        # Amusement
    'Sad': 1,          # Stress
    'Surprise': 1,     # Stress
    'Fear': 1,         # Stress
    'Disgust': 1,      # Stress
    'Anger': 1,        # Stress
    'Contempt': 1      # Stress
}

# Valence (positivity) mapping
EMOTION_VALENCE = {
    'Neutral': 0.5,
    'Happy': 1.0,
    'Sad': 0.0,
    'Surprise': 0.6,
    'Fear': 0.1,
    'Disgust': 0.2,
    'Anger': 0.2,
    'Contempt': 0.3
}

# Arousal mapping (intensity)
EMOTION_AROUSAL = {
    'Neutral': 0.3,
    'Happy': 0.8,
    'Sad': 0.4,
    'Surprise': 0.9,
    'Fear': 0.9,
    'Disgust': 0.7,
    'Anger': 0.9,
    'Contempt': 0.6
}


class PhysiologicalFeatures:
    """Extract physiological features from ECG, EDA, Temperature signals."""
    
    def __init__(self, ecg_sr=700, eda_sr=4):
        """
        Args:
            ecg_sr: ECG sampling rate (Hz)
            eda_sr: EDA sampling rate (Hz)
        """
        self.ecg_sr = ecg_sr
        self.eda_sr = eda_sr
    
    # ===== ECG FEATURES =====
    def extract_ecg_features(self, ecg_signal):
        """
        Extract ECG/HR features.
        
        Features:
        - Heart Rate (mean, std, min, max)
        - HRV (Heart Rate Variability): SDNN, RMSSD, pNN50
        - HR trend (acceleration/deceleration)
        
        Args:
            ecg_signal: Preprocessed ECG signal (1D array)
        
        Returns:
            Dictionary of ECG features
        """
        features = {}
        
        try:
            # Peak detection
            peaks, _ = find_peaks(ecg_signal, distance=self.ecg_sr//2)
            
            if len(peaks) < 2:
                return self._get_nan_ecg_features()
            
            # Calculate RR intervals (time between peaks in seconds)
            rr_intervals = np.diff(peaks) / self.ecg_sr
            
            # Heart Rate features
            hr = 60 / rr_intervals  # Convert to bpm
            features['hr_mean'] = np.mean(hr)
            features['hr_std'] = np.std(hr)
            features['hr_min'] = np.min(hr)
            features['hr_max'] = np.max(hr)
            features['hr_range'] = np.max(hr) - np.min(hr)
            
            # ===== HRV FEATURES =====
            # Time domain
            features['hrv_sdnn'] = np.std(rr_intervals)  # Standard deviation of NN intervals
            features['hrv_rmssd'] = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))  # Root mean square of successive differences
            features['hrv_nn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)  # NN intervals > 50ms
            features['hrv_pnn50'] = 100 * features['hrv_nn50'] / len(rr_intervals)  # Percentage of NN50
            
            # Frequency domain (via FFT)
            if len(rr_intervals) > 4:
                freqs, psd = welch(rr_intervals, fs=1/np.mean(rr_intervals), nperseg=min(len(rr_intervals), 64))
                
                # HRV frequency bands (normalized)
                vlf_idx = freqs < 0.04  # Very Low Frequency
                lf_idx = (freqs >= 0.04) & (freqs < 0.15)  # Low Frequency
                hf_idx = (freqs >= 0.15) & (freqs < 0.4)  # High Frequency
                
                features['hrv_vlf'] = np.sum(psd[vlf_idx])
                features['hrv_lf'] = np.sum(psd[lf_idx])
                features['hrv_hf'] = np.sum(psd[hf_idx])
                features['hrv_lf_hf_ratio'] = features['hrv_lf'] / (features['hrv_hf'] + 1e-6)
            
            # HR dynamics
            features['hr_acceleration'] = np.mean(np.diff(hr))  # Rate of change
            features['hr_deceleration'] = np.min(np.diff(hr))
            
            return features
        
        except Exception as e:
            print(f"  ⚠ ECG feature extraction error: {e}")
            return self._get_nan_ecg_features()
    
    def _get_nan_ecg_features(self):
        """Return NaN dictionary for ECG features."""
        ecg_feat_names = [
            'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
            'hrv_sdnn', 'hrv_rmssd', 'hrv_nn50', 'hrv_pnn50',
            'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio',
            'hr_acceleration', 'hr_deceleration'
        ]
        return {name: np.nan for name in ecg_feat_names}
    
    # ===== EDA FEATURES =====
    def extract_eda_features(self, eda_signal):
        """
        Extract EDA (Electrodermal Activity) features.
        
        Features:
        - EDA level (mean, std, trend)
        - EDA peaks (number, height, rise time)
        - EDA phasic activity
        
        Args:
            eda_signal: Preprocessed EDA signal (1D array)
        
        Returns:
            Dictionary of EDA features
        """
        features = {}
        
        try:
            # Overall statistics
            features['eda_mean'] = np.mean(eda_signal)
            features['eda_std'] = np.std(eda_signal)
            features['eda_min'] = np.min(eda_signal)
            features['eda_max'] = np.max(eda_signal)
            features['eda_range'] = np.max(eda_signal) - np.min(eda_signal)
            
            # Trend (linear regression slope)
            x = np.arange(len(eda_signal))
            z = np.polyfit(x, eda_signal, 1)
            features['eda_trend'] = z[0]  # Slope of EDA over time
            
            # Peak detection (peaks indicate emotional response)
            peaks, peak_properties = find_peaks(eda_signal, distance=self.eda_sr, prominence=0.1)
            
            features['eda_peak_count'] = len(peaks)
            
            if len(peaks) > 0:
                features['eda_peak_mean_height'] = np.mean(peak_properties['prominences'])
                features['eda_peak_mean_interval'] = np.mean(np.diff(peaks)) / self.eda_sr if len(peaks) > 1 else 0
            else:
                features['eda_peak_mean_height'] = 0
                features['eda_peak_mean_interval'] = 0
            
            # Rise time (time from baseline to peak)
            if len(peaks) > 0:
                rise_times = []
                for peak in peaks:
                    baseline_idx = max(0, peak - 2 * self.eda_sr)
                    rise_time = (peak - baseline_idx) / self.eda_sr
                    rise_times.append(rise_time)
                features['eda_mean_rise_time'] = np.mean(rise_times)
            else:
                features['eda_mean_rise_time'] = 0
            
            # Variability (related to emotional arousal)
            features['eda_slope_variability'] = np.std(np.diff(eda_signal))
            
            return features
        
        except Exception as e:
            print(f"  ⚠ EDA feature extraction error: {e}")
            return self._get_nan_eda_features()
    
    def _get_nan_eda_features(self):
        """Return NaN dictionary for EDA features."""
        eda_feat_names = [
            'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
            'eda_trend', 'eda_peak_count', 'eda_peak_mean_height',
            'eda_peak_mean_interval', 'eda_mean_rise_time', 'eda_slope_variability'
        ]
        return {name: np.nan for name in eda_feat_names}
    
    # ===== TEMPERATURE FEATURES =====
    def extract_temperature_features(self, temp_signal):
        """
        Extract skin temperature features.
        
        Features:
        - Temperature level and variability
        - Temperature dynamics/trends
        
        Args:
            temp_signal: Preprocessed temperature signal (1D array)
        
        Returns:
            Dictionary of temperature features
        """
        features = {}
        
        try:
            features['temp_mean'] = np.mean(temp_signal)
            features['temp_std'] = np.std(temp_signal)
            features['temp_min'] = np.min(temp_signal)
            features['temp_max'] = np.max(temp_signal)
            features['temp_range'] = np.max(temp_signal) - np.min(temp_signal)
            
            # Trend
            x = np.arange(len(temp_signal))
            z = np.polyfit(x, temp_signal, 1)
            features['temp_trend'] = z[0]
            
            # Rising/falling rate
            diffs = np.diff(temp_signal)
            features['temp_rise_rate'] = np.mean(diffs[diffs > 0]) if np.sum(diffs > 0) > 0 else 0
            features['temp_fall_rate'] = np.mean(diffs[diffs < 0]) if np.sum(diffs < 0) > 0 else 0
            
            return features
        
        except Exception as e:
            print(f"  ⚠ Temperature feature extraction error: {e}")
            return self._get_nan_temp_features()
    
    def _get_nan_temp_features(self):
        """Return NaN dictionary for temperature features."""
        temp_feat_names = [
            'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range',
            'temp_trend', 'temp_rise_rate', 'temp_fall_rate'
        ]
        return {name: np.nan for name in temp_feat_names}
    
    # ===== COMBINED PHYSIOLOGICAL FEATURES =====
    def extract_all_physiological_features(self, ecg_signal, eda_signal, temp_signal):
        """
        Extract all physiological features.
        
        Args:
            ecg_signal: Preprocessed ECG (1D array)
            eda_signal: Preprocessed EDA (1D array)
            temp_signal: Preprocessed temperature (1D array)
        
        Returns:
            Combined feature dictionary
        """
        features = {}
        
        # Extract individual signal features
        features.update(self.extract_ecg_features(ecg_signal))
        features.update(self.extract_eda_features(eda_signal))
        features.update(self.extract_temperature_features(temp_signal))
        
        # ===== CROSS-SIGNAL FEATURES =====
        # Correlations between signals
        ecg_norm = (ecg_signal - np.mean(ecg_signal)) / (np.std(ecg_signal) + 1e-6)
        eda_norm = (eda_signal - np.mean(eda_signal)) / (np.std(eda_signal) + 1e-6)
        temp_norm = (temp_signal - np.mean(temp_signal)) / (np.std(temp_signal) + 1e-6)
        
        features['ecg_eda_correlation'] = np.corrcoef(ecg_norm, eda_norm)[0, 1]
        features['ecg_temp_correlation'] = np.corrcoef(ecg_norm, temp_norm)[0, 1]
        features['eda_temp_correlation'] = np.corrcoef(eda_norm, temp_norm)[0, 1]
        
        return features


class FacialFeatures:
    """Extract facial features from preprocessed frames - supports AffectNet emotions."""
    
    def __init__(self):
        """Initialize facial feature extractor."""
        pass
    
    # ===== FRAME-LEVEL FEATURES =====
    def extract_frame_statistics(self, frame):
        """
        Extract basic statistics from frame (color, brightness, texture).
        
        Args:
            frame: Preprocessed frame (H, W, 3) in range [0, 1]
        
        Returns:
            Dictionary of frame statistics
        """
        features = {}
        
        try:
            # Per-channel statistics
            for c_idx, channel in enumerate(['red', 'green', 'blue']):
                channel_data = frame[:, :, c_idx].flatten()
                features[f'{channel}_mean'] = np.mean(channel_data)
                features[f'{channel}_std'] = np.std(channel_data)
                features[f'{channel}_entropy'] = stats.entropy(np.histogram(channel_data, bins=16)[0])
            
            # Overall brightness
            gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
            features['brightness'] = np.mean(gray)
            features['contrast'] = np.std(gray)
            
            # Edge density (indicates expression changes and facial muscle activity)
            edges_x = np.abs(np.diff(gray, axis=0))
            edges_y = np.abs(np.diff(gray, axis=1))
            features['edge_density'] = (np.mean(edges_x) + np.mean(edges_y)) / 2
            
            # Texture energy (related to skin tension, emotion intensity)
            features['texture_energy'] = np.var(gray)
            
            return features
        
        except Exception as e:
            print(f"  ⚠ Frame statistics error: {e}")
            return {}
    
    # ===== TEMPORAL DYNAMICS FEATURES =====
    def extract_temporal_features(self, frame_sequence):
        """
        Extract temporal/motion features from sequence of frames.
        
        Features relate to facial expression changes over time:
        - Higher motion = more emotional expression
        - Consistency indicates emotion stability
        
        Args:
            frame_sequence: List of preprocessed frames (T, H, W, 3)
        
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        try:
            if len(frame_sequence) < 2:
                return features
            
            # Frame-to-frame optical flow approximation
            frame_diffs = []
            for i in range(len(frame_sequence) - 1):
                diff = np.mean(np.abs(frame_sequence[i+1] - frame_sequence[i]))
                frame_diffs.append(diff)
            
            features['frame_change_mean'] = np.mean(frame_diffs)
            features['frame_change_std'] = np.std(frame_diffs)
            features['frame_change_max'] = np.max(frame_diffs)
            features['frame_change_min'] = np.min(frame_diffs)
            
            # Motion intensity (related to emotional arousal)
            features['motion_intensity'] = np.sum(frame_diffs) / len(frame_sequence)
            
            # Temporal consistency (lower = more expression variation = more stress)
            features['temporal_consistency'] = 1 - features['motion_intensity']
            
            # Motion acceleration (change in motion rate)
            if len(frame_diffs) > 1:
                motion_diffs = np.diff(frame_diffs)
                features['motion_acceleration'] = np.mean(np.abs(motion_diffs))
            else:
                features['motion_acceleration'] = 0
            
            return features
        
        except Exception as e:
            print(f"  ⚠ Temporal features error: {e}")
            return {}
    
    # ===== REGIONAL FACIAL FEATURES =====
    def extract_facial_regions(self, frame_sequence):
        """
        Extract features from different facial regions.
        
        Regions:
        - Upper face: eyes, eyebrows (linked to intensity/surprise)
        - Lower face: mouth, jaw (linked to happiness/speech)
        - Left/Right symmetry (authentic emotions show asymmetry)
        
        Args:
            frame_sequence: List of preprocessed frames (T, H, W, 3)
        
        Returns:
            Dictionary of regional features
        """
        features = {}
        
        try:
            if len(frame_sequence) == 0:
                return features
            
            # Average frame for analysis
            avg_frame = np.mean(frame_sequence, axis=0)
            h, w, c = avg_frame.shape
            
            # Upper face region (eyes, eyebrows)
            upper_face = avg_frame[:h//3, :, :]
            features['upper_face_brightness'] = np.mean(upper_face)
            features['upper_face_contrast'] = np.std(upper_face)
            
            # Lower face region (mouth, jaw)
            lower_face = avg_frame[2*h//3:, :, :]
            features['lower_face_brightness'] = np.mean(lower_face)
            features['lower_face_contrast'] = np.std(lower_face)
            
            # Central region (important for most emotions)
            central_face = avg_frame[h//4:3*h//4, w//4:3*w//4, :]
            features['central_brightness'] = np.mean(central_face)
            features['central_contrast'] = np.std(central_face)
            
            # Left-right symmetry (sign of authentic emotion)
            left_half = avg_frame[:, :w//2, :].mean(axis=2)
            right_half = avg_frame[:, w//2:, :].mean(axis=2)
            
            if left_half.shape[1] == right_half.shape[1]:
                try:
                    symmetry = np.corrcoef(
                        left_half.flatten(),
                        np.fliplr(right_half).flatten()
                    )[0, 1]
                    features['facial_symmetry'] = float(symmetry) if not np.isnan(symmetry) else 0.5
                except:
                    features['facial_symmetry'] = 0.5
            
            # Upper/lower activity ratio (related to expression type)
            upper_activity = np.var(np.mean(frame_sequence[:, :h//3, :, :], axis=0))
            lower_activity = np.var(np.mean(frame_sequence[:, 2*h//3:, :, :], axis=0))
            features['upper_lower_activity_ratio'] = upper_activity / (lower_activity + 1e-6)
            
            # Activity in mouth region (speech, smile intensity)
            mouth_region = np.mean(frame_sequence[:, h//2:, :, :], axis=(0, 3))
            features['mouth_activity'] = np.std(mouth_region)
            
            return features
        
        except Exception as e:
            print(f"  ⚠ Facial regions error: {e}")
            return {}
    
    # ===== EMOTION-SPECIFIC VISUAL INDICATORS =====
    def extract_emotion_indicators(self, frame_sequence):
        """
        Extract visual indicators linked to specific emotions/stress.
        
        Stress indicators from facial analysis:
        - Eye tension (higher edge density around eyes)
        - Mouth tightness (lower mouth region activity)
        - Brow tension (upper face activity)
        - Overall facial muscle engagement
        
        Args:
            frame_sequence: Sequence of frames (T, H, W, 3)
        
        Returns:
            Dictionary of emotion indicators
        """
        features = {}
        
        try:
            if len(frame_sequence) == 0:
                return features
            
            # Average frame for analysis
            avg_frame = np.mean(frame_sequence, axis=0)
            h, w, _ = avg_frame.shape
            
            # Convert to grayscale
            gray = 0.299 * avg_frame[:, :, 0] + 0.587 * avg_frame[:, :, 1] + 0.114 * avg_frame[:, :, 2]
            
            # Eye region analysis (upper 1/3)
            eye_region = gray[:h//3, :]
            features['eye_region_sharpness'] = np.mean(np.abs(np.diff(eye_region)))
            
            # Mouth region analysis (lower 1/3)
            mouth_region = gray[2*h//3:, :]
            features['mouth_region_sharpness'] = np.mean(np.abs(np.diff(mouth_region)))
            
            # Overall facial expression intensity (based on edge density)
            edges = np.abs(np.diff(gray, axis=0)) + np.abs(np.diff(gray, axis=1))
            features['facial_expression_intensity'] = np.mean(edges)
            
            # Facial muscle engagement (variance across frame sequence)
            gray_sequence = []
            for frame in frame_sequence:
                g = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
                gray_sequence.append(g)
            gray_sequence = np.array(gray_sequence)
            
            # Temporal variance (how much face changes)
            features['facial_muscle_engagement'] = np.mean(np.std(gray_sequence, axis=0))
            
            # Wrinkle/texture indication (related to stress, aging)
            features['facial_texture_complexity'] = np.std(gray)
            
            return features
        
        except Exception as e:
            print(f"  ⚠ Emotion indicators error: {e}")
            return {}
    
    # ===== AFFECTNET EMOTION FEATURES =====
    def extract_affectnet_emotion_features(self, emotion_label, emotion_confidence=None):
        """
        Extract features based on AffectNet emotion classification.
        
        AffectNet emotions (8 categories):
        0: Neutral, 1: Happy, 2: Sad, 3: Surprise, 
        4: Fear, 5: Disgust, 6: Anger, 7: Contempt
        
        Args:
            emotion_label: Integer 0-7 or emotion name string
            emotion_confidence: Optional confidence/probability [0, 1]
        
        Returns:
            Dictionary of emotion-based features
        """
        features = {}
        
        try:
            # Convert label to emotion name
            if isinstance(emotion_label, int):
                emotion_name = AFFECTNET_EMOTIONS.get(emotion_label, 'Unknown')
            else:
                emotion_name = str(emotion_label)
            
            # Map to stress level (0=baseline, 1=stress, 2=amusement, 3=meditation)
            stress_level = EMOTION_TO_STRESS_MAP.get(emotion_name, 0)
            features['emotion_stress_level'] = stress_level
            
            # Valence (positivity): 0=negative, 1=positive
            valence = EMOTION_VALENCE.get(emotion_name, 0.5)
            features['emotion_valence'] = valence
            
            # Arousal (intensity): 0=calm, 1=intense
            arousal = EMOTION_AROUSAL.get(emotion_name, 0.5)
            features['emotion_arousal'] = arousal
            
            # Confidence/certainty
            if emotion_confidence is not None:
                features['emotion_confidence'] = float(emotion_confidence)
            else:
                features['emotion_confidence'] = 1.0
            
            # One-hot encoding for each emotion
            for emotion_id, emo_name in AFFECTNET_EMOTIONS.items():
                features[f'emotion_{emo_name.lower()}'] = 1.0 if emo_name == emotion_name else 0.0
            
            return features
        
        except Exception as e:
            print(f"  ⚠ AffectNet emotion extraction error: {e}")
            return {}



class StressIndicators:
    """Compute multimodal stress indicators combining physiological and facial features."""
    
    @staticmethod
    def compute_stress_index(physio_features):
        """
        Compute stress index from physiological features.
        
        Higher value = higher stress
        
        Args:
            physio_features: Dictionary from PhysiologicalFeatures.extract_all_physiological_features()
        
        Returns:
            stress_index: float [0, 1]
        """
        components = []
        weights = []
        
        try:
            # Heart rate elevation (relative to baseline)
            if 'hr_mean' in physio_features and not np.isnan(physio_features['hr_mean']):
                hr_stress = min(physio_features['hr_mean'] / 100, 1.0)  # Normalized to 100 bpm
                components.append(hr_stress)
                weights.append(0.25)
            
            # Heart rate variability (lower HRV = higher stress)
            if 'hrv_sdnn' in physio_features and not np.isnan(physio_features['hrv_sdnn']):
                hrv_stress = 1 - min(physio_features['hrv_sdnn'] / 100, 1.0)
                components.append(hrv_stress)
                weights.append(0.20)
            
            # EDA peaks (more peaks = higher arousal/stress)
            if 'eda_peak_count' in physio_features and not np.isnan(physio_features['eda_peak_count']):
                eda_stress = min(physio_features['eda_peak_count'] / 20, 1.0)
                components.append(eda_stress)
                weights.append(0.25)
            
            # Temperature rise (increased temp = stress)
            if 'temp_trend' in physio_features and not np.isnan(physio_features['temp_trend']):
                temp_stress = min(max(physio_features['temp_trend'] * 10, 0), 1.0)
                components.append(temp_stress)
                weights.append(0.15)
            
            # LF/HF ratio (higher ratio = sympathetic dominance = stress)
            if 'hrv_lf_hf_ratio' in physio_features and not np.isnan(physio_features['hrv_lf_hf_ratio']):
                lf_hf_stress = min(physio_features['hrv_lf_hf_ratio'] / 5, 1.0)
                components.append(lf_hf_stress)
                weights.append(0.15)
            
            if len(components) == 0:
                return 0.5
            
            # Weighted average
            weights = np.array(weights) / np.sum(weights)
            stress_index = np.sum(np.array(components) * weights)
            
            return float(np.clip(stress_index, 0, 1))
        
        except Exception as e:
            print(f"  ⚠ Stress index computation error: {e}")
            return 0.5
    
    @staticmethod
    def compute_facial_stress_index(facial_features):
        """
        Compute stress index from facial features.
        
        Facial indicators of stress:
        - High eye tension (eye_region_sharpness)
        - Low mouth region activity (tension)
        - Low facial symmetry (tension)
        - High facial expression intensity
        - Facial emotion classification
        
        Args:
            facial_features: Dictionary from FacialFeatures methods
        
        Returns:
            facial_stress_index: float [0, 1]
        """
        components = []
        weights = []
        
        try:
            # Emotion-based stress level from AffectNet
            if 'emotion_stress_level' in facial_features:
                # Normalize: 0=baseline, 1=stress, 2=amusement, 3=meditation
                emotion_stress = 1.0 if facial_features['emotion_stress_level'] == 1 else 0.0
                components.append(emotion_stress)
                weights.append(0.30)
            
            # Negative valence indicates stress
            if 'emotion_valence' in facial_features:
                valence_stress = 1 - facial_features['emotion_valence']  # Low valence = high stress
                components.append(valence_stress)
                weights.append(0.20)
            
            # High arousal can indicate stress (combined with negative valence)
            if 'emotion_arousal' in facial_features and 'emotion_valence' in facial_features:
                # High arousal + low valence = stress
                arousal_valence_stress = facial_features['emotion_arousal'] * (1 - facial_features['emotion_valence'])
                components.append(arousal_valence_stress)
                weights.append(0.20)
            
            # Facial expression intensity (higher = more expressive = potentially stressed)
            if 'facial_expression_intensity' in facial_features:
                expr_stress = min(facial_features['facial_expression_intensity'] * 2, 1.0)
                components.append(expr_stress)
                weights.append(0.15)
            
            # Low symmetry indicates tension/stress
            if 'facial_symmetry' in facial_features:
                symmetry = facial_features['facial_symmetry']
                if not np.isnan(symmetry):
                    asymmetry_stress = 1 - (symmetry + 1) / 2  # Convert from [-1, 1] to [0, 1]
                    components.append(asymmetry_stress)
                    weights.append(0.15)
            
            if len(components) == 0:
                return 0.5
            
            weights = np.array(weights) / np.sum(weights)
            facial_stress = np.sum(np.array(components) * weights)
            
            return float(np.clip(facial_stress, 0, 1))
        
        except Exception as e:
            print(f"  ⚠ Facial stress computation error: {e}")
            return 0.5

    
    @staticmethod
    def compute_arousal_level(eda_features):
        """
        Compute arousal level from EDA features.
        
        Args:
            eda_features: Dictionary containing EDA features
        
        Returns:
            arousal: float [0, 1]
        """
        try:
            arousal_components = []
            
            # EDA level (baseline arousal)
            if 'eda_mean' in eda_features and not np.isnan(eda_features['eda_mean']):
                arousal_components.append(min(eda_features['eda_mean'] / 5, 1.0))
            
            # EDA peak activity
            if 'eda_peak_count' in eda_features and not np.isnan(eda_features['eda_peak_count']):
                arousal_components.append(min(eda_features['eda_peak_count'] / 10, 1.0))
            
            if len(arousal_components) == 0:
                return 0.5
            
            return float(np.mean(arousal_components))
        
        except:
            return 0.5
    
    @staticmethod
    def compute_engagement_level(facial_features):
        """
        Compute engagement/expression level from facial features.
        
        Args:
            facial_features: Dictionary from FacialFeatures methods
        
        Returns:
            engagement: float [0, 1]
        """
        try:
            engagement_components = []
            
            # Motion intensity (more motion = more engaged)
            if 'motion_intensity' in facial_features:
                engagement_components.append(min(facial_features['motion_intensity'] * 2, 1.0))
            
            # Edge density (expression visibility)
            if 'edge_density' in facial_features:
                engagement_components.append(facial_features['edge_density'])
            
            # Central activity (eye/mouth region activity)
            if 'mouth_activity' in facial_features:
                engagement_components.append(facial_features['mouth_activity'])
            
            # Facial muscle engagement
            if 'facial_muscle_engagement' in facial_features:
                engagement_components.append(facial_features['facial_muscle_engagement'])
            
            if len(engagement_components) == 0:
                return 0.5
            
            return float(np.mean(engagement_components))
        
        except:
            return 0.5
    
    @staticmethod
    def compute_multimodal_stress(physio_features, facial_features, weights=None):
        """
        Compute combined stress index from both physiological and facial features.
        
        Args:
            physio_features: Dictionary from PhysiologicalFeatures
            facial_features: Dictionary from FacialFeatures
            weights: Optional dict with keys 'physio', 'facial' (should sum to 1.0)
        
        Returns:
            multimodal_stress_index: float [0, 1]
        """
        if weights is None:
            weights = {'physio': 0.6, 'facial': 0.4}  # Physio is more reliable for stress
        
        try:
            physio_stress = StressIndicators.compute_stress_index(physio_features)
            facial_stress = StressIndicators.compute_facial_stress_index(facial_features)
            
            multimodal_stress = (
                weights['physio'] * physio_stress + 
                weights['facial'] * facial_stress
            )
            
            return float(np.clip(multimodal_stress, 0, 1))
        except:
            return 0.5



class FeatureExtractor:
    """Complete feature extraction pipeline - supports both unimodal and multimodal."""
    
    def __init__(self, ecg_sr=700, eda_sr=4):
        """
        Args:
            ecg_sr: ECG sampling rate
            eda_sr: EDA sampling rate
        """
        self.physio_feat = PhysiologicalFeatures(ecg_sr=ecg_sr, eda_sr=eda_sr)
        self.facial_feat = FacialFeatures()
        self.stress_ind = StressIndicators()
    
    def extract_features_from_window(self, window_data, label=None):
        """
        Extract all physiological features from a single preprocessed window.
        
        Args:
            window_data: Dictionary with 'ecg', 'eda', 'temperature' signals
            label: Optional stress label (0=baseline, 1=stress, etc.)
        
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Physiological features
        physio_feats = self.physio_feat.extract_all_physiological_features(
            window_data['ecg'],
            window_data['eda'],
            window_data['temperature']
        )
        features.update(physio_feats)
        
        # Stress indicators
        features['stress_index'] = self.stress_ind.compute_stress_index(physio_feats)
        features['arousal_level'] = self.stress_ind.compute_arousal_level(physio_feats)
        
        # Add label if provided
        if label is not None:
            features['label'] = int(label)
        
        return features
    
    def extract_features_from_facial_window(self, frame_sequence, emotion_label=None, emotion_confidence=None, label=None):
        """
        Extract facial features from frame sequence (supports AffectNet).
        
        Args:
            frame_sequence: List of preprocessed frames (T, H, W, 3)
            emotion_label: Optional AffectNet emotion (0-7 or string)
            emotion_confidence: Optional confidence of emotion prediction [0, 1]
            label: Optional stress label (for training)
        
        Returns:
            Dictionary of facial features
        """
        features = {}
        
        if len(frame_sequence) == 0:
            return features
        
        # Frame-level statistics
        for frame in frame_sequence:
            frame_stats = self.facial_feat.extract_frame_statistics(frame)
            for key, val in frame_stats.items():
                if key not in features:
                    features[key] = []
                features[key].append(val)
        
        # Average frame statistics
        for key in features:
            if isinstance(features[key], list):
                features[key] = np.mean(features[key])
        
        # Temporal features
        features.update(self.facial_feat.extract_temporal_features(frame_sequence))
        
        # Facial regional features
        features.update(self.facial_feat.extract_facial_regions(frame_sequence))
        
        # Emotion indicators
        features.update(self.facial_feat.extract_emotion_indicators(frame_sequence))
        
        # AffectNet emotion features
        if emotion_label is not None:
            features.update(self.facial_feat.extract_affectnet_emotion_features(
                emotion_label, emotion_confidence
            ))
        
        # Engagement level
        features['engagement_level'] = self.stress_ind.compute_engagement_level(features)
        
        # Facial stress index
        features['facial_stress_index'] = self.stress_ind.compute_facial_stress_index(features)
        
        if label is not None:
            features['label'] = int(label)
        
        return features
    
    def extract_multimodal_features(self, window_data, frame_sequence, 
                                   emotion_label=None, emotion_confidence=None, label=None,
                                   physio_weights=None, modal_weights=None):
        """
        Extract features from both physiological and facial modalities.
        
        This is the primary method for multimodal stress detection.
        
        Args:
            window_data: Dictionary with 'ecg', 'eda', 'temperature' signals
            frame_sequence: List of preprocessed video frames
            emotion_label: Optional AffectNet emotion label
            emotion_confidence: Optional emotion confidence
            label: Optional ground truth stress label
            physio_weights: Optional weights for modalities in final computation
            modal_weights: Optional weights for physio vs facial (default: 0.6/0.4)
        
        Returns:
            Dictionary of combined multimodal features
        """
        features = {}
        
        # Extract physiological features
        physio_feats = self.extract_features_from_window(window_data, label=None)
        features.update({f'physio_{k}': v for k, v in physio_feats.items()})
        
        # Extract facial features
        facial_feats = self.extract_features_from_facial_window(
            frame_sequence, emotion_label, emotion_confidence, label=None
        )
        features.update({f'facial_{k}': v for k, v in facial_feats.items()})
        
        # Compute multimodal stress index
        features['multimodal_stress_index'] = self.stress_ind.compute_multimodal_stress(
            physio_feats, facial_feats, weights=modal_weights
        )
        
        # Add physiological stress
        features['physiological_stress'] = self.stress_ind.compute_stress_index(physio_feats)
        
        # Add facial stress
        features['facial_stress_component'] = self.stress_ind.compute_facial_stress_index(facial_feats)
        
        # Add label if provided
        if label is not None:
            features['label'] = int(label)
        
        return features


def get_physiological_feature_names():
    """Get list of physiological feature names."""
    ecg_features = [
        'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
        'hrv_sdnn', 'hrv_rmssd', 'hrv_nn50', 'hrv_pnn50',
        'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio',
        'hr_acceleration', 'hr_deceleration'
    ]
    
    eda_features = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
        'eda_trend', 'eda_peak_count', 'eda_peak_mean_height',
        'eda_peak_mean_interval', 'eda_mean_rise_time', 'eda_slope_variability'
    ]
    
    temp_features = [
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range',
        'temp_trend', 'temp_rise_rate', 'temp_fall_rate'
    ]
    
    cross_features = [
        'ecg_eda_correlation', 'ecg_temp_correlation', 'eda_temp_correlation'
    ]
    
    stress_features = ['stress_index', 'arousal_level']
    
    return ecg_features + eda_features + temp_features + cross_features + stress_features


def get_facial_feature_names():
    """Get list of facial feature names."""
    color_features = [
        'red_mean', 'red_std', 'red_entropy',
        'green_mean', 'green_std', 'green_entropy',
        'blue_mean', 'blue_std', 'blue_entropy',
        'brightness', 'contrast', 'edge_density', 'texture_energy'
    ]
    
    temporal_features = [
        'frame_change_mean', 'frame_change_std', 'frame_change_max', 'frame_change_min',
        'motion_intensity', 'temporal_consistency', 'motion_acceleration'
    ]
    
    regional_features = [
        'upper_face_brightness', 'upper_face_contrast',
        'lower_face_brightness', 'lower_face_contrast',
        'central_brightness', 'central_contrast',
        'facial_symmetry', 'upper_lower_activity_ratio', 'mouth_activity'
    ]
    
    emotion_indicators = [
        'eye_region_sharpness', 'mouth_region_sharpness',
        'facial_expression_intensity', 'facial_muscle_engagement',
        'facial_texture_complexity'
    ]
    
    affectnet_features = [
        'emotion_stress_level', 'emotion_valence', 'emotion_arousal', 'emotion_confidence'
    ]
    
    # Add one-hot emotion encoding
    for emotion_name in AFFECTNET_EMOTIONS.values():
        affectnet_features.append(f'emotion_{emotion_name.lower()}')
    
    stress_indicators = ['engagement_level', 'facial_stress_index']
    
    return (color_features + temporal_features + regional_features + 
            emotion_indicators + affectnet_features + stress_indicators)


def get_feature_names():
    """
    Get list of all feature names (physiological + facial).
    
    Returns:
        List of feature names (strings)
    """
    physio_names = get_physiological_feature_names()
    facial_names = get_facial_feature_names()
    stress_names = ['physiological_stress', 'facial_stress_component', 'multimodal_stress_index']
    
    return physio_names + facial_names + stress_names


def get_multimodal_feature_names():
    """
    Get list of multimodal feature names (with modal prefixes).
    
    Returns:
        List of multimodal feature names
    """
    physio_names = [f'physio_{name}' for name in get_physiological_feature_names()]
    facial_names = [f'facial_{name}' for name in get_facial_feature_names()]
    stress_names = ['physiological_stress', 'facial_stress_component', 'multimodal_stress_index']
    
    return physio_names + facial_names + stress_names


def get_feature_names():
    """
    Get list of all feature names.
    
    Returns:
        List of feature names (strings)
    """
    # Physiological features
    ecg_features = [
        'hr_mean', 'hr_std', 'hr_min', 'hr_max', 'hr_range',
        'hrv_sdnn', 'hrv_rmssd', 'hrv_nn50', 'hrv_pnn50',
        'hrv_vlf', 'hrv_lf', 'hrv_hf', 'hrv_lf_hf_ratio',
        'hr_acceleration', 'hr_deceleration'
    ]
    
    eda_features = [
        'eda_mean', 'eda_std', 'eda_min', 'eda_max', 'eda_range',
        'eda_trend', 'eda_peak_count', 'eda_peak_mean_height',
        'eda_peak_mean_interval', 'eda_mean_rise_time', 'eda_slope_variability'
    ]
    
    temp_features = [
        'temp_mean', 'temp_std', 'temp_min', 'temp_max', 'temp_range',
        'temp_trend', 'temp_rise_rate', 'temp_fall_rate'
    ]
    
    cross_features = [
        'ecg_eda_correlation', 'ecg_temp_correlation', 'eda_temp_correlation'
    ]
    
    stress_features = ['stress_index', 'arousal_level']
    
    facial_features = [
        'red_mean', 'red_std', 'green_mean', 'green_std', 'blue_mean', 'blue_std',
        'brightness', 'contrast', 'edge_density',
        'frame_change_mean', 'frame_change_std', 'frame_change_max',
        'motion_intensity', 'temporal_consistency',
        'facial_symmetry', 'upper_lower_activity_ratio', 'central_activity',
        'engagement_level'
    ]
    
    all_features = (ecg_features + eda_features + temp_features + 
                   cross_features + stress_features + facial_features)
    
    return all_features


if __name__ == "__main__":
    print("\n🚀 Testing Multimodal Feature Engineering\n")
    
    # ===== TEST 1: Physiological Features =====
    print("="*70)
    print("TEST 1: Physiological Feature Extraction (WESAD)")
    print("="*70)
    
    np.random.seed(42)
    ecg = np.sin(np.linspace(0, 100*np.pi, 1000)) + np.random.randn(1000) * 0.1
    eda = np.cumsum(np.random.randn(1000) * 0.01) 
    temp = 37 + np.cumsum(np.random.randn(1000) * 0.001)
    
    extractor = FeatureExtractor()
    
    window_data = {
        'ecg': ecg,
        'eda': eda,
        'temperature': temp
    }
    
    physio_features = extractor.extract_features_from_window(window_data, label=1)
    
    print(f"\n✓ Extracted {len(physio_features)} physiological features")
    print(f"  - Stress Index: {physio_features['stress_index']:.4f}")
    print(f"  - Arousal Level: {physio_features['arousal_level']:.4f}")
    print(f"  - HR Mean: {physio_features['hr_mean']:.2f} bpm")
    print(f"  - HRV (SDNN): {physio_features['hrv_sdnn']:.4f}")
    print(f"  - EDA Peaks: {physio_features['eda_peak_count']:.0f}")
    
    # ===== TEST 2: Facial Features (AffectNet) =====
    print("\n" + "="*70)
    print("TEST 2: Facial Feature Extraction (AffectNet)")
    print("="*70)
    
    # Create sample frame sequence (10 frames, 224x224, RGB)
    frame_sequence = np.random.rand(10, 224, 224, 3).astype(np.float32)
    
    # Test with different AffectNet emotions
    emotions_to_test = [
        (0, "Neutral"),
        (1, "Happy"),
        (4, "Fear"),
        (6, "Anger")
    ]
    
    for emotion_id, emotion_name in emotions_to_test:
        facial_features = extractor.extract_features_from_facial_window(
            frame_sequence, 
            emotion_label=emotion_id,
            emotion_confidence=0.9,
            label=EMOTION_TO_STRESS_MAP[emotion_name]
        )
        
        print(f"\n  Emotion: {emotion_name} ({emotion_id})")
        print(f"    - Engagement: {facial_features.get('engagement_level', 0):.4f}")
        print(f"    - Facial Stress: {facial_features.get('facial_stress_index', 0):.4f}")
        print(f"    - Valence: {facial_features.get('emotion_valence', 0):.4f}")
        print(f"    - Arousal: {facial_features.get('emotion_arousal', 0):.4f}")
    
    # ===== TEST 3: Multimodal Features =====
    print("\n" + "="*70)
    print("TEST 3: Multimodal Feature Extraction (WESAD + AffectNet)")
    print("="*70)
    
    multimodal_features = extractor.extract_multimodal_features(
        window_data=window_data,
        frame_sequence=frame_sequence,
        emotion_label=1,  # Happy
        emotion_confidence=0.85,
        label=1  # Stress
    )
    
    print(f"\n✓ Extracted {len(multimodal_features)} multimodal features")
    print(f"  - Physiological Stress: {multimodal_features['physiological_stress']:.4f}")
    print(f"  - Facial Stress: {multimodal_features['facial_stress_component']:.4f}")
    print(f"  - Multimodal Stress (combined): {multimodal_features['multimodal_stress_index']:.4f}")
    print(f"  - Label: {multimodal_features['label']}")
    
    # ===== TEST 4: Feature Statistics =====
    print("\n" + "="*70)
    print("TEST 4: Feature Set Statistics")
    print("="*70)
    
    physio_names = get_physiological_feature_names()
    facial_names = get_facial_feature_names()
    all_names = get_feature_names()
    multimodal_names = get_multimodal_feature_names()
    
    print(f"\n✓ Physiological features: {len(physio_names)}")
    print(f"✓ Facial features: {len(facial_names)}")
    print(f"✓ Total unimodal features: {len(all_names)}")
    print(f"✓ Total multimodal features: {len(multimodal_names)}")
    
    print(f"\nPhysiological feature breakdown:")
    print(f"  - ECG: 15 features")
    print(f"  - EDA: 11 features")
    print(f"  - Temperature: 8 features")
    print(f"  - Cross-signal: 3 features")
    print(f"  - Stress indicators: 2 features")
    
    print(f"\nFacial feature breakdown:")
    print(f"  - Color/texture: 13 features")
    print(f"  - Temporal dynamics: 7 features")
    print(f"  - Regional analysis: 9 features")
    print(f"  - Emotion indicators: 5 features")
    print(f"  - AffectNet emotions: 12 features (4 + 8 one-hot)")
    print(f"  - Stress indicators: 2 features")
    
    print("\n" + "="*70)
    print("✅ MULTIMODAL FEATURE ENGINEERING COMPLETE")
    print("="*70)
    print("\nDatasets integrated:")
    print("  ✓ WESAD: Physiological signals (ECG, EDA, Temperature)")
    print("  ✓ AffectNet: Facial emotions (8 categories)")
    print("  ✓ Multimodal fusion for stress detection")
    print("="*70 + "\n")

