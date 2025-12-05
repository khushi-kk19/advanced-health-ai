"""
FEATURE EXTRACTION: Stress indicators from multimodal data
- Physiological: HRV, EDA, Temperature
- Facial: Eyes, Mouth, Head pose
"""
import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks
import cv2
import mediapipe as mp

class PhysiologicalFeatures:
    """Extract stress features from physiological signals."""
    
    def __init__(self, ecg_sampling_rate=700, eda_sampling_rate=4):
        self.ecg_sr = ecg_sampling_rate
        self.eda_sr = eda_sampling_rate
    
    def extract_hrv_features(self, ecg_signal):
        """HRV features indicate stress level (low HRV = high stress)."""
        try:
            # Find R-peaks in ECG
            peaks, _ = find_peaks(ecg_signal, distance=self.ecg_sr//3)
            
            if len(peaks) < 2:
                return {'rmssd': 0, 'pnn50': 0, 'sdnn': 0, 'hr_mean': 0, 'lf_hf_ratio': 0}
            
            # RR intervals in milliseconds
            rr_intervals = np.diff(peaks) / self.ecg_sr * 1000
            
            # RMSSD: Root Mean Square of Successive Differences
            successive_diffs = np.diff(rr_intervals)
            rmssd = np.sqrt(np.mean(successive_diffs**2))
            
            # pNN50: % of RR intervals > 50ms
            pnn50 = 100 * np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs)
            
            # SDNN: Standard deviation of RR intervals
            sdnn = np.std(rr_intervals)
            
            # Heart rate
            hr_values = 60000 / rr_intervals
            hr_mean = np.mean(hr_values)
            
            # LF/HF ratio (stress indicator)
            freqs, psd = signal.welch(rr_intervals, fs=4)
            lf_power = np.sum(psd[(freqs >= 0.04) & (freqs < 0.15)])
            hf_power = np.sum(psd[(freqs >= 0.15) & (freqs < 0.4)])
            lf_hf_ratio = lf_power / (hf_power + 1e-6)
            
            return {
                'rmssd': float(rmssd),
                'pnn50': float(pnn50),
                'sdnn': float(sdnn),
                'hr_mean': float(hr_mean),
                'lf_hf_ratio': float(lf_hf_ratio)
            }
        except Exception as e:
            print(f"HRV error: {e}")
            return {'rmssd': 0, 'pnn50': 0, 'sdnn': 0, 'hr_mean': 0, 'lf_hf_ratio': 0}
    
    def extract_eda_features(self, eda_signal):
        """EDA features: skin conductance indicates arousal/stress."""
        try:
            # Find peaks (EDA spikes indicate stress response)
            peaks, properties = find_peaks(eda_signal, distance=self.eda_sr, 
                                          prominence=0.1)
            
            features = {
                'scl_mean': float(np.mean(eda_signal)),
                'scl_std': float(np.std(eda_signal)),
                'scl_range': float(np.max(eda_signal) - np.min(eda_signal)),
                'peak_count': len(peaks),
                'peak_frequency': len(peaks) / max(len(eda_signal) / (self.eda_sr * 60), 0.1)
            }
            
            if len(peaks) > 0:
                features['peak_amplitude_mean'] = float(np.mean(eda_signal[peaks]))
            else:
                features['peak_amplitude_mean'] = 0
            
            return features
        except Exception as e:
            print(f"EDA error: {e}")
            return {'scl_mean': 0, 'scl_std': 0, 'scl_range': 0, 'peak_count': 0, 
                   'peak_frequency': 0, 'peak_amplitude_mean': 0}
    
    def extract_temperature_features(self, temp_signal):
        """Temperature features: elevated temp indicates stress."""
        return {
            'temp_mean': float(np.mean(temp_signal)),
            'temp_std': float(np.std(temp_signal)),
            'temp_trend': float(np.polyfit(range(len(temp_signal)), temp_signal, 1)[0])
        }
    
    def extract_statistical_features(self, signal_data):
        """General statistical features."""
        return {
            'mean': float(np.mean(signal_data)),
            'std': float(np.std(signal_data)),
            'min': float(np.min(signal_data)),
            'max': float(np.max(signal_data)),
            'skewness': float(stats.skew(signal_data)),
            'kurtosis': float(stats.kurtosis(signal_data))
        }


class FacialFeatures:
    """Extract stress features from facial video."""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def extract_eye_features(self, landmarks):
        """Eye metrics indicate stress (wide eyes, rapid blinking)."""
        try:
            # Left eye
            left_eye = [landmarks[i] for i in [33, 160, 158, 133, 153, 144]]
            # Right eye
            right_eye = [landmarks[i] for i in [362, 385, 387, 362, 380, 374]]
            
            def eye_aspect_ratio(eye):
                p1 = np.array([eye[1].x, eye[1].y])
                p2 = np.array([eye[4].x, eye[4].y])
                p3 = np.array([eye[0].x, eye[0].y])
                p4 = np.array([eye[3].x, eye[3].y])
                
                dist_vertical = np.linalg.norm(p1 - p2)
                dist_horizontal = np.linalg.norm(p3 - p4)
                
                return dist_vertical / (dist_horizontal + 1e-6)
            
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            
            return {
                'left_eye_aspect_ratio': float(left_ear),
                'right_eye_aspect_ratio': float(right_ear),
                'avg_eye_aspect_ratio': float((left_ear + right_ear) / 2),
                'eye_openness': float((left_ear + right_ear) / 2)
            }
        except:
            return {'left_eye_aspect_ratio': 0, 'right_eye_aspect_ratio': 0,
                   'avg_eye_aspect_ratio': 0, 'eye_openness': 0}
    
    def extract_mouth_features(self, landmarks):
        """Mouth metrics: open mouth, tension indicate stress."""
        try:
            # Mouth landmarks
            top = np.array([landmarks[13].x, landmarks[13].y])
            bottom = np.array([landmarks[14].x, landmarks[14].y])
            left = np.array([landmarks[78].x, landmarks[78].y])
            right = np.array([landmarks[308].x, landmarks[308].y])
            
            mouth_vertical = np.linalg.norm(top - bottom)
            mouth_horizontal = np.linalg.norm(left - right)
            
            mouth_aspect_ratio = mouth_vertical / (mouth_horizontal + 1e-6)
            
            # Smile intensity (mouth corners position)
            mouth_left_corner = np.array([landmarks[61].x, landmarks[61].y])
            mouth_right_corner = np.array([landmarks[291].x, landmarks[291].y])
            smile_height = -(mouth_left_corner[1] + mouth_right_corner[1]) / 2
            
            return {
                'mouth_aspect_ratio': float(mouth_aspect_ratio),
                'mouth_openness': float(mouth_vertical),
                'smile_intensity': float(max(0, smile_height))
            }
        except:
            return {'mouth_aspect_ratio': 0, 'mouth_openness': 0, 'smile_intensity': 0}
    
    def extract_head_pose_features(self, landmarks):
        """Head pose: forward/down = stress."""
        try:
            nose = np.array([landmarks[1].x, landmarks[1].y])
            chin = np.array([landmarks[152].x, landmarks[152].y])
            left_ear = np.array([landmarks[234].x, landmarks[234].y])
            right_ear = np.array([landmarks[454].x, landmarks[454].y])
            
            # Head tilt
            head_tilt = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])
            
            # Head yaw (left-right)
            head_yaw = (left_ear[0] - right_ear[0])
            
            # Head pitch (up-down)
            head_pitch = (chin[1] - nose[1])
            
            return {
                'head_tilt': float(head_tilt),
                'head_yaw': float(head_yaw),
                'head_pitch': float(head_pitch),
                'head_down': 1.0 if head_pitch > 0.1 else 0.0
            }
        except:
            return {'head_tilt': 0, 'head_yaw': 0, 'head_pitch': 0, 'head_down': 0}
    
    def process_frame(self, frame):
        """Extract all facial features from frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.face_landmarks:
            return None
        
        landmarks = results.face_landmarks.landmark
        
        features = {
            'eyes': self.extract_eye_features(landmarks),
            'mouth': self.extract_mouth_features(landmarks),
            'head_pose': self.extract_head_pose_features(landmarks)
        }
        
        return features


class StressIndicators:
    """Combine multimodal features to compute stress score."""
    
    @staticmethod
    def compute_stress_score(physio_features, facial_features=None):
        """
        Compute overall stress score (0-1).
        
        Stress indicators:
        - High HRV = low stress, Low HRV = high stress
        - High EDA = high stress
        - High HR = high stress
        - Wide eyes = stress
        - Open mouth = stress
        - Head down = stress
        """
        stress_score = 0
        weight_count = 0
        
        # Physiological indicators
        if physio_features:
            # HRV indicator (inverse: low HRV = high stress)
            hrv_score = 1 - (physio_features['hrv']['rmssd'] / (physio_features['hrv']['rmssd'] + 100))
            stress_score += 0.3 * hrv_score
            
            # EDA indicator (high EDA = high stress)
            eda_score = physio_features['eda']['peak_frequency'] / 10  # Normalize
            stress_score += 0.3 * min(eda_score, 1.0)
            
            # HR indicator
            hr_score = (physio_features['hrv']['hr_mean'] - 60) / 80  # Normalize 60-140 BPM
            stress_score += 0.2 * np.clip(hr_score, 0, 1)
            
            weight_count += 3
        
        # Facial indicators
        if facial_features:
            # Eye wideness (high = stress)
            eye_score = np.clip(facial_features['eyes']['eye_openness'], 0, 1)
            stress_score += 0.1 * eye_score
            
            # Head down (positive = stress)
            head_score = facial_features['head_pose']['head_down']
            stress_score += 0.1 * head_score
            
            weight_count += 2
        
        return np.clip(stress_score / weight_count if weight_count > 0 else 0, 0, 1)
