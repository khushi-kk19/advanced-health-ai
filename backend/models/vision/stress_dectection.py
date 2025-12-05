"""
Stress detection from facial expressions using MediaPipe.

Detects:
- Eye opening (closed eyes = stress/fatigue)
- Mouth openness (clenched jaw = stress)
- Head pose (tension patterns)
- Blink rate (elevated = stress)
"""
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from datetime import datetime, timedelta

class StressDetector:
    """
    Detect stress from facial expressions.
    """
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Stress indicators history
        self.eye_aspect_ratio_history = deque(maxlen=30)
        self.mouth_aspect_ratio_history = deque(maxlen=30)
        self.blink_history = deque(maxlen=100)
        
        self.stress_score = 0
        self.last_blink_time = datetime.now()
        self.blink_count = 0
        self.eyes_closed = False
        
    def get_eye_aspect_ratio(self, landmarks, eye_indices):
        """
        Calculate eye aspect ratio (EAR).
        High EAR = open eyes, Low EAR = closed eyes
        
        Eye indices (MediaPipe Face Mesh):
        Left eye: [33, 160, 158, 133, 153, 144]
        Right eye: [362, 385, 387, 263, 373, 380]
        """
        p1 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
        p2 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])
        p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
        p4 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        p5 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        p6 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        
        # Euclidean distances
        dist1 = np.linalg.norm(p1 - p2)
        dist2 = np.linalg.norm(p3 - p4)
        dist3 = np.linalg.norm(p5 - p6)
        
        # EAR = (||p1 - p2|| + ||p3 - p4||) / (2 * ||p5 - p6||)
        ear = (dist1 + dist2) / (2.0 * dist3 + 1e-6)
        
        return ear
    
    def get_mouth_aspect_ratio(self, landmarks):
        """
        Calculate mouth aspect ratio (MAR).
        High MAR = open mouth, Low MAR = closed mouth
        """
        # Mouth landmarks (MediaPipe)
        top_lip = np.array([landmarks[13].x, landmarks[13].y])
        bottom_lip = np.array([landmarks[14].x, landmarks[14].y])
        left_mouth = np.array([landmarks[61].x, landmarks[61].y])
        right_mouth = np.array([landmarks[291].x, landmarks[291].y])
        
        vertical_dist = np.linalg.norm(top_lip - bottom_lip)
        horizontal_dist = np.linalg.norm(left_mouth - right_mouth)
        
        mar = vertical_dist / (horizontal_dist + 1e-6)
        
        return mar
    
    def process_frame(self, frame):
        """
        Process frame and detect stress indicators.
        
        Returns:
            stress_score: 0-100 (0=relaxed, 100=very stressed)
            indicators: dict of individual indicators
        """
        h, w, c = frame.shape
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if not results.multi_face_landmarks:
            return 0, {}
        
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Left and right eye indices (MediaPipe Face Mesh)
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        
        # Calculate metrics
        left_ear = self.get_eye_aspect_ratio(landmarks, left_eye_indices)
        right_ear = self.get_eye_aspect_ratio(landmarks, right_eye_indices)
        ear = (left_ear + right_ear) / 2.0
        
        mar = self.get_mouth_aspect_ratio(landmarks)
        
        self.eye_aspect_ratio_history.append(ear)
        self.mouth_aspect_ratio_history.append(mar)
        
        # Detect blinks (sudden drop in EAR)
        if len(self.eye_aspect_ratio_history) > 2:
            current_ear = self.eye_aspect_ratio_history[-1]
            prev_ear = self.eye_aspect_ratio_history[-2]
            
            if current_ear < 0.15 and prev_ear > 0.2:  # Eyes closing
                if not self.eyes_closed:
                    self.eyes_closed = True
            elif current_ear > 0.2 and prev_ear < 0.15:  # Eyes opening
                if self.eyes_closed:
                    self.eyes_closed = False
                    self.blink_count += 1
                    self.blink_history.append(datetime.now())
        
        # Calculate blink rate (blinks per minute)
        blink_rate = 0
        if len(self.blink_history) > 1:
            time_span = (self.blink_history[-1] - self.blink_history[0]).total_seconds()
            if time_span > 0:
                blink_rate = (len(self.blink_history) / time_span) * 60
        
        # Calculate stress score (0-100)
        # Components:
        # 1. Low EAR (closed eyes) = stress indicator
        # 2. Low MAR (clenched jaw) = stress indicator
        # 3. High blink rate (>25 bpm) = stress indicator
        
        ear_stress = max(0, (1.0 - ear) * 100)  # Low EAR = high stress
        mar_stress = max(0, (0.5 - mar) * 100)  # Low MAR = high stress
        blink_stress = max(0, (blink_rate - 15) / 20 * 100)  # >25 bpm = stress
        
        stress_score = (ear_stress * 0.4 + mar_stress * 0.3 + blink_stress * 0.3)
        stress_score = np.clip(stress_score, 0, 100)
        
        self.stress_score = stress_score
        
        indicators = {
            'eye_aspect_ratio': float(ear),
            'mouth_aspect_ratio': float(mar),
            'blink_rate': float(blink_rate),
            'stress_score': float(stress_score),
            'stress_level': self._classify_stress(stress_score)
        }
        
        return stress_score, indicators
    
    def _classify_stress(self, score):
        """Classify stress level."""
        if score < 20:
            return "Relaxed"
        elif score < 40:
            return "Calm"
        elif score < 60:
            return "Neutral"
        elif score < 80:
            return "Stressed"
        else:
            return "Very Stressed"
    
    def visualize(self, frame, landmarks, stress_score, indicators):
        """Draw visualization on frame."""
        h, w, c = frame.shape
        
        # Draw stress level bar
        bar_height = 30
        bar_y = 20
        bar_width = int((stress_score / 100) * 200)
        
        # Color based on stress level
        if stress_score < 30:
            color = (0, 255, 0)  # Green
        elif stress_score < 60:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        cv2.rectangle(frame, (20, bar_y), (220, bar_y + bar_height), (200, 200, 200), -1)
        cv2.rectangle(frame, (20, bar_y), (20 + bar_width, bar_y + bar_height), color, -1)
        cv2.putText(frame, f"Stress: {stress_score:.0f}", (230, bar_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw indicators
        y_offset = 80
        cv2.putText(frame, f"Level: {indicators['stress_level']}", (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Blink Rate: {indicators['blink_rate']:.1f}/min", 
                   (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        return frame

if __name__ == "__main__":
    # Test stress detection
    cap = cv2.VideoCapture(0)
    detector = StressDetector()
    
    print("Stress Detection - Press 'q' to quit")
    
    while True:
        ret, frame = cv2.imread(frame)
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        stress_score, indicators = detector.process_frame(frame)
        
        if stress_score > 0:
            frame = detector.visualize(frame, None, stress_score, indicators)
        
        cv2.imshow("Stress Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()