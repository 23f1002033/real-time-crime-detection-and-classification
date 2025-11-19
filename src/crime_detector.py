import torch
import torch.nn as nn
import cv2
import numpy as np
from collections import deque
import time
import os

class SimpleCrimeDetector(nn.Module):
    """
    Simplified crime detection model that works without complex dependencies.
    Uses a lightweight 3D CNN for real-time video analysis.
    """
    def __init__(self, num_classes=14):  # 13 crime types + 1 normal
        super(SimpleCrimeDetector, self).__init__()
        
        # Simple 3D CNN for video analysis
        self.conv3d1 = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        
        self.conv3d2 = nn.Conv3d(64, 128, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 2, 2), padding=(0, 1, 1))
        
        self.conv3d3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        
        # Crime labels based on UCF-Crime dataset
        self.crime_labels = [
            'Normal', 'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
            'Explosion', 'Fighting', 'RoadAccident', 'Robbery', 
            'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
        ]
        
    def forward(self, x):
        # x shape: (batch_size, channels, frames, height, width)
        x = torch.relu(self.conv3d1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv3d2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3d3(x))
        x = self.pool3(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def predict_crime(self, frame_sequence):
        """
        Predict crime type from a sequence of frames
        Args:
            frame_sequence: numpy array of shape (frames, height, width, channels)
        Returns:
            tuple: (predicted_label, confidence_score, is_crime)
        """
        self.eval()
        with torch.no_grad():
            # Preprocess frames
            if len(frame_sequence) < 16:
                # Pad sequence if too short
                while len(frame_sequence) < 16:
                    frame_sequence.append(frame_sequence[-1])
            elif len(frame_sequence) > 16:
                # Sample frames if too long
                indices = np.linspace(0, len(frame_sequence)-1, 16, dtype=int)
                frame_sequence = [frame_sequence[i] for i in indices]
            
            # Convert to tensor
            frames = np.stack(frame_sequence)  # (16, H, W, 3)
            frames = frames.transpose(3, 0, 1, 2)  # (3, 16, H, W)
            frames = torch.from_numpy(frames).float().unsqueeze(0)  # (1, 3, 16, H, W)
            
            # Normalize
            frames = frames / 255.0
            
            # Predict
            outputs = self.forward(frames)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_idx = predicted.item()
            confidence_score = confidence.item()
            predicted_label = self.crime_labels[predicted_idx]
            is_crime = predicted_idx != 0  # 0 is 'Normal'
            
            return predicted_label, confidence_score, is_crime

class MotionDetector:
    """
    Real-time motion detection for triggering crime analysis
    """
    def __init__(self, min_area=2000, threshold=25):
        self.min_area = min_area
        self.threshold = threshold
        self.background = None
        
    def detect_motion(self, frame):
        """
        Detect motion in frame
        Returns: (has_motion, processed_frame, motion_areas)
        """
        # Validate frame
        if frame is None or len(frame.shape) != 3:
            return False, frame if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8), []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.background is None:
                self.background = gray
                return False, frame, []
            
            # Ensure background and current frame have same size
            if self.background.shape != gray.shape:
                self.background = cv2.resize(self.background, (gray.shape[1], gray.shape[0]))
            
            # Compute frame difference
            frame_delta = cv2.absdiff(self.background, gray)
            thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_areas = []
            has_significant_motion = False
            
            for contour in contours:
                if cv2.contourArea(contour) < self.min_area:
                    continue
                    
                (x, y, w, h) = cv2.boundingRect(contour)
                motion_areas.append((x, y, w, h))
                has_significant_motion = True
            
            return has_significant_motion, thresh, motion_areas
            
        except cv2.error as e:
            print(f"Motion detection error: {e}")
            return False, frame, []

class FrameBuffer:
    """
    Maintains a rolling buffer of frames for analysis
    """
    def __init__(self, max_frames=16):
        self.frames = deque(maxlen=max_frames)
        self.max_frames = max_frames
        
    def add_frame(self, frame):
        # Validate frame before processing
        if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
            return  # Skip invalid frames
        
        # Resize frame for efficiency
        try:
            resized_frame = cv2.resize(frame, (224, 224))
            self.frames.append(resized_frame)
        except cv2.error:
            return  # Skip frames that can't be resized
        
    def get_sequence(self):
        return list(self.frames)
    
    def is_full(self):
        return len(self.frames) >= self.max_frames

class CrimeDetectionPipeline:
    """
    Complete pipeline integrating motion detection, frame processing, and ML inference
    """
    def __init__(self):
        self.crime_detector = SimpleCrimeDetector()
        self.motion_detector = MotionDetector()
        self.frame_buffer = FrameBuffer()
        
        # Initialize with random weights (in real scenario, load trained weights)
        self._initialize_model()
        
        # Detection state
        self.detection_active = False
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
    def _initialize_model(self):
        """Initialize model with trained weights if available"""
        # Look for model in the models directory
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(current_dir, 'models', 'crime_model.pth')
        
        if os.path.exists(model_path):
            try:
                print(f"Loading trained model from {model_path}...")
                checkpoint = torch.load(model_path, map_location='cpu')
                self.crime_detector.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Trained model loaded successfully!")
                val_acc = checkpoint.get('val_acc', checkpoint.get('accuracy', 0))
                if isinstance(val_acc, (int, float)) and val_acc > 0:
                    print(f"   Model accuracy: {val_acc:.2f}%")
                else:
                    print("   Model training completed successfully")
            except Exception as e:
                print(f"⚠️  Failed to load trained model: {e}")
                print("   Using random weights (for demonstration only)")
        else:
            print(f"ℹ️  No trained model found at {model_path}")
            print("   Using random weights (for demonstration only)")
            print("   Run training pipeline to get actual crime detection!")
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline
        Returns: (crime_detected, crime_type, confidence, motion_areas)
        """
        # Validate frame first
        if frame is None:
            return False, "Normal", 0.0, []
        
        try:
            # Add frame to buffer (now has validation)
            self.frame_buffer.add_frame(frame)
            
            # Detect motion (now has validation)
            has_motion, motion_mask, motion_areas = self.motion_detector.detect_motion(frame)
            
            crime_detected = False
            crime_type = "Normal"
            confidence = 0.0
            
            # Only analyze for crimes if we have motion and enough frames
            current_time = time.time()
            if (has_motion and 
                self.frame_buffer.is_full() and 
                self.detection_active and
                current_time - self.last_detection_time > self.detection_cooldown):
                
                try:
                    # Get frame sequence for analysis
                    frame_sequence = self.frame_buffer.get_sequence()
                    
                    # Predict crime
                    crime_type, confidence, crime_detected = self.crime_detector.predict_crime(frame_sequence)
                    
                    if crime_detected:
                        self.last_detection_time = current_time
                        
                except Exception as e:
                    print(f"Crime detection error: {e}")
                    crime_detected = False
            
            return crime_detected, crime_type, confidence, motion_areas
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return False, "Normal", 0.0, []
    
    def start_detection(self):
        self.detection_active = True
        
    def stop_detection(self):
        self.detection_active = False
        
    def is_detecting(self):
        return self.detection_active