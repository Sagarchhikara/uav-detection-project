from ultralytics import YOLO
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import yaml

class DroneDetector:
    """YOLOv8-based drone detector"""
    
    def __init__(self, model_path: str, conf_threshold: float = 0.5, device: str = 'auto'):
        """
        Initialize drone detector
        
        Args:
            model_path: Path to YOLO model weights
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('auto', 'cpu', 'cuda:0')
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        
        # Auto-detect device
        if device == 'auto':
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model()
        
        print(f"✓ DroneDetector initialized")
        print(f"  Model: {model_path}")
        print(f"  Device: {self.device}")
        print(f"  Confidence threshold: {conf_threshold}")
    
    def _load_model(self) -> YOLO:
        """Load YOLO model"""
        try:
            model = YOLO(self.model_path)
            # Warm up model
            dummy_input = torch.zeros(1, 3, 640, 640)
            if self.device.startswith('cuda'):
                dummy_input = dummy_input.cuda()
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to pre-trained YOLOv8s...")
            return YOLO('yolov8s.pt')
    
    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """
        Detect drones in frame
        
        Args:
            frame: Input frame (BGR format)
        
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                detections.append((*box, conf))
        
        return detections
    
    def detect_and_visualize(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect drones and return annotated frame
        
        Returns:
            detections, annotated_frame
        """
        detections = self.detect(frame)
        annotated_frame = self.draw_detections(frame.copy(), detections)
        return detections, annotated_frame
    
    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """Draw detection boxes on frame"""
        for x1, y1, x2, y2, conf in detections:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw confidence
            label = f"Drone: {conf:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame

def load_model(model_path: str, device: str = 'auto') -> YOLO:
    """Load YOLO model with error handling"""
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"Warning: Could not load {model_path}: {e}")
        print("Using pre-trained YOLOv8s instead...")
        return YOLO('yolov8s.pt')

def prepare_dataset(dataset_path: str, output_format: str = 'yolo') -> None:
    """Prepare dataset for training (placeholder)"""
    print(f"Dataset preparation for {dataset_path} -> {output_format}")
    print("This function should convert your Anti-UAV dataset to YOLO format")
    print("See scripts/convert_to_yolo.py for implementation")

def train_model(data_yaml: str, epochs: int = 100, device: str = 'auto') -> YOLO:
    """Train YOLO model on drone dataset"""
    model = YOLO('yolov8s.pt')  # Start with pre-trained weights
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        device=device,
        project='models/finetuned',
        name='drone_detector'
    )
    
    return model