import cv2
import numpy as np
from typing import List, Tuple
from src.detection.yolo_detector import DroneDetector
from src.tracking.tracker import SimpleTracker
from src.behavior.behavior_classifier import BehaviorClassifier
from src.alerts.alert_manager import AlertManager

class DroneDetectorTracker:
    """Combined detection + tracking + behavior analysis pipeline"""
    
    def __init__(self, model_path, conf_threshold=0.5, restricted_zones=None):
        self.detector = DroneDetector(model_path, conf_threshold)
        self.tracker = SimpleTracker()
        self.behavior_classifier = BehaviorClassifier(fps=30, restricted_zones=restricted_zones)
        self.alert_manager = AlertManager()
        self.frame_count = 0
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Returns:
            tracks: List of (track_id, x1, y1, x2, y2, conf)
            annotated_frame: Frame with visualizations
            alerts: List of current alerts
        """
        self.frame_count += 1
        
        # Run detection
        detections = self.detector.detect(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Analyze behavior and generate alerts
        alerts = []
        for track_id, x1, y1, x2, y2, conf in tracks:
            trajectory = self.tracker.get_track_history(track_id)
            if len(trajectory) > 5:  # Only analyze if we have enough history
                analysis = self.behavior_classifier.analyze(track_id, trajectory)
                alert = self.alert_manager.generate_alert(analysis, self.frame_count)
                if alert:
                    alerts.append(alert)
        
        # Annotate frame
        annotated_frame = self._annotate_frame(frame.copy(), tracks, alerts)
        
        return tracks, annotated_frame, alerts
    
    def _annotate_frame(self, frame, tracks, alerts):
        """Annotate frame with tracks and alerts"""
        # Draw tracks
        for track_id, x1, y1, x2, y2, conf in tracks:
            # Determine color based on alerts
            color = (0, 255, 0)  # Green by default
            alert_level = 'NORMAL'
            
            for alert in alerts:
                if alert['track_id'] == track_id:
                    if alert['alert_level'] == 'HIGH':
                        color = (0, 0, 255)  # Red
                        alert_level = 'HIGH'
                    elif alert['alert_level'] == 'MEDIUM':
                        color = (0, 165, 255)  # Orange
                        alert_level = 'MEDIUM'
                    elif alert['alert_level'] == 'LOW':
                        color = (0, 255, 255)  # Yellow
                        alert_level = 'LOW'
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            # Draw track ID and alert level
            label = f"ID:{track_id} {alert_level} ({conf:.2f})"
            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw trajectory
            history = self.tracker.get_track_history(track_id)
            if len(history) > 1:
                points = [(int(cx), int(cy)) for _, cx, cy in history]
                for i in range(len(points) - 1):
                    cv2.line(frame, points[i], points[i+1], (255, 0, 0), 2)
        
        # Draw restricted zones if any
        if hasattr(self.behavior_classifier, 'zone_checker') and self.behavior_classifier.zone_checker:
            for zone in self.behavior_classifier.zone_checker.zones:
                points = np.array(zone, np.int32)
                cv2.polylines(frame, [points], True, (255, 255, 0), 2)
        
        return frame

def process_video_with_tracking(model_path, video_path, output_path, restricted_zones=None):
    """Process entire video with tracking and behavior analysis"""
    
    detector = DroneDetectorTracker(model_path, restricted_zones=restricted_zones)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    total_alerts = 0
    
    print(f"Processing video: {video_path}")
    print(f"Output: {output_path}")
    print(f"Total frames: {total_frames}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracks, annotated_frame, alerts = detector.process_frame(frame)
        out.write(annotated_frame)
        
        total_alerts += len(alerts)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames, "
                  f"{len(tracks)} active tracks, {total_alerts} total alerts")
    
    cap.release()
    out.release()
    
    # Print final statistics
    stats = detector.alert_manager.get_statistics()
    print(f"\n✓ Processing complete!")
    print(f"  Output: {output_path}")
    print(f"  Total alerts: {stats.get('total_alerts', 0)}")
    print(f"  High alerts: {stats.get('high_alerts', 0)}")
    print(f"  Medium alerts: {stats.get('medium_alerts', 0)}")
    print(f"  Low alerts: {stats.get('low_alerts', 0)}")

if __name__ == "__main__":
    # Example usage
    process_video_with_tracking(
        model_path='runs/detect/models/finetuned/drone_detector/weights/best.pt',
        video_path='data/raw/test_video.mp4',
        output_path='outputs/videos/tracked_video.mp4',
        restricted_zones=[
            [(100, 100), (300, 100), (300, 300), (100, 300)]  # Example zone
        ]
    )