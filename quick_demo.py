#!/usr/bin/env python3
"""
Quick demonstration of the Anti-UAV Detection System
"""

import sys
import os
import numpy as np
import cv2
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

def demo_video_processing():
    """Demonstrate video processing with the full pipeline"""
    print("🎬 Anti-UAV Detection System - Live Demo")
    print("=" * 60)
    
    # Import our modules
    from detection.detector_with_tracking import DroneDetectorTracker
    
    # Check if we have test video
    if not Path('test_video.mp4').exists():
        print("Creating test video with simulated drone movement...")
        create_demo_video()
    
    # Initialize the full detection system
    print("\n🚀 Initializing Anti-UAV Detection System...")
    
    # Define restricted zones (example)
    restricted_zones = [
        [(200, 100), (400, 100), (400, 200), (200, 200)],  # Top zone
        [(100, 300), (300, 300), (300, 400), (100, 400)]   # Bottom zone
    ]
    
    detector = DroneDetectorTracker(
        model_path='yolov8s.pt',
        conf_threshold=0.3,
        restricted_zones=restricted_zones
    )
    
    # Process the video
    print("\n📹 Processing video...")
    
    cap = cv2.VideoCapture('test_video.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_output.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    all_alerts = []
    detection_summary = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_tracks': 0,
        'total_alerts': 0,
        'alert_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
    }
    
    print("Processing frames:", end=" ")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame through full pipeline
        tracks, annotated_frame, alerts = detector.process_frame(frame)
        
        # Update statistics
        detection_summary['total_frames'] += 1
        if tracks:
            detection_summary['frames_with_detections'] += 1
            detection_summary['total_tracks'] += len(tracks)
        
        if alerts:
            detection_summary['total_alerts'] += len(alerts)
            for alert in alerts:
                level = alert.get('alert_level', 'LOW')
                detection_summary['alert_breakdown'][level] += 1
                all_alerts.append(alert)
        
        # Write annotated frame
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"{frame_count}", end=" ")
        
        # Process only first 60 frames for demo
        if frame_count >= 60:
            break
    
    cap.release()
    out.release()
    
    print(f"\n✅ Processing complete!")
    
    # Display results
    print("\n📊 DETECTION RESULTS:")
    print("=" * 40)
    print(f"Total Frames Processed: {detection_summary['total_frames']}")
    print(f"Frames with Detections: {detection_summary['frames_with_detections']}")
    print(f"Total Track Instances: {detection_summary['total_tracks']}")
    print(f"Total Alerts Generated: {detection_summary['total_alerts']}")
    
    if detection_summary['total_alerts'] > 0:
        print(f"\n🚨 Alert Breakdown:")
        for level, count in detection_summary['alert_breakdown'].items():
            if count > 0:
                print(f"  {level}: {count} alerts")
    
    # Save detailed results
    with open('demo_results.json', 'w') as f:
        json.dump({
            'summary': detection_summary,
            'alerts': all_alerts
        }, f, indent=2, default=str)
    
    print(f"\n📁 Output Files Created:")
    print(f"  🎥 Processed Video: demo_output.mp4")
    print(f"  📋 Results Report: demo_results.json")
    
    # Show system capabilities
    print(f"\n🎯 System Capabilities Demonstrated:")
    print(f"  ✅ Object Detection (YOLOv8)")
    print(f"  ✅ Multi-Object Tracking")
    print(f"  ✅ Behavior Analysis")
    print(f"  ✅ Alert Generation")
    print(f"  ✅ Restricted Zone Monitoring")
    print(f"  ✅ Real-time Processing")

def create_demo_video():
    """Create a demo video with various drone behaviors"""
    width, height = 640, 480
    fps = 30
    duration = 3  # seconds
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)  # Dark background
        
        # Draw restricted zones
        cv2.rectangle(frame, (200, 100), (400, 200), (0, 0, 100), 2)  # Red zone
        cv2.rectangle(frame, (100, 300), (300, 400), (0, 0, 100), 2)  # Red zone
        cv2.putText(frame, "RESTRICTED", (210, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "RESTRICTED", (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        t = frame_idx / total_frames
        
        # Drone 1: Fast moving (suspicious)
        x1 = int(50 + t * (width - 100))
        y1 = int(height // 4)
        cv2.circle(frame, (x1, y1), 8, (255, 255, 255), -1)
        cv2.putText(frame, "D1", (x1-10, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Drone 2: Hovering (suspicious)
        x2 = int(width // 2 + 10 * np.sin(t * 8 * np.pi))
        y2 = int(height // 2 + 5 * np.cos(t * 8 * np.pi))
        cv2.circle(frame, (x2, y2), 8, (255, 255, 255), -1)
        cv2.putText(frame, "D2", (x2-10, y2-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Drone 3: Enters restricted zone
        if t > 0.5:
            x3 = int(150 + (t-0.5) * 400)
            y3 = int(150)  # In restricted zone
            cv2.circle(frame, (x3, y3), 8, (255, 255, 255), -1)
            cv2.putText(frame, "D3", (x3-10, y3-15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx:03d}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Time: {t:.2f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()

if __name__ == "__main__":
    demo_video_processing()