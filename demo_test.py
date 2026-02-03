#!/usr/bin/env python3
"""
Demo test of the Anti-UAV Detection System
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add src to path
sys.path.append('src')

def create_test_video():
    """Create a simple test video with moving objects"""
    print("🎬 Creating test video...")
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_video.mp4', fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Create frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # Dark gray background
        
        # Add moving "drone" (white rectangle)
        t = frame_idx / total_frames
        x = int(50 + t * (width - 100))  # Move from left to right
        y = int(height // 2 + 50 * np.sin(t * 4 * np.pi))  # Sine wave motion
        
        # Draw "drone"
        cv2.rectangle(frame, (x-15, y-10), (x+15, y+10), (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Test video created: test_video.mp4")

def test_detection():
    """Test the detection system"""
    print("🎯 Testing detection system...")
    
    try:
        from detection.yolo_detector import DroneDetector
        
        # Initialize detector with pre-trained YOLO
        detector = DroneDetector('yolov8s.pt', conf_threshold=0.3)
        
        # Test on a single frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)
        
        # Add a white rectangle (simulated object)
        cv2.rectangle(frame, (200, 150), (250, 200), (255, 255, 255), -1)
        
        # Run detection
        detections = detector.detect(frame)
        
        print(f"✅ Detection test passed - Found {len(detections)} objects")
        
        # Test visualization
        annotated_frame = detector.draw_detections(frame.copy(), detections)
        cv2.imwrite('detection_test.jpg', annotated_frame)
        print("✅ Saved detection test result: detection_test.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_tracking():
    """Test the tracking system"""
    print("🔗 Testing tracking system...")
    
    try:
        from tracking.tracker import SimpleTracker
        
        tracker = SimpleTracker()
        
        # Simulate detections across frames
        frame1_detections = [(100, 100, 150, 150, 0.9)]  # x1, y1, x2, y2, conf
        frame2_detections = [(105, 105, 155, 155, 0.8)]  # Moved slightly
        frame3_detections = [(110, 110, 160, 160, 0.85)]  # Moved again
        
        # Update tracker
        tracks1 = tracker.update(frame1_detections)
        tracks2 = tracker.update(frame2_detections)
        tracks3 = tracker.update(frame3_detections)
        
        print(f"✅ Tracking test passed")
        print(f"  Frame 1: {len(tracks1)} tracks")
        print(f"  Frame 2: {len(tracks2)} tracks")
        print(f"  Frame 3: {len(tracks3)} tracks")
        
        # Test trajectory
        if tracks3:
            track_id = tracks3[0][0]
            history = tracker.get_track_history(track_id)
            print(f"  Track {track_id} history: {len(history)} points")
        
        return True
        
    except Exception as e:
        print(f"❌ Tracking test failed: {e}")
        return False

def test_behavior_analysis():
    """Test behavior analysis"""
    print("🧠 Testing behavior analysis...")
    
    try:
        from behavior.behavior_classifier import BehaviorClassifier
        
        # Create classifier
        classifier = BehaviorClassifier(fps=30)
        
        # Create test trajectory (high speed movement)
        trajectory = [
            (0, 100, 100),   # frame, x, y
            (1, 120, 100),
            (2, 140, 100),
            (3, 160, 100),
            (4, 180, 100),
        ]
        
        # Analyze behavior
        analysis = classifier.analyze(track_id=1, trajectory=trajectory)
        
        print(f"✅ Behavior analysis test passed")
        print(f"  Suspicious: {analysis.is_suspicious}")
        print(f"  Alert Level: {analysis.alert_level}")
        print(f"  Speed Flag: {analysis.speed_flag}")
        print(f"  Speed Value: {analysis.speed_value:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Behavior analysis test failed: {e}")
        return False

def test_full_pipeline():
    """Test the complete pipeline"""
    print("🚀 Testing full pipeline...")
    
    if not Path('test_video.mp4').exists():
        create_test_video()
    
    try:
        from detection.detector_with_tracking import DroneDetectorTracker
        
        # Initialize full pipeline
        detector = DroneDetectorTracker('yolov8s.pt', conf_threshold=0.3)
        
        # Open test video
        cap = cv2.VideoCapture('test_video.mp4')
        
        frame_count = 0
        total_tracks = 0
        total_alerts = 0
        
        # Process first 30 frames
        while frame_count < 30:
            ret, frame = cap.read()
            if not ret:
                break
            
            tracks, annotated_frame, alerts = detector.process_frame(frame)
            
            total_tracks += len(tracks)
            total_alerts += len(alerts)
            
            # Save first annotated frame
            if frame_count == 0:
                cv2.imwrite('pipeline_test.jpg', annotated_frame)
            
            frame_count += 1
        
        cap.release()
        
        print(f"✅ Full pipeline test passed")
        print(f"  Processed {frame_count} frames")
        print(f"  Total track detections: {total_tracks}")
        print(f"  Total alerts: {total_alerts}")
        print("✅ Saved pipeline test result: pipeline_test.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚁 Anti-UAV Detection System - Demo Test")
    print("=" * 60)
    
    tests = [
        ("Detection", test_detection),
        ("Tracking", test_tracking),
        ("Behavior Analysis", test_behavior_analysis),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your Anti-UAV system is working!")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("\n🚀 Next steps:")
    print("1. Add real drone dataset for training")
    print("2. Train custom model: python scripts/train_model.py")
    print("3. Launch web interface: streamlit run ui/streamlit_app.py")

if __name__ == "__main__":
    main()