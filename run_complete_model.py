#!/usr/bin/env python3
"""
Run the complete Anti-UAV Detection System ML model
This runs the full pipeline: Detection + Tracking + Behavior Analysis + Alerts
"""

import sys
import cv2
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

from detection.detector_with_tracking import DroneDetectorTracker

def run_complete_model():
    """Run the complete Anti-UAV ML model on video"""
    
    print("🚁 ANTI-UAV DETECTION SYSTEM - COMPLETE ML MODEL")
    print("=" * 70)
    
    # Check if we have a test video
    video_path = 'test_drone_video.mp4'
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Creating test video first...")
        create_test_video()
    
    # Define restricted zones for behavior analysis
    restricted_zones = [
        [(200, 150), (400, 150), (400, 250), (200, 250)],  # Zone 1
        [(800, 400), (1000, 400), (1000, 500), (800, 500)]  # Zone 2
    ]
    
    print(f"📹 Input Video: {video_path}")
    print(f"🚫 Restricted Zones: {len(restricted_zones)} zones defined")
    print(f"🎯 Model: YOLOv8s (Pre-trained)")
    print(f"🔍 Confidence Threshold: 0.3")
    print("-" * 70)
    
    # Initialize the complete detection system
    print("🚀 Initializing Anti-UAV Detection System...")
    detector = DroneDetectorTracker(
        model_path='yolov8s.pt',
        conf_threshold=0.3,
        restricted_zones=restricted_zones
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"📊 Video Properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {duration:.1f}s")
    print(f"   Total Frames: {total_frames}")
    print("-" * 70)
    
    # Create output video
    output_path = 'outputs/videos/complete_model_output.mp4'
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Processing statistics
    stats = {
        'frames_processed': 0,
        'total_detections': 0,
        'total_tracks': 0,
        'total_alerts': 0,
        'alert_breakdown': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'unique_tracks': set(),
        'processing_times': []
    }
    
    print("🔄 PROCESSING VIDEO...")
    print("Frame | Tracks | Alerts | Alert Level | Processing Time")
    print("-" * 70)
    
    frame_idx = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # Process frame through complete ML pipeline
        tracks, annotated_frame, alerts = detector.process_frame(frame)
        
        frame_time = time.time() - frame_start
        stats['processing_times'].append(frame_time)
        
        # Update statistics
        stats['frames_processed'] += 1
        stats['total_tracks'] += len(tracks)
        stats['total_alerts'] += len(alerts)
        
        # Track unique drone IDs
        for track_id, _, _, _, _, _ in tracks:
            stats['unique_tracks'].add(track_id)
        
        # Count alert levels
        alert_levels = []
        for alert in alerts:
            level = alert.get('alert_level', 'UNKNOWN')
            stats['alert_breakdown'][level] = stats['alert_breakdown'].get(level, 0) + 1
            alert_levels.append(level)
        
        # Write processed frame
        out.write(annotated_frame)
        
        # Print progress every 10 frames
        if frame_idx % 10 == 0 or alerts:
            alert_str = ', '.join(alert_levels) if alert_levels else 'None'
            print(f"{frame_idx:5d} | {len(tracks):6d} | {len(alerts):6d} | {alert_str:11s} | {frame_time:.3f}s")
        
        frame_idx += 1
        
        # Process only first 100 frames for demo (remove this limit for full processing)
        if frame_idx >= 100:
            break
    
    # Cleanup
    cap.release()
    out.release()
    
    total_time = time.time() - start_time
    avg_fps = stats['frames_processed'] / total_time
    avg_frame_time = sum(stats['processing_times']) / len(stats['processing_times'])
    
    # Final Results
    print("=" * 70)
    print("🎯 COMPLETE ML MODEL RESULTS")
    print("=" * 70)
    
    print(f"📊 Processing Statistics:")
    print(f"   Frames Processed: {stats['frames_processed']}")
    print(f"   Total Processing Time: {total_time:.2f}s")
    print(f"   Average FPS: {avg_fps:.1f}")
    print(f"   Average Frame Time: {avg_frame_time:.3f}s")
    
    print(f"\n🎯 Detection & Tracking:")
    print(f"   Total Track Instances: {stats['total_tracks']}")
    print(f"   Unique Drones Detected: {len(stats['unique_tracks'])}")
    print(f"   Unique Track IDs: {sorted(list(stats['unique_tracks']))}")
    
    print(f"\n🚨 Alert System:")
    print(f"   Total Alerts Generated: {stats['total_alerts']}")
    print(f"   HIGH Priority: {stats['alert_breakdown'].get('HIGH', 0)}")
    print(f"   MEDIUM Priority: {stats['alert_breakdown'].get('MEDIUM', 0)}")
    print(f"   LOW Priority: {stats['alert_breakdown'].get('LOW', 0)}")
    
    print(f"\n📁 Output Files:")
    print(f"   Processed Video: {output_path}")
    print(f"   Alert Logs: outputs/logs/alerts.json")
    
    # Get final statistics from alert manager
    final_stats = detector.alert_manager.get_statistics()
    if final_stats:
        print(f"\n📈 Alert Manager Statistics:")
        for key, value in final_stats.items():
            print(f"   {key}: {value}")
    
    print("\n🎉 COMPLETE ML MODEL EXECUTION FINISHED!")
    print("=" * 70)
    
    return stats

def create_test_video():
    """Create test video if it doesn't exist"""
    import numpy as np
    
    width, height = 1280, 720
    fps = 25
    duration = 4  # Shorter for demo
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_drone_video.mp4', fourcc, fps, (width, height))
    
    print("Creating test video...")
    
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (135, 206, 235)  # Sky blue
        
        t = frame_idx / total_frames
        
        # Drone 1: Fast movement
        x1 = int(50 + t * (width - 100))
        y1 = int(height // 3)
        cv2.circle(frame, (x1, y1), 12, (50, 50, 50), -1)
        
        # Drone 2: Hovering
        x2 = int(width // 2 + 15 * np.sin(t * 12 * np.pi))
        y2 = int(height // 2 + 8 * np.cos(t * 12 * np.pi))
        cv2.circle(frame, (x2, y2), 10, (70, 70, 70), -1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print("✅ Test video created")

if __name__ == "__main__":
    try:
        stats = run_complete_model()
        print(f"\n✅ SUCCESS: Complete ML model executed successfully!")
        print(f"   Processed {stats['frames_processed']} frames")
        print(f"   Detected {len(stats['unique_tracks'])} unique drones")
        print(f"   Generated {stats['total_alerts']} alerts")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()