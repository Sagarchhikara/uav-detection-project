#!/usr/bin/env python3
"""
Create a test video for the Anti-UAV system
"""

import cv2
import numpy as np

def create_test_video():
    """Create a test video with moving objects that look like drones"""
    
    # Video parameters
    width, height = 1280, 720  # Match the resolution from your screenshot
    fps = 25
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_drone_video.mp4', fourcc, fps, (width, height))
    
    print(f"Creating test video: {width}x{height} @ {fps}fps, {duration}s ({total_frames} frames)")
    
    for frame_idx in range(total_frames):
        # Create frame with sky-like background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (135, 206, 235)  # Sky blue background
        
        # Add some clouds
        cv2.ellipse(frame, (200, 150), (80, 40), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (600, 100), (60, 30), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(frame, (1000, 200), (100, 50), 0, 0, 360, (255, 255, 255), -1)
        
        t = frame_idx / total_frames
        
        # Drone 1: Fast moving across screen (suspicious behavior)
        x1 = int(50 + t * (width - 100))
        y1 = int(height // 3)
        # Draw drone body
        cv2.ellipse(frame, (x1, y1), (15, 8), 0, 0, 360, (50, 50, 50), -1)
        # Draw propellers
        cv2.circle(frame, (x1-10, y1-5), 3, (100, 100, 100), -1)
        cv2.circle(frame, (x1+10, y1-5), 3, (100, 100, 100), -1)
        cv2.circle(frame, (x1-10, y1+5), 3, (100, 100, 100), -1)
        cv2.circle(frame, (x1+10, y1+5), 3, (100, 100, 100), -1)
        
        # Drone 2: Hovering with small movements (suspicious behavior)
        x2 = int(width // 2 + 20 * np.sin(t * 10 * np.pi))
        y2 = int(height // 2 + 10 * np.cos(t * 10 * np.pi))
        # Draw drone body
        cv2.ellipse(frame, (x2, y2), (12, 6), 0, 0, 360, (70, 70, 70), -1)
        # Draw propellers
        cv2.circle(frame, (x2-8, y2-4), 2, (120, 120, 120), -1)
        cv2.circle(frame, (x2+8, y2-4), 2, (120, 120, 120), -1)
        cv2.circle(frame, (x2-8, y2+4), 2, (120, 120, 120), -1)
        cv2.circle(frame, (x2+8, y2+4), 2, (120, 120, 120), -1)
        
        # Drone 3: Normal speed movement
        if t > 0.3:  # Appears later
            x3 = int(100 + (t-0.3) * 300)
            y3 = int(height * 0.7 + 30 * np.sin((t-0.3) * 4 * np.pi))
            # Draw drone body
            cv2.ellipse(frame, (x3, y3), (10, 5), 0, 0, 360, (60, 60, 60), -1)
            # Draw propellers
            cv2.circle(frame, (x3-6, y3-3), 2, (110, 110, 110), -1)
            cv2.circle(frame, (x3+6, y3-3), 2, (110, 110, 110), -1)
            cv2.circle(frame, (x3-6, y3+3), 2, (110, 110, 110), -1)
            cv2.circle(frame, (x3+6, y3+3), 2, (110, 110, 110), -1)
        
        # Add some birds for contrast (should not be detected as drones)
        if frame_idx % 60 < 30:  # Flapping wings
            bird_x = int(300 + t * 200)
            bird_y = int(100 + 20 * np.sin(t * 8 * np.pi))
            cv2.ellipse(frame, (bird_x, bird_y), (8, 3), 0, 0, 360, (0, 0, 0), -1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_idx:04d}/{total_frames}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.2f}s", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Anti-UAV Test Video", (20, height-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_idx % 50 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%)")
    
    out.release()
    print("✅ Test video created: test_drone_video.mp4")
    print(f"Video specs: {width}x{height}, {fps}fps, {duration}s, {total_frames} frames")

if __name__ == "__main__":
    create_test_video()