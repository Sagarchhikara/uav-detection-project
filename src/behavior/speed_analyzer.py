import numpy as np
from typing import List, Tuple

class SpeedAnalyzer:
    """Analyze drone speed from trajectory"""
    
    def __init__(self, fps=30, speed_threshold_pixels=50):
        """
        Args:
            fps: Frames per second
            speed_threshold_pixels: Speed threshold in pixels/frame
        """
        self.fps = fps
        self.speed_threshold = speed_threshold_pixels
    
    def calculate_speed(self, trajectory: List[Tuple[int, float, float]]) -> float:
        """
        Calculate average speed from trajectory
        
        Args:
            trajectory: [(frame_num, center_x, center_y), ...]
        
        Returns:
            Average speed in pixels/frame
        """
        if len(trajectory) < 2:
            return 0.0
        
        speeds = []
        for i in range(len(trajectory) - 1):
            frame1, x1, y1 = trajectory[i]
            frame2, x2, y2 = trajectory[i + 1]
            
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            time_diff = (frame2 - frame1) / self.fps  # seconds
            
            if time_diff > 0:
                speed = distance / time_diff  # pixels/second
                speeds.append(speed)
        
        return float(np.mean(speeds)) if speeds else 0.0
    
    def is_high_speed(self, trajectory: List[Tuple]) -> bool:
        """Check if drone is moving at suspicious speed"""
        speed = self.calculate_speed(trajectory)
        return bool(speed > self.speed_threshold)
    
    def get_instant_speed(self, trajectory: List[Tuple], window=5) -> float:
        """Get instantaneous speed over last N frames"""
        if len(trajectory) < 2:
            return 0.0
        
        recent = trajectory[-min(window, len(trajectory)):]
        return float(self.calculate_speed(recent))