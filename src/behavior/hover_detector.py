import numpy as np
from typing import List, Tuple

class HoverDetector:
    """Detect hovering behavior"""
    
    def __init__(self, radius_threshold=10, min_frames=30):
        """
        Args:
            radius_threshold: Maximum movement radius (pixels)
            min_frames: Minimum frames to qualify as hovering
        """
        self.radius_threshold = radius_threshold
        self.min_frames = min_frames
    
    def is_hovering(self, trajectory: List[Tuple[int, float, float]]) -> bool:
        """
        Check if drone is hovering
        
        Args:
            trajectory: [(frame_num, center_x, center_y), ...]
        
        Returns:
            True if hovering detected
        """
        if len(trajectory) < self.min_frames:
            return False
        
        # Check recent trajectory
        recent = trajectory[-self.min_frames:]
        positions = np.array([(x, y) for _, x, y in recent])
        
        # Calculate centroid
        centroid = positions.mean(axis=0)
        
        # Calculate maximum distance from centroid
        distances = np.sqrt(((positions - centroid)**2).sum(axis=1))
        max_distance = distances.max()
        
        return bool(max_distance < self.radius_threshold)
    
    def calculate_movement_variance(self, trajectory: List[Tuple]) -> float:
        """Calculate variance in position (low = hovering)"""
        if len(trajectory) < 2:
            return 0.0
        
        positions = np.array([(x, y) for _, x, y in trajectory])
        return float(np.var(positions))