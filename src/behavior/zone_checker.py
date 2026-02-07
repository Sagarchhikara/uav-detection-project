import numpy as np
from typing import List, Tuple

class ZoneChecker:
    """Check if drone enters restricted zones"""
    
    def __init__(self, restricted_zones: List[List[Tuple[int, int]]]):
        """
        Args:
            restricted_zones: List of polygons, each polygon is list of (x, y) points
                             Example: [[(100, 100), (200, 100), (200, 200), (100, 200)]]
        """
        self.zones = restricted_zones
        self.zone_names = [f"Zone_{i}" for i in range(len(restricted_zones))]
    
    def check_position(self, x: float, y: float) -> Tuple[bool, str]:
        """
        Check if position is in any restricted zone
        
        Returns:
            (is_restricted, zone_name)
        """
        point = (x, y)
        
        for i, zone in enumerate(self.zones):
            if self._point_in_polygon(point, zone):
                return bool(True), str(self.zone_names[i])
        
        return bool(False), str("")
    
    def check_trajectory(self, trajectory: List[Tuple[int, float, float]]) -> bool:
        """Check if any point in trajectory enters restricted zone"""
        for _, x, y in trajectory:
            is_restricted, _ = self.check_position(x, y)
            if is_restricted:
                return bool(True)
        return bool(False)
    
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """
        Check if point is inside polygon using ray casting algorithm
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return bool(inside)