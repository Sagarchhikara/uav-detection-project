from dataclasses import dataclass
from typing import List, Tuple
from src.behavior.speed_analyzer import SpeedAnalyzer
from src.behavior.hover_detector import HoverDetector
from src.behavior.zone_checker import ZoneChecker

@dataclass
class BehaviorAnalysis:
    """Results of behavior analysis"""
    track_id: int
    is_suspicious: bool
    speed_flag: bool
    hover_flag: bool
    zone_flag: bool
    speed_value: float
    alert_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    zone_name: str = ""

class BehaviorClassifier:
    """Classify drone behavior as normal or suspicious"""
    
    def __init__(self, fps=30, restricted_zones=None):
        self.speed_analyzer = SpeedAnalyzer(fps=fps, speed_threshold_pixels=50)
        self.hover_detector = HoverDetector(radius_threshold=10, min_frames=30)
        
        if restricted_zones:
            self.zone_checker = ZoneChecker(restricted_zones)
        else:
            self.zone_checker = None
    
    def analyze(self, track_id: int, trajectory: List[Tuple]) -> BehaviorAnalysis:
        """
        Analyze trajectory and classify behavior
        
        Args:
            track_id: Track ID
            trajectory: [(frame_num, center_x, center_y), ...]
        
        Returns:
            BehaviorAnalysis object
        """
        # Analyze speed
        speed = self.speed_analyzer.calculate_speed(trajectory)
        speed_flag = self.speed_analyzer.is_high_speed(trajectory)
        
        # Check hovering
        hover_flag = self.hover_detector.is_hovering(trajectory)
        
        # Check restricted zones
        zone_flag = False
        zone_name = ""
        if self.zone_checker and trajectory:
            _, last_x, last_y = trajectory[-1]
            zone_flag, zone_name = self.zone_checker.check_position(last_x, last_y)
        
        # Determine alert level
        flags_count = int(sum([bool(speed_flag), bool(hover_flag), bool(zone_flag)]))
        
        if bool(zone_flag):
            alert_level = 'HIGH'
        elif flags_count >= 2:
            alert_level = 'MEDIUM'
        elif flags_count == 1:
            alert_level = 'LOW'
        else:
            alert_level = 'NORMAL'
        
        is_suspicious = bool(alert_level != 'NORMAL')
        
        return BehaviorAnalysis(
            track_id=int(track_id),
            is_suspicious=bool(is_suspicious),
            speed_flag=bool(speed_flag),
            hover_flag=bool(hover_flag),
            zone_flag=bool(zone_flag),
            speed_value=float(speed),
            alert_level=str(alert_level),
            zone_name=str(zone_name)
        )