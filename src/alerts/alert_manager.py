import json
from datetime import datetime
from pathlib import Path
from typing import List
from src.behavior.behavior_classifier import BehaviorAnalysis

class AlertManager:
    """Manage alerts and logging"""
    
    def __init__(self, log_file='outputs/logs/alerts.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.alerts = []
    
    def generate_alert(self, analysis: BehaviorAnalysis, frame_num: int):
        """Generate alert from behavior analysis"""
        if not analysis.is_suspicious:
            return None
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'frame_num': frame_num,
            'track_id': analysis.track_id,
            'alert_level': analysis.alert_level,
            'speed_flag': analysis.speed_flag,
            'hover_flag': analysis.hover_flag,
            'zone_flag': analysis.zone_flag,
            'speed_value': analysis.speed_value,
            'zone_name': analysis.zone_name
        }
        
        self.alerts.append(alert)
        self._log_alert(alert)
        
        return alert
    
    def _log_alert(self, alert):
        """Append alert to log file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def get_statistics(self):
        """Get alert statistics"""
        if not self.alerts:
            return {}
        
        return {
            'total_alerts': len(self.alerts),
            'high_alerts': sum(1 for a in self.alerts if a['alert_level'] == 'HIGH'),
            'medium_alerts': sum(1 for a in self.alerts if a['alert_level'] == 'MEDIUM'),
            'low_alerts': sum(1 for a in self.alerts if a['alert_level'] == 'LOW'),
            'speed_violations': sum(1 for a in self.alerts if a['speed_flag']),
            'hover_detections': sum(1 for a in self.alerts if a['hover_flag']),
            'zone_violations': sum(1 for a in self.alerts if a['zone_flag'])
        }