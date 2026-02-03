import json
from datetime import datetime
from pathlib import Path
from typing import List
import numpy as np
from src.behavior.behavior_classifier import BehaviorAnalysis

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

class AlertManager:
    """Manage alerts and logging"""
    
    def __init__(self, log_file='outputs/logs/alerts.json'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.alerts = []
    
    def _safe_serialize(self, obj):
        """Safely convert any object to JSON-serializable format"""
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, str):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._safe_serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._safe_serialize(value) for key, value in obj.items()}
        else:
            return str(obj)  # Fallback to string representation
    
    def generate_alert(self, analysis: BehaviorAnalysis, frame_num: int):
        """Generate alert from behavior analysis"""
        if not analysis.is_suspicious:
            return None
        
        # Create alert with safe serialization
        alert = {
            'timestamp': datetime.now().isoformat(),
            'frame_num': self._safe_serialize(frame_num),
            'track_id': self._safe_serialize(analysis.track_id),
            'alert_level': self._safe_serialize(analysis.alert_level),
            'speed_flag': self._safe_serialize(analysis.speed_flag),
            'hover_flag': self._safe_serialize(analysis.hover_flag),
            'zone_flag': self._safe_serialize(analysis.zone_flag),
            'speed_value': self._safe_serialize(analysis.speed_value),
            'zone_name': self._safe_serialize(analysis.zone_name)
        }
        
        self.alerts.append(alert)
        self._log_alert(alert)
        
        return alert
    
    def _log_alert(self, alert):
        """Append alert to log file"""
        try:
            # Use standard JSON dumps since alert is already safely serialized
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert) + '\n')
        except Exception as e:
            # Fallback to NumpyEncoder if there are still issues
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert, cls=NumpyEncoder) + '\n')
    
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