
import unittest
import numpy as np
import sys
import os
from typing import List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.tracking.tracker import SimpleTracker
from src.behavior.behavior_classifier import BehaviorClassifier, BehaviorAnalysis
from src.alerts.alert_manager import AlertManager

class MockDetector:
    """Mocks the YOLO detector by returning predefined detections"""
    def __init__(self, detections_per_frame: List[List[Tuple]]):
        self.detections = detections_per_frame
        self.frame_idx = 0

    def detect(self, frame):
        if self.frame_idx < len(self.detections):
            dets = self.detections[self.frame_idx]
            self.frame_idx += 1
            return dets
        return []

class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Setup output directory
        self.test_log = 'outputs/logs/test_alerts.json'
        if os.path.exists(self.test_log):
            os.remove(self.test_log)
            
        # Initialize components
        # Set min_hits=1 so we get tracks immediately for testing
        self.tracker = SimpleTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        
        # Define a zone: square from (100, 100) to (300, 300)
        self.zones = [[(100, 100), (300, 100), (300, 300), (100, 300)]]
        self.classifier = BehaviorClassifier(fps=30, restricted_zones=self.zones)
        self.alert_manager = AlertManager(log_file=self.test_log)

    def test_speed_detection(self):
        """Test if high speed object triggers alert"""
        # Create a trajectory moving FAST but trackable
        # Threshold is 50 pixels per frame
        # We move 60 pixels per frame using larger box
        
        detections = []
        for i in range(10):
            x = i * 60
            detections.append([(x, 0, x+200, 200, 0.9)])
            
        alerts_triggered = False
        
        for frame_idx, dets in enumerate(detections):
            tracks = self.tracker.update(dets)
            
            for track_id, x1, y1, x2, y2, conf in tracks:
                history = self.tracker.get_track_history(track_id)
                if len(history) >= 2:
                    analysis = self.classifier.analyze(track_id, history)
                    if analysis.speed_flag:
                        alerts_triggered = True
                        
        self.assertTrue(alerts_triggered, "Should detect high speed")

    def test_hovering_detection(self):
        """Test if stationary object triggers hovering alert"""
        # Stationary object at 400, 400 (outside zone)
        detections = []
        for i in range(40): # Need at least 30 frames
             # Add tiny noise (jitter)
            jitter = (i % 2) 
            x = 400 + jitter
            detections.append([(x, 400, x+20, 420, 0.9)])
            
        alerts_triggered = False
        for frame_idx, dets in enumerate(detections):
            tracks = self.tracker.update(dets)
            for track_id, x1, y1, x2, y2, conf in tracks:
                history = self.tracker.get_track_history(track_id)
                if len(history) > 30:
                    analysis = self.classifier.analyze(track_id, history)
                    if analysis.hover_flag:
                        alerts_triggered = True
                        
        self.assertTrue(alerts_triggered, "Should detect hovering")

    def test_zone_checker_direct(self):
        """Test ZoneChecker independently"""
        from src.behavior.zone_checker import ZoneChecker
        zones = [[(100, 100), (300, 100), (300, 300), (100, 300)]]
        checker = ZoneChecker(zones)
        
        # Test inside
        is_in, name = checker.check_position(150, 150)
        self.assertTrue(is_in, "Direct check: 150,150 should be inside")
        
        # Test outside
        is_in, name = checker.check_position(10, 10)
        self.assertFalse(is_in, "Direct check: 10,10 should be outside")

    def test_zone_entry(self):
        """Test if object entering zone triggers alert"""
        # Object starts outside and enters zone
        detections = []
        # Frame 0: Outside (0,0) - (20,20)
        detections.append([(0, 0, 20, 20, 0.9)])
        # Frame 1: Inside (150, 150)
        # Note: If tracker loses track, it creates a new ID.
        detections.append([(150, 150, 170, 170, 0.9)])
        
        alerts_triggered = False
        for frame_idx, dets in enumerate(detections):
            tracks = self.tracker.update(dets)
            for track_id, x1, y1, x2, y2, conf in tracks:
                history = self.tracker.get_track_history(track_id)
                analysis = self.classifier.analyze(track_id, history)
                
                if analysis.zone_flag:
                    alerts_triggered = True
                    self.assertEqual(analysis.alert_level, 'HIGH', "Zone entry should be HIGH alert")
                    
        self.assertTrue(alerts_triggered, "Should detect zone entry")

if __name__ == '__main__':
    unittest.main()
