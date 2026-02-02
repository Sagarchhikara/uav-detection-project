import numpy as np
from collections import defaultdict, deque
from typing import List, Tuple, Dict

class SimpleTracker:
    """
    Simple IoU-based tracker for drone detection
    Simplified version suitable for hackathon
    """
    
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum detections before track is confirmed
            iou_threshold: Minimum IoU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = {}  # track_id -> Track object
        self.next_id = 0
        self.frame_count = 0
    
    def update(self, detections: List[Tuple[float, float, float, float, float]]):
        """
        Update tracks with new detections
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence]
        
        Returns:
            List of active tracks: [(track_id, x1, y1, x2, y2, confidence), ...]
        """
        self.frame_count += 1
        
        # Match detections to existing tracks
        if detections and self.tracks:
            matches, unmatched_detections, unmatched_tracks = self._match(detections)
        else:
            matches = []
            unmatched_detections = list(range(len(detections)))
            unmatched_tracks = list(self.tracks.keys())
        
        # Update matched tracks
        for det_idx, track_id in matches:
            self.tracks[track_id].update(detections[det_idx], self.frame_count)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._create_track(detections[det_idx])
        
        # Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            self.tracks[track_id].mark_lost()
        
        # Remove dead tracks
        self._remove_dead_tracks()
        
        # Return confirmed tracks
        return self._get_active_tracks()
    
    def _match(self, detections):
        """Match detections to tracks using IoU"""
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        
        track_ids = list(self.tracks.keys())
        for d_idx, det in enumerate(detections):
            for t_idx, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id].bbox
                iou_matrix[d_idx, t_idx] = self._calculate_iou(det[:4], track_bbox)
        
        # Simple greedy matching
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(track_ids)))
        
        while True:
            if iou_matrix.size == 0:
                break
            
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            
            d_idx, t_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matches.append((d_idx, track_ids[t_idx]))
            
            unmatched_detections.remove(d_idx)
            unmatched_tracks.remove(t_idx)
            
            iou_matrix[d_idx, :] = 0
            iou_matrix[:, t_idx] = 0
        
        unmatched_tracks = [track_ids[i] for i in unmatched_tracks]
        
        return matches, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _create_track(self, detection):
        """Create new track"""
        track = Track(self.next_id, detection, self.frame_count)
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _remove_dead_tracks(self):
        """Remove tracks that haven't been seen for too long"""
        dead_tracks = []
        for track_id, track in self.tracks.items():
            if self.frame_count - track.last_seen > self.max_age:
                dead_tracks.append(track_id)
        
        for track_id in dead_tracks:
            del self.tracks[track_id]
    
    def _get_active_tracks(self):
        """Get confirmed active tracks"""
        active = []
        for track_id, track in self.tracks.items():
            if track.hits >= self.min_hits:
                active.append((track_id, *track.bbox, track.confidence))
        return active
    
    def get_track_history(self, track_id):
        """Get trajectory history for a track"""
        if track_id in self.tracks:
            return list(self.tracks[track_id].history)
        return []


class Track:
    """Single tracked object"""
    
    def __init__(self, track_id, detection, frame_num):
        self.track_id = track_id
        self.bbox = detection[:4]  # x1, y1, x2, y2
        self.confidence = detection[4]
        self.hits = 1
        self.age = 0
        self.last_seen = frame_num
        
        # Trajectory history: [(frame_num, center_x, center_y), ...]
        center = self._get_center(self.bbox)
        self.history = deque(maxlen=100)  # Keep last 100 positions
        self.history.append((frame_num, *center))
    
    def update(self, detection, frame_num):
        """Update track with new detection"""
        self.bbox = detection[:4]
        self.confidence = detection[4]
        self.hits += 1
        self.last_seen = frame_num
        
        center = self._get_center(self.bbox)
        self.history.append((frame_num, *center))
    
    def mark_lost(self):
        """Mark track as lost (not detected this frame)"""
        self.age += 1
    
    def _get_center(self, bbox):
        """Get center point of bbox"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)