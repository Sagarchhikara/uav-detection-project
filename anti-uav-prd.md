# Anti-UAV Detection System - Project Requirements Document (PRD)

## Project Overview

**Project Name:** Anti-UAV Computer Vision Detection System  
**Duration:** 5 days prep + 24-hour hackathon  
**Goal:** Detect drones in video footage that evade traditional radar systems and identify suspicious behavior (high speed, hovering)

---

## Problem Statement

Current radar systems cannot detect small, low-altitude drones effectively. This system uses computer vision to:
1. Detect drones in video footage (including infrared/thermal)
2. Track drones across video frames
3. Analyze behavior to identify suspicious activity (speeding, hovering, restricted zone entry)
4. Alert operators in real-time

---

## Technical Stack

### Core Technologies
- **Language:** Python 3.9+
- **Deep Learning Framework:** PyTorch 2.0+
- **Object Detection:** YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV 4.8+
- **Video Processing:** FFmpeg
- **UI Framework:** Streamlit OR Flask
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Plotly

### Hardware Requirements
- **GPU:** NVIDIA RTX 4050 (for training)
- **RAM:** 16GB minimum
- **Storage:** 100GB free (for 50GB dataset + outputs)

### Development Environment
- **OS:** Windows/Linux/Mac
- **IDE:** VS Code with Python extensions
- **Version Control:** Git

---

## Dataset Information

**Source:** Anti-UAV GitHub dataset (https://github.com/ZhaoJ9014/Anti-UAV)  
**Size:** 50GB  
**Format:** Video sequences with frame-by-frame annotations  
**Content:** 
- Infrared/thermal drone footage
- Regular RGB drone footage
- Annotations in JSON format (bounding boxes, visibility flags)
- Three drone sizes: Large, Normal, Tiny

**Data Structure:**
```
dataset/
├── train/
│   ├── video_name/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── IR_label.json
├── test/
└── validation/
```

---

## System Architecture

### Phase 1: Object Detection (Days 1-2)

**Component:** Drone Detection Module

**Input:** Video file or video stream  
**Output:** Bounding boxes around detected drones with confidence scores

**Requirements:**
1. Load pre-trained YOLOv8 model (yolov8n.pt or yolov8s.pt)
2. Fine-tune model on Anti-UAV dataset
3. Support both image and video input
4. Handle infrared/thermal and RGB video
5. Output format: `[frame_id, x1, y1, x2, y2, confidence, class]`

**Acceptance Criteria:**
- [ ] Model can detect drones in test video with >80% accuracy
- [ ] Processing speed: >15 FPS on RTX 4050
- [ ] Works on both thermal and RGB footage
- [ ] Can handle different drone sizes (tiny to large)

**Key Functions to Implement:**
```python
def load_model(model_path: str, device: str) -> YOLO
def prepare_dataset(dataset_path: str, output_format: str) -> None
def train_model(model: YOLO, data_yaml: str, epochs: int) -> None
def detect_drones(model: YOLO, video_path: str) -> List[Detection]
def draw_detections(frame: np.ndarray, detections: List) -> np.ndarray
```

---

### Phase 2: Object Tracking (Day 3)

**Component:** Multi-Object Tracking Module

**Input:** Per-frame detections from Phase 1  
**Output:** Tracked objects with unique IDs across frames

**Requirements:**
1. Implement tracking algorithm (DeepSORT, ByteTrack, or SORT)
2. Assign unique ID to each detected drone
3. Maintain track across occlusions (up to 30 frames)
4. Store trajectory data (positions over time)

**Acceptance Criteria:**
- [ ] Each drone maintains consistent ID across video
- [ ] Can handle multiple drones simultaneously (up to 10)
- [ ] Trajectory stored as time-series data
- [ ] Handle drone re-entry after leaving frame

**Key Functions to Implement:**
```python
def initialize_tracker(tracker_type: str) -> Tracker
def update_tracks(tracker: Tracker, detections: List, frame: np.ndarray) -> List[Track]
def get_trajectory(track_id: int) -> List[Tuple[int, int]]
def visualize_tracks(frame: np.ndarray, tracks: List[Track]) -> np.ndarray
```

**Data Structure for Tracks:**
```python
class Track:
    track_id: int
    bbox_history: List[Tuple[int, int, int, int]]  # [(x1,y1,x2,y2), ...]
    frame_ids: List[int]
    confidence_scores: List[float]
    last_seen: int
```

---

### Phase 3: Behavior Analysis (Days 3-4)

**Component:** Suspicious Behavior Detection

**Input:** Tracked drone trajectories  
**Output:** Behavior classification (Normal / Suspicious) with specific flags

**Suspicious Behaviors to Detect:**

#### 1. **High-Speed Detection**
- Calculate velocity between frames
- Formula: `speed = distance_pixels / time_between_frames`
- Threshold: Flag if speed > 50 pixels/frame (adjustable)

#### 2. **Hovering Detection**
- Detect minimal movement over extended period
- Check if drone stays within 10-pixel radius for >30 frames
- Calculate using centroid distance variance

#### 3. **Restricted Zone Entry**
- Define restricted zones as polygons
- Check if drone centroid enters restricted area
- Support multiple restricted zones

#### 4. **Erratic Movement**
- Detect sudden direction changes
- Calculate acceleration/deceleration
- Flag if direction change > 45° in <5 frames

**Acceptance Criteria:**
- [ ] Speed calculated in pixels/frame (with option to convert to m/s if metadata available)
- [ ] Hovering detected with <5% false positive rate
- [ ] Support user-defined restricted zones
- [ ] Real-time behavior classification (<100ms per frame)

**Key Functions to Implement:**
```python
def calculate_speed(trajectory: List[Tuple], fps: int) -> float
def detect_hovering(trajectory: List[Tuple], threshold: int) -> bool
def check_restricted_zone(position: Tuple, zones: List[Polygon]) -> bool
def detect_erratic_movement(trajectory: List[Tuple]) -> bool
def classify_behavior(track: Track) -> BehaviorLabel

class BehaviorLabel:
    is_suspicious: bool
    speed_flag: bool
    hover_flag: bool
    restricted_zone_flag: bool
    erratic_movement_flag: bool
    confidence: float
```

---

### Phase 4: Alert System (Day 4)

**Component:** Real-time Alert and Notification

**Requirements:**
1. Generate alerts when suspicious behavior detected
2. Log all detections with timestamp
3. Visual alerts on video output
4. Optional: Audio alerts, email notifications

**Alert Levels:**
- **LOW:** Single suspicious indicator
- **MEDIUM:** 2+ suspicious indicators
- **HIGH:** Drone in restricted zone OR extreme speed

**Acceptance Criteria:**
- [ ] Alerts generated within 200ms of detection
- [ ] Alert log saved to file (CSV/JSON)
- [ ] Visual overlay on video showing alert level
- [ ] Alert history accessible during playback

**Key Functions to Implement:**
```python
def generate_alert(behavior: BehaviorLabel, track: Track) -> Alert
def log_alert(alert: Alert, log_file: str) -> None
def display_alert_overlay(frame: np.ndarray, alerts: List[Alert]) -> np.ndarray
def send_notification(alert: Alert, method: str) -> bool

class Alert:
    timestamp: datetime
    track_id: int
    alert_level: str  # LOW, MEDIUM, HIGH
    behavior_flags: List[str]
    position: Tuple[int, int]
    snapshot: np.ndarray
```

---

### Phase 5: User Interface (Day 5)

**Component:** Web-based Dashboard

**Features Required:**

#### Video Upload & Processing
- Upload video file (MP4, AVI, MOV)
- Select processing options (detection only, detection + tracking, full analysis)
- Progress bar showing processing status

#### Live Display
- Show processed video with:
  - Bounding boxes around drones
  - Track IDs
  - Behavior labels
  - Alert overlays
  
#### Analytics Dashboard
- Total drones detected
- Number of suspicious events
- Timeline of detections
- Heatmap of drone activity
- Behavior breakdown (pie chart)

#### Controls
- Play/pause video
- Adjust detection sensitivity
- Define restricted zones (click to draw polygon)
- Export results (video + JSON report)

**Technology Choice:**
- **Streamlit** (recommended - faster to build)
- OR Flask + JavaScript frontend

**Acceptance Criteria:**
- [ ] Can upload and process video end-to-end
- [ ] Real-time visualization during processing
- [ ] All analytics update dynamically
- [ ] Export feature works (video + report)
- [ ] Runs on localhost without errors

**Key Pages/Components:**
```python
# Streamlit structure
def main_page():
    # File upload
    # Processing options
    # Start processing button

def video_display_page():
    # Video player with overlays
    # Current detection info
    # Alert panel

def analytics_page():
    # Statistics summary
    # Charts and graphs
    # Detection timeline
    # Export button
```

---

## File Structure

```
anti-uav-system/
├── README.md
├── requirements.txt
├── config/
│   ├── model_config.yaml
│   └── detection_config.yaml
├── data/
│   ├── raw/              # Original videos
│   ├── processed/        # Processed frames
│   └── annotations/      # YOLO format labels
├── models/
│   ├── pretrained/       # Downloaded YOLO weights
│   ├── finetuned/        # Your trained model
│   └── tracker/          # Tracking model weights
├── src/
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── yolo_detector.py
│   │   └── data_prep.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── tracker.py
│   │   └── trajectory.py
│   ├── behavior/
│   │   ├── __init__.py
│   │   ├── speed_analyzer.py
│   │   ├── hover_detector.py
│   │   └── zone_checker.py
│   ├── alerts/
│   │   ├── __init__.py
│   │   └── alert_manager.py
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py
│       └── visualization.py
├── ui/
│   ├── streamlit_app.py
│   └── components/
│       ├── video_player.py
│       ├── analytics.py
│       └── controls.py
├── scripts/
│   ├── train_model.py
│   ├── test_model.py
│   └── process_video.py
├── tests/
│   └── test_detection.py
├── outputs/
│   ├── videos/           # Processed videos
│   ├── logs/            # Alert logs
│   └── reports/         # JSON reports
└── notebooks/
    └── exploratory_analysis.ipynb
```

---

## Implementation Timeline

### **Day 1: Setup & Dataset Preparation** (8 hours)

**Morning (4h):**
- [ ] Install all dependencies (Python, PyTorch, CUDA, OpenCV, ultralytics)
- [ ] Download YOLOv8 pre-trained weights
- [ ] Clone Anti-UAV dataset
- [ ] Verify GPU is working with PyTorch

**Afternoon (4h):**
- [ ] Explore dataset structure
- [ ] Convert annotations to YOLO format if needed
- [ ] Create data.yaml file for training
- [ ] Split data (train/val/test: 70/20/10)
- [ ] Test loading a few images/videos

**Deliverable:** Dataset ready for training, environment verified

---

### **Day 2: Model Fine-tuning** (10 hours)

**Morning (3h):**
- [ ] Write training script
- [ ] Configure hyperparameters (epochs=50-100, batch=16, img=640)
- [ ] Start training on RTX 4050
- [ ] Monitor training progress (check mAP, loss curves)

**Afternoon (3h):**
- [ ] Continue training (can run overnight)
- [ ] Write inference script
- [ ] Test model on sample videos
- [ ] Evaluate performance (precision, recall, mAP)

**Evening (4h):**
- [ ] Fine-tune if needed (adjust learning rate, augmentation)
- [ ] Save best model weights
- [ ] Create baseline detection pipeline
- [ ] Test on different drone sizes

**Deliverable:** Trained YOLO model with >80% accuracy

---

### **Day 3: Tracking Implementation** (10 hours)

**Morning (5h):**
- [ ] Implement tracking algorithm (recommend ByteTrack or DeepSORT)
- [ ] Integrate with detection pipeline
- [ ] Test multi-object tracking on sample video
- [ ] Store trajectory data structure

**Afternoon (5h):**
- [ ] Implement trajectory storage and retrieval
- [ ] Add visualization (draw track IDs, paths)
- [ ] Test tracking persistence across occlusions
- [ ] Handle edge cases (drone leaving/entering frame)

**Deliverable:** Working detection + tracking pipeline

---

### **Day 4: Behavior Analysis & Alerts** (10 hours)

**Morning (5h):**
- [ ] Implement speed calculation
- [ ] Implement hovering detection
- [ ] Create restricted zone definition system
- [ ] Test each behavior detector independently

**Afternoon (5h):**
- [ ] Combine all behavior detectors
- [ ] Implement alert generation logic
- [ ] Create alert logging system
- [ ] Add visual overlays for alerts
- [ ] Test full pipeline on multiple videos

**Deliverable:** Complete behavior analysis system

---

### **Day 5: UI & Demo Preparation** (10 hours)

**Morning (5h):**
- [ ] Build Streamlit dashboard
- [ ] Implement video upload
- [ ] Create live processing display
- [ ] Add analytics visualizations

**Afternoon (5h):**
- [ ] Test entire system end-to-end
- [ ] Fix bugs and polish UI
- [ ] Create demo video
- [ ] Write documentation
- [ ] Prepare presentation slides

**Deliverable:** Complete working demo

---

### **Hackathon Day (24 hours)**

**Hours 0-4:** Final testing, minor fixes  
**Hours 4-8:** Presentation preparation, demo rehearsal  
**Hours 8-20:** Available for questions, debugging, polish  
**Hours 20-24:** Final presentation, demo to judges

---

## Key Configuration Files

### `requirements.txt`
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
plotly>=5.14.0
streamlit>=1.25.0
Pillow>=9.5.0
scikit-learn>=1.3.0
scipy>=1.11.0
filterpy>=1.4.5
lapx>=0.5.0
```

### `config/model_config.yaml`
```yaml
model:
  type: "yolov8"
  variant: "yolov8s"  # n, s, m, l, x (s is good balance)
  pretrained: true
  
training:
  epochs: 100
  batch_size: 16
  img_size: 640
  device: "cuda:0"
  workers: 8
  patience: 20
  
  optimizer:
    type: "AdamW"
    lr: 0.001
    weight_decay: 0.0005
  
  augmentation:
    hsv_h: 0.015
    hsv_s: 0.7
    hsv_v: 0.4
    degrees: 10.0
    translate: 0.1
    scale: 0.5
    flipud: 0.5
    fliplr: 0.5
```

### `config/detection_config.yaml`
```yaml
detection:
  confidence_threshold: 0.5
  iou_threshold: 0.45
  max_detections: 50
  
tracking:
  tracker_type: "bytetrack"  # or "deepsort"
  max_age: 30
  min_hits: 3
  iou_threshold: 0.3
  
behavior:
  speed:
    threshold_pixels_per_frame: 50
    fps: 30
  
  hovering:
    radius_pixels: 10
    min_frames: 30
  
  restricted_zones:
    - name: "Building A"
      polygon: [[100, 100], [200, 100], [200, 200], [100, 200]]
    - name: "Airspace Red Zone"
      polygon: [[300, 300], [500, 300], [500, 500], [300, 500]]
  
  alert_levels:
    low: ["speed_flag"]
    medium: ["speed_flag", "hover_flag"]
    high: ["restricted_zone_flag"]
```

---

## Testing Checklist

### Unit Tests
- [ ] Detection on single image
- [ ] Detection on video
- [ ] Tracking with single object
- [ ] Tracking with multiple objects
- [ ] Speed calculation accuracy
- [ ] Hovering detection accuracy
- [ ] Zone intersection logic

### Integration Tests
- [ ] Full pipeline on short video (30 sec)
- [ ] Full pipeline on long video (5 min)
- [ ] Multiple drone scenarios
- [ ] Edge cases (drone entering/leaving, occlusion)

### Performance Tests
- [ ] FPS on RTX 4050
- [ ] Memory usage
- [ ] Processing time for 1-minute video
- [ ] UI responsiveness

### Demo Tests
- [ ] Upload video → process → view results
- [ ] All analytics display correctly
- [ ] Export functionality works
- [ ] No crashes during demo

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size, use smaller YOLO variant (yolov8n), process fewer frames

### Issue 2: Poor Detection Accuracy
**Solution:** More training epochs, adjust confidence threshold, verify data quality

### Issue 3: Lost Tracks
**Solution:** Tune tracker parameters (max_age, iou_threshold), improve detection quality

### Issue 4: Slow Processing
**Solution:** Use GPU, reduce input resolution, skip frames (process every 2nd frame)

### Issue 5: False Hovering Detections
**Solution:** Adjust radius threshold, require more consecutive frames

---

## Deliverables for Hackathon

### Code
- [ ] Complete GitHub repository
- [ ] README with setup instructions
- [ ] All code documented
- [ ] Requirements.txt

### Demo
- [ ] Working web interface
- [ ] Pre-processed demo videos (3-5 examples)
- [ ] Backup demo video (in case live demo fails)

### Presentation
- [ ] Problem statement slide
- [ ] Technical approach slide
- [ ] Demo walkthrough
- [ ] Results/metrics slide
- [ ] Future improvements slide

### Documentation
- [ ] System architecture diagram
- [ ] How it works (flow chart)
- [ ] Results summary (accuracy, speed, etc.)

---

## Success Metrics

### Technical Metrics
- **Detection Accuracy:** >80% mAP on test set
- **Processing Speed:** >15 FPS on RTX 4050
- **Tracking Accuracy:** >70% MOTA (Multiple Object Tracking Accuracy)
- **False Positive Rate:** <10% for behavior detection

### Demo Metrics
- **System Uptime:** Demo runs without crashes
- **Response Time:** <2 seconds from upload to start processing
- **User Experience:** Intuitive UI, clear visualizations

---

## Future Enhancements (Mention in Presentation)

1. **Real-time streaming:** Live camera feed processing
2. **Drone classification:** Identify drone type/model
3. **3D trajectory:** Estimate altitude using multiple cameras
4. **Counter-measures:** Integration with jamming/net systems
5. **Edge deployment:** Run on Raspberry Pi / Jetson Nano
6. **Multi-camera fusion:** Combine multiple viewpoints
7. **Behavioral learning:** Train ML model on labeled suspicious patterns

---

## References & Resources

### Documentation
- YOLOv8: https://docs.ultralytics.com/
- OpenCV: https://docs.opencv.org/
- ByteTrack: https://github.com/ifzhang/ByteTrack
- Anti-UAV Dataset: https://github.com/ZhaoJ9014/Anti-UAV

### Tutorials
- Fine-tuning YOLO: https://docs.ultralytics.com/modes/train/
- Object Tracking: https://learnopencv.com/object-tracking-using-opencv-cpp-python/
- Streamlit: https://docs.streamlit.io/

### Papers
- YOLO: "You Only Look Once: Unified, Real-Time Object Detection"
- ByteTrack: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"

---

## Notes for AI Coding Assistant

When implementing this project, follow these guidelines:

1. **Use type hints** in all Python functions
2. **Add docstrings** to all functions and classes
3. **Error handling:** Wrap video processing in try-except blocks
4. **Logging:** Use Python logging module for debugging
5. **Configuration:** Load parameters from YAML files, not hardcoded
6. **Modular code:** Each component should be independently testable
7. **Comments:** Explain complex logic, especially in behavior analysis
8. **Optimize:** Use NumPy vectorization where possible
9. **GPU usage:** Always check if CUDA is available before using GPU

### Code Style
- Follow PEP 8
- Use meaningful variable names
- Keep functions under 50 lines when possible
- Separate concerns (detection, tracking, analysis)

### Performance Tips
- Process frames in batches when possible
- Use OpenCV's VideoWriter for efficient output
- Cache model inference results if reprocessing same video
- Use multiprocessing for non-GPU tasks

---

## Emergency Fallback Plan

If advanced features don't work in time:

**Minimum Viable Product:**
1. Detection working (even if accuracy is 60%)
2. Basic tracking (even if some tracks are lost)
3. Simple speed calculation (even if not perfectly calibrated)
4. Basic UI (even if just file upload + processed video output)

**Presentation Strategy:**
- Lead with what works
- Show demo video of working features
- Explain what you learned and what you'd improve
- Focus on the problem-solving approach, not perfection

---

**Good luck! You got this! 🚀**