# Anti-UAV System - Complete Tech Stack Workflow & Implementation Guide

**For:** AI-Assisted Coding (Cursor, GitHub Copilot, ChatGPT)  
**Timeline:** 5 Days Prep + 24 Hour Hackathon  
**Hardware:** RTX 4050, 16GB RAM, 100GB Storage

---

## Table of Contents

1. [Environment Setup (Day 1 Morning)](#day-1-morning-environment-setup)
2. [Dataset Preparation (Day 1 Afternoon)](#day-1-afternoon-dataset-preparation)
3. [YOLO Training Pipeline (Day 2)](#day-2-yolo-training-pipeline)
4. [Object Tracking Implementation (Day 3)](#day-3-object-tracking)
5. [Behavior Analysis System (Day 4)](#day-4-behavior-analysis)
6. [UI Development (Day 5)](#day-5-ui-development)
7. [Integration & Testing](#integration-testing)
8. [Troubleshooting Guide](#troubleshooting)

---

## Tech Stack Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT VIDEO STREAM                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VIDEO PREPROCESSING                           │
│  Tools: OpenCV, FFmpeg                                          │
│  - Frame extraction                                             │
│  - Resize/normalize                                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DRONE DETECTION (YOLOv8)                     │
│  Framework: PyTorch + Ultralytics                               │
│  Input: Individual frames                                       │
│  Output: [x1, y1, x2, y2, confidence, class]                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OBJECT TRACKING (ByteTrack)                  │
│  Library: ByteTrack / DeepSORT                                  │
│  Input: Per-frame detections                                    │
│  Output: Tracked objects with IDs + trajectories                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BEHAVIOR ANALYSIS                            │
│  Tools: NumPy, SciPy                                            │
│  - Speed calculation                                            │
│  - Hovering detection                                           │
│  - Zone checking                                                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ALERT GENERATION                             │
│  - Classification (Normal/Suspicious)                           │
│  - Alert logging                                                │
│  - Notification triggers                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    VISUALIZATION & UI                           │
│  Framework: Streamlit / Flask                                   │
│  - Video player with overlays                                   │
│  - Analytics dashboard                                          │
│  - Export functionality                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Day 1 Morning: Environment Setup

### Step 1: Install Python & Core Tools

**Windows:**
```bash
# Download Python 3.10 from python.org
# During installation: Check "Add Python to PATH"

# Verify installation
python --version  # Should show Python 3.10.x
pip --version

# Install Git
# Download from git-scm.com
git --version
```

**Linux/Mac:**
```bash
# Install Python 3.10
sudo apt update
sudo apt install python3.10 python3.10-pip python3.10-venv

# Verify
python3.10 --version
pip3 --version
```

### Step 2: Create Project Structure

```bash
# Create project directory
mkdir anti-uav-system
cd anti-uav-system

# Initialize Git
git init
git branch -M main

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Your terminal should now show (venv) prefix
```

### Step 3: Install CUDA & PyTorch (Critical for GPU)

**Check CUDA Compatibility:**
```bash
# Windows - Open Command Prompt
nvidia-smi

# Look for CUDA Version (e.g., CUDA Version: 12.1)
# This tells you what CUDA your RTX 4050 supports
```

**Install PyTorch with CUDA:**
```bash
# For CUDA 11.8 (most common for RTX 4050)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify GPU is accessible
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Expected output:
# True
# NVIDIA GeForce RTX 4050
```

**If the above shows `False` - STOP and troubleshoot before continuing!**

### Step 4: Install Core Dependencies

Create `requirements.txt`:
```txt
# Deep Learning
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.100

# Computer Vision
opencv-python>=4.8.0
opencv-contrib-python>=4.8.0

# Data Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.11.0

# Tracking
filterpy>=1.4.5
lap>=0.4.0

# Visualization
matplotlib>=3.7.0
plotly>=5.14.0
seaborn>=0.12.0

# UI
streamlit>=1.25.0

# Utils
pillow>=9.5.0
tqdm>=4.65.0
pyyaml>=6.0
```

Install everything:
```bash
pip install -r requirements.txt

# Verify key packages
python -c "import ultralytics; print(ultralytics.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"
```

### Step 5: Download Pre-trained YOLO Weights

```bash
# Create models directory
mkdir -p models/pretrained

# Download YOLOv8 weights (this happens automatically on first use, but you can do it manually)
python -c "from ultralytics import YOLO; model = YOLO('yolov8s.pt')"

# This downloads yolov8s.pt to your local cache
# Available variants: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium)
# For RTX 4050: yolov8s.pt is recommended (good balance)
```

### Step 6: Create Initial Project Structure

```bash
# Create all directories
mkdir -p data/{raw,processed,annotations}
mkdir -p models/{pretrained,finetuned}
mkdir -p src/{detection,tracking,behavior,alerts,utils}
mkdir -p ui/components
mkdir -p scripts
mkdir -p outputs/{videos,logs,reports}
mkdir -p config
mkdir -p tests

# Create __init__.py files for Python modules
touch src/__init__.py
touch src/detection/__init__.py
touch src/tracking/__init__.py
touch src/behavior/__init__.py
touch src/alerts/__init__.py
touch src/utils/__init__.py
```

**Verification Checkpoint:**
```bash
# Run this to verify everything is set up
python << EOF
import torch
import cv2
from ultralytics import YOLO
import numpy as np
import streamlit

print("✓ PyTorch:", torch.__version__)
print("✓ CUDA Available:", torch.cuda.is_available())
print("✓ OpenCV:", cv2.__version__)
print("✓ NumPy:", np.__version__)
print("✓ Ultralytics installed")
print("✓ Streamlit installed")
print("\n🎉 Environment setup complete!")
EOF
```

---

## Day 1 Afternoon: Dataset Preparation

### Understanding the Anti-UAV Dataset

The dataset structure looks like this:
```
Anti-UAV/
├── train/
│   ├── video1/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── IR_label.json
│   ├── video2/
│   └── ...
├── test/
└── validation/
```

### Step 1: Download and Organize Dataset

```bash
# Navigate to data directory
cd data/raw

# Clone the Anti-UAV dataset (or download from source)
git clone https://github.com/ZhaoJ9014/Anti-UAV.git

# Or if you already have the 50GB dataset, copy it here
# cp -r /path/to/your/dataset/* .
```

### Step 2: Explore Dataset Structure

Create `scripts/explore_dataset.py`:
```python
import json
import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

def explore_dataset(dataset_path):
    """Explore Anti-UAV dataset structure"""
    
    dataset_path = Path(dataset_path)
    
    # Count videos and frames
    video_folders = list(dataset_path.glob("*/"))
    total_frames = 0
    labeled_frames = 0
    
    print(f"Found {len(video_folders)} video sequences\n")
    
    for video_folder in video_folders[:5]:  # Check first 5
        frames = list(video_folder.glob("*.jpg"))
        label_file = video_folder / "IR_label.json"
        
        print(f"Video: {video_folder.name}")
        print(f"  Frames: {len(frames)}")
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            print(f"  Labeled frames: {len(labels)}")
            
            # Show sample annotation
            if labels:
                sample_key = list(labels.keys())[0]
                print(f"  Sample annotation: {labels[sample_key]}")
        
        print()
        total_frames += len(frames)
    
    print(f"Total frames (first 5 videos): {total_frames}")
    
    # Visualize a sample
    visualize_sample(video_folders[0])

def visualize_sample(video_folder):
    """Visualize a sample frame with annotation"""
    
    # Load first frame
    frames = sorted(video_folder.glob("*.jpg"))
    if not frames:
        return
    
    img = cv2.imread(str(frames[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load annotation
    label_file = video_folder / "IR_label.json"
    if label_file.exists():
        with open(label_file, 'r') as f:
            labels = json.load(f)
        
        frame_name = frames[0].stem
        if frame_name in labels:
            bbox = labels[frame_name]['bbox']  # Format may vary
            # Draw bounding box
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Sample from {video_folder.name}")
    plt.axis('off')
    plt.savefig('dataset_sample.png')
    print("Saved visualization to dataset_sample.png")

if __name__ == "__main__":
    explore_dataset("data/raw/Anti-UAV/train")
```

Run it:
```bash
python scripts/explore_dataset.py
```

### Step 3: Convert to YOLO Format

YOLO expects annotations in this format:
```
class_id center_x center_y width height
```

All coordinates normalized to [0, 1].

Create `scripts/convert_to_yolo.py`:
```python
import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil

def convert_antiuav_to_yolo(dataset_path, output_path, split='train'):
    """
    Convert Anti-UAV dataset to YOLO format
    
    Args:
        dataset_path: Path to Anti-UAV dataset
        output_path: Where to save YOLO format data
        split: 'train', 'val', or 'test'
    """
    
    dataset_path = Path(dataset_path) / split
    output_path = Path(output_path) / split
    
    # Create YOLO structure
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    video_folders = list(dataset_path.glob("*/"))
    
    for video_folder in tqdm(video_folders, desc=f"Converting {split}"):
        label_file = video_folder / "IR_label.json"
        
        if not label_file.exists():
            continue
        
        with open(label_file, 'r') as f:
            labels = json.load(f)
        
        frames = sorted(video_folder.glob("*.jpg"))
        
        for frame_path in frames:
            frame_name = frame_path.stem
            
            if frame_name not in labels:
                continue
            
            # Copy image
            img_dest = output_path / 'images' / f"{video_folder.name}_{frame_name}.jpg"
            shutil.copy(frame_path, img_dest)
            
            # Get image dimensions
            import cv2
            img = cv2.imread(str(frame_path))
            img_height, img_width = img.shape[:2]
            
            # Convert annotation
            annotation = labels[frame_name]
            
            # Parse bbox (format depends on dataset)
            # Common format: {'bbox': [x, y, w, h], 'exist': 1}
            if 'exist' in annotation and annotation['exist'] == 0:
                # No drone in frame - create empty label file
                label_dest = output_path / 'labels' / f"{video_folder.name}_{frame_name}.txt"
                label_dest.touch()
                continue
            
            bbox = annotation.get('bbox', annotation.get('gt_bbox', []))
            
            if not bbox or len(bbox) != 4:
                continue
            
            x, y, w, h = bbox
            
            # Convert to YOLO format (normalized center coords + dimensions)
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            norm_w = w / img_width
            norm_h = h / img_height
            
            # Class 0 = drone
            yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            
            # Save label
            label_dest = output_path / 'labels' / f"{video_folder.name}_{frame_name}.txt"
            with open(label_dest, 'w') as f:
                f.write(yolo_line)
    
    print(f"✓ Converted {split} set")
    print(f"  Images: {len(list((output_path / 'images').glob('*.jpg')))}")
    print(f"  Labels: {len(list((output_path / 'labels').glob('*.txt')))}")

if __name__ == "__main__":
    # Convert all splits
    for split in ['train', 'val', 'test']:
        if Path(f"data/raw/Anti-UAV/{split}").exists():
            convert_antiuav_to_yolo(
                "data/raw/Anti-UAV",
                "data/processed/yolo_format",
                split=split
            )
```

Run conversion:
```bash
python scripts/convert_to_yolo.py
```

### Step 4: Create YOLO Configuration File

Create `config/data.yaml`:
```yaml
# Dataset paths
path: /absolute/path/to/anti-uav-system/data/processed/yolo_format
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: drone

# Number of classes
nc: 1
```

**Important:** Replace `/absolute/path/to/` with your actual path. Use:
```bash
pwd  # On Linux/Mac
cd  # On Windows
```

### Step 5: Verify Data Pipeline

Create `scripts/verify_data.py`:
```python
from ultralytics import YOLO
import yaml
import cv2
import matplotlib.pyplot as plt

def verify_data():
    """Verify YOLO data is correctly formatted"""
    
    # Load config
    with open('config/data.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("Data config loaded:")
    print(f"  Train: {config['train']}")
    print(f"  Val: {config['val']}")
    print(f"  Classes: {config['names']}")
    
    # Try to load a sample
    from pathlib import Path
    train_images = Path(config['path']) / config['train']
    sample_images = list(train_images.glob('*.jpg'))[:5]
    
    if not sample_images:
        print("❌ No training images found!")
        return
    
    print(f"\n✓ Found {len(sample_images)} sample images")
    
    # Visualize samples with labels
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for idx, img_path in enumerate(sample_images):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load corresponding label
        label_path = img_path.parent.parent / 'labels' / (img_path.stem + '.txt')
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, cx, cy, w, h = map(float, line.strip().split())
                    
                    # Convert back to pixel coords
                    img_h, img_w = img.shape[:2]
                    x1 = int((cx - w/2) * img_w)
                    y1 = int((cy - h/2) * img_h)
                    x2 = int((cx + w/2) * img_w)
                    y2 = int((cy + h/2) * img_h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[idx].imshow(img)
        axes[idx].axis('off')
        axes[idx].set_title(img_path.stem)
    
    plt.tight_layout()
    plt.savefig('data_verification.png')
    print("✓ Saved visualization to data_verification.png")

if __name__ == "__main__":
    verify_data()
```

Run verification:
```bash
python scripts/verify_data.py
```

**Checkpoint:** You should see 5 images with green bounding boxes around drones.

---

## Day 2: YOLO Training Pipeline

### Understanding Training Process

```
Input: Images + Labels → YOLO Model → Training Loop → Trained Weights
                            ↓
                    Validates on Val Set
                            ↓
                    Saves Best Weights
```

### Step 1: Create Training Script

Create `scripts/train_model.py`:
```python
from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

def train_drone_detector(
    model_size='yolov8s',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='cuda:0'
):
    """
    Train YOLOv8 on drone dataset
    
    Args:
        model_size: 'yolov8n', 'yolov8s', 'yolov8m' (s recommended for RTX 4050)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        img_size: Input image size
        device: 'cuda:0' or 'cpu'
    """
    
    # Verify GPU
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("⚠️  CUDA not available! Falling back to CPU (will be SLOW)")
            device = 'cpu'
        else:
            print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load pre-trained model
    model = YOLO(f'{model_size}.pt')
    print(f"✓ Loaded pre-trained {model_size} model")
    
    # Training configuration
    results = model.train(
        data='config/data.yaml',
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,  # Initial learning rate
        lrf=0.01,   # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        
        # Augmentation
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,    # Saturation
        hsv_v=0.4,    # Value
        degrees=10.0,  # Rotation
        translate=0.1, # Translation
        scale=0.5,     # Scaling
        flipud=0.5,    # Vertical flip
        fliplr=0.5,    # Horizontal flip
        mosaic=1.0,    # Mosaic augmentation
        
        # Validation
        val=True,
        patience=20,  # Early stopping patience
        
        # Saving
        save=True,
        save_period=10,  # Save every N epochs
        project='models/finetuned',
        name='drone_detector',
        exist_ok=True,
        
        # Visualization
        plots=True,
        
        # Performance
        workers=8,  # Data loading workers
        cache=False  # Don't cache images (too much RAM for 50GB)
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: {results.save_dir}/weights/best.pt")
    print(f"Last model saved to: {results.save_dir}/weights/last.pt")
    print("\nKey Metrics:")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return results

if __name__ == "__main__":
    # Start training
    results = train_drone_detector(
        model_size='yolov8s',  # Change to 'yolov8n' if running out of memory
        epochs=100,
        batch_size=16,  # Reduce to 8 if OOM
        img_size=640,
        device='cuda:0'
    )
```

### Step 2: Start Training

```bash
# Activate your environment first
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Start training (this will run for several hours)
python scripts/train_model.py

# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi  # Linux
# or check Task Manager > Performance > GPU on Windows
```

**What to expect:**
```
Epoch 1/100: 100%|████████| 500/500 [02:15<00:00,  3.70it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all        500        732      0.654      0.512      0.598     0.341

Epoch 2/100: 100%|████████| 500/500 [02:12<00:00,  3.78it/s]
      Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
        all        500        732      0.701      0.598      0.663     0.402
...
```

### Step 3: Monitor Training Progress

While training runs, YOLO saves plots in `models/finetuned/drone_detector/`:
- `results.png` - Loss and metric curves
- `confusion_matrix.png` - Confusion matrix
- `PR_curve.png` - Precision-Recall curve
- `F1_curve.png` - F1 score curve

**Check these periodically to ensure training is progressing!**

### Step 4: Evaluate Trained Model

Create `scripts/test_model.py`:
```python
from ultralytics import YOLO
import cv2
from pathlib import Path

def test_model(model_path, test_video_path, output_path='outputs/test_detection.mp4'):
    """
    Test trained model on video
    
    Args:
        model_path: Path to trained weights (best.pt)
        test_video_path: Path to test video
        output_path: Where to save output video
    """
    
    # Load model
    model = YOLO(model_path)
    print(f"✓ Loaded model from {model_path}")
    
    # Open video
    cap = cv2.VideoCapture(test_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    detections_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=0.5, verbose=False)
        
        # Draw results
        annotated_frame = results[0].plot()
        
        # Count detections
        if len(results[0].boxes) > 0:
            detections_count += 1
        
        # Write frame
        out.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames ({detections_count} with detections)")
    
    cap.release()
    out.release()
    
    print(f"\n✓ Detection complete!")
    print(f"  Output saved to: {output_path}")
    print(f"  Frames with detections: {detections_count}/{total_frames} ({100*detections_count/total_frames:.1f}%)")

if __name__ == "__main__":
    test_model(
        model_path='models/finetuned/drone_detector/weights/best.pt',
        test_video_path='data/raw/test_video.mp4'
    )
```

Run test:
```bash
python scripts/test_model.py
```

**Success Criteria for Day 2:**
- [ ] Model trained without errors
- [ ] mAP50 > 0.7 (70% accuracy)
- [ ] Test video shows detections with bounding boxes
- [ ] Processing speed >15 FPS on GPU

---

## Day 3: Object Tracking

### Why Tracking Matters

Detection gives you: "There's a drone at position X"  
Tracking gives you: "THIS SPECIFIC drone (ID=5) has moved from A→B→C"

You need tracking to:
1. Calculate speed (requires trajectory)
2. Detect hovering (requires position history)
3. Maintain identity across frames

### Step 1: Install Tracking Dependencies

```bash
pip install filterpy lap

# ByteTrack (best for your use case)
pip install git+https://github.com/ifzhang/ByteTrack.git
```

### Step 2: Implement Tracking Module

Create `src/tracking/tracker.py`:
```python
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
            List of active tracks: [(track_id, x1, y1, x2, y2), ...]
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
            return self.tracks[track_id].history
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
```

### Step 3: Integrate Tracking with Detection

Create `src/detection/detector_with_tracking.py`:
```python
from ultralytics import YOLO
import cv2
import numpy as np
from src.tracking.tracker import SimpleTracker

class DroneDetectorTracker:
    """Combined detection + tracking pipeline"""
    
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.tracker = SimpleTracker()
        self.conf_threshold = conf_threshold
    
    def process_frame(self, frame):
        """
        Process single frame
        
        Returns:
            tracks: List of (track_id, x1, y1, x2, y2, conf)
            annotated_frame: Frame with visualizations
        """
        # Run detection
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]
        
        # Extract detections
        detections = []
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                detections.append((*box, conf))
        
        # Update tracker
        tracks = self.tracker.update(detections)
        
        # Annotate frame
        annotated_frame = frame.copy()
        for track_id, x1, y1, x2, y2, conf in tracks:
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (int(x1), int(y1)), (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Draw track ID
            label = f"ID:{track_id} ({conf:.2f})"
            cv2.putText(annotated_frame, label,
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw trajectory
            history = self.tracker.get_track_history(track_id)
            if len(history) > 1:
                points = [(int(cx), int(cy)) for _, cx, cy in history]
                for i in range(len(points) - 1):
                    cv2.line(annotated_frame, points[i], points[i+1], 
                            (255, 0, 0), 2)
        
        return tracks, annotated_frame

def process_video_with_tracking(model_path, video_path, output_path):
    """Process entire video with tracking"""
    
    detector = DroneDetectorTracker(model_path)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracks, annotated_frame = detector.process_frame(frame)
        out.write(annotated_frame)
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx} frames, {len(tracks)} active tracks")
    
    cap.release()
    out.release()
    print(f"✓ Tracking complete: {output_path}")

if __name__ == "__main__":
    process_video_with_tracking(
        model_path='models/finetuned/drone_detector/weights/best.pt',
        video_path='data/raw/test_video.mp4',
        output_path='outputs/tracked_video.mp4'
    )
```

Run tracking test:
```bash
python src/detection/detector_with_tracking.py
```

**Success Criteria:**
- [ ] Each drone gets unique ID
- [ ] IDs persist across frames
- [ ] Trajectory lines drawn
- [ ] No flickering IDs (track switches)

---

## Day 4: Behavior Analysis

### Understanding Behavior Detection

```
Trajectory Data → Calculate Metrics → Apply Thresholds → Generate Alerts

Metrics:
- Speed: Distance / Time
- Hovering: Low variance in position
- Zone: Point-in-polygon test
```

### Step 1: Speed Analyzer

Create `src/behavior/speed_analyzer.py`:
```python
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
        
        return np.mean(speeds) if speeds else 0.0
    
    def is_high_speed(self, trajectory: List[Tuple]) -> bool:
        """Check if drone is moving at suspicious speed"""
        speed = self.calculate_speed(trajectory)
        return speed > self.speed_threshold
    
    def get_instant_speed(self, trajectory: List[Tuple], window=5) -> float:
        """Get instantaneous speed over last N frames"""
        if len(trajectory) < 2:
            return 0.0
        
        recent = trajectory[-min(window, len(trajectory)):]
        return self.calculate_speed(recent)
```

### Step 2: Hovering Detector

Create `src/behavior/hover_detector.py`:
```python
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
        
        return max_distance < self.radius_threshold
    
    def calculate_movement_variance(self, trajectory: List[Tuple]) -> float:
        """Calculate variance in position (low = hovering)"""
        if len(trajectory) < 2:
            return 0.0
        
        positions = np.array([(x, y) for _, x, y in trajectory])
        return np.var(positions)
```

### Step 3: Zone Checker

Create `src/behavior/zone_checker.py`:
```python
import numpy as np
from typing import List, Tuple
from shapely.geometry import Point, Polygon

class ZoneChecker:
    """Check if drone enters restricted zones"""
    
    def __init__(self, restricted_zones: List[List[Tuple[int, int]]]):
        """
        Args:
            restricted_zones: List of polygons, each polygon is list of (x, y) points
                             Example: [[(100, 100), (200, 100), (200, 200), (100, 200)]]
        """
        self.zones = [Polygon(zone) for zone in restricted_zones]
        self.zone_names = [f"Zone_{i}" for i in range(len(restricted_zones))]
    
    def check_position(self, x: float, y: float) -> Tuple[bool, str]:
        """
        Check if position is in any restricted zone
        
        Returns:
            (is_restricted, zone_name)
        """
        point = Point(x, y)
        
        for i, zone in enumerate(self.zones):
            if zone.contains(point):
                return True, self.zone_names[i]
        
        return False, ""
    
    def check_trajectory(self, trajectory: List[Tuple[int, float, float]]) -> bool:
        """Check if any point in trajectory enters restricted zone"""
        for _, x, y in trajectory:
            is_restricted, _ = self.check_position(x, y)
            if is_restricted:
                return True
        return False
```

### Step 4: Integrated Behavior Classifier

Create `src/behavior/behavior_classifier.py`:
```python
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
        flags_count = sum([speed_flag, hover_flag, zone_flag])
        
        if zone_flag:
            alert_level = 'HIGH'
        elif flags_count >= 2:
            alert_level = 'MEDIUM'
        elif flags_count == 1:
            alert_level = 'LOW'
        else:
            alert_level = 'NORMAL'
        
        is_suspicious = alert_level != 'NORMAL'
        
        return BehaviorAnalysis(
            track_id=track_id,
            is_suspicious=is_suspicious,
            speed_flag=speed_flag,
            hover_flag=hover_flag,
            zone_flag=zone_flag,
            speed_value=speed,
            alert_level=alert_level,
            zone_name=zone_name
        )
```

### Step 5: Alert System

Create `src/alerts/alert_manager.py`:
```python
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
```

**Success Criteria for Day 4:**
- [ ] Speed calculation works
- [ ] Hovering detection works
- [ ] Zone checking works
- [ ] Alerts generated correctly
- [ ] All components tested individually

---

## Day 5: UI Development

*(Continuing in next message due to length...)*

Would you like me to continue with Day 5 (UI), integration, and the troubleshooting guide? Or should I focus on a specific section you need more detail on?

Also - quick check: Are you planning to use **Streamlit** or **Flask** for your UI? Streamlit is WAY faster to build (I recommend it for hackathons).
