from ultralytics import YOLO
import cv2
from pathlib import Path
import sys
import os
import glob
import numpy as np

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector_with_tracking import process_video_with_tracking

class FrameStreamer:
    """Helper to stream frames from video file or image directory"""
    def __init__(self, source_path):
        self.source_path = Path(source_path)
        self.is_video = self.source_path.is_file()
        self.cap = None
        self.image_files = []
        self.current_idx = 0
        
        if self.is_video:
            self.cap = cv2.VideoCapture(str(self.source_path))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            # Assume directory of images
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                self.image_files.extend(list(self.source_path.glob(ext)))
            
            # Sort to ensure correct order
            self.image_files.sort(key=lambda x: str(x))
            
            if not self.image_files:
                raise ValueError(f"No images found in {self.source_path}")
            
            # Read first image to get dims
            img = cv2.imread(str(self.image_files[0]))
            if img is None:
                raise ValueError(f"Could not read first image {self.image_files[0]}")
                
            self.height, self.width = img.shape[:2]
            self.fps = 30  # Default for images
            self.total_frames = len(self.image_files)
            
    def read(self):
        if self.is_video:
            return self.cap.read()
        else:
            if self.current_idx >= len(self.image_files):
                return False, None
            
            img_path = self.image_files[self.current_idx]
            frame = cv2.imread(str(img_path))
            self.current_idx += 1
            
            if frame is None:
                return False, None
            return True, frame
            
    def release(self):
        if self.is_video and self.cap:
            self.cap.release()

def test_model_simple(model_path, test_input_path, output_path='outputs/videos/test_detection.mp4'):
    """
    Simple test of trained model on video or image directory (detection only)
    
    Args:
        model_path: Path to trained weights (best.pt)
        test_input_path: Path to test video OR directory of images
        output_path: Where to save output video
    """
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return
    
    if not Path(test_input_path).exists():
        print(f"❌ Input not found: {test_input_path}")
        return
    
    # Load model
    model = YOLO(model_path)
    print(f"✓ Loaded model from {model_path}")
    
    # Setup input stream
    try:
        streamer = FrameStreamer(test_input_path)
    except Exception as e:
        print(f"❌ Error setting up input: {e}")
        return

    print(f"Input: {streamer.width}x{streamer.height} @ {streamer.fps}fps, {streamer.total_frames} frames")
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, streamer.fps, (streamer.width, streamer.height))
    
    frame_idx = 0
    detections_count = 0
    
    print("Processing...")
    
    while True:
        ret, frame = streamer.read()
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
            print(f"Processed {frame_idx}/{streamer.total_frames} frames ({detections_count} with detections)")
            
    streamer.release()
    out.release()
    
    print(f"\n✓ Detection complete!")
    print(f"  Output saved to: {output_path}")
    if streamer.total_frames > 0:
        print(f"  Frames with detections: {detections_count}/{streamer.total_frames} ({100*detections_count/streamer.total_frames:.1f}%)")

def test_full_pipeline(model_path, test_input_path, output_path='outputs/videos/full_pipeline.mp4'):
    """
    Test full pipeline with tracking and behavior analysis
    """
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return
    
    if not Path(test_input_path).exists():
        print(f"❌ Input not found: {test_input_path}")
        return
        
    print("Testing full pipeline (detection + tracking + behavior analysis)...")
    
    # Define some example restricted zones
    restricted_zones = [
        [(100, 100), (300, 100), (300, 300), (100, 300)],  # Top-left zone
        [(500, 400), (700, 400), (700, 600), (500, 600)]   # Bottom-right zone
    ]
    
    # Note: process_video_with_tracking might expect a video file path string.
    # If the src module specifically requires a file path and uses cv2.VideoCapture internally,
    # we might need to modify THAT function too. 
    # For now, let's assume we can only run simple detection on directory of images
    # unless we modify the core detector.
    
    # Check if input is a file (video)
    if Path(test_input_path).is_file():
        process_video_with_tracking(
            model_path=model_path,
            video_path=test_input_path,
            output_path=output_path,
            restricted_zones=restricted_zones
        )
    else:
        print("⚠️  Full pipeline currently only supports video files as input.")
        print("   Running simple detection test instead...")
        test_model_simple(model_path, test_input_path, output_path)

if __name__ == "__main__":
    # Test paths
    model_path = 'runs/detect/models/finetuned/drone_detector/weights/best.pt'
    
    # Default test video
    test_video = 'data/raw/test_video.mp4'
    
    # Alternative: Processed validation data (as a proxy for test data)
    val_images_dir = 'data/processed/yolo_format/val/images'
    
    target_input = None
    
    print("Anti-UAV Model Testing")
    print("=" * 50)
    
    # Determine input source
    if Path(test_video).exists():
        target_input = test_video
        print(f"✓ Found test video: {test_video}")
    elif Path(val_images_dir).exists() and any(Path(val_images_dir).iterdir()):
        target_input = val_images_dir
        print(f"✓ Found processed validation images: {val_images_dir}")
        print("  Using validation images for testing since test video is missing.")
    else:
        print(f"❌ No test video found at {test_video}")
        print(f"❌ No validation images found at {val_images_dir}")
        print("Please add a test video file to data/raw/test_video.mp4 or prepare the dataset.")
        sys.exit(1)

    # Check if we have a trained model
    if Path(model_path).exists():
        print("✓ Found trained model")
        
        # Run simple detection test
        print("\n1. Testing simple detection...")
        test_model_simple(model_path, target_input)
        
        # Run full pipeline test
        if Path(target_input).is_file():
            print("\n2. Testing full pipeline...")
            test_full_pipeline(model_path, target_input)
        else:
            print("\n2. Skipping full pipeline (requires video file input)")
            
    else:
        print(f"❌ No trained model found at {model_path}")
        print("Please train the model first:")
        print("  python scripts/train_model.py")
        
        # Test with pre-trained YOLO as fallback
        print("\nTesting with pre-trained YOLOv8s (will detect general objects, not specifically drones)...")
        # Start download if not exists
        if not Path("yolov8s.pt").exists():
            YOLO("yolov8s.pt") # triggers download
            
        test_model_simple('yolov8s.pt', target_input, 'outputs/videos/pretrained_test.mp4')