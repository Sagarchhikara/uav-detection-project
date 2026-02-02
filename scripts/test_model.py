from ultralytics import YOLO
import cv2
from pathlib import Path
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.detection.detector_with_tracking import process_video_with_tracking

def test_model_simple(model_path, test_video_path, output_path='outputs/videos/test_detection.mp4'):
    """
    Simple test of trained model on video (detection only)
    
    Args:
        model_path: Path to trained weights (best.pt)
        test_video_path: Path to test video
        output_path: Where to save output video
    """
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return
    
    if not Path(test_video_path).exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("Please add a test video to data/raw/")
        return
    
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
    
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    detections_count = 0
    
    print("Processing video...")
    
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

def test_full_pipeline(model_path, test_video_path, output_path='outputs/videos/full_pipeline.mp4'):
    """
    Test full pipeline with tracking and behavior analysis
    """
    
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using scripts/train_model.py")
        return
    
    if not Path(test_video_path).exists():
        print(f"❌ Test video not found: {test_video_path}")
        print("Please add a test video to data/raw/")
        return
    
    print("Testing full pipeline (detection + tracking + behavior analysis)...")
    
    # Define some example restricted zones
    restricted_zones = [
        [(100, 100), (300, 100), (300, 300), (100, 300)],  # Top-left zone
        [(500, 400), (700, 400), (700, 600), (500, 600)]   # Bottom-right zone
    ]
    
    process_video_with_tracking(
        model_path=model_path,
        video_path=test_video_path,
        output_path=output_path,
        restricted_zones=restricted_zones
    )

if __name__ == "__main__":
    # Test paths
    model_path = 'models/finetuned/drone_detector/weights/best.pt'
    test_video = 'data/raw/test_video.mp4'
    
    print("Anti-UAV Model Testing")
    print("=" * 50)
    
    # Check if we have a trained model
    if Path(model_path).exists():
        print("✓ Found trained model")
        
        # Check if we have test video
        if Path(test_video).exists():
            print("✓ Found test video")
            
            # Run simple detection test
            print("\n1. Testing simple detection...")
            test_model_simple(model_path, test_video)
            
            # Run full pipeline test
            print("\n2. Testing full pipeline...")
            test_full_pipeline(model_path, test_video)
            
        else:
            print(f"❌ No test video found at {test_video}")
            print("Please add a test video file to data/raw/test_video.mp4")
    
    else:
        print(f"❌ No trained model found at {model_path}")
        print("Please train the model first:")
        print("  python scripts/train_model.py")
        
        # Test with pre-trained YOLO as fallback
        print("\nTesting with pre-trained YOLOv8 (will detect general objects, not specifically drones)...")
        if Path(test_video).exists():
            test_model_simple('yolov8s.pt', test_video, 'outputs/videos/pretrained_test.mp4')