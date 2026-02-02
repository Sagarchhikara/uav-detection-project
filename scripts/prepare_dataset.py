import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2

def explore_dataset(dataset_path):
    """Explore Anti-UAV dataset structure"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download the Anti-UAV dataset and place it in data/raw/")
        return
    
    # Count videos and frames
    video_folders = list(dataset_path.glob("*/"))
    total_frames = 0
    labeled_frames = 0
    
    print(f"Found {len(video_folders)} video sequences\n")
    
    for video_folder in video_folders[:5]:  # Check first 5
        if not video_folder.is_dir():
            continue
            
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
    
    if not dataset_path.exists():
        print(f"❌ Dataset split not found: {dataset_path}")
        return
    
    # Create YOLO structure
    (output_path / 'images').mkdir(parents=True, exist_ok=True)
    (output_path / 'labels').mkdir(parents=True, exist_ok=True)
    
    video_folders = [f for f in dataset_path.glob("*/") if f.is_dir()]
    
    converted_count = 0
    
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
            img = cv2.imread(str(frame_path))
            if img is None:
                continue
            img_height, img_width = img.shape[:2]
            
            # Convert annotation
            annotation = labels[frame_name]
            
            # Parse bbox (format depends on dataset)
            # Common format: {'bbox': [x, y, w, h], 'exist': 1}
            if 'exist' in annotation and annotation['exist'] == 0:
                # No drone in frame - create empty label file
                label_dest = output_path / 'labels' / f"{video_folder.name}_{frame_name}.txt"
                label_dest.touch()
                converted_count += 1
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
            
            # Ensure coordinates are within [0, 1]
            center_x = max(0, min(1, center_x))
            center_y = max(0, min(1, center_y))
            norm_w = max(0, min(1, norm_w))
            norm_h = max(0, min(1, norm_h))
            
            # Class 0 = drone
            yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            
            # Save label
            label_dest = output_path / 'labels' / f"{video_folder.name}_{frame_name}.txt"
            with open(label_dest, 'w') as f:
                f.write(yolo_line)
            
            converted_count += 1
    
    print(f"✓ Converted {split} set")
    print(f"  Images: {len(list((output_path / 'images').glob('*.jpg')))}")
    print(f"  Labels: {len(list((output_path / 'labels').glob('*.txt')))}")
    print(f"  Converted frames: {converted_count}")

def create_sample_dataset():
    """Create a small sample dataset for testing if real dataset is not available"""
    
    print("Creating sample dataset for testing...")
    
    # Create sample structure
    sample_path = Path("data/raw/sample_dataset")
    sample_path.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        split_path = sample_path / split
        split_path.mkdir(exist_ok=True)
        
        # Create a sample video folder
        video_path = split_path / "sample_video_001"
        video_path.mkdir(exist_ok=True)
        
        # Create sample images (just colored rectangles)
        for i in range(10):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            img[:] = (50, 50, 50)  # Dark gray background
            
            # Add a "drone" (white rectangle) at random position
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            w, h = 30, 20
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), -1)
            
            # Save image
            cv2.imwrite(str(video_path / f"{i:06d}.jpg"), img)
        
        # Create sample labels
        labels = {}
        for i in range(10):
            x = np.random.randint(50, 590)
            y = np.random.randint(50, 430)
            w, h = 30, 20
            
            labels[f"{i:06d}"] = {
                "bbox": [x, y, w, h],
                "exist": 1
            }
        
        with open(video_path / "IR_label.json", 'w') as f:
            json.dump(labels, f)
    
    print(f"✓ Sample dataset created at {sample_path}")
    return sample_path

if __name__ == "__main__":
    import numpy as np
    
    print("Anti-UAV Dataset Preparation")
    print("=" * 50)
    
    # Check if Anti-UAV dataset exists
    dataset_path = Path("data/raw/Anti-UAV")
    
    if dataset_path.exists():
        print("✓ Found Anti-UAV dataset")
        
        # Explore dataset
        print("\n1. Exploring dataset structure...")
        explore_dataset(dataset_path)
        
        # Convert to YOLO format
        print("\n2. Converting to YOLO format...")
        for split in ['train', 'val', 'test']:
            if (dataset_path / split).exists():
                convert_antiuav_to_yolo(
                    "data/raw/Anti-UAV",
                    "data/processed/yolo_format",
                    split=split
                )
        
        # Update data.yaml with absolute path
        config_path = Path("config/data.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Replace relative path with absolute path
            abs_path = Path("data/processed/yolo_format").resolve()
            content = content.replace("./data/processed/yolo_format", str(abs_path))
            
            with open(config_path, 'w') as f:
                f.write(content)
            
            print(f"✓ Updated data.yaml with absolute path: {abs_path}")
    
    else:
        print("❌ Anti-UAV dataset not found")
        print("Creating sample dataset for testing...")
        
        sample_path = create_sample_dataset()
        
        # Convert sample dataset
        convert_antiuav_to_yolo(
            str(sample_path),
            "data/processed/yolo_format",
            split="train"
        )
        
        print("\n📝 To use the real Anti-UAV dataset:")
        print("1. Download from: https://github.com/ZhaoJ9014/Anti-UAV")
        print("2. Extract to: data/raw/Anti-UAV/")
        print("3. Run this script again")
    
    print("\n✓ Dataset preparation complete!")
    print("You can now train the model using: python scripts/train_model.py")