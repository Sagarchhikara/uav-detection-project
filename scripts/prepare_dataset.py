import json
import os
from pathlib import Path
from tqdm import tqdm
import shutil
import cv2
import numpy as np

def explore_dataset(dataset_path):
    """Explore Anti-UAV dataset structure"""
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please download the Anti-UAV dataset and place it in data/raw/")
        return
    
    # Check for train/val/test splits or flat structure
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    # If splits exist (train/val/test), explore those
    if any(s.name in ['train', 'val', 'test'] for s in subdirs):
        print(f"Found splits: {[s.name for s in subdirs if s.name in ['train', 'val', 'test']]}\n")
        # Just look at train for exploration
        target_dir = dataset_path / 'train'
        if not target_dir.exists():
            target_dir = dataset_path
    else:
        target_dir = dataset_path
    
    video_folders = [f for f in target_dir.glob("*") if f.is_dir() and f.name not in ['train', 'val', 'test']]
    
    print(f"Found {len(video_folders)} video sequences in {target_dir.name}\n")
    
    for video_folder in video_folders[:3]:  # Check first 3
        # Check for video files
        mp4_files = list(video_folder.glob("*.mp4"))
        jpg_files = list(video_folder.glob("*.jpg"))
        
        label_file = video_folder / "IR_label.json"
        if not label_file.exists():
            label_file = video_folder / "infrared.json"
            
        print(f"Video: {video_folder.name}")
        if mp4_files:
            print(f"  Video files: {[f.name for f in mp4_files]}")
        if jpg_files:
            print(f"  Extracted frames: {len(jpg_files)}")
            
        if label_file.exists():
            with open(label_file, 'r') as f:
                labels = json.load(f)
            
            if isinstance(labels, dict):
                 print(f"  Label format: Dictionary (keys={list(labels.keys())[:3]}...)")
            elif isinstance(labels, list):
                 print(f"  Label format: List (length={len(labels)})")
                 if len(labels) > 0 and isinstance(labels[0], dict):
                     print(f"  First item keys: {list(labels[0].keys())}")
            elif isinstance(labels, dict) and "exist" in labels:
                 # The complex structure seen in infrared.json
                 print(f"  Label format: Complex Dictionary (keys={list(labels.keys())})")
                 print(f"  'exist' length: {len(labels.get('exist', []))}")
        else:
             print("  ⚠️ No label file found")
        
        print()

def convert_antiuav_to_yolo(dataset_path, output_path):
    """
    Convert Anti-UAV dataset to YOLO format
    Handles recursively finding video folders.
    """
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return
    
    # Define splits relative to dataset_path
    # If dataset has train/val/test subdirs, use them.
    # Otherwise, split randomly.
    
    # First, gather all video folders
    all_video_folders = []
    
    # Recursive search for folders containing 'infrared.json' or 'IR_label.json'
    print("Searching for video folders...")
    for root, dirs, files in os.walk(dataset_path):
        if "infrared.json" in files or "IR_label.json" in files:
            all_video_folders.append(Path(root))
            
    print(f"Found {len(all_video_folders)} video folders total.")
    
    # Determine split for each folder
    # If they are already in 'train'/'val'/'test' folders, respect that.
    # Else, do 70/20/10 split
    
    random_split = False
    if not any(p.name in ['train', 'val', 'test'] for p in dataset_path.iterdir() if p.is_dir()):
        print("No split directories found. Performing random 70/20/10 split.")
        random_split = True
        np.random.shuffle(all_video_folders)
        n = len(all_video_folders)
        idx1 = int(n * 0.7)
        idx2 = int(n * 0.9)
        train_folders = all_video_folders[:idx1]
        val_folders = all_video_folders[idx1:idx2]
        test_folders = all_video_folders[idx2:]
    else:
        train_folders = [f for f in all_video_folders if 'train' in f.parts]
        val_folders = [f for f in all_video_folders if 'val' in f.parts]
        test_folders = [f for f in all_video_folders if 'test' in f.parts]
        
        # Fallback if some are missed
        remaining = [f for f in all_video_folders if f not in train_folders + val_folders + test_folders]
        if remaining:
            print(f"Found {len(remaining)} folders outside split dirs. Adding to train.")
            train_folders.extend(remaining)

    splits = {
        'train': train_folders,
        'val': val_folders,
        'test': test_folders
    }
    
    for split_name, folders in splits.items():
        if not folders:
            continue
            
        print(f"\nProcessing {split_name} split ({len(folders)} videos)...")
        
        split_out = output_path / split_name
        (split_out / 'images').mkdir(parents=True, exist_ok=True)
        (split_out / 'labels').mkdir(parents=True, exist_ok=True)
        
        converted_count = 0
        
        for video_folder in tqdm(folders):
            # 1. Determine Label File
            label_file = video_folder / "IR_label.json"
            if not label_file.exists():
                label_file = video_folder / "infrared.json"
                
            if not label_file.exists():
                continue
                
            # 2. Extract frames if needed
            # Prefer infrared.mp4
            video_file = video_folder / "infrared.mp4"
            if not video_file.exists():
                # Check for any mp4
                mp4s = list(video_folder.glob("*.mp4"))
                if mp4s:
                    video_file = mp4s[0]
            
            with open(label_file, 'r') as f:
                labels_data = json.load(f)
            
            # Parse Labels
            parsed_labels = {} # frame_idx -> [bbox]
            
            if isinstance(labels_data, dict):
                if "exist" in labels_data:
                    # Structure found in infrared.json
                    exists = labels_data.get("exist", [])
                    # gt_rect or gt_bbox
                    rects = labels_data.get("gt_rect", [])
                    if not rects:
                        rects = labels_data.get("gt_bbox", [])
                        
                    for idx, (exist, rect) in enumerate(zip(exists, rects)):
                        if exist == 1 and len(rect) == 4:
                            parsed_labels[idx] = rect
                else:
                    # Generic dictionary keys
                    pass
            
            # Process Video
            if video_file.exists():
                cap = cv2.VideoCapture(str(video_file))
                frame_idx = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    if frame_idx in parsed_labels:
                        bbox = parsed_labels[frame_idx]
                        
                        # Save Image
                        img_filename = f"{video_folder.name}_{frame_idx:06d}.jpg"
                        img_path = split_out / 'images' / img_filename
                        cv2.imwrite(str(img_path), frame)
                        
                        # Convert to YOLO
                        img_h, img_w = frame.shape[:2]
                        x, y, w, h = bbox
                        
                        # Normalize
                        center_x = (x + w/2) / img_w
                        center_y = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        
                        # Clamp
                        center_x = max(0, min(1, center_x))
                        center_y = max(0, min(1, center_y))
                        norm_w = max(0, min(1, norm_w))
                        norm_h = max(0, min(1, norm_h))
                        
                        yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                        
                        label_path = split_out / 'labels' / f"{video_folder.name}_{frame_idx:06d}.txt"
                        with open(label_path, 'w') as lf:
                            lf.write(yolo_line)
                            
                        converted_count += 1
                        
                    frame_idx += 1
                cap.release()
            else:
                # Iterate over images if no video file
                jpgs = sorted(list(video_folder.glob("*.jpg")))
                if not jpgs:
                    continue
                    
                for i, jpg_path in enumerate(jpgs):
                    idx = i # Assumption
                    if idx in parsed_labels:
                        bbox = parsed_labels[idx]
                        
                         # Save Image
                        img_filename = f"{video_folder.name}_{idx:06d}.jpg"
                        img_path = split_out / 'images' / img_filename
                        shutil.copy(jpg_path, img_path)
                        
                        # Read dims
                        img = cv2.imread(str(img_path))
                        if img is None: continue
                        img_h, img_w = img.shape[:2]
                        
                        x, y, w, h = bbox
                         # Normalize
                        center_x = (x + w/2) / img_w
                        center_y = (y + h/2) / img_h
                        norm_w = w / img_w
                        norm_h = h / img_h
                        
                        yolo_line = f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n"
                        
                        label_path = split_out / 'labels' / f"{video_folder.name}_{idx:06d}.txt"
                        with open(label_path, 'w') as lf:
                            lf.write(yolo_line)
                            
                        converted_count += 1
                        
        print(f"✓ Converted {split_name}: {converted_count} labeled frames")

def create_sample_dataset():
    """Create a small sample dataset for testing if real dataset is not available"""
    print("Creating sample dataset for testing...")
    sample_path = Path("data/raw/sample_dataset")
    sample_path.mkdir(parents=True, exist_ok=True)
    
    # Create structure matching what we expect (infrared.json)
    video_path = sample_path / "sample_video_001"
    video_path.mkdir(exist_ok=True)
    
    # Create dummy video using cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path / 'infrared.mp4'), fourcc, 20.0, (640, 480))
    
    exists = []
    gt_rect = []
    
    for i in range(10):
        frame = np.zeros((480, 640, 3), dtype=np.uint8) + 50
        x = np.random.randint(50, 590)
        y = np.random.randint(50, 430)
        w, h = 30, 20
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), -1)
        out.write(frame)
        
        exists.append(1)
        gt_rect.append([x, y, w, h])
        
    out.release()
    
    labels = {
        "exist": exists,
        "gt_rect": gt_rect
    }
    
    with open(video_path / "infrared.json", 'w') as f:
        json.dump(labels, f)
        
    print(f"✓ Sample dataset created at {sample_path}")
    return sample_path

if __name__ == "__main__":
    
    print("Anti-UAV Dataset Preparation")
    print("=" * 50)
    
    dataset_path = Path("data/raw/Anti-UAV")
    
    if not dataset_path.exists():
        print("Dataset not valid, checking current folder...")
        # Fallback check
        if Path("Anti-UAV").exists():
             shutil.move("Anti-UAV", "data/raw/Anti-UAV")
             dataset_path = Path("data/raw/Anti-UAV")
    
    if dataset_path.exists():
        print("✓ Found Anti-UAV dataset")
        explore_dataset(dataset_path)
        
        print("\nConverting to YOLO format...")
        convert_antiuav_to_yolo(dataset_path, "data/processed/yolo_format")
        
        # Update config
        config_path = Path("config/data.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                content = f.read()
            abs_path = Path("data/processed/yolo_format").resolve()
            # Regex or safe replace
            import re
            content = re.sub(r"path: .*", f"path: {abs_path}", content)
            
            with open(config_path, 'w') as f:
                f.write(content)
            print(f"✓ Updated data.yaml")
            
    else:
        print("❌ Anti-UAV dataset not found")
        sample = create_sample_dataset()
        convert_antiuav_to_yolo(sample, "data/processed/yolo_format")
    
    print("\n✓ Dataset preparation complete!")