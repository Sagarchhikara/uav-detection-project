#!/usr/bin/env python3
"""
Verify that the Anti-UAV Detection System is properly set up
Run this script on a fresh laptop to check everything is working
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print("✅ Python version:", f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print("❌ Python 3.8+ required. Current:", f"{version.major}.{version.minor}.{version.micro}")
        return False

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        'torch', 'torchvision', 'ultralytics', 'cv2', 'numpy', 
        'pandas', 'streamlit', 'plotly', 'scipy', 'sklearn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_project_structure():
    """Check if project structure is correct"""
    required_dirs = [
        'config', 'data', 'models', 'src', 'scripts', 'ui', 'outputs'
    ]
    
    required_files = [
        'requirements.txt', 'README.md', 'run_app.py',
        'config/model_config.yaml', 'config/detection_config.yaml', 'config/data.yaml',
        'scripts/train_model.py', 'scripts/test_model.py', 'scripts/prepare_dataset.py',
        'ui/streamlit_app.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    for directory in required_dirs:
        if Path(directory).exists():
            print(f"✅ Directory: {directory}/")
        else:
            print(f"❌ Directory: {directory}/")
            missing_dirs.append(directory)
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File: {file_path}")
            missing_files.append(file_path)
    
    return len(missing_dirs) == 0 and len(missing_files) == 0

def check_gpu():
    """Check GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  No GPU detected - training will be slower on CPU")
            return False
    except:
        print("❌ Cannot check GPU status")
        return False

def check_dataset_location():
    """Check if dataset location exists"""
    dataset_path = Path("data/raw")
    if dataset_path.exists():
        print("✅ Dataset directory exists: data/raw/")
        
        # Check if Anti-UAV dataset is present
        anti_uav_path = dataset_path / "Anti-UAV"
        if anti_uav_path.exists():
            print("✅ Anti-UAV dataset found")
            
            # Check subdirectories
            for split in ['train', 'val', 'test']:
                split_path = anti_uav_path / split
                if split_path.exists():
                    video_count = len(list(split_path.glob("*/")))
                    print(f"✅ {split}/ directory: {video_count} video folders")
                else:
                    print(f"⚠️  {split}/ directory not found")
        else:
            print("⚠️  No Anti-UAV dataset found in data/raw/")
            print("   Upload your dataset to: data/raw/Anti-UAV/")
    else:
        print("❌ Dataset directory missing: data/raw/")

def main():
    """Main verification function"""
    print("🚁 Anti-UAV Detection System - Setup Verification")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    print("\n1. Python Version:")
    if not check_python_version():
        all_good = False
    
    # Check dependencies
    print("\n2. Dependencies:")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        all_good = False
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("   Install with: pip install -r requirements.txt")
    
    # Check project structure
    print("\n3. Project Structure:")
    if not check_project_structure():
        all_good = False
    
    # Check GPU
    print("\n4. GPU Status:")
    check_gpu()
    
    # Check dataset
    print("\n5. Dataset Location:")
    check_dataset_location()
    
    # Final status
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 SETUP VERIFICATION COMPLETE!")
        print("✅ Your system is ready for training!")
        print("\nNext steps:")
        print("1. Upload dataset to data/raw/Anti-UAV/")
        print("2. Run: python scripts/prepare_dataset.py")
        print("3. Run: python scripts/train_model.py")
    else:
        print("❌ SETUP INCOMPLETE!")
        print("Please fix the issues above before proceeding.")
        print("Refer to SETUP_AND_TRAINING_GUIDE.txt for detailed instructions.")

if __name__ == "__main__":
    main()