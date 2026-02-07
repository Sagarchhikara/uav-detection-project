#!/usr/bin/env python3
"""
Environment setup and verification script for Anti-UAV Detection System
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            # Extract CUDA version from nvidia-smi output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    print(f"✅ CUDA Version: {cuda_version}")
                    return True
        else:
            print("⚠️  No NVIDIA GPU detected - will use CPU (slower)")
            return False
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - no GPU acceleration")
        return False

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def verify_installation():
    """Verify key packages are installed correctly"""
    packages = {
        'torch': 'PyTorch',
        'ultralytics': 'YOLOv8',
        'cv2': 'OpenCV',
        'streamlit': 'Streamlit',
        'numpy': 'NumPy'
    }
    
    all_good = True
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} installed")
        except ImportError:
            print(f"❌ {name} not found")
            all_good = False
    
    return all_good

def check_pytorch_cuda():
    """Check if PyTorch can use CUDA"""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ PyTorch CUDA available - {device_name}")
            return True
        else:
            print("⚠️  PyTorch CUDA not available - using CPU")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'models/pretrained',
        'models/finetuned',
        'outputs/videos',
        'outputs/logs',
        'outputs/reports'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def download_yolo_weights():
    """Download pre-trained YOLO weights"""
    try:
        from ultralytics import YOLO
        print("📥 Downloading YOLOv8 weights...")
        
        # This will download yolov8s.pt to the ultralytics cache
        model = YOLO('yolov8s.pt')
        print("✅ YOLOv8 weights downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download YOLO weights: {e}")
        return False

def main():
    """Main setup function"""
    print("🚁 Anti-UAV Detection System - Environment Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check CUDA
    has_cuda = check_cuda()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("\n❌ Setup failed during package installation")
        sys.exit(1)
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Setup failed during verification")
        sys.exit(1)
    
    # Check PyTorch CUDA
    pytorch_cuda = check_pytorch_cuda()
    
    # Download YOLO weights
    if not download_yolo_weights():
        print("\n⚠️  YOLO weights download failed, but you can continue")
    
    # Final status
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETE!")
    print("=" * 60)
    
    print("\n📋 System Status:")
    print(f"  Python: ✅")
    print(f"  CUDA: {'✅' if has_cuda else '⚠️ '}")
    print(f"  PyTorch CUDA: {'✅' if pytorch_cuda else '⚠️ '}")
    print(f"  Packages: ✅")
    
    print("\n🚀 Next Steps:")
    print("1. Prepare your dataset:")
    print("   python scripts/prepare_dataset.py")
    print("\n2. Train the model:")
    print("   python scripts/train_model.py")
    print("\n3. Test the model:")
    print("   python scripts/test_model.py")
    print("\n4. Launch the web interface:")
    print("   streamlit run ui/streamlit_app.py")
    
    if not pytorch_cuda:
        print("\n⚠️  Note: No GPU acceleration detected.")
        print("   Training and inference will be slower on CPU.")
        print("   Consider installing CUDA and PyTorch with GPU support.")

if __name__ == "__main__":
    main()