from ultralytics import YOLO
import torch
from pathlib import Path
import yaml

def train_drone_detector(
    model_size='yolov8s',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='auto'
):
    """
    Train YOLOv8 on drone dataset
    
    Args:
        model_size: 'yolov8n', 'yolov8s', 'yolov8m' (s recommended for RTX 4050)
        epochs: Number of training epochs
        batch_size: Batch size (reduce if OOM)
        img_size: Input image size
        device: 'auto', 'cuda:0' or 'cpu'
    """
    
    # Auto-detect device
    if device == 'auto':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
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
    
    # Check if data.yaml exists
    data_yaml = 'config/data.yaml'
    if not Path(data_yaml).exists():
        print(f"❌ Data config not found: {data_yaml}")
        print("Please create the data.yaml file with your dataset paths")
        return None
    
    # Training configuration
    try:
        results = model.train(
            data=data_yaml,
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
        
        return results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if dataset exists and paths are correct in config/data.yaml")
        print("2. Reduce batch_size if getting CUDA out of memory")
        print("3. Use yolov8n instead of yolov8s for lower memory usage")
        return None

if __name__ == "__main__":
    # Start training
    print("Starting Anti-UAV Drone Detection Model Training...")
    print("This will take several hours depending on your dataset size and hardware.")
    
    results = train_drone_detector(
        model_size='yolov8s',  # Change to 'yolov8n' if running out of memory
        epochs=100,
        batch_size=16,  # Reduce to 8 if OOM
        img_size=640,
        device='auto'
    )
    
    if results:
        print("\n🎉 Training completed successfully!")
        print("You can now test your model using scripts/test_model.py")
    else:
        print("\n❌ Training failed. Please check the error messages above.")