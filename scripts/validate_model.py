from ultralytics import YOLO
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def validate_model():
    """
    Run validation on the trained model to generate metrics and plots.
    Produces: confusion_matrix, F1_curve, PR_curve, val_batch predictions, etc.
    """
    
    # Path to your trained model
    model_path = 'runs/detect/models/finetuned/drone_detector/weights/best.pt'
    
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    print("Starting validation...")
    # metrics will be saved to runs/detect/val
    metrics = model.val(
        data='config/data.yaml',
        split='val',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.6,
        device=0
    )
    
    print("\n✓ Validation complete!")
    print(f"  Results saved to: {metrics.save_dir}")
    print("  Generated plots:")
    print("  - Confusion Matrix")
    print("  - F1 / Precision / Recall Curves")
    print("  - Validation batch predictions (val_batch*.jpg)")

if __name__ == "__main__":
    validate_model()
