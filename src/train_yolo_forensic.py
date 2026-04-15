import os
from ultralytics import YOLO
import torch

def train_yolo_precision_v5():
    # 1. GPU Setup
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 2. V4 Intelligence Migration
    # We are initializing from the recovered v4 intelligence but starting a STABLE v5 run
    print("Migrating recovered V4 intelligence into Stable V5 Epochs...")
    weight_path = r'D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\models\precision_recovery_v4.pt'
    
    if os.path.exists(weight_path):
        model = YOLO(weight_path)
    else:
        # Fallback only if recovery file was missing
        print("WARNING: Recovery weights not found. Falling back to baseline v2.")
        model = YOLO('runs/detect/forensic_evidence_model/weights/best.pt') 

    # 3. Training Configuration
    DATA_YAML = r"D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\data\crime_scene_yolov8\data.yaml"
    
    print("Launching Phase 5: Judicial Stability Run (v5)...")
    try:
        results = model.train(
            data=DATA_YAML, 
            epochs=80,       # Remaining epochs from v4 plus buffer
            imgsz=640,      
            batch=4,         # VRAM Optimized
            name='forensic_evidence_model_v5', 
            device=device,
            exist_ok=True,
            resume=False,    # Starting fresh v5 run with recovered weights
            amp=False,       # CUDA Stabilization
            workers=0,       # Threading Stabilization
            # --- MAINTAINING HIGH PRECISION SETTINGS ---
            box=7.5,
            cls=1.5,
            hsv_s=0.5,
            hsv_v=0.3,
            mosaic=0.5,
            scale=0.3,
            patience=50      
        )
        print("\nPrecision Recovery Stage 5 complete!")
        print(f"Final Judicial Weights saved in: {results.save_dir}")
        
    except Exception as e:
        print(f"An error occurred during V5 migration: {e}")

if __name__ == "__main__":
    train_yolo_precision_v5()
