import os
from ultralytics import YOLO
import torch

def train_forensic_yolo11():
    # 1. Hardware Check
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"INITIALIZING YOLOv11-L HIGH-SPEED TRAINING ON: {device}")
    
    # 2. Load the latest YOLOv11 Small Backbone (Full Audit)
    # YOLOv11s provides the best balance for full-dataset forensic scanning
    model = YOLO("yolo11n.pt")
    
    # 3. Training Path Configuration
    DATA_YAML = r"D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\data\crime_scene_yolov8\data.yaml"
    
    print("COMMENCING STAGE 6.5: YOLOv11-L HIGH-SPEED DISCOVERY AUDIT...")
    
    try:
        # Full Dataset Forensic Audit Configuration
        results = model.train(
            data=DATA_YAML,
            epochs=50,        # Judicial limit as requested
            patience=10,      # Early stopping trigger
            imgsz=512,        # Balanced Forensic Resolution
            batch=24,         # Optimized for 6GB RTX 4050 throughput
            name='yolo11_forensic_audit',
            device=device,
            exist_ok=True,
            workers=0,         # Windows Stability Mode
            # --- ADVANCED FORENSIC TUNING ---
            box=7.5,
            cls=1.5,
            hsv_s=0.5,
            hsv_v=0.3,
            mosaic=0.8,
            scale=0.3,
            amp=True           # Fast training activation
        )
        print("\nYOLOv11-L Forensic Training Complete!")
        print(f"Weights saved in: {results.save_dir}")
        
    except Exception as e:
        print(f"An error occurred during YOLOv11 training: {e}")

if __name__ == "__main__":
    train_forensic_yolo11()
