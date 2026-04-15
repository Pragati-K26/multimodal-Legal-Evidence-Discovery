import os
from ultralytics import RTDETR
import torch

def train_forensic_rtdetr():
    # 1. Hardware Check & Optimization
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"INITIALIZING RT-DETR FORENSIC TRAINING ON: {device}")
    
    # 2. Load Transformer Backbone (Large for Maximum Recall)
    # We use rtdetr-l.pt as the gold-standard for Transformer detection
    model = RTDETR("rtdetr-l.pt")
    
    # 3. Training Path Configuration
    DATA_YAML = r"D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\data\crime_scene_yolov8\data.yaml"
    
    print("COMMENCING PHASE 6: TRANSFORMER-BASED JUDICIAL AUDIT...")
    
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=100,      
            patience=50,    
            imgsz=1024,      
            batch=1,         
            name='rtdetr_forensic_audit',
            device=device,
            exist_ok=True,
            workers=0,       # Windows Stability Mode (Bypasses pickling errors)
            # --- TRANSFORMER AUGMENTATIONS ---
            mosaic=1.0,      # Blend scenes for complex context
            mixup=0.2,       # Overlay transparency for fluid traces
            scale=0.5,       # Multi-scale awareness
            flipud=0.0,      # Maintain forensic orientation
            fliplr=0.5,      # Lateral augmentation
            amp=True         # Automatic Mixed Precision for laptop speed
        )
        print("\nRT-DETR Forensic Shift Complete!")
        print(f"Judicial Transformer Weights saved in: {results.save_dir}")
        
    except Exception as e:
        print(f"An error occurred during RT-DETR training: {e}")

if __name__ == "__main__":
    train_forensic_rtdetr()
