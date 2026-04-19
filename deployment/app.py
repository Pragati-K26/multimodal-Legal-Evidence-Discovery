import os
import cv2
import base64
import asyncio
import threading
import queue
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ultralytics import YOLO
from google import genai
import uvicorn
import json
import time
import yt_dlp
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=2)
insight_queue = asyncio.Queue(maxsize=5)

# 1. Cloud Configuration (Hugging Face)
API_KEY = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
# IMPORTANT: Manually upload your best.pt files to the /models folder in your Space
AUDIT_WEIGHTS = os.path.join(MODELS_ROOT, "best_yolo11.pt")
LEGACY_WEIGHTS = os.path.join(MODELS_ROOT, "best_yolo8.pt")

GEMINI_MODEL = "gemini-1.5-flash" # Optimized for free-tier speed

# 2. Forensic Profiles (V7.4 - Dual Mode)
CRITICAL_CLASSES = {'Blood', 'Handgun', 'Shotgun', 'Knife', 'Hammer', 'Rope', 'Victim', 'Human-body', 'Finger-print', 'Shoe-print'}

# PROFILE A: IMAGE AUDIT (Hyper-Sensitive)
IMAGE_THRESHOLDS = {k: 0.01 for k in CRITICAL_CLASSES}
IMAGE_THRESHOLDS['Blood'] = 0.005      
IMAGE_THRESHOLDS['Human-body'] = 0.003
IMAGE_THRESHOLDS['Victim'] = 0.003     
IMAGE_THRESHOLDS['Handgun'] = 0.03     
IMAGE_DEFAULT_THRESH = 0.01

# PROFILE B: VIDEO MONITOR (High-Stability)
VIDEO_THRESHOLDS = {k: 0.25 for k in CRITICAL_CLASSES}
VIDEO_THRESHOLDS['Blood'] = 0.01       
VIDEO_THRESHOLDS['Finger-print'] = 0.001 
VIDEO_THRESHOLDS['Shoe-print'] = 0.001
VIDEO_THRESHOLDS['Handgun'] = 0.5     
VIDEO_THRESHOLDS['Shotgun'] = 0.5     
VIDEO_THRESHOLDS['Knife'] = 0.5      
VIDEO_THRESHOLDS['Human-body'] = 0.2 
VIDEO_THRESHOLDS['Hammer'] = 0.5 
VIDEO_DEFAULT_THRESH = 0.25

UPLOAD_DIR = "/tmp/uploads"
RECORDING_DIR = "/tmp/recordings"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RECORDING_DIR, exist_ok=True)

# 3. Model Initialization (CPU Mode Force)
print(f"INITIALIZING JUDICIAL CLOUD (CPU MODE)...")
model_audit = YOLO(AUDIT_WEIGHTS) if os.path.exists(AUDIT_WEIGHTS) else None
model_legacy = YOLO(LEGACY_WEIGHTS) if os.path.exists(LEGACY_WEIGHTS) else None
client = genai.Client(api_key=API_KEY)

# 4. Frontend Assets
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend_cloud")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

def resolve_judicial_stream(url):
    """
    Acts as a Judicial Stream Handshake. If the input is a web link (YouTube, etc.),
    it resolves it into a direct MP4/stream URL for cv2 integration.
    """
    if not url.startswith(("http://", "https://")):
        return url
    
    print(f"Resolving Judicial Stream: {url}")
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        print(f"Stream Resolution Failure: {e}")
        return url

@app.get("/")
def get_dashboard():
    html_path = os.path.join(FRONTEND_DIR, "evidence_scanner.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return "Forensic Dashboard Pending Build..."

@app.post("/upload_evidence")
async def upload_evidence(file: UploadFile = File(...)):
    import shutil
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "SUCCESS", "path": file_path}

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0]); yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2]); yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# 6. Neural Fusion Logic
def fuse_detections(raw_detections, iou_threshold=0.5):
    """V7.6 Judicial Consensus: Weighted Box Fusion (WBF-lite). 
    Averages overlapping boxes based on confidence weight."""
    if not raw_detections: return []
    final_fused = []
    labels = set(d['label'] for d in raw_detections)
    
    # Model Weights (V7.9): Balanced Ensemble Consensus (50/50)
    MODEL_WEIGHTS = {"AUDIT": 1.0, "LEGACY": 1.0}
    
    for label in labels:
        label_dets = [d for d in raw_detections if d['label'] == label]
        while label_dets:
            primary = label_dets.pop(0)
            cluster = [primary]; keep = []
            for other in label_dets:
                if calculate_iou(primary['box'], other['box']) > iou_threshold:
                    cluster.append(other)
                else:
                    keep.append(other)
            label_dets = keep
            
            # Weighted Averaging based on Confidence AND Model Priority
            sum_weighted_conf = sum(d['conf'] * MODEL_WEIGHTS.get(d['model'], 1.0) for d in cluster)
            avg_box = [0, 0, 0, 0]
            for d in cluster:
                weight = d['conf'] * MODEL_WEIGHTS.get(d['model'], 1.0)
                for i in range(4): avg_box[i] += d['box'][i] * weight
            
            avg_box = [int(v / sum_weighted_conf) for v in avg_box]
            final_fused.append({
                "label": label, 
                "conf": max(d['conf'] for d in cluster), 
                "box": avg_box
            })
    return final_fused

def process_yolo_only(frame, timestamp, is_video=True):
    if not model_audit or not model_legacy: return [], frame
    
    # Select Forensic Profile
    thresholds = VIDEO_THRESHOLDS if is_video else IMAGE_THRESHOLDS
    def_thresh = VIDEO_DEFAULT_THRESH if is_video else IMAGE_DEFAULT_THRESH
    
    # UPGRADE: Judicial resolution jump to 800px for forensic parity (V7.9)
    res_audit = model_audit(frame, conf=0.01, iou=0.25, imgsz=800, device='cpu', verbose=False)[0]
    res_legacy = model_legacy(frame, conf=0.01, iou=0.25, imgsz=800, device='cpu', verbose=False)[0]
    
    raw_hits = []
    # Audit Engine (YOLOv11) - WEIGHT: 2.0
    for box in res_audit.boxes:
        conf, cls_id = float(box.conf[0]), int(box.cls[0])
        label = res_audit.names[cls_id]
        if conf > thresholds.get(label, def_thresh):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "HUMAN BODY" if label in ["Human-body", "Victim"] else label
            raw_hits.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2], "model": "AUDIT"})
            
    # Legacy Engine (YOLOv8) - WEIGHT: 1.0
    for box in res_legacy.boxes:
        conf, cls_id = float(box.conf[0]), int(box.cls[0])
        label = res_legacy.names[cls_id]
        if conf > thresholds.get(label, def_thresh):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "HUMAN BODY" if label in ["Human-body", "Victim"] else label
            raw_hits.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2], "model": "LEGACY"})
                
    fused_hits = fuse_detections(raw_hits)
    detections = []
    for det in fused_hits:
        x1, y1, x2, y2 = det['box']
        label, conf = det['label'], det['conf']
        color = (0, 0, 255) if label in CRITICAL_CLASSES or label == 'HUMAN BODY' else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # CLOUD NEURAL ZOOM: High-resolution judicial capture
        h_orig, w_orig = frame.shape[:2]
        pad_x, pad_y = max(10, int((x2 - x1) * 0.15)), max(10, int((y2 - y1) * 0.15))
        cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        cx2, cy2 = min(w_orig, x2 + pad_x), min(h_orig, y2 + pad_y)
        
        crop = frame[cy1:cy2, cx1:cx2]
        crop_b64 = None
        if crop.size > 0:
            _, c_buf = cv2.imencode('.jpg', crop)
            crop_b64 = base64.b64encode(c_buf).decode('utf-8')
            
        detections.append({
            "label": label, 
            "conf": conf, 
            "timestamp": timestamp,
            "crop": crop_b64,
            "type": "NEURAL"
        })
    return detections, frame

@app.websocket("/ws/discover")
async def websocket_discovery(websocket: WebSocket):
    await websocket.accept()
    raw_path = websocket.query_params.get("video_path")
    if not raw_path: await websocket.close(); return
    
    # Judicial Stream Handshake (YouTube Support)
    video_path = resolve_judicial_stream(raw_path)
    
    # Automatic Source Detection
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _, ext = os.path.splitext(video_path.lower().split('?')[0])
    is_video = ext not in img_exts
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            detections, annotated = process_yolo_only(frame, timestamp, is_video=is_video)
            _, buf = cv2.imencode('.jpg', annotated)
            frame_b64 = base64.b64encode(buf).decode('utf-8')
            await websocket.send_text(json.dumps({"frame": frame_b64, "detections": detections, "status": "DISCOVERING"}))
            await asyncio.sleep(0.1) # Throttle for CPU
    except Exception as e:
        print(f"Cloud WebSocket Error: {e}")
    finally:
        cap.release()
        await websocket.send_text(json.dumps({"status": "COMPLETE"}))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
