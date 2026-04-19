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
from video_streamer import ThreadedStreamer
import uvicorn
import json
import time
import yt_dlp
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import io

app = FastAPI()
executor = ThreadPoolExecutor(max_workers=3)
insight_queue = asyncio.Queue(maxsize=10) # Prevent memory bloat from queued high-res frames

# 1. Configuration (DUAL AUDIT MODE - V7.0)
API_KEY = "AIzaSyCjvcOS-cNjHAlP9l4h2ylZ4OX-NN5kmN0"
MODELS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
AUDIT_WEIGHTS = r"D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\runs\detect\yolo11_forensic_audit\weights\best.pt"
LEGACY_WEIGHTS = r"D:\deep learning\Multimodal-Legal-Discovery\Legal_Evidence_Discovery\models\forensic_evidence_model_v5\weights\best.pt"

GEMINI_MODEL = "gemini-3.1-pro-preview"

# 2. Forensic Profiles (V7.4 - Dual Mode)
CRITICAL_CLASSES = {'Blood', 'Handgun', 'Shotgun', 'Knife', 'Hammer', 'Rope', 'Victim', 'Human-body', 'Finger-print', 'Shoe-print'}

# PROFILE A: IMAGE AUDIT (Hyper-Sensitive)
IMAGE_THRESHOLDS = {k: 0.01 for k in CRITICAL_CLASSES}
IMAGE_THRESHOLDS['Blood'] = 0.005      # Extreme Recall
IMAGE_THRESHOLDS['Human-body'] = 0.003 # V7.2 Partial Body
IMAGE_THRESHOLDS['Victim'] = 0.003     # V7.2 Face/Upper Torso
IMAGE_THRESHOLDS['Handgun'] = 0.03     # Per User Noise Filter
IMAGE_DEFAULT_THRESH = 0.01

# PROFILE B: VIDEO MONITOR (High-Stability)
# Designed to eliminate "random detections" during live movement
VIDEO_THRESHOLDS = {k: 0.25 for k in CRITICAL_CLASSES}
VIDEO_THRESHOLDS['Blood'] = 0.01       # RECAL OVERRIDE (Per User Request)
VIDEO_THRESHOLDS['Finger-print'] = 0.001 
VIDEO_THRESHOLDS['Shoe-print'] = 0.001
VIDEO_THRESHOLDS['Handgun'] = 0.5     # Eliminate transient weapon ghosts
VIDEO_THRESHOLDS['Shotgun'] = 0.5     
VIDEO_THRESHOLDS['Knife'] = 0.5      
VIDEO_THRESHOLDS['Human-body'] = 0.2 

VIDEO_THRESHOLDS['Hammer'] = 0.5 # Stabilize silhouette jitter
VIDEO_DEFAULT_THRESH = 0.25

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 3. Model Initialization (DUAL DISCOVERY)
print(f"INITIALIZING JUDICIAL HYBRID (V7.0)...")
print(f" - Loading Audit Engine (V11): {AUDIT_WEIGHTS}")
model_audit = YOLO(AUDIT_WEIGHTS)
print(f" - Loading Legacy Engine (V5): {LEGACY_WEIGHTS}")
model_legacy = YOLO(LEGACY_WEIGHTS)
client = genai.Client(api_key=API_KEY)

# 4. Storage & Static Assets
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# 5. Background Recording Worker (Zero Frame Loss)
def recording_worker(frame_queue, output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recorder = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"JUDICIAL TAPE START: {output_path}")
    
    while True:
        try:
            item = frame_queue.get(timeout=2)
            if isinstance(item, str) and item == "STOP": break
            recorder.write(item)
        except queue.Empty:
            continue
    
    recorder.release()
    print(f"JUDICIAL TAPE SEALED: {output_path}")

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
    with open(os.path.join(FRONTEND_DIR, "evidence_scanner.html"), "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.post("/upload_evidence")
async def upload_evidence(file: UploadFile = File(...)):
    import shutil
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "SUCCESS", "path": file_path}

# 6. Forensic Clarity Utilities (Neural Fusion)
def calculate_iou(box1, box2):
    # box1, box2: [x1, y1, x2, y2]
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def fuse_detections(raw_detections, iou_threshold=0.5):
    """V7.6 Judicial Consensus: Weighted Box Fusion (WBF-lite). 
    Averages overlapping boxes based on confidence weight AND model priority."""
    if not raw_detections: return []
    
    # Model Weights (V7.9): Balanced Forensic Consensus (50/50)
    MODEL_WEIGHTS = {"AUDIT": 1.0, "LEGACY": 1.0}
    
    final_fused = []
    labels = set(d['label'] for d in raw_detections)
    
    for label in labels:
        label_dets = [d for d in raw_detections if d['label'] == label]
        
        while label_dets:
            primary = label_dets.pop(0)
            cluster = [primary]
            keep = []
            
            for other in label_dets:
                if calculate_iou(primary['box'], other['box']) > iou_threshold:
                    cluster.append(other)
                else:
                    keep.append(other)
            label_dets = keep
            
            # Weighted Averaging: (Coord * Conf * ModelWeight) / TotalWeight
            sum_weighted_conf = sum(d['conf'] * MODEL_WEIGHTS.get(d['model'], 1.0) for d in cluster)
            avg_box = [0, 0, 0, 0]
            for d in cluster:
                weight = d['conf'] * MODEL_WEIGHTS.get(d['model'], 1.0)
                for i in range(4):
                    avg_box[i] += d['box'][i] * weight
            
            avg_box = [int(v / sum_weighted_conf) for v in avg_box]
            
            final_fused.append({
                "label": label,
                "conf": max(d['conf'] for d in cluster),
                "box": avg_box
            })
            
    return final_fused

# 7. Judicial Discovery Logic (Hybrid Recall)
# 5. Background Recording Worker (Zero Frame Loss)
def recording_worker(frame_queue, output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    recorder = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"JUDICIAL TAPE START: {output_path}")
    
    while True:
        try:
            item = frame_queue.get(timeout=2)
            if isinstance(item, str) and item == "STOP": break
            recorder.write(item)
        except queue.Empty:
            continue
    
    recorder.release()
    print(f"JUDICIAL TAPE SEALED: {output_path}")

def process_yolo_only(frame, timestamp, is_video=True):
    annotated = frame.copy()
    
    # Select Forensic Profile
    thresholds = VIDEO_THRESHOLDS if is_video else IMAGE_THRESHOLDS
    def_thresh = VIDEO_DEFAULT_THRESH if is_video else IMAGE_DEFAULT_THRESH
    
    # --- STAGE 7.2: HYPER-RESOLUTION HYBRID SWEEP ---
    # We upscale to 800px to capture thin ropes and partial bodies
    res_audit = model_audit(frame, conf=0.01, iou=0.25, imgsz=800, verbose=False)[0]
    res_legacy = model_legacy(frame, conf=0.01, iou=0.25, imgsz=800, verbose=False)[0]
    
    raw_hits = []
    
    # Process Audit Hits (v11)
    for box in res_audit.boxes:
        conf, cls_id = float(box.conf[0]), int(box.cls[0])
        label = res_audit.names[cls_id]
        if conf > thresholds.get(label, def_thresh):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "HUMAN BODY" if label in ["Human-body", "Victim"] else label
            raw_hits.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2], "model": "AUDIT"})

    # Process Legacy Hits (v5)
    for box in res_legacy.boxes:
        conf, cls_id = float(box.conf[0]), int(box.cls[0])
        label = res_legacy.names[cls_id]
        if conf > thresholds.get(label, def_thresh):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = "HUMAN BODY" if label in ["Human-body", "Victim"] else label
            raw_hits.append({"label": label, "conf": conf, "box": [x1, y1, x2, y2], "model": "LEGACY"})

    # --- APPLY NEURAL FUSION ---
    # Merges any overlapping hits from the two different models
    fused_hits = fuse_detections(raw_hits)
    detections = []
    
    for det in fused_hits:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['conf']
        
        # Red for critical evidence, Green for others
        color = (0, 0, 255) if label in CRITICAL_CLASSES or label == 'HUMAN BODY' else (0, 255, 0)
        
        # Thinner lines for clarity (V6.8)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label.upper()} {conf:.2f}", (x1, y1-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # IVEL UPGRADE: Extra zoomed-in crop
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
            
    return detections, annotated

def generate_multimodal_brief_sync(frame, detections):
    if not detections: return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    prompt = (
        f"Act as a Forensic Judicial Specialist. Analyze this frame carefully. \n"
        f"1. FORNSIC OCR: Extract all text (Plates, signs, markers, labels).\n"
        f"2. FLUID ANALYTICS: Specifically audit for any blood stains, spatter markers, or fluid traces missing from these detections: {detections}.\n"
        f"3. SUMMARY: Briefly state the discovery for a legal log."
    )
    
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL, 
            contents=[prompt, pil_img]
        )
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return "Insight unavailable."

async def insight_worker(websocket, queue):
    """Background Judicial Reasoner: Processes frames and sends insights without blocking the stream."""
    print("INSIGHT WORKER: Active and awaiting forensic targets...")
    while True:
        try:
            # Wait for next forensic target
            frame, detections, timestamp = await queue.get()
            
            # Perform multimodal analysis in executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            brief = await loop.run_in_executor(executor, generate_multimodal_brief_sync, frame, detections)
            
            # Generate thumbnail for the historic feed (Post-process)
            s = cv2.resize(frame, (320, 180))
            _, buf = cv2.imencode('.jpg', s)
            thumbnail = base64.b64encode(buf).decode('utf-8')
            
            # Send the brief back immediately as a separate judicial update
            await websocket.send_text(json.dumps({
                "status": "INSIGHT",
                "timestamp": timestamp,
                "legal_brief": brief,
                "thumbnail": thumbnail,
                "detections": detections # Resending detections for UI context
            }))
            
            queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"INSIGHT WORKER FAILURE: {e}")
            await asyncio.sleep(1)

# 7. WebSocket Scanner
@app.websocket("/ws/discover")
async def websocket_discovery(websocket: WebSocket):
    await websocket.accept()
    video_path = websocket.query_params.get("video_path")
    if not video_path:
        await websocket.close(); return

    # --- JUDICIAL STATE INITIALIZATION ---
    print(f"COMMENCING TAPE-LOCK DISCOVERY: {video_path}")
    
    # Judicial Stream Handshake (V7.9)
    video_path = resolve_judicial_stream(video_path)
    
    # V7.4: Automatic Source Detection
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    _, ext = os.path.splitext(video_path.lower())
    is_video = ext not in img_exts
    print(f"SOURCE TYPE: {'VIDEO SCAN' if is_video else 'IMAGE AUDIT'}")
    
    streamer = ThreadedStreamer(video_path)
    frame_count = 0
    last_brief_time = 0
    recorder = None
    
    # --- RECORDER INITIALIZATION (JUDICIAL RECORD) ---
    processed_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "processed_evidence")
    os.makedirs(processed_dir, exist_ok=True)
    session_id = f"discovery_{int(time.time())}.avi"
    video_out_path = os.path.join(processed_dir, session_id)

    # Initialize frame queue for zero-loss recording
    frame_queue = queue.Queue(maxsize=128)
    h_init, w_init = 640, 480 # Fallback
    
    # --- START BACKGROUND WORKERS ---
    worker_task = asyncio.create_task(insight_worker(websocket, insight_queue))
    
    recording_thread = None
    
    try:
        while True:
            data = streamer.read()
            if data == "COMPLETE":
                await websocket.send_text(json.dumps({"status": "COMPLETE"})); break
            if data is None:
                if streamer.error_message:
                    await websocket.send_text(json.dumps({
                        "status": "ERROR", 
                        "message": streamer.error_message
                    }))
                    break
                if streamer.stopped: break # Safety break
                await asyncio.sleep(0.01); continue
                
            frame, timestamp = data
            frame_count += 1
            
            # 1. ANALYZE FRAME (YOLO DETECTIONS) - INSTANT
            try:
                detections, annotated = process_yolo_only(frame, timestamp, is_video=is_video)
            except Exception as e:
                print(f"YOLO Discovery Failure: {e}")
                continue
            
            # 2. ASYNC RECORDING (ZERO LOSS)
            if recording_thread is None:
                h, w = frame.shape[:2]
                recording_thread = threading.Thread(
                    target=recording_worker, 
                    args=(frame_queue, video_out_path, 20.0, w, h)
                )
                recording_thread.start()
            
            frame_queue.put(annotated)
            
            # 3. ASYNCHRONOUS MULTIMODAL ANALYTICS (NON-BLOCKING)
            has_critical = any(d['label'] in CRITICAL_CLASSES for d in detections)
            
            if detections:
                # Deduplication logic (Triggered every 5 seconds OR on critical detection)
                if (has_critical and timestamp - last_brief_time > 5.0) or (frame_count % 120 == 0):
                    # Push to background worker without waiting
                    try:
                        insight_queue.put_nowait((frame.copy(), detections, timestamp))
                        last_brief_time = timestamp
                    except asyncio.QueueFull:
                        # Skip if queue is overwhelmed to maintain real-time integrity
                        pass
            
            # 4. DASHBOARD UPDATE (LIVE DISCOVERY SYNC) - HIGH SPEED
            if frame_count % 4 == 0 or frame_count == 1:
                _, buf = cv2.imencode('.jpg', annotated)
                frame_b64 = base64.b64encode(buf).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "frame": frame_b64,
                    "timestamp": timestamp,
                    "detections": detections,
                    "status": "DISCOVERING",
                    "session_record": session_id
                }))
            
            # Tiny sleep to yield for UI responsiveness
            await asyncio.sleep(0.001) 
            
    except WebSocketDisconnect:
        print("Judicial Discovery Aborted by user/network.")
    except Exception as e:
        print(f"CRITICAL DISCOVERY ERROR: {e}")
    finally:
        worker_task.cancel()
        if recording_thread:
            frame_queue.put("STOP")
            recording_thread.join()
        streamer.stop()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8989)
