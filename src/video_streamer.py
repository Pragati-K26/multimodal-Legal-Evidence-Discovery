import cv2
import threading
import queue
import time
import yt_dlp
import os
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageSequence

class ThreadedStreamer:
    def __init__(self, video_path, target_fps=10):
        self.video_path = video_path
        self.target_fps = target_fps
        self.frame_queue = queue.Queue(maxsize=256)
        self.stopped = False
        self.fps = 30 
        self.cumulative_frame_count = 0
        self.error_message = None
        
        # Start the grabber thread
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)
        self.thread.start()

    def _get_stream_url(self, path):
        if "youtube.com" in path or "youtu.be" in path:
            print(f"Resolving Judicial Stream: {path}")
            # Format 18 is a stable, single-container MP4 (360p) that doesn't require ffmpeg to merge
            ydl_opts = {
                'format': '18/best[ext=mp4]',
                'quiet': True,
                'no_warnings': True,
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(path, download=False)
                    url = info.get('url', path)
                    print(f"Handshake Successful. Buffered URL obtained.")
                    return url
            except Exception as e:
                print(f"Handshake Failure: {e}")
                return path
        return path

    def _grab_frames(self):
        """Sequential Multi-Source Handshake with Content-Type awareness"""
        try:
            url_lower = self.video_path.lower()
            
            # --- PHASE 1: DYNAMIC HEADER PROBING ---
            is_image = False
            is_gif = False
            
            if self.video_path.startswith('http'):
                try:
                    head = requests.head(self.video_path, timeout=5, allow_redirects=True)
                    content_type = head.headers.get('Content-Type', '').lower()
                    if 'image/gif' in content_type:
                        is_gif = True
                    elif 'image/' in content_type:
                        is_image = True
                except:
                    pass # Fallback to extension check if HEAD fails

            # --- PHASE 2: EXTENSION FALLBACK ---
            if not is_image and not is_gif:
                path_strip = url_lower.split('?')[0]
                is_image = any(path_strip.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp'])
                is_gif = path_strip.endswith('.gif')

            # --- CASE 1: STATIC IMAGE OR ANIMATED GIF ---
            if is_image or is_gif:
                print(f"Initializing Asset Discovery: {self.video_path}")
                if self.video_path.startswith('http'):
                    response = requests.get(self.video_path, timeout=10)
                    data = BytesIO(response.content)
                else:
                    data = self.video_path
                
                img = Image.open(data)
                
                if is_image:
                    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    self.frame_queue.put((frame, 0.0))
                    self.stopped = True
                else:
                    frames = ImageSequence.Iterator(img)
                    ts = 0.0
                    for f in frames:
                        if self.stopped: break
                        frame = cv2.cvtColor(np.array(f.convert('RGB')), cv2.COLOR_RGB2BGR)
                        self.frame_queue.put((frame, ts))
                        ts += 0.1
                        time.sleep(0.05)
                    self.stopped = True
                return

            # --- CASE 2: VIDEO OR LIVE STREAM ---
            stream_source = self._get_stream_url(self.video_path)
            
            cap = cv2.VideoCapture(stream_source)
            if not cap.isOpened():
                if "tenor.com/search" in url_lower or "google.com/search" in url_lower:
                    self.error_message = "Search pages are not supported. Please provide a direct link to a single GIF or Video."
                else:
                    self.error_message = "Handshake failed. Verify the URL is a direct media link."
                print(f"CRITICAL: {self.error_message}")
                self.stopped = True
                return

            source_fps = cap.get(cv2.CAP_PROP_FPS)
            self.fps = source_fps if source_fps > 0 else 30
            print(f"Eyes Open. Stream Calibrated at {self.fps} FPS.")

            retries = 0
            MAX_SOCKET_RETRIES = 5

            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    if retries < MAX_SOCKET_RETRIES:
                        print(f"Judicial Socket Failure. Attempting re-resolution and reconnection ({retries+1}/{MAX_SOCKET_RETRIES})...")
                        time.sleep(1.5) # Allow network buffer to clear
                        # Re-handshake with a FRESH signed URL
                        cap.release()
                        stream_source = self._get_stream_url(self.video_path)
                        cap = cv2.VideoCapture(stream_source)
                        
                        # Note: Seek by frames is unreliable on streams, but we try a best-effort 
                        # to skip forward or just continue from the new buffer head.
                        if self.cumulative_frame_count > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, self.cumulative_frame_count)
                        
                        retries += 1
                        continue
                    else:
                        print("End of discovery tape reached or permanent socket failure.")
                        self.stopped = True
                        break
                
                retries = 0 # Reset on success
                msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                timestamp = (msec / 1000.0) if msec > 0 else (self.cumulative_frame_count / self.fps)
                self.cumulative_frame_count += 1
                
                # Judicial Sequence: No frame loss allowed for batch processing
                self.frame_queue.put((frame, timestamp))
                        
            cap.release()
                     
        except Exception as e:
            print(f"Vision Failure: {e}")
            self.stopped = True

    def read(self):
        if self.frame_queue.empty() and self.stopped:
            return "COMPLETE"
        if self.frame_queue.empty():
            return None
        return self.frame_queue.get()

    def stop(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=2)
