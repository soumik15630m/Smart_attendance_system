import os
import sys
import time
import threading
import cv2
import numpy as np
import requests
import warnings
import asyncio
import websockets

# --- CUDA Setup ---
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    try: os.add_dll_directory(cuda_bin)
    except: pass

warnings.filterwarnings("ignore")
from insightface.app import FaceAnalysis

API_URL = "http://127.0.0.1:8000/attendance/identify"
WS_URL = "ws://127.0.0.1:8000/ws/video-input"
CAMERA_ID = 0

class AsyncWebSocketClient:
    def __init__(self, uri):
        self.uri = uri
        self.loop = asyncio.new_event_loop()
        self.queue = asyncio.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_loop())

    async def _main_loop(self):
        while True:
            try:
                async with websockets.connect(self.uri) as websocket:
                    print(f"Connected to Relay: {self.uri}")
                    while True:
                        frame_bytes = await self.queue.get()
                        await websocket.send(frame_bytes)
            except Exception as e:
                print(f"WebSocket Connection Lost. Retrying... ({e})")
                await asyncio.sleep(2)

    def send_frame(self, frame_bytes):
        if self.loop.is_running():
            if self.queue.full():
                try: self.queue.get_nowait()
                except: pass
            self.loop.call_soon_threadsafe(self.queue.put_nowait, frame_bytes)

app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

ws_client = AsyncWebSocketClient(WS_URL)

latest_frame = None
detected_faces = []
recognition_results = {}
results_lock = threading.Lock()
state_lock = threading.Lock()
last_api_call = 0  # Initialize variable
running = True

def verify_face_worker(embedding_list, face_key):
    global recognition_results
    try:
        payload = {"embedding": embedding_list, "camera_id": "Pro_Cam_01"}
        response = requests.post(API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            data = response.json()
            name = data.get("person_name", "Unknown")
            # Green for success/ignored, Red for unknown
            color = (0, 255, 0) if data.get("status") in ["success", "ignored"] else (0, 0, 255)

            with results_lock:
                recognition_results[face_key] = {
                    "name": name,
                    "color": color,
                    "expiry": time.time() + 10.0
                }
    except Exception as e:
        print(f"API Error for {face_key}: {e}")

def ai_worker():
    global detected_faces, last_api_call
    while running:
        # Access the shared frame safely
        with state_lock:
            if latest_frame is None:
                img_copy = None
            else:
                # We copy strictly here to avoid holding the lock during inference
                img_copy = latest_frame.copy()

        if img_copy is None:
            time.sleep(0.01)
            continue

        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        current_time = time.time()

        for face in faces:
            bbox = face.bbox.astype(int)
            # Use larger grid (e.g., 50px) or center point to reduce flickering
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            face_key = f"{center_x//50}_{center_y//50}"

            with results_lock:
                if face_key not in recognition_results or current_time > recognition_results[face_key]['expiry']:
                    # Debounce API calls (every 2.0 seconds globally for simplicity, or per face)
                    if (current_time - last_api_call) > 2.0:
                        last_api_call = current_time
                        threading.Thread(
                            target=verify_face_worker,
                            args=(face.embedding.tolist(), face_key),
                            daemon=True
                        ).start()

        with state_lock:
            detected_faces = faces

def start_camera():
    global latest_frame, running
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threading.Thread(target=ai_worker, daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Standardize input (Mirror)
        frame = cv2.flip(frame, 1)

        # 2. Update Shared State for AI (CLEAN FRAME ONLY)
        with state_lock:
            latest_frame = frame # Pass reference to the clean raw frame
            faces_to_draw = detected_faces # Get latest detections

        # 3. Create a Visualization Copy
        # CRITICAL FIX: We draw on a COPY, not the original frame.
        # This keeps 'latest_frame' clean for the AI thread.
        vis_frame = frame.copy()

        # 4. Draw UI on the Visualization Frame
        for face in faces_to_draw:
            bbox = face.bbox.astype(int)

            # Robust Key Generation (using Center Point)
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            face_key = f"{center_x//50}_{center_y//50}"

            name, color = "Scanning...", (0, 255, 255) # Default Yellow

            with results_lock:
                # Try exact match
                if face_key in recognition_results:
                    res = recognition_results[face_key]
                    if time.time() < res['expiry']:
                        name, color = res['name'], res['color']

            # Explicit integer casting for OpenCV to ensure box draws correctly
            start_point = (int(bbox[0]), int(bbox[1]))
            end_point = (int(bbox[2]), int(bbox[3]))

            cv2.rectangle(vis_frame, start_point, end_point, color, 2)
            cv2.putText(vis_frame, name, (int(bbox[0]), int(bbox[1])-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 5. Send VISUALIZATION frame to Web Relay & Local Display
        _, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ws_client.send_frame(buffer.tobytes())

        cv2.imshow('Face Attendance', vis_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()