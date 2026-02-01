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
        self.queue = asyncio.Queue(maxsize=1) # Limit queue to keep stream real-time
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_loop())

    async def _main_loop(self):
        # it auto-reconnects
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
        """Thread-safe method to push frames from the sync loop."""
        if self.loop.is_running():
            # If queue is full, discard old frame to prioritize current one
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
running = True

def verify_face_worker(embedding_list, face_key):
    """Background thread to handle server communication."""
    global recognition_results
    try:
        payload = {"embedding": embedding_list, "camera_id": "Pro_Cam_01"}
        # Increased timeout for Neon DB latency
        response = requests.post(API_URL, json=payload, timeout=15)

        if response.status_code == 200:
            data = response.json()
            name = data.get("person_name", "Unknown")
            color = (0, 255, 0) if data.get("status") in ["success", "ignored"] else (0, 0, 255)

            with results_lock:
                # Store result for 10 seconds of persistence
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
        if latest_frame is None:
            time.sleep(0.01)
            continue

        with state_lock:
            img_copy = latest_frame.copy()

        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        current_time = time.time()

        for face in faces:
            bbox = face.bbox.astype(int)
            # Create a spatial key (X_Y) to track the result even as the object refreshes
            face_key = f"{bbox[0]//40}_{bbox[1]//40}"

            with results_lock:
                # If we don't have a result or it's expired, trigger API
                if face_key not in recognition_results or current_time > recognition_results[face_key]['expiry']:
                    if (current_time - last_api_call) > 3.0:
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
        frame = cv2.flip(frame, 1)

        with state_lock:
            latest_frame = frame
            faces_to_draw = detected_faces

        # Drawing logic for UI...
        for face in faces_to_draw:
            bbox = face.bbox.astype(int)
            face_key = f"{bbox[0]//40}_{bbox[1]//40}"
            name, color = "Scanning...", (0, 255, 255)

            with results_lock:
                if face_key in recognition_results:
                    res = recognition_results[face_key]
                    if time.time() < res['expiry']:
                        name, color = res['name'], res['color']

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # STREAMING ENGINE: Send processed frame to Web Relay
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ws_client.send_frame(buffer.tobytes())

        cv2.imshow('Face Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_camera()