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
from dotenv import load_dotenv

# 1. Load Environment Variables
load_dotenv()

SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))

# Settings for V1 Quality Control
MIN_BRIGHTNESS = 60  # 0-255 (Below this = "Too Dark")
MIN_FACE_WIDTH = 60  # Pixels (Below this = "Too Far")
MIN_DET_SCORE = 0.60  # 0-1.0 (Below this = "Not a clear face")

API_URL = f"http://{SERVER_IP}:{SERVER_PORT}/attendance/identify"
WS_URL = f"ws://{SERVER_IP}:{SERVER_PORT}/ws/video-input"

# --- CUDA Setup (Platform Safe) ---
cuda_bin = os.getenv("CUDA_PATH_BIN", "")

if cuda_bin and os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

    if sys.platform == "win32":
        add_dll = getattr(os, "add_dll_directory", None)
        if add_dll:
            try:
                add_dll(cuda_bin)
            except Exception:
                pass

warnings.filterwarnings("ignore")
# noqa: E402 tells Ruff to ignore the import position check
from insightface.app import FaceAnalysis  # noqa: E402
import onnxruntime as ort  # noqa: E402


# --- Optimized Threaded Camera ---
class ThreadedCamera:
    """
    Reads camera frames in a separate thread to prevent I/O blocking
    the main UI loop. Critical for smooth performance.
    """

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        # Buffer size 1 ensuring we always get the *latest* frame, not an old buffered one
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.ret, self.frame = self.capture.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.capture.isOpened():
                self.stop()
                break

            ret, frame = self.capture.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                self.stop()

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return self.ret, None

    def stop(self):
        self.stopped = True
        self.capture.release()


# --- WebSocket Client ---
class AsyncWebSocketClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.loop = asyncio.new_event_loop()
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self) -> None:
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._main_loop())

    async def _main_loop(self) -> None:
        while True:
            try:
                async with websockets.connect(self.uri) as websocket:
                    print("Connected to Stream Relay")
                    while True:
                        frame_bytes = await self.queue.get()
                        await websocket.send(frame_bytes)
            except Exception:
                await asyncio.sleep(2)

    def send_frame(self, frame_bytes: bytes) -> None:
        if self.loop.is_running():
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
            self.loop.call_soon_threadsafe(self.queue.put_nowait, frame_bytes)


# --- AI Setup (Auto-Switching) ---
print("[-] Loading AI Models...")

# Default Settings (CPU Safe)
provider_list = ["CPUExecutionProvider"]
ctx_id = -1  # -1 = CPU
det_size = (320, 320)  # Low res for CPU speed
mode_name = "CPU OPTIMIZED"

try:
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("    CUDA Detected! Attempting to initialize GPU mode...")
        provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        det_size = (640, 640)
        mode_name = "GPU PURE PRECISION"
    else:
        print("    CUDA Not Found. Using Multi-threaded CPU mode.")

except Exception as e:
    print(f"    Error checking CUDA: {e}. Falling back to CPU.")

print(f"[-] Mode: {mode_name} | Resolution: {det_size}")

app = FaceAnalysis(name="buffalo_s", providers=provider_list)
app.prepare(ctx_id=ctx_id, det_size=det_size)
print(" AI Ready.")

ws_client = AsyncWebSocketClient(WS_URL)

# Global State
latest_frame = None
detected_faces: list = []
recognition_results: dict = {}
results_lock = threading.Lock()
state_lock = threading.Lock()
last_api_call = 0
running = True


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def verify_face_worker(embedding_list, face_key):
    global recognition_results
    try:
        headers = {"X-API-Key": os.getenv("API_KEY", "")}
        payload = {"embedding": embedding_list, "camera_id": "Pro_Cam_01"}

        response = requests.post(API_URL, json=payload, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            name = data.get("person_name", "Unknown")
            status = data.get("status")
            color = (0, 255, 0) if status in ["success", "ignored"] else (0, 0, 255)

            with results_lock:
                recognition_results[face_key] = {
                    "name": name,
                    "color": color,
                    "expiry": time.time() + 5.0,
                }
    except Exception:
        pass


def ai_worker():
    global detected_faces, last_api_call
    while running:
        with state_lock:
            if latest_frame is None:
                img_copy = None
            else:
                img_copy = latest_frame.copy()

        if img_copy is None:
            time.sleep(0.01)
            continue

        brightness = get_brightness(img_copy)
        if brightness < MIN_BRIGHTNESS:
            with state_lock:
                detected_faces = []
            time.sleep(0.1)
            continue

        # AI Inference (Heavy Task)
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        current_time = time.time()

        valid_faces = []

        for face in faces:
            bbox = face.bbox.astype(int)
            width = bbox[2] - bbox[0]

            if face.det_score < MIN_DET_SCORE:
                continue
            if width < MIN_FACE_WIDTH:
                continue

            valid_faces.append(face)

            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            face_key = f"{center_x // 50}_{center_y // 50}"

            with results_lock:
                if (
                    face_key not in recognition_results
                    or current_time > recognition_results[face_key]["expiry"]
                ):
                    if (current_time - last_api_call) > 1.0:
                        last_api_call = current_time
                        threading.Thread(
                            target=verify_face_worker,
                            args=(face.embedding.tolist(), face_key),
                            daemon=True,
                        ).start()

        with state_lock:
            detected_faces = valid_faces


def start_camera():
    global latest_frame, running

    # Use Threaded Camera instead of blocking cv2.VideoCapture
    print("[-] Starting Threaded Camera...")
    cam = ThreadedCamera(CAMERA_INDEX).start()

    # Start AI Thread
    threading.Thread(target=ai_worker, daemon=True).start()

    print("[-] System Online. Press 'q' to exit.")

    while True:
        # Non-blocking read (reads latest available frame from thread)
        ret, frame = cam.read()

        if not ret or frame is None:
            time.sleep(0.01)  # Wait slightly if camera hasn't initialized
            continue

        frame = cv2.flip(frame, 1)

        # Update global frame for AI worker
        with state_lock:
            latest_frame = frame
            # Quick copy for UI to minimize lock time
            faces_to_draw = list(detected_faces)

        vis_frame = frame.copy()

        brightness = get_brightness(vis_frame)
        if brightness < MIN_BRIGHTNESS:
            cv2.putText(
                vis_frame,
                "TOO DARK",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                3,
            )

        for face in faces_to_draw:
            bbox = face.bbox.astype(int)
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            face_key = f"{center_x // 50}_{center_y // 50}"

            name, color = "Scanning...", (0, 255, 255)

            with results_lock:
                if face_key in recognition_results:
                    res = recognition_results[face_key]
                    if time.time() < res["expiry"]:
                        name, color = res["name"], res["color"]

            cv2.rectangle(vis_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(
                vis_frame,
                name,
                (bbox[0], bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # Web Stream (Optional - only send if needed to save bandwidth)
        _, buffer = cv2.imencode(".jpg", vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        ws_client.send_frame(buffer.tobytes())

        cv2.imshow("Face Attendance Client (V2 Multi-Threaded)", vis_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            running = False
            cam.stop()
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_camera()
