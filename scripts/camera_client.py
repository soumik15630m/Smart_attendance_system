import asyncio
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import websockets
from dotenv import load_dotenv

load_dotenv()

SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))

MIN_BRIGHTNESS = int(os.getenv("MIN_BRIGHTNESS", "60"))
MIN_DET_SCORE = float(os.getenv("MIN_DET_SCORE", "0.65"))

# --- Tracking / smoothing tuning -------------------------------------------------
# These control how a detected face is turned into a stable "track" across
# frames, instead of re-identifying it from scratch every frame based on a
# raw grid position (which is what caused the label/box jitter).
#
# Everything below is expressed as a ratio or a time duration rather than a
# fixed pixel/frame count, so the same defaults behave sensibly whether the
# camera is 480p or 4K, and whether it's running at 12fps on CPU or 60fps on
# GPU. Where a value genuinely can't be derived from something measurable,
# it's still exposed as an env var instead of buried as a magic number.
IOU_MATCH_THRESHOLD = float(os.getenv("IOU_MATCH_THRESHOLD", "0.3"))
TRACK_TIMEOUT = float(os.getenv("TRACK_TIMEOUT_SECONDS", "1.0"))
BBOX_SMOOTHING_ALPHA = float(os.getenv("BBOX_SMOOTHING_ALPHA", "0.35"))

# A face must hold still for this long (not a fixed frame count) before
# we'll trust it enough to attempt recognition. Converted to an actual
# frame count at runtime using the camera's *measured* fps.
STABILITY_SECONDS = float(os.getenv("STABILITY_SECONDS", "0.18"))

# Movement tolerance as a fraction of the face's own width, so a face that
# fills a small part of the frame isn't held to the same pixel tolerance as
# one that fills most of it.
MOVEMENT_TOLERANCE_RATIO = float(os.getenv("MOVEMENT_TOLERANCE_RATIO", "0.10"))

# Minimum face size as a fraction of frame width rather than raw pixels, so
# it doesn't need retuning if the camera resolution changes.
MIN_FACE_WIDTH_RATIO = float(os.getenv("MIN_FACE_WIDTH_RATIO", "0.06"))

# Sharpness (blur) gate: rather than one fixed Laplacian-variance number,
# we track a running mean/std of the scores this camera actually produces
# and require a candidate frame to sit comfortably above that camera's own
# baseline. See RollingStats below.
BLUR_FLOOR = float(os.getenv("BLUR_FLOOR", "25.0"))
BLUR_STD_MARGIN = float(os.getenv("BLUR_STD_MARGIN", "0.5"))
BLUR_WARMUP_SAMPLES = int(os.getenv("BLUR_WARMUP_SAMPLES", "20"))

# Cooldown before re-querying an unlocked track backs off exponentially on
# repeated "unknown" results (no point hammering a face the server keeps
# not recognizing), and resets once it locks in a match.
RETRY_BASE_SECONDS = float(os.getenv("RETRY_BASE_SECONDS", "1.5"))
RETRY_MAX_SECONDS = float(os.getenv("RETRY_MAX_SECONDS", "10.0"))

# Pick a recognition model automatically based on available compute unless
# the user pins one explicitly: the larger backbone is worth it on GPU, but
# can be a real drag on CPU-only boxes.
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "")

API_URL = f"http://{SERVER_IP}:{SERVER_PORT}/attendance/identify"
WS_URL = f"ws://{SERVER_IP}:{SERVER_PORT}/ws/video-input"

cuda_bin = os.getenv("CUDA_PATH_BIN", "")

if cuda_bin and os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

    if sys.platform == "win32":
        add_dll = getattr(os, "add_dll_directory", None)
        if add_dll:
            try:
                add_dll(cuda_bin)
            except Exception:  # noqa: BLE001, S110 - CUDA path is optional
                pass

warnings.filterwarnings("ignore")
import onnxruntime as ort  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402


class ThreadedCamera:
    """Read frames on a background thread."""

    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
            except Exception:  # noqa: BLE001 - any connection error triggers reconnect
                await asyncio.sleep(2)

    def send_frame(self, frame_bytes: bytes) -> None:
        if self.loop.is_running():
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except Exception:  # noqa: BLE001, S110 - queue already drained
                    pass
            self.loop.call_soon_threadsafe(self.queue.put_nowait, frame_bytes)


print("[-] Loading AI Models...")

provider_list = ["CPUExecutionProvider"]
ctx_id = -1
det_size = (416, 416)
mode_name = "CPU OPTIMIZED"
auto_model = "buffalo_s"

try:
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("    CUDA Detected! Attempting to initialize GPU mode...")
        provider_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0
        det_size = (640, 640)
        mode_name = "GPU PURE PRECISION"
        auto_model = "buffalo_l"
    else:
        print("    CUDA Not Found. Using Multi-threaded CPU mode.")

except Exception as e:  # noqa: BLE001 - any provider-check failure falls back to CPU
    print(f"    Error checking CUDA: {e}. Falling back to CPU.")

# Larger backbone (buffalo_l) when we've got GPU headroom, smaller
# (buffalo_s) on CPU-only hardware where every millisecond counts. An
# explicit FACE_MODEL_NAME env var always wins.
resolved_model_name = FACE_MODEL_NAME or auto_model

print(f"[-] Mode: {mode_name} | Resolution: {det_size} | Model: {resolved_model_name}")

app = FaceAnalysis(name=resolved_model_name, providers=provider_list)
app.prepare(ctx_id=ctx_id, det_size=det_size)
print(" AI Ready.")

ws_client = AsyncWebSocketClient(WS_URL)

latest_frame = None
frame_lock = threading.Lock()
running = True

tracks: dict = {}
tracks_lock = threading.Lock()
_next_track_id = 0

# Bounds worst-case thread/resource usage for recognition API calls instead
# of spawning one unmanaged thread per call. Submissions are now gated per
# track (only stable, sharp, high-confidence crops get sent) rather than by
# a single global timer, so 2 workers is still ample headroom.
recognition_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="recog")


def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def get_blur_score(gray_crop):
    """Higher = sharper. Motion blur / out-of-focus frames score low."""
    if gray_crop.size == 0:
        return 0.0
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var()


def iou(box_a, box_b):
    xa, ya = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xb, yb = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class RollingStats:
    """Streaming mean/variance (Welford's algorithm), used to turn a fixed
    "magic number" threshold into one that calibrates itself to whatever
    camera and lighting this is actually running under.
    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, value):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def std(self):
        if self.n < 2:
            return 0.0
        return (self.m2 / (self.n - 1)) ** 0.5


class FpsMeter:
    """EMA of the interval between calls, so time-based settings (e.g. "hold
    still for 180ms") can be converted into a frame count that adapts to
    however fast this particular machine/camera actually runs.
    """

    def __init__(self, alpha=0.1, assumed_fps=15.0):
        self._alpha = alpha
        self._avg_interval = 1.0 / assumed_fps
        self._last_tick = None

    def tick(self):
        now = time.time()
        if self._last_tick is not None:
            interval = now - self._last_tick
            if 0 < interval < 1.0:
                self._avg_interval = (
                    self._alpha * interval + (1 - self._alpha) * self._avg_interval
                )
        self._last_tick = now

    @property
    def fps(self):
        return 1.0 / self._avg_interval if self._avg_interval > 0 else 15.0

    def frames_for(self, seconds):
        return max(1, round(seconds * self.fps))


blur_stats = RollingStats()
fps_meter = FpsMeter()


class FaceTrack:
    """Persistent identity for a face across frames.

    Replaces the old "grid cell as identity" approach, which reset state
    (and therefore the on-screen label) every time a face's center crossed
    a 50px grid line. A track survives small movement, so the label and box
    stop flickering, and we only bother calling the recognition API once
    the track has proven it's holding still and looks sharp.
    """

    def __init__(self, track_id, bbox):
        self.id = track_id
        self.smoothed_bbox = np.array(bbox, dtype=float)
        self.center = self._center(bbox)
        self.stable_frames = 0
        self.last_seen = time.time()

        self.name = "Scanning..."
        self.color = (0, 255, 255)
        self.locked = False  # True once confidently identified; stops re-querying
        self.last_attempt = 0.0
        self.consecutive_misses = 0  # drives exponential retry backoff
        self._matched_face = None  # last detected face object matched to this track

    @staticmethod
    def _center(bbox):
        return np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])

    def update(self, bbox):
        new_center = self._center(bbox)
        moved = np.linalg.norm(new_center - self.center)

        # Tolerance scales with the face's own size: a face that's 400px
        # wide can wobble more in absolute pixels than a 60px-wide one and
        # still be "the same amount" of jitter.
        face_width = max(1.0, bbox[2] - bbox[0])
        tolerance = MOVEMENT_TOLERANCE_RATIO * face_width

        self.smoothed_bbox = (
            BBOX_SMOOTHING_ALPHA * np.array(bbox, dtype=float)
            + (1 - BBOX_SMOOTHING_ALPHA) * self.smoothed_bbox
        )
        self.center = new_center
        self.last_seen = time.time()

        if moved <= tolerance:
            cap = fps_meter.frames_for(STABILITY_SECONDS) * 4
            self.stable_frames = min(self.stable_frames + 1, cap)
        else:
            # Real movement (someone walking through frame) resets the
            # "ready to identify" counter but does NOT wipe an already
            # locked-in name, so the label doesn't drop out mid-motion.
            self.stable_frames = 0

    def is_stable(self):
        return self.stable_frames >= fps_meter.frames_for(STABILITY_SECONDS)

    def can_attempt(self, now):
        if self.locked:
            return False
        backoff = min(
            RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**self.consecutive_misses)
        )
        return (now - self.last_attempt) > backoff

    def blur_is_acceptable(self, score):
        """Accept once we have enough history to judge; otherwise fall back
        to a conservative floor so we're not blind on the very first faces
        of a session.
        """
        if blur_stats.n < BLUR_WARMUP_SAMPLES:
            return score >= BLUR_FLOOR
        return score >= max(
            BLUR_FLOOR, blur_stats.mean - BLUR_STD_MARGIN * blur_stats.std
        )


def verify_face_worker(embedding_list, track_id):
    try:
        headers = {"X-API-Key": os.getenv("API_KEY", "")}
        payload = {"embedding": embedding_list, "camera_id": "Pro_Cam_01"}

        response = requests.post(API_URL, json=payload, headers=headers, timeout=5)

        if response.status_code == 200:
            data = response.json()
            name = data.get("person_name", "Unknown")
            status = data.get("status")
            recognized = status in ("success", "ignored")
            color = (0, 255, 0) if recognized else (0, 0, 255)

            with tracks_lock:
                track = tracks.get(track_id)
                if track is not None:
                    track.name = name
                    track.color = color
                    # Lock on any confirmed identity match so the label
                    # stops re-querying (and thus stops flickering) for the
                    # rest of this person's time in frame. "unknown" is
                    # left unlocked so it naturally retries after a
                    # backoff that grows with repeated misses, e.g. if the
                    # first attempt caught a bad angle.
                    if recognized:
                        track.locked = True
                        track.consecutive_misses = 0
                    else:
                        track.consecutive_misses += 1
        elif response.status_code == 401:
            print(
                "[!] 401 Unauthorized from server. Check that API_KEY in your "
                ".env matches the server's API_KEY."
            )
    except Exception:  # noqa: BLE001 - request failure counts as a miss, not a crash
        with tracks_lock:
            track = tracks.get(track_id)
            if track is not None:
                track.consecutive_misses += 1


def ai_worker():
    global _next_track_id
    while running:
        img_copy = latest_frame_snapshot()

        if img_copy is None:
            time.sleep(0.01)
            continue

        fps_meter.tick()

        brightness = get_brightness(img_copy)
        if brightness < MIN_BRIGHTNESS:
            with tracks_lock:
                tracks.clear()
            time.sleep(0.1)
            continue

        frame_width = img_copy.shape[1]
        min_face_width = MIN_FACE_WIDTH_RATIO * frame_width

        gray_frame = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        now = time.time()

        valid_faces = [
            f
            for f in faces
            if f.det_score >= MIN_DET_SCORE
            and (f.bbox[2] - f.bbox[0]) >= min_face_width
        ]

        with tracks_lock:
            # Drop tracks nothing has matched to in a while.
            for tid in [
                t for t, tr in tracks.items() if now - tr.last_seen > TRACK_TIMEOUT
            ]:
                del tracks[tid]

            unmatched_faces = list(valid_faces)
            for track in tracks.values():
                if not unmatched_faces:
                    break
                best_face, best_iou = None, IOU_MATCH_THRESHOLD
                for face in unmatched_faces:
                    score = iou(track.smoothed_bbox, face.bbox)
                    if score > best_iou:
                        best_face, best_iou = face, score
                if best_face is not None:
                    track.update(best_face.bbox)
                    track._matched_face = best_face
                    unmatched_faces.remove(best_face)
                else:
                    track._matched_face = None

            for face in unmatched_faces:
                _next_track_id += 1
                new_track = FaceTrack(_next_track_id, face.bbox)
                new_track._matched_face = face
                tracks[new_track.id] = new_track

            # Decide which stable/sharp tracks are worth sending to the
            # recognition API right now. This is the "one good frame is
            # enough" gate: only fire once a face has held still for
            # roughly STABILITY_SECONDS (converted to frames via measured
            # fps) AND the crop is sharp relative to this camera's own
            # recent sharpness baseline, instead of an arbitrary global 1s
            # timer that could catch any frame.
            for track in tracks.values():
                face = getattr(track, "_matched_face", None)
                if face is None or not track.is_stable() or not track.can_attempt(now):
                    continue

                x1, y1, x2, y2 = [int(max(0, v)) for v in face.bbox]
                crop = gray_frame[y1:y2, x1:x2]
                blur_score = get_blur_score(crop)
                blur_stats.update(blur_score)
                if not track.blur_is_acceptable(blur_score):
                    continue

                track.last_attempt = now
                recognition_executor.submit(
                    verify_face_worker, face.embedding.tolist(), track.id
                )


def latest_frame_snapshot():
    with frame_lock:
        return None if latest_frame is None else latest_frame.copy()


def start_camera():
    global latest_frame, running

    print("[-] Starting Threaded Camera...")
    cam = ThreadedCamera(CAMERA_INDEX).start()

    threading.Thread(target=ai_worker, daemon=True).start()

    print("[-] System Online. Press 'q' to exit.")

    while True:
        ret, frame = cam.read()

        if not ret or frame is None:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame

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

        with tracks_lock:
            tracks_to_draw = [
                (t.smoothed_bbox.copy(), t.name, t.color) for t in tracks.values()
            ]

        # Drawn from the smoothed (EMA) bbox rather than the raw per-frame
        # detection box, so the rectangle glides instead of jittering with
        # every small detector wobble.
        for bbox, name, color in tracks_to_draw:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis_frame,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

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
