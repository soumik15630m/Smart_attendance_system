import argparse
import os
import warnings

import cv2
import numpy as np
import onnxruntime as ort
import requests
from dotenv import load_dotenv
from insightface.app import FaceAnalysis

load_dotenv()

SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")
REG_URL = f"http://{SERVER_IP}:{SERVER_PORT}/persons/register"

# Registration quality gates. A single-snapshot embedding is fragile: one
# so-so angle or slight blur becomes that person's permanent match
# template. Instead of asking for a fixed sample count, we keep accepting
# samples until the running average template stops changing meaningfully
# (cosine similarity between successive averages clears CONVERGENCE_COSINE),
# within a sane min/max so it can't stop too early or run forever.
MIN_SAMPLES = int(os.getenv("MIN_SAMPLES", "3"))
MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "8"))
CONVERGENCE_COSINE = float(os.getenv("CONVERGENCE_COSINE", "0.999"))
MIN_DET_SCORE = float(os.getenv("MIN_DET_SCORE", "0.65"))

# Blur gate calibrates to this session's own footage instead of one fixed
# number: once we've seen a reasonably sharp frame, later frames need to
# clear a fraction of the best sharpness observed so far.
BLUR_FLOOR = float(os.getenv("BLUR_FLOOR", "25.0"))
BLUR_RELATIVE_FRACTION = float(os.getenv("BLUR_RELATIVE_FRACTION", "0.5"))

default_cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
cuda_bin = os.getenv("CUDA_PATH_BIN", default_cuda_path)

if os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(cuda_bin)
        except Exception:  # noqa: BLE001, S110 - CUDA path is optional
            pass

print("--------------------------------------------------")
print("FACE REGISTRATION CLIENT")
print(f"Target Server: {REG_URL}")
print("--------------------------------------------------")

warnings.filterwarnings("ignore")

# Must match camera_client.py's model choice exactly. Different insightface
# model packs produce embeddings in different vector spaces, so registering
# with one model and matching against another silently breaks recognition
# (cosine distances become meaningless). We apply the same auto-selection
# logic as camera_client.py (GPU -> buffalo_l, CPU-only -> buffalo_s) so
# that, absent an explicit override, both scripts agree without the user
# having to keep two env vars in sync by hand.
try:
    _providers = ort.get_available_providers()
except Exception:  # noqa: BLE001 - falls back to CPU model choice
    _providers = []
_auto_model = "buffalo_l" if "CUDAExecutionProvider" in _providers else "buffalo_s"
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "") or _auto_model

app = FaceAnalysis(
    name=FACE_MODEL_NAME, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))


def get_blur_score(gray_crop):
    if gray_crop.size == 0:
        return 0.0
    return cv2.Laplacian(gray_crop, cv2.CV_64F).var()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register one face to the attendance API."
    )
    parser.add_argument("--name", type=str, help="Person name to register.")
    parser.add_argument("--employee-id", type=str, help="Employee ID to assign.")
    return parser.parse_args()


def enroll(name_override: str | None = None, employee_id_override: str | None = None):
    # Gather User Info
    if name_override:
        name = name_override.strip()
        print(f"Using provided name: {name}")
    else:
        name = input("Enter Person Name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    if employee_id_override is not None:
        emp_id = employee_id_override.strip()
        if emp_id:
            print(f"Using provided employee ID: {emp_id}")
    else:
        emp_id = input("Enter Employee ID: ").strip()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n[Controls]")
    print(
        f"  's' -> Capture sample (min {MIN_SAMPLES}, up to {MAX_SAMPLES}; vary angle slightly)"
    )
    print("  'q' -> Quit")

    collected_embeddings: list[np.ndarray] = []
    running_average = None
    best_blur_seen = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        faces = app.get(frame)
        status_color = (0, 0, 255)
        status_text = "No Face Detected"
        ready_to_capture = False

        if len(faces) == 1:
            face = faces[0]
            bbox = face.bbox.astype(int)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = [max(0, v) for v in bbox]
            blur_score = get_blur_score(gray[y1:y2, x1:x2])
            best_blur_seen = max(best_blur_seen, blur_score)
            # Requires clearing both an absolute floor (guards against a
            # uniformly blurry session) and a fraction of the sharpest
            # frame seen so far this session (adapts to this camera/lighting).
            blur_gate = max(BLUR_FLOOR, BLUR_RELATIVE_FRACTION * best_blur_seen)

            if face.det_score < MIN_DET_SCORE:
                status_color = (0, 255, 255)
                status_text = "Move closer / face camera more directly"
            elif blur_score < blur_gate:
                status_color = (0, 255, 255)
                status_text = "Too blurry - hold still"
            else:
                status_color = (0, 255, 0)
                status_text = (
                    f"Ready (Press 's')  [{len(collected_embeddings)}/{MIN_SAMPLES}+]"
                )
                ready_to_capture = True

            cv2.rectangle(
                display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), status_color, 2
            )
        elif len(faces) > 1:
            status_color = (0, 255, 255)
            status_text = "Too many faces! Only 1 allowed."

        cv2.putText(
            display_frame,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        cv2.imshow("Registration", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and ready_to_capture:
            new_embedding = faces[0].embedding.astype(np.float64)
            collected_embeddings.append(new_embedding)
            sample_count = len(collected_embeddings)

            previous_average = running_average
            running_average = np.mean(collected_embeddings, axis=0)

            print(f"  Captured sample {sample_count} (max {MAX_SAMPLES})")

            converged = False
            if previous_average is not None and sample_count >= MIN_SAMPLES:
                # Cosine similarity between the template before and after
                # this sample tells us whether the average is still moving
                # meaningfully or has settled - i.e. more samples would add
                # noise rather than signal.
                denom = np.linalg.norm(previous_average) * np.linalg.norm(
                    running_average
                )
                similarity = (
                    float(np.dot(previous_average, running_average) / denom)
                    if denom > 0
                    else 1.0
                )
                converged = similarity >= CONVERGENCE_COSINE

            if converged or sample_count >= MAX_SAMPLES:
                # Re-normalize to unit length so the result stays on the
                # same hypersphere the individual (already L2-normalized)
                # embeddings live on - this keeps cosine distance
                # comparisons well-behaved downstream.
                norm = np.linalg.norm(running_average)
                final_embedding = (
                    running_average / norm if norm > 0 else running_average
                )

                payload = {
                    "name": name,
                    "employee_id": emp_id if emp_id else None,
                    "role": "Employee",
                    "embedding": final_embedding.tolist(),
                }

                print(
                    f"Sending averaged template ({sample_count} samples) to server..."
                )
                try:
                    headers = {"X-API-Key": os.getenv("API_KEY", "")}
                    response = requests.post(
                        REG_URL, json=payload, headers=headers, timeout=10
                    )

                    if response.status_code == 200:
                        print(
                            f" Success! {name} registered from {sample_count} samples."
                        )
                    elif response.status_code == 401:
                        print(
                            " Failed: 401 Unauthorized. Check that API_KEY in your "
                            ".env matches the server's API_KEY."
                        )
                    else:
                        print(f" Failed: {response.text}")
                except Exception as e:  # noqa: BLE001 - any request failure is reported
                    print(f" Connection Error: {e}")
                break

        elif key == ord("q"):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cli_args = parse_args()
    enroll(name_override=cli_args.name, employee_id_override=cli_args.employee_id)
