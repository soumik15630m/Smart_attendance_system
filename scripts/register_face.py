import os
import cv2
import requests
import warnings
from insightface.app import FaceAnalysis

SERVER_IP = os.getenv("SERVER_IP", "127.0.0.1")
SERVER_PORT = os.getenv("SERVER_PORT", "8000")
REG_URL = f"http://{SERVER_IP}:{SERVER_PORT}/persons/register"

default_cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
cuda_bin = os.getenv("CUDA_PATH_BIN", default_cuda_path)

if os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(cuda_bin)
        except Exception:  # Use Exception instead of a bare except
            pass

print("--------------------------------------------------")
print("FACE REGISTRATION CLIENT")
print(f"Target Server: {REG_URL}")
print("--------------------------------------------------")

# Suppress ONNX warnings
warnings.filterwarnings("ignore")

app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def enroll():
    # Gather User Info
    name = input("Enter Person Name: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    emp_id = input("Enter Employee ID: ").strip()

    # Start Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n[Controls]")
    print("  's' -> Save Snapshot & Register")
    print("  'q' -> Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        faces = app.get(frame)
        status_color = (0, 0, 255) # Red (No Face)
        status_text = "No Face Detected"

        if len(faces) == 1:
            status_color = (0, 255, 0) # Green (Good)
            status_text = "Ready to Register (Press 's')"
            bbox = faces[0].bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), status_color, 2)
        elif len(faces) > 1:
            status_color = (0, 255, 255) # Yellow (Too Many)
            status_text = "Too many faces! Only 1 allowed."

        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.imshow("Registration", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) == 1:
            face = faces[0]
            payload = {
                "name": name,
                "employee_id": emp_id if emp_id else None,
                "role": "Employee",
                "embedding": face.embedding.tolist()
            }

            print("Sending data to server...")
            try:
                headers = {"X-API-Key": os.getenv("API_KEY", "")}
                response = requests.post(REG_URL, json=payload, headers=headers, timeout=10)

                if response.status_code == 200:
                    print(f" Success! {name} registered.")
                    break
                else:
                    print(f" Failed: {response.text}")
            except Exception as e:
                print(f" Connection Error: {e}")

        elif key == ord('q'):
            print("Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll()