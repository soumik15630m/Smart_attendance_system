import os
import cv2
import requests
import numpy as np
from insightface.app import FaceAnalysis


cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
if os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]
    try: os.add_dll_directory(cuda_bin)
    except: pass

REG_URL = "http://127.0.0.1:8000/persons/register"

app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

def enroll():
    print("--------------------------------------------------")
    print("FACE REGISTRATION SYSTEM")
    print("--------------------------------------------------")
    name = input("Enter Person Name: ")
    emp_id = input("Enter Employee ID: ")

    cap = cv2.VideoCapture(0)
    print("\nLook at the camera. Press 's' to take a snapshot, or 'q' to cancel.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # Quick detection for UI feedback
        faces = app.get(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        cv2.imshow("Registration - Snapshot", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(faces) > 0:
            # We use the embedding from the highest-score face
            face = faces[0]
            payload = {
                "name": name,
                "employee_id": emp_id,
                "role": "Employee",
                "embedding": face.embedding.tolist()
            }

            try:
                response = requests.post(REG_URL, json=payload)
                if response.status_code == 200:
                    print(f"Success! {name} has been registered in Neon DB.")
                else:
                    print(f"Failed: {response.text}")
            except Exception as e:
                print(f"Connection Error: {e}")
            break

        elif key == ord('q'):
            print("Registration cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    enroll()