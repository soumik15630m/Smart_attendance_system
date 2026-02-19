import os
import sys

import onnxruntime as ort
from dotenv import load_dotenv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
env_path = os.path.join(project_root, ".env")
load_dotenv(env_path)

print("----------------------------------------------------------------")
print(f"ONNX Runtime Version: {ort.__version__}")
print("----------------------------------------------------------------")


cuda_env_val = os.getenv("CUDA_PATH_BIN", "")
cudnn_env_val = os.getenv("CUDNN_PATH_BIN", "")

paths_to_check = []
if cuda_env_val:
    paths_to_check.extend(cuda_env_val.split(os.pathsep))
if cudnn_env_val:
    paths_to_check.extend(cudnn_env_val.split(os.pathsep))

paths_to_check = [p.strip() for p in paths_to_check if p.strip()]

found_paths: list[str] = []

if not paths_to_check:
    print("WARNING: No 'CUDA_PATH_BIN' or 'CUDNN_PATH_BIN' set in .env file.")
    print("         Please define them to point to your bin folders.")
else:
    print(f"Scanning {len(paths_to_check)} configured paths from .env...")

for path in paths_to_check:
    if os.path.exists(path):
        print(f"   Found folder: {path}")

        if sys.platform == "win32":
            add_dll = getattr(os, "add_dll_directory", None)
            if add_dll:
                try:
                    add_dll(path)
                    found_paths.append(path)
                except Exception:
                    pass
    else:
        print(f"    Folder defined in env NOT found: {path}")

if not found_paths and sys.platform == "win32":
    print("\nCRITICAL: No valid DLL folders successfully added.")

required_files = {
    "cublas64_12.dll": "CUDA (Main Driver)",
    "cudnn64_9.dll": "cuDNN (Neural Network)",
    "zlibwapi.dll": "ZLIB (Required Helper)",
}

print("\nChecking for specific files in detected folders:")
missing_files = []

for filename, desc in required_files.items():
    found = False
    for folder in found_paths:
        full_path = os.path.join(folder, filename)
        if os.path.exists(full_path):
            print(f"    Found {filename} ({desc}) in {folder}")
            found = True
            break

    if not found:
        print(f"    (Not found in explicit paths): {filename}")
        missing_files.append(filename)

print("\n Attempting to load NVIDIA GPU...")
try:
    if "CUDAExecutionProvider" in ort.get_available_providers():
        print("   -> Provider listed by ONNX Runtime. Validating...")

        try:
            ort.InferenceSession(
                "dummy_model.onnx",
                providers=["CUDAExecutionProvider"],
            )
        except Exception as e:
            if "Load model" in str(e) or "No such file" in str(e):
                print("\n SUCCESS! CUDA Provider initialized successfully.")
            else:
                print(f"\n FAILURE: {e}")
    else:
        print("\n CUDA Provider NOT listed. Python didn't find the library at all.")

except Exception as e:
    print(f"\n Error: {e}")

print("----------------------------------------------------------------")
