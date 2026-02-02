import os
import onnxruntime as ort

print("----------------------------------------------------------------")
print(f"ONNX Runtime Version: {ort.__version__}")
print("----------------------------------------------------------------")
paths_to_check = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.7\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.6\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.5\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.4\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.3\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.2\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.1\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9.0\bin",
]

found_paths = []

print("Scanning for DLL folders...")
for path in paths_to_check:
    if os.path.exists(path):
        print(f"   ✅ Found folder: {path}")
        try:
            os.add_dll_directory(path)
            found_paths.append(path)
        except AttributeError:
            pass  # Python < 3.8
    else:
        # Only print missing if it's the main CUDA one, otherwise it gets spammy
        if "Toolkit" in path:
            print(f" Folder NOT found: {path}")

if not found_paths:
    print(
        "\nCRITICAL: No CUDA or cuDNN folders found. Did you install them to a custom location?"
    )

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
        print(f"    MISSING: {filename} ({desc})")
        missing_files.append(filename)

print("\n Attempting to load NVIDIA GPU...")
try:
    providers = ["CUDAExecutionProvider"]
    # We must try to create a session to trigger the actual load error
    # Creating a dummy check isn't enough
    if "CUDAExecutionProvider" in ort.get_available_providers():
        # Just because it's listed doesn't mean it works.
        # We need to verify if it throws an error when used.
        # Simple test: Can we set the provider?
        print("   -> Provider listed by ONNX Runtime. Validating...")

        if len(missing_files) > 0:
            print(
                f"\n WARNING: It will likely fail because you are missing: {missing_files}"
            )
            print(
                "   Python can see the 'option' to use CUDA, but the driver will crash on launch."
            )
        else:
            print("\n SUCCESS! All files present. GPU is ready!")

    else:
        print("\n CUDA Provider NOT listed. Python didn't find the library at all.")

except Exception as e:
    print(f"\n Error: {e}")

print("----------------------------------------------------------------")
