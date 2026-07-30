# Installation Guide — SENITAL AI

## 1. Prerequisites
- **Python 3.9 – 3.12** (64-bit). Check: `python --version`
  (on Windows, tick **"Add Python to PATH"** during install).
- ~2 GB free disk.

## 2. Get the project
Copy the `SENITAL-AI` folder to your machine and open a terminal in it:
```bash
cd SENITAL-AI
```

## 3. Create a virtual environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If activation is blocked, run once and retry:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

## 4. Install PyTorch (for the object detector)
A CPU build is enough:
```bash
pip install torch torchvision
```
(For an NVIDIA GPU, pick the CUDA command from https://pytorch.org/get-started/locally/.)

## 5. Install the rest
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Verify
```bash
python -c "import onnxruntime, rtmlib, cv2, flask, ultralytics, torch; print('OK')"
```

## 7. First run
```bash
python app.py
```
Open http://127.0.0.1:5000. On first use the **RTMPose** and object-detector
weights download automatically (a few tens of MB) and are cached for later runs.

---

## Dependencies
| Package | Purpose |
|---------|---------|
| rtmlib + onnxruntime | RTMPose pose engine (keypoints) |
| torch, torchvision, ultralytics | YOLO object detector (held-object box / strict mode) |
| opencv-python | video / image I/O, camera streams, drawing |
| numpy (<2.0) | array processing |
| imageio-ffmpeg | bundled ffmpeg for the siren + H.264 output |
| flask | web application |
| pygrabber | camera names in the live picker (Windows) |
| onvif-zeep, wsdiscovery | network-camera ONVIF lookup + LAN discovery |

## Common issues
- **`Activate.ps1` blocked:** run the `Set-ExecutionPolicy` command in step 3.
- **`pip install torch` fails:** use the exact command from the PyTorch website.
- **Port 5000 in use:** stop the other process, or change the port in the last
  line of `app.py`.
