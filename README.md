# SENITAL AI — Activity Recognition

Real-time human-activity detection from body pose. SENITAL AI finds every person
in a video or live camera feed, works out what they are doing, draws a bounding
box, and raises an on-screen alert plus a siren when a monitored activity occurs.

It uses **RTMPose** (a fast, accurate top-down pose model that runs on CPU) to
get each person's skeleton, applies transparent geometric rules to classify the
activity, and stabilises the result over time so labels don't flicker. No heavy
action-recognition model is required.

---

## Detected activities

| Activity | Recognised from the skeleton |
|----------|------------------------------|
| **Using phone** | one hand held **at the ear** |
| **Taking picture** | **both hands together** (small camera-sized gap allowed), raised to the upper belly or higher |
| **Placing object** | the person **bends / reaches down**, or reaches **one hand far out** from the body |

All three work from the **front and in side profile**, and for **small / distant
people** (RTMPose localises the wrists reliably even far from the camera).
When a monitored activity is detected the frame gets a red alert border and a
siren sounds.

---

## Features

- **Video upload** — upload a clip in the browser and watch the annotated result.
- **Live camera** — process your webcam in real time.
- **Network / IP camera** — connect an Ethernet/Wi-Fi camera by IP (RTSP or
  HTTP/MJPEG), with ONVIF auto-discovery and a guided setup panel.
- **Two model options** — **Balanced** (fast) and **Accurate**.
- **Temporal stabilisation** — an activity is only reported once it persists, so
  brief false flickers are suppressed.
- **Detection mode** — *Pose only* (default) or *Require object in hand* (strict:
  only flag when a held object is also detected).
- **On-screen alerts + siren**.
- **CPU-only** — no GPU required.

---

## Requirements

- **Python 3.9 – 3.12** (64-bit)
- Windows, macOS, or Linux (camera device names in the live picker are
  Windows-only via `pygrabber`; the rest is cross-platform)
- ~2 GB free disk for the Python packages and model weights

### Supported input formats
- **Video files:** MP4, MOV, AVI, MKV, WEBM
- **Network cameras:** RTSP (H.264 / H.265) and HTTP / MJPEG (RTSP over TCP)

All decoded via the bundled FFmpeg.

---

## Installation

```bash
cd SENITAL-AI

# 1. Create and activate a virtual environment
python -m venv venv
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

# 2. Install PyTorch for your hardware (CPU build is fine)
#    from https://pytorch.org/get-started/locally/
pip install torch torchvision

# 3. Install the rest
pip install -r requirements.txt
```

All model weights (RTMPose ONNX + the object detector) **download automatically
on first run** and are cached — nothing is bundled in the repo.

---

## Running

### Web app (recommended)

```bash
python app.py
```

Open **http://127.0.0.1:5000**.

- **Video Upload** — drop a video, set options, click **Analyze**.
- **Live Camera** — choose a camera, click **Start camera**, **Allow** the prompt.
- **Network Camera** — **Scan network** or enter IP + login, **Test connection**,
  then **Start detection**. See the on-page **Setup guide**.

Stop with `Ctrl + C`.

### Command line

```bash
python main.py --input data/input/clip.mp4
# accurate model:
python main.py --input data/input/clip.mp4 --model accurate
```

The annotated video is written to `data/output/<name>_activity.mp4`.

---

## Controls (every page)

| Control | Meaning |
|---------|---------|
| Detection model | **Balanced** (fast) or **Accurate** |
| Detection mode | *Pose only* (default) or *Require object in hand* (strict) |
| Detection accuracy | person-detection confidence threshold |
| Object threshold | object-detection confidence threshold (strict mode) |

---

## How it works

```
frame ─▶ RTMPose ─▶ 17 keypoints per person
                       │
             geometric activity rules (phone / picture / placing, side-profile aware)
                       │
             temporal hold (activity must persist ⇒ no flicker)
                       │
             draw box + label + alert + siren
```

For the live camera the browser captures frames and sends them to the server.
For network cameras the server pulls the RTSP/HTTP stream (browsers can't open
RTSP) and streams the annotated result back.

---

## Project structure

```
SENITAL-AI/
├── app.py                  # Flask web app (upload + live + network camera)
├── main.py                 # command-line entry point
├── requirements.txt
├── src/
│   ├── config.py           # settings and thresholds
│   ├── detector.py         # RTMPose + object detector
│   ├── pipeline.py         # activity rules, temporal stabilisation, drawing, siren
│   ├── netcam.py           # network camera: URL building, ONVIF, discovery
│   └── alarm.py            # siren generation and audio muxing
├── templates/              # index / live / network pages
├── static/style.css
└── data/{input,output}/
```

See [INSTALL.md](INSTALL.md) and [USAGE.md](USAGE.md) for details.

---

## Troubleshooting

- **An activity is not detected:** use the **Accurate** model, and keep
  **Detection mode** on *Pose only* (default).
- **Too slow:** use the **Balanced** model, or raise the upload **Processing
  interval**.
- **Network camera won't connect:** open the **Setup guide** on the Network
  Camera page; check power, same subnet, and login. RTSP = port 554, HTTP = 80.
- **No siren in the downloaded video:** ensure `imageio-ffmpeg` is installed;
  play the file in VLC / Windows Media Player.

## License
See [LICENSE](LICENSE).
