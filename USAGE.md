# Usage Guide — SENITAL AI

## Start the web app
```bash
# activate the venv first, then:
python app.py
```
Open **http://127.0.0.1:5000**. Stop with `Ctrl + C`. Three pages, linked at the
top of each: **Video Upload** (`/`), **Live Camera** (`/live`), **Network
Camera** (`/network`).

### Settings on every page
- **Detection model** — **Balanced** (fast) or **Accurate**.
- **Detection mode** — *Pose only* (default; detect from body pose) or *Require
  object in hand* (strict: only flag when a held object is also detected).
- **Detection accuracy** — person-detection confidence threshold.
- **Object threshold** — object-detection confidence threshold (strict mode).

---

## Video upload
1. Drag a video onto the upload area (or **browse**).
2. Choose the model and, optionally, the **Processing interval** (detect every
   Nth frame — higher = faster).
3. Click **Analyze**. When done, the annotated video plays; **Download** it or
   **Process another video**.

The output has the boxes, activity labels, red alert overlay, and an embedded
siren during monitored activities (play in VLC / Windows Media Player for sound).

---

## Live camera
1. Choose the model, then your **Camera**.
2. Click **Start camera** and **Allow** the browser permission.
3. The annotated feed appears with a live FPS readout; a red **Alert** badge +
   siren play on a monitored activity. Click **Stop** to end.

The browser captures the camera and sends frames to the server, so any webcam
your browser supports works.

---

## Network / IP camera
1. Open **Network Camera**. Read **Before you start** and **Setup guide** for the
   Ethernet wiring and IP configuration.
2. **Scan network** to auto-discover ONVIF cameras (click one to fill its IP), or
   type the **IP + username + password**. Advanced: paste a full RTSP/HTTP URL.
3. **Test connection** (a preview appears), then **Start detection**.

The server pulls the RTSP/HTTP stream, processes it, and streams the annotated
result back with the same alerts and siren.

---

## Command line
```bash
python main.py --input data/input/clip.mp4
```
Output → `data/output/clip_activity.mp4`.

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | (required) | input video path |
| `--output` | `data/output/<name>_activity.mp4` | output path |
| `--model` | `balanced` | `balanced` (fast) or `accurate` |
| `--person-thresh` | `0.4` | person-detection confidence |
| `--process-every` | `1` | run detection every Nth frame |
| `--require-object` | off | strict: only flag if a held object is detected |
| `--debug-skeleton` | off | overlay the detected skeleton (debugging) |
| `--no-alarm` | off | do not embed the siren |
| `--no-alert-visuals` | off | do not draw the alert overlay |

Run `python main.py --help` for the full list.
