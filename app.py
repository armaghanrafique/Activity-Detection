#!/usr/bin/env python
"""Web UI for the pose activity detector.

Upload a video in the browser, watch it get processed, then play the annotated
result (bounding boxes + activity labels + red alert overlay + siren audio)
right on the page.

    python app.py
    # then open http://127.0.0.1:5000
"""

import subprocess
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

from src.config import PipelineConfig
from src.pipeline import ALERT_CATS, ActivityPosePipeline

BASE = Path(__file__).resolve().parent
UPLOADS = BASE / "uploads"
OUTPUTS = BASE / "data" / "output"
UPLOADS.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB upload cap

# In-memory job registry: job_id -> {status, progress, output, error}
JOBS = {}
_ALLOWED = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _transcode_web(src: Path, dst: Path) -> None:
    """Re-encode to browser-friendly H.264 + AAC (mp4v from OpenCV won't play
    reliably in <video>). Keeps the siren audio track if present."""
    cmd = [
        _ffmpeg(), "-y", "-i", str(src),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def _f(v, default, lo, hi):
    """Parse a float from a request value, clamped to [lo, hi]."""
    try:
        return max(lo, min(hi, float(v)))
    except (TypeError, ValueError):
        return default


def _apply_thresholds(pipe, conf, objconf):
    """Apply the accuracy + object thresholds to a live pipeline (no reload)."""
    pipe.pose.score_thresh = conf
    pipe.obj.score_thresh = objconf


def _process(job_id, in_path, out_raw, out_web, opts):
    job = JOBS[job_id]
    try:
        job["status"] = "processing"
        cfg = PipelineConfig(
            input_video=str(in_path),
            output_video=str(out_raw),
            device="cpu",
            yolo_weights=opts["weights"],
            process_every=opts["every"],
            require_object=opts.get("object", False),
            person_score_thresh=opts.get("conf", 0.4),
            object_score_thresh=opts.get("objconf", 0.15),
        )
        pipe = ActivityPosePipeline(cfg)

        def on_progress(done, total):
            # Reserve the last 8% for the transcode step.
            job["progress"] = min(92, int(done / total * 92))

        pipe.run(on_progress=on_progress)

        job["status"] = "transcoding"
        job["progress"] = 95
        _transcode_web(out_raw, out_web)
        try:
            out_raw.unlink(missing_ok=True)
        except Exception:
            pass

        job["output"] = out_web.name
        job["progress"] = 100
        job["status"] = "done"
    except Exception as exc:  # surface the error to the UI
        job["status"] = "error"
        job["error"] = str(exc)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="No video file received."), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED:
        return jsonify(error=f"Unsupported file type '{ext}'."), 400

    job_id = uuid.uuid4().hex[:12]
    in_path = UPLOADS / f"{job_id}{ext}"
    f.save(in_path)

    opts = {
        "weights": request.form.get("weights", "rtmpose"),
        "every": max(1, int(request.form.get("every", 3))),
        "object": request.form.get("object", "0") != "0",
        "conf": _f(request.form.get("conf"), 0.4, 0.05, 0.95),
        "objconf": _f(request.form.get("objconf"), 0.15, 0.05, 0.95),
    }
    out_raw = OUTPUTS / f"{job_id}_raw.mp4"
    out_web = OUTPUTS / f"{job_id}_web.mp4"
    JOBS[job_id] = {"status": "queued", "progress": 0, "output": None, "error": None}
    threading.Thread(
        target=_process, args=(job_id, in_path, out_raw, out_web, opts), daemon=True
    ).start()
    return jsonify(job_id=job_id)


@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job."), 404
    return jsonify(job)


@app.route("/video/<job_id>")
def video(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done" or not job.get("output"):
        abort(404)
    path = OUTPUTS / job["output"]
    if not path.exists():
        abort(404)
    return send_file(path, mimetype="video/mp4", conditional=True)


# ------------------------------------------------------------------ #
# Live webcam mode
# ------------------------------------------------------------------ #
LIVE = {"alert": False, "labels": [], "fps": 0.0}   # live state (for the UI)
_live_lock = threading.Lock()                  # only one webcam stream at a time
_live_lock_state = threading.Lock()            # guards the LIVE dict
_live_pipe = None                              # cached pipeline (loads pose once)
_live_pipe_lock = threading.Lock()
_proc_lock = threading.Lock()                  # serialise pose inference
_net_lock = threading.Lock()                   # one network stream at a time
_NET = {"url": None}                           # resolved network-camera URL


def list_cameras():
    """List connected cameras as [{index, name}]. Uses pygrabber for real
    device names on Windows; falls back to probing indices."""
    cams = []
    try:
        # pygrabber uses DirectShow/COM; COM must be initialised on this thread
        # (Flask serves each request on a worker thread).
        try:
            import comtypes
            comtypes.CoInitialize()
        except Exception:
            pass
        from pygrabber.dshow_graph import FilterGraph

        for i, name in enumerate(FilterGraph().get_input_devices()):
            cams.append({"index": i, "name": name})
    except Exception as exc:
        print(f"[cameras] name enumeration unavailable ({exc}); probing indices")
    if not cams:
        for i in range(6):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap and cap.isOpened():
                    cams.append({"index": i, "name": f"Camera {i}"})
                if cap:
                    cap.release()
            except Exception:
                pass
    print(f"[cameras] found: {cams}")
    return cams


def _get_live_pipeline(weights: str) -> ActivityPosePipeline:
    global _live_pipe
    with _live_pipe_lock:
        if _live_pipe is None or _live_pipe.cfg.yolo_weights != weights:
            _live_pipe = ActivityPosePipeline(PipelineConfig(yolo_weights=weights))
            _live_pipe._tracks = []  # fresh tracker
        return _live_pipe


def _open_camera(index: int):
    """Open the webcam, trying multiple Windows backends, and warm it up.

    Returns (cap, first_good_frame) or (None, None). OpenCV on Windows often
    returns black frames for the first ~10 reads while the sensor initialises,
    so we read and discard until we get a non-black frame.
    """
    import time as _t

    import numpy as _np

    # (backend, use-MJPG). Cameras differ a lot: some only work on DirectShow,
    # some only on Media Foundation (MSMF); some need the MJPG pixel format,
    # while forcing MJPG makes others hang. We try every backend WITHOUT MJPG
    # first (safest, and enough for most cams incl. many that DShow shows black
    # for), then fall back to MJPG for cams that need it.
    MSMF = ("CAP_MSMF", getattr(cv2, "CAP_MSMF", 1400))
    DSHOW = ("CAP_DSHOW", cv2.CAP_DSHOW)
    DEFAULT = ("default", None)
    # NOTE: MSMF + forced MJPG is deliberately excluded — it hangs on some
    # webcams (e.g. Creative Live! Cam).
    attempts = [
        (*MSMF, False), (*DSHOW, False), (*DEFAULT, False),   # no-MJPG first
        (*DSHOW, True), (*DEFAULT, True),                     # then MJPG (safe combos)
    ]

    def _try(name, be, mjpg, out):
        # Runs in a worker thread so a blocking cap.read() can't hang forever.
        try:
            cap = cv2.VideoCapture(index) if be is None else cv2.VideoCapture(index, be)
            if not cap or not cap.isOpened():
                if cap:
                    cap.release()
                return
            if mjpg:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            for _ in range(50):  # warm up (USB cams start black)
                ok, frame = cap.read()
                if ok and frame is not None and float(_np.mean(frame)) > 4.0:
                    out["cap"], out["frame"] = cap, frame
                    return
                _t.sleep(0.05)
            cap.release()
        except Exception as exc:
            print(f"[live] {name} mjpg={mjpg}: error {exc}")

    for attempt_i, (name, be, mjpg) in enumerate(attempts):
        if attempt_i > 0:
            _t.sleep(0.6)  # let the camera driver settle between backends
        out = {"cap": None, "frame": None}
        th = threading.Thread(target=_try, args=(name, be, mjpg, out), daemon=True)
        th.start()
        th.join(5.0)  # per-backend budget; a hung read is abandoned here
        if th.is_alive():
            print(f"[live] {name} mjpg={mjpg}: timed out (backend stuck) — skipping")
            continue
        if out["cap"] is not None:
            c = out["cap"]
            w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[live] camera opened via {name} (mjpg={mjpg}) at {w}x{h}")
            return out["cap"], out["frame"]
        print(f"[live] {name} mjpg={mjpg}: no usable frame")
    print(f"[live] could not get a usable frame from camera index {index}")
    return None, None


def _live_frames(weights: str, cam_index: int, imgsz: int = 640, require_object: bool = True,
                 conf: float = 0.4, objconf: float = 0.15):
    """MJPEG generator: read webcam, annotate, yield JPEG frames."""
    import time as _t

    if not _live_lock.acquire(blocking=False):
        print("[live] a stream is already running — ignoring new request")
        return  # a stream is already running

    cap = None
    stop = threading.Event()
    holder = {"frame": None}          # always the LATEST camera frame
    hlock = threading.Lock()

    def _capture_loop():
        # Continuously grab the newest frame so the buffer never backs up.
        # The processing loop reads whatever is latest here, so output latency
        # stays bounded to ~one processing cycle instead of growing over time.
        while not stop.is_set():
            ok, f = cap.read()
            if not ok or f is None:
                stop.set()
                break
            with hlock:
                holder["frame"] = f

    try:
        print(f"[live] starting — model={weights}, camera={cam_index}, imgsz={imgsz}")
        pipe = _get_live_pipeline(weights)
        pipe.pose.imgsz = int(imgsz)  # apply processing scale (no reload)
        pipe.require_object = require_object
        _apply_thresholds(pipe, conf, objconf)
        cap, first = _open_camera(cam_index)
        if cap is None or first is None:
            return  # no usable camera (reason already logged)
        with hlock:
            holder["frame"] = first

        threading.Thread(target=_capture_loop, daemon=True).start()

        idx, last_alert_t, last_labels = 0, 0.0, []
        win_start, win_count = _t.time(), 0
        while not stop.is_set():
            with hlock:
                frame = holder["frame"]
            if frame is None:
                _t.sleep(0.005)
                continue
            frame = frame.copy()

            people, labels = pipe.process_frame(frame)   # the slow (CPU) step
            now = _t.time()
            if labels:
                last_alert_t, last_labels = now, labels
            alert_active = (now - last_alert_t) <= 1.0   # hold ~1s (time-based)

            # Rolling processing-FPS estimate.
            win_count += 1
            fps = win_count / (now - win_start) if now > win_start else 0.0
            if now - win_start >= 1.5:
                win_start, win_count = now, 0

            with _live_lock_state:
                LIVE["alert"] = bool(alert_active)
                LIVE["labels"] = last_labels if alert_active else []
                LIVE["fps"] = round(fps, 1)

            annotated = pipe._draw(frame, people, idx, alert_active, last_labels)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            idx += 1
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                   + jpg + b"\r\n")
            if idx % 100 == 0:
                print(f"[live] processed {idx} frames (~{fps:.1f} fps)")
    except GeneratorExit:
        print("[live] client disconnected — stopping stream")
    finally:
        stop.set()
        if cap is not None:
            cap.release()
        with _live_lock_state:
            LIVE["alert"] = False
            LIVE["labels"] = []
            LIVE["fps"] = 0.0
        _live_lock.release()
        print("[live] camera released")


@app.route("/live")
def live():
    return render_template("live.html")


@app.route("/cameras")
def cameras():
    return jsonify(cameras=list_cameras())


@app.route("/live_feed")
def live_feed():
    weights = request.args.get("weights", "rtmpose")
    cam = int(request.args.get("cam", 0))
    imgsz = int(request.args.get("imgsz", 640))
    require_object = request.args.get("object", "0") != "0"
    conf = _f(request.args.get("conf"), 0.4, 0.05, 0.95)
    objconf = _f(request.args.get("objconf"), 0.15, 0.05, 0.95)
    return Response(_live_frames(weights, cam, imgsz, require_object, conf, objconf),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/live_status")
def live_status():
    with _live_lock_state:
        return jsonify(alert=LIVE["alert"], labels=LIVE["labels"], fps=LIVE["fps"])


@app.route("/process_frame", methods=["POST"])
def process_frame_route():
    """Browser-camera mode: receive one JPEG frame, run pose + activity
    detection, return the annotated JPEG. Alert state goes in the headers.

    This is the reliable path for the live camera — the BROWSER captures the
    webcam (robust, handles all drivers) and just sends us frames to process.
    """
    weights = request.form.get("weights", "rtmpose")
    try:
        imgsz = int(request.form.get("imgsz", 480))
    except Exception:
        imgsz = 480
    f = request.files.get("frame")
    if not f:
        return "no frame", 400
    data = np.frombuffer(f.read(), np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if frame is None:
        return "bad frame", 400

    with _proc_lock:  # YOLO predict + tracker are not concurrency-safe
        pipe = _get_live_pipeline(weights)
        pipe.pose.imgsz = int(imgsz)
        pipe.require_object = request.form.get("object", "0") != "0"
        _apply_thresholds(
            pipe,
            _f(request.form.get("conf"), 0.4, 0.05, 0.95),
            _f(request.form.get("objconf"), 0.15, 0.05, 0.95),
        )
        people, labels = pipe.process_frame(frame)
        annotated = pipe._draw(frame, people, 0, bool(labels), labels)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        return "encode failed", 500
    resp = Response(buf.tobytes(), mimetype="image/jpeg")
    resp.headers["X-Alert"] = "1" if labels else "0"
    resp.headers["X-Labels"] = ", ".join(dict.fromkeys(labels))
    return resp


# ------------------------------------------------------------------ #
# Network (IP / Ethernet) camera
# ------------------------------------------------------------------ #
def _mask_url(url: str) -> str:
    """Hide credentials in a URL before showing it in the UI."""
    from urllib.parse import urlparse, urlunparse
    p = urlparse(url)
    if "@" in (p.netloc or ""):
        host = p.netloc.split("@", 1)[1]
        return urlunparse((p.scheme, "***:***@" + host, p.path, p.params,
                           p.query, p.fragment))
    return url


@app.route("/network")
def network():
    return render_template("network.html")


@app.route("/net_scan")
def net_scan():
    try:
        from src import netcam
        return jsonify(cameras=netcam.discover())
    except Exception as exc:  # discovery is best-effort — never 500
        print(f"[net] scan error: {exc}")
        return jsonify(cameras=[], error="Scan unavailable."), 200


@app.route("/net_test", methods=["POST"])
def net_test():
    _fail = jsonify(ok=False,
                    error="Could not connect or no video received. Check the IP, "
                          "credentials, port, and that the camera is reachable."), 200
    try:
        from src import netcam
        cfg = {k: request.form.get(k, "") for k in
               ("url", "ip", "port", "user", "password", "protocol", "path",
                "onvif_port")}
        if not (cfg["url"].strip() or cfg["ip"].strip()):
            return jsonify(ok=False, error="Enter a camera IP address or a full "
                                           "stream URL."), 200
        url, cap, frame = netcam.resolve(cfg)
        if cap is None or frame is None:
            return _fail
        ok, buf = cv2.imencode(".jpg", frame)
        cap.release()
        if not ok:
            return _fail
        with _live_lock_state:
            _NET["url"] = url
        import base64
        preview = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
        return jsonify(ok=True, url=_mask_url(url), preview=preview)
    except Exception as exc:  # any failure -> clean error, never a 500 crash
        print(f"[net] test error: {exc}")
        return _fail


def _net_frames(url, weights, imgsz, require_object=True, conf=0.4, objconf=0.15):
    """MJPEG generator for a network camera: server pulls RTSP/HTTP, processes,
    streams the annotated result to the browser."""
    import time as _t
    from src import netcam

    if not _net_lock.acquire(blocking=False):
        return
    stop = threading.Event()
    holder = {"frame": None}
    hlock = threading.Lock()
    cap = None

    def _capture_loop():
        while not stop.is_set():
            ok, f = cap.read()
            if not ok or f is None:
                stop.set()
                break
            with hlock:
                holder["frame"] = f

    try:
        pipe = _get_live_pipeline(weights)
        pipe.pose.imgsz = int(imgsz)
        pipe.require_object = require_object
        _apply_thresholds(pipe, conf, objconf)
        cap, first = netcam.open_stream(url)
        if cap is None or first is None:
            print("[net] could not open stream")
            return
        with hlock:
            holder["frame"] = first
        threading.Thread(target=_capture_loop, daemon=True).start()
        print(f"[net] streaming {_mask_url(url)}")

        idx, last_alert_t, last_labels = 0, 0.0, []
        win_start, win_count = _t.time(), 0
        while not stop.is_set():
            with hlock:
                frame = holder["frame"]
            if frame is None:
                _t.sleep(0.005)
                continue
            frame = frame.copy()
            people, labels = pipe.process_frame(frame)
            now = _t.time()
            if labels:
                last_alert_t, last_labels = now, labels
            alert_active = (now - last_alert_t) <= 1.0
            win_count += 1
            fps = win_count / (now - win_start) if now > win_start else 0.0
            if now - win_start >= 1.5:
                win_start, win_count = now, 0
            with _live_lock_state:
                LIVE["alert"] = bool(alert_active)
                LIVE["labels"] = last_labels if alert_active else []
                LIVE["fps"] = round(fps, 1)
            annotated = pipe._draw(frame, people, idx, alert_active, last_labels)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            idx += 1
            if not ok:
                continue
            jpg = buf.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                   + b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                   + jpg + b"\r\n")
    except GeneratorExit:
        pass
    finally:
        stop.set()
        if cap is not None:
            cap.release()
        with _live_lock_state:
            LIVE["alert"] = False
            LIVE["labels"] = []
            LIVE["fps"] = 0.0
        _net_lock.release()
        print("[net] stream released")


@app.route("/net_stream")
def net_stream():
    weights = request.args.get("weights", "rtmpose")
    try:
        imgsz = int(request.args.get("imgsz", 480))
    except Exception:
        imgsz = 480
    require_object = request.args.get("object", "0") != "0"
    conf = _f(request.args.get("conf"), 0.4, 0.05, 0.95)
    objconf = _f(request.args.get("objconf"), 0.15, 0.05, 0.95)
    url = _NET.get("url")
    if not url:
        return "Not connected. Run 'Test connection' first.", 409
    return Response(_net_frames(url, weights, imgsz, require_object, conf, objconf),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/snapshot")
def snapshot():
    """Diagnostic: grab ONE raw frame from the camera and return it as a JPEG,
    so you can verify the camera image directly (open /snapshot?cam=0)."""
    cam = int(request.args.get("cam", 0))
    if not _live_lock.acquire(blocking=False):
        return "Camera is busy (a live stream is running). Stop it first.", 409
    cap = None
    try:
        cap, frame = _open_camera(cam)
        if cap is None or frame is None:
            return "Could not open a usable camera (see server log).", 503
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return "Failed to encode frame.", 500
        return Response(buf.tobytes(), mimetype="image/jpeg")
    finally:
        if cap is not None:
            cap.release()
        _live_lock.release()


if __name__ == "__main__":
    print("\n  SENITAL AI (Activity Recognition)")
    print("  Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, threaded=True)
