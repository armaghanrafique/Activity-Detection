"""Pose-based activity detection pipeline (YOLO-pose + geometric rules).

Reads activities directly from the YOLO-pose skeleton with simple geometric
rules — no action-recognition model — so it runs fast. Detects: using phone,
taking picture, placing object.

The geometry proposes the activity; a second YOLO object detector then confirms
the implied object (a phone / camera / any held object) is actually present near
the hands, so an empty-handed pose does not trigger a false activity.

These are HEURISTICS: fast and transparent, tuned for a live demo.
"""

import os
import time
from collections import Counter, deque
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from . import alarm
from .config import PipelineConfig
from .detector import RTMPoseDetector, YoloObjectDetector


# Categories that raise an ALERT (the three target activities).
ALERT_CATS = {"phone", "picture", "pickdrop"}

# Amber box drawn around the object that confirms an activity.
_OBJ_COLOR = (0, 215, 255)

# COCO-17 skeleton edges (for the debug overlay).
_SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (0, 5), (0, 6),                           # neck to shoulders
]

# Classes that count as a "phone" for "using phone". yolov8n often labels a
# hand-held phone as "remote" (or "cell phone"), so accept both.
_PHONE_CLASSES = {"cell phone", "remote"}

# Portable COCO objects a person can hold / carry / place. Used to confirm
# "taking picture" and "placing object" — furniture, vehicles, animals, etc.
# are excluded so sitting near a couch does not confirm an activity.
_HOLDABLE = {
    "cell phone", "laptop", "keyboard", "mouse", "remote", "book", "clock",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "potted plant",
    "frisbee", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "tennis racket", "snowboard", "skis",
}


def _hand_points(kp, conf: float = 0.15):
    """Wrist + head points (image coords) and shoulder width for one person."""
    nose, lear, rear = kp[0], kp[3], kp[4]
    lsh, rsh = kp[5], kp[6]
    lw, rw = kp[9], kp[10]
    wrists = [(float(w[0]), float(w[1])) for w in (lw, rw) if w[2] >= conf]
    heads = [(float(h[0]), float(h[1])) for h in (nose, lear, rear) if h[2] >= conf]
    if lsh[2] >= conf and rsh[2] >= conf:
        sw = max(abs(float(lsh[0]) - float(rsh[0])), 1.0)
    else:
        sw = 40.0
    return wrists, heads, sw


def _object_near(obox, points, sw: float) -> bool:
    """True if any point is inside (a margin around) the object box or close to
    its center — i.e. the object is in/near the person's hand. Generous, because
    the wrist keypoint and the object box rarely line up exactly."""
    ox1, oy1, ox2, oy2 = float(obox[0]), float(obox[1]), float(obox[2]), float(obox[3])
    m = 0.6 * max(ox2 - ox1, oy2 - oy1)
    cx, cy = (ox1 + ox2) / 2.0, (oy1 + oy2) / 2.0
    for px, py in points:
        if ox1 - m <= px <= ox2 + m and oy1 - m <= py <= oy2 + m:
            return True
        if ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5 < 2.2 * sw:
            return True
    return False


def _match_object(objects, points, sw, phone_only, person_box):
    """Find an object near ``points`` and inside the person's region.

    Returns ``(box, name)`` or ``None``. ``phone_only`` restricts to phones
    (for "using phone"); otherwise any held object counts (camera / bottle /
    cup / book ... for "taking picture" and "placing object").
    """
    px1, py1, px2, py2 = [float(v) for v in person_box]
    pm = 0.25 * max(px2 - px1, py2 - py1)
    for obox, name, _conf in objects:
        if phone_only:
            if name not in _PHONE_CLASSES:
                continue
        elif name not in _HOLDABLE:
            continue
        cx = (float(obox[0]) + float(obox[2])) / 2.0
        cy = (float(obox[1]) + float(obox[3])) / 2.0
        if not (px1 - pm <= cx <= px2 + pm and py1 - pm <= cy <= py2 + pm):
            continue
        if _object_near(obox, points, sw):
            return obox, name
    return None


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# BGR colors by category.
_COLORS = {
    "phone": (40, 40, 220),      # red
    "picture": (40, 120, 240),   # orange
    "pickdrop": (200, 60, 200),  # magenta
    "none": (60, 180, 75),       # green — person detected, no target activity
}


def classify_activity(kpts: np.ndarray, box, conf: float = 0.15) -> Tuple[str, str, bool]:
    """Return (display_label, category, needs_object) from one person's keypoints.

    ``needs_object`` marks a pose that should only be reported if the object
    detector finds something in the hand.

    COCO keypoints: 0 nose, 3/4 ears, 5/6 shoulders, 9/10 wrists, 11/12 hips,
    13/14 knees.
    """
    def ok(p):
        return p[2] >= conf

    nose, lear, rear = kpts[0], kpts[3], kpts[4]
    lsh, rsh = kpts[5], kpts[6]
    lel, rel = kpts[7], kpts[8]
    lw, rw = kpts[9], kpts[10]
    lh, rh = kpts[11], kpts[12]

    bx_w = max(float(box[2]) - float(box[0]), 1.0)
    bx_h = max(float(box[3]) - float(box[1]), 1.0)

    # Shoulder line + center (fall back to box geometry if missing).
    if ok(lsh) and ok(rsh):
        sw_raw = abs(lsh[0] - rsh[0])
        sh_y = (lsh[1] + rsh[1]) / 2.0; sh_x = (lsh[0] + rsh[0]) / 2.0
    elif ok(lsh) or ok(rsh):
        s = lsh if ok(lsh) else rsh
        sw_raw = 0.0; sh_y = float(s[1]); sh_x = float(s[0])
    else:
        sw_raw = 0.0; sh_y = float(box[1]) + 0.22 * bx_h; sh_x = float(box[0]) + 0.5 * bx_w

    # Hip line + center.
    if ok(lh) and ok(rh):
        hip_y = (lh[1] + rh[1]) / 2.0; hip_x = (lh[0] + rh[0]) / 2.0
    elif ok(lh) or ok(rh):
        hh = lh if ok(lh) else rh
        hip_y = float(hh[1]); hip_x = float(hh[0])
    else:
        hip_y = float(box[1]) + 0.6 * bx_h; hip_x = sh_x
    torso = max(hip_y - sh_y, 0.30 * bx_h, 1.0)
    # Robust body unit: the shoulder width when facing the camera, but it never
    # collapses in a SIDE PROFILE (where the shoulders overlap in x) — it then
    # falls back to the torso height, which stays stable side-on.
    sw = max(sw_raw, 0.33 * torso, 0.30 * bx_w)
    body_cx = (sh_x + hip_x) / 2.0

    # Side profile: only one hip is visible (the far one is occluded) and/or the
    # shoulders overlap. The two hands then overlap too, so the two-hand "picture"
    # test cannot fire — a single raised hand has to stand in for it.
    n_hips = int(ok(lh)) + int(ok(rh))
    side_profile = (n_hips <= 1) or (sw_raw < 0.30 * torso)

    wrists = [w for w in (lw, rw) if ok(w)]
    heads = [h for h in (nose, lear, rear) if ok(h)]
    head_x = (sum(h[0] for h in heads) / len(heads)) if heads else sh_x

    # Tight hand-to-EAR test: the wrist must be right at an ear keypoint (not the
    # nose / chin / general head area). The tracked wrist sits at the jaw when a
    # phone is held to the ear, ~0.6-0.9 sw away, so that is the tightest bound.
    ears = [e for e in (lear, rear) if ok(e)]

    def on_ear(w):
        for e in ears:
            if ((w[0] - e[0]) ** 2 + (w[1] - e[1]) ** 2) ** 0.5 < 0.90 * sw:
                return True
        return False

    # Both wrists clearly above the shoulders => a two-hand pose, not hand-to-ear.
    up = [w for w in wrists if w[1] < sh_y - 0.05 * torso]

    # 1) USING PHONE — ONE hand tightly on the EAR. Nothing else counts.
    if len(up) <= 1 and any(on_ear(w) for w in wrists):
        return "using phone", "phone", False

    # 2) TAKING PICTURE — BOTH hands close together (a small camera-sized gap is
    #    allowed), raised so their midpoint sits at the upper belly or higher.
    if len(wrists) == 2:
        w0, w1 = wrists
        gap = ((w0[0] - w1[0]) ** 2 + (w0[1] - w1[1]) ** 2) ** 0.5
        wm_y = (w0[1] + w1[1]) / 2.0
        if gap < 0.85 * sw and wm_y <= sh_y + 0.55 * torso:
            return "taking picture", "picture", False

    # 3) PLACING OBJECT — the person bends / reaches DOWN (a hand well below the
    #    hips), OR reaches one hand far OUT from the body (and not up). Pose only;
    #    both use scale/center that stay stable in a side profile.
    for w in wrists:
        if w[1] > hip_y + 0.35 * torso:                  # bending / reaching down
            return "placing object", "pickdrop", False
    for w in wrists:
        if abs(w[0] - body_cx) > 1.3 * sw and w[1] > sh_y + 0.10 * torso:
            return "placing object", "pickdrop", False   # one hand far from body

    # 4) No target activity — the person is still detected (box only, no label).
    return "", "none", False


class ActivityPosePipeline:
    # UI "Detection model" value -> RTMPose size. RTMPose is a top-down pose
    # model (accurate wrist localisation, works for small/far people on CPU).
    _RTM_MODE = {"rtmpose": "lightweight", "rtmpose-acc": "balanced"}

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.device = cfg.device or "cpu"
        mode = self._RTM_MODE.get(cfg.yolo_weights, getattr(cfg, "rtmpose_mode", "lightweight"))
        print(f"[activity] loading pose model (RTMPose / {mode})...")
        self.pose = RTMPoseDetector(
            self.device, mode=mode, score_thresh=cfg.person_score_thresh,
        )
        # Object detector — used only to draw the box on a held object and for
        # the optional "require object in hand" strict mode. It never gates the
        # pose-based activities by default.
        print(f"[activity] loading object detector ({cfg.object_weights})...")
        self.obj = YoloObjectDetector(
            self.device, cfg.object_score_thresh, cfg.object_weights, cfg.object_imgsz,
        )
        self.require_object = bool(cfg.require_object)
        self._obj_hist = deque(maxlen=max(1, int(cfg.object_memory_frames)))
        self._obj_tick = 0
        self._frame_objects = []
        self._frame_kpts = np.zeros((0, 17, 3), dtype=np.float32)
        # IoU tracker + per-track label history for temporal stabilisation.
        self.smooth_window = int(getattr(cfg, "smooth_window", 9))
        self.hold_min = int(getattr(cfg, "hold_min_frames", 3))
        self._tracks = []

    def _smooth(self, detections):
        """Match detections to tracks by IoU and majority-vote the labels."""
        used = set()
        out = []
        for box, raw in detections:
            best_j, best_iou = -1, 0.3
            for j, tr in enumerate(self._tracks):
                if j in used:
                    continue
                i = _iou(box, tr["box"])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0:
                tr = self._tracks[best_j]
                used.add(best_j)
            else:
                tr = {"box": box, "hist": deque(maxlen=self.smooth_window)}
                self._tracks.append(tr)
                used.add(len(self._tracks) - 1)
            tr["box"] = box
            tr["hist"].append(raw)
            # Temporal HOLD: report an activity only if its category persisted
            # across at least hold_min of the recent window — a 1-2 frame blip is
            # suppressed as "none". Among sustained activities, pick the strongest.
            cat_counts = Counter(c for _, c in tr["hist"] if c in ALERT_CATS)
            best_cat, best_n = (cat_counts.most_common(1)[0] if cat_counts else (None, 0))
            if best_cat is not None and best_n >= self.hold_min:
                cat = best_cat
                label = next(l for l, c in reversed(tr["hist"]) if c == best_cat)
            else:
                label, cat = "", "none"
            out.append((box, (label, cat)))
        self._tracks = [tr for j, tr in enumerate(self._tracks) if j in used]
        return out

    def _detect_objects_roi(self, rgb, boxes):
        """Detect objects inside each person's region (zoomed in) so small /
        distant objects like a phone are large enough to detect. Object boxes
        are mapped back to full-frame coordinates."""
        h, w = rgb.shape[:2]
        out = []
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            mx, my = 0.35 * bw, 0.20 * bh   # a little context around the person
            cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
            cx2, cy2 = min(w, int(x2 + mx)), min(h, int(y2 + my))
            crop = rgb[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            for obox, name, conf in self.obj.detect(crop):
                out.append(([obox[0] + cx1, obox[1] + cy1,
                             obox[2] + cx1, obox[3] + cy1], name, conf))
        return out

    def process_frame(self, frame):
        """Detect + classify one frame. Returns (people, alert_labels).

        Shared by the file pipeline and the live webcam stream. ``people`` is
        the smoothed list of (box, (label, category)); ``alert_labels`` are the
        target-activity labels present this frame (empty if none).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self.pose.detect(rgb)
        kpts = np.asarray(self.pose.last_keypoints, dtype=np.float32)
        self._frame_kpts = kpts   # kept for the (debug-only) skeleton overlay

        # Always run the object detector (when loaded) so its box can sync with
        # a detected activity. It only VETOES an activity in strict mode
        # (require_object) — otherwise geometry drives detection and the object
        # box is a non-blocking bonus, so distant cameras (phone too small to
        # detect) still detect the activity.
        recent_objects = []
        if self.obj is not None and len(boxes):
            # Zoom into each person (ROI) so small/distant objects like a phone
            # are detectable — run every Nth frame to keep the frame rate up and
            # reuse recent results in between.
            self.obj.imgsz = int(self.cfg.object_imgsz)
            if self._obj_tick % max(1, int(self.cfg.object_every)) == 0:
                self._obj_hist.append(self._detect_objects_roi(rgb, boxes))
            self._obj_tick += 1
            # Pool objects across recent passes so a phone that drops out of a
            # single detection still confirms the activity.
            for objs in self._obj_hist:
                recent_objects.extend(objs)

        detections = []
        frame_objs = []
        for i, box in enumerate(boxes):
            kp = kpts[i] if i < len(kpts) else np.zeros((17, 3), np.float32)
            label, cat, needs_object = classify_activity(
                kp, box, conf=self.cfg.kpt_conf_thresh)
            # Object confirmation: draw the object's box when it is found near
            # the hands (syncs with the activity). Veto the activity when no
            # object is found IF this pose needs one (needs_object) OR the user
            # chose strict "require object in hand" mode.
            if cat in ALERT_CATS and self.obj is not None:
                wrists, heads, sw = _hand_points(kp)
                phone_only = cat == "phone"
                points = wrists + heads if cat != "pickdrop" else wrists
                match = _match_object(recent_objects, points, sw, phone_only, box)
                if match is not None:
                    frame_objs.append((match[0], match[1], cat))
                elif needs_object or self.require_object:
                    label, cat = "", "none"       # no object -> veto
            elif cat in ALERT_CATS and needs_object:
                # object detector unavailable but this pose requires one
                label, cat = "", "none"
            detections.append((box, (label, cat)))

        people = self._smooth(detections)
        alert_labels = [lbl for _, (lbl, cat) in people if cat in ALERT_CATS]
        # Show an object box only for a category the smoothed output is actually
        # displaying, so the amber box and the red activity label stay in sync.
        active_cats = {cat for _, (lbl, cat) in people if cat in ALERT_CATS}
        self._frame_objects = [o for o in frame_objs if o[2] in active_cats]
        return people, alert_labels

    def _draw(self, frame, people, frame_idx=0, alert_active=False, alert_text=None):
        out = frame.copy()

        # Full skeleton overlay of what the pose model is seeing (for debugging).
        if getattr(self.cfg, "draw_keypoints", False):
            kc = float(self.cfg.kpt_conf_thresh)
            for kp in getattr(self, "_frame_kpts", []):
                for a, b in _SKELETON:
                    if kp[a][2] >= kc and kp[b][2] >= kc:
                        cv2.line(out, (int(kp[a][0]), int(kp[a][1])),
                                 (int(kp[b][0]), int(kp[b][1])), (0, 255, 140), 2, cv2.LINE_AA)
                for j in range(17):
                    if kp[j][2] >= kc:
                        # wrists larger + amber (activity-driving), rest small.
                        r, col = (5, (0, 165, 255)) if j in (9, 10) else (3, (0, 255, 255))
                        cv2.circle(out, (int(kp[j][0]), int(kp[j][1])), r, col, -1)

        for box, (label, cat) in people:
            x1, y1, x2, y2 = [int(v) for v in box]
            color = _COLORS.get(cat, (60, 180, 75))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
            # Only draw a label when the person is doing a target activity;
            # otherwise it's just a detected-person box (no text).
            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
                cv2.putText(out, label, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Boxes around the objects that confirmed an activity this frame —
        # box only, no label, just to point out the object being used.
        for obox, _name, _cat in getattr(self, "_frame_objects", []):
            ox1, oy1, ox2, oy2 = [int(v) for v in obox]
            cv2.rectangle(out, (ox1, oy1), (ox2, oy2), _OBJ_COLOR, 2)

        if alert_active and self.cfg.alert_visuals:
            self._draw_alert(out, frame_idx, alert_text or [])
        return out

    @staticmethod
    def _draw_alert(img, frame_idx, alert_labels):
        """Loud on-screen alert: pulsing red border + top banner."""
        h, w = img.shape[:2]
        red = (0, 0, 255)
        pulse = (frame_idx // 5) % 2
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), red, 14 if pulse else 8)

        unique = list(dict.fromkeys(alert_labels))
        text = "! ALERT ! " + ", ".join(unique) if unique else "! ALERT !"
        fs = max(0.8, w / 1000.0)
        (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        pad = 12
        cv2.rectangle(img, (0, 0), (w, tht + 2 * pad),
                      red if pulse else (0, 0, 200), -1)
        cv2.putText(img, text, (max(pad, (w - tw) // 2), pad + tht),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)

    def run(self, on_progress=None) -> str:
        """Process the video. ``on_progress(done_frames, total_frames)`` is
        called periodically (used by the web UI for a progress bar)."""
        cfg = self.cfg
        if not Path(cfg.input_video).exists():
            raise FileNotFoundError(f"Input video not found: {cfg.input_video}")

        cap = cv2.VideoCapture(cfg.input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        Path(cfg.output_video).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            cfg.output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

        start = time.time()
        every = max(1, int(self.cfg.process_every))
        hold_frames = int(self.cfg.alert_hold_sec * fps)
        frame_idx, n_report = 0, 0
        people = []
        last_alert_frame = -10 ** 9
        last_alert_labels = []
        alert_flags = []  # per output frame: is the alert active?
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % every == 0:
                # Detect + classify + object-gate (also refreshes self._frame_objects).
                people, _ = self.process_frame(frame)

            # Alert = any person doing an alert activity (not sit/stand).
            cur_labels = [lbl for _, (lbl, cat) in people if cat in ALERT_CATS]
            if cur_labels:
                last_alert_frame = frame_idx
                last_alert_labels = cur_labels
            alert_active = (frame_idx - last_alert_frame) <= hold_frames
            alert_flags.append(alert_active)

            writer.write(self._draw(
                frame, people, frame_idx=frame_idx,
                alert_active=alert_active, alert_text=last_alert_labels,
            ))
            frame_idx += 1
            if frame_idx - n_report >= 50:
                n_report = frame_idx
                print(f"[activity] {frame_idx} frames processed")
            if on_progress and total_frames and frame_idx % 10 == 0:
                on_progress(frame_idx, total_frames)

        cap.release()
        writer.release()
        print(f"[activity] wrote {frame_idx} frames -> {cfg.output_video}")

        if cfg.alarm_audio:
            self._add_siren(cfg.output_video, alert_flags, fps, frame_idx / fps)

        print(f"[activity] done in {time.time() - start:.1f}s")
        return cfg.output_video

    def _add_siren(self, video_path, alert_flags, fps, duration) -> None:
        """Generate a siren for the alert windows and mux it into the video."""
        intervals = alarm.flags_to_intervals(alert_flags, fps)
        if not intervals:
            print("[alert] no alert activities detected — no siren added.")
            return
        if not alarm.has_ffmpeg():
            print("[alert] ffmpeg not found — install it for the siren:\n"
                  "        pip install imageio-ffmpeg")
            return
        out = Path(video_path)
        wav = str(out.with_suffix(".siren.wav"))
        tmp = str(out.with_suffix(".snd.mp4"))
        try:
            alarm.build_siren_wav(wav, duration, intervals)
            alarm.mux_audio(video_path, wav, tmp)
            os.replace(tmp, video_path)
            os.remove(wav)
            print(f"[alert] siren embedded at {len(intervals)} alert window(s).")
        except Exception as exc:
            print(f"[alert] could not embed siren ({exc}).")