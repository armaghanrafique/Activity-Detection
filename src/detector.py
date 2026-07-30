"""Detectors: RTMPose (body keypoints) + a YOLO object detector.

RTMPose (top-down, ONNX) is the pose engine — it localises wrists reliably for
small / distant people on CPU. The YOLO object detector is only used to draw a
box on a held object / for the optional "require object" mode.
"""

import numpy as np
import torch


class YoloObjectDetector:
    """YOLOv8 object detector (COCO). Returns the objects a person could be
    holding — a phone (to confirm "using phone" / "taking picture") or any
    other object (to confirm "placing object") — so activities are only
    asserted when the matching object is actually present in the frame.

    ``detect`` returns a list of ``(box_xyxy, class_name, confidence)``, with
    the ``person`` class dropped (that is the pose model's job).
    """

    def __init__(self, device: str, score_thresh: float = 0.25,
                 weights: str = "yolov8n.pt", imgsz: int = 640):
        from ultralytics import YOLO

        self.device = device
        self.score_thresh = score_thresh
        self.imgsz = int(imgsz)
        self.model = YOLO(weights)
        self.model.to(device)
        self.names = self.model.names  # {class_id: name}

    @torch.no_grad()
    def detect(self, frame_rgb: np.ndarray):
        frame_bgr = frame_rgb[:, :, ::-1]
        results = self.model.predict(
            frame_bgr, conf=self.score_thresh, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )
        out = []
        if not results or results[0].boxes is None:
            return out
        b = results[0].boxes
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        cls = b.cls.cpu().numpy().astype(int)
        cnf = b.conf.cpu().numpy().astype(np.float32)
        for box, c, s in zip(xyxy, cls, cnf):
            name = self.names.get(int(c), str(c))
            if name == "person":
                continue
            out.append((box, name, float(s)))
        return out

class RTMPoseDetector:
    """RTMPose (top-down) via rtmlib/onnxruntime — a stronger, CPU-friendly pose
    estimator that places wrist keypoints far more reliably than YOLO-pose,
    especially for distant people in loose clothing. It runs its own person
    detector, so ``detect`` returns person boxes and stores their 17 COCO
    keypoints in ``last_keypoints`` with shape ``(N, 17, 3)`` as ``(x, y, conf)``.
    """

    def __init__(self, device: str = "cpu", mode: str = "balanced",
                 score_thresh: float = 0.4, imgsz: int = 640):
        from rtmlib import Body

        self.device = device
        self.score_thresh = score_thresh
        self.imgsz = int(imgsz)   # kept for interface parity (rtmlib self-sizes)
        self.model = Body(mode=mode, backend="onnxruntime", device=device)
        self.last_keypoints = np.zeros((0, 17, 3), dtype=np.float32)

    def detect(self, frame_rgb: np.ndarray, imgsz=None) -> np.ndarray:
        img_bgr = frame_rgb[:, :, ::-1]          # rtmlib expects BGR (cv2 order)
        kpts, scores = self.model(img_bgr)
        n = 0 if kpts is None else len(kpts)
        if n == 0:
            self.last_keypoints = np.zeros((0, 17, 3), dtype=np.float32)
            return np.zeros((0, 4), dtype=np.float32)
        K = np.zeros((n, 17, 3), dtype=np.float32)
        K[:, :, :2] = np.asarray(kpts)[:, :17, :2]
        K[:, :, 2] = np.asarray(scores)[:, :17]
        self.last_keypoints = K
        # Person boxes are derived from the visible keypoints (rtmlib's Body API
        # returns keypoints, not boxes) — good enough for scale + ROI cropping.
        boxes = np.zeros((n, 4), dtype=np.float32)
        for i in range(n):
            v = K[i][K[i][:, 2] > 0.3]
            if len(v) >= 2:
                x1, y1 = v[:, 0].min(), v[:, 1].min()
                x2, y2 = v[:, 0].max(), v[:, 1].max()
                pw, ph = (x2 - x1) * 0.12, (y2 - y1) * 0.12
                boxes[i] = [x1 - pw, y1 - ph, x2 + pw, y2 + ph]
        return boxes
