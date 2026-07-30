"""Configuration for the pose-based activity detector."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"


@dataclass
class PipelineConfig:
    """All tunable settings for the activity pipeline."""

    # --- I/O ---------------------------------------------------------------
    input_video: str = ""
    output_video: str = ""
    device: str = "cpu"

    # --- Pose model --------------------------------------------------------
    # RTMPose (top-down, ONNX, CPU-friendly). The UI value selects the size:
    #   "rtmpose"     -> "lightweight"  (Balanced — fast, the default)
    #   "rtmpose-acc" -> "balanced"     (Accurate — a bit slower)
    yolo_weights: str = "rtmpose"
    # Fallback RTMPose size if yolo_weights is just "rtmpose".
    rtmpose_mode: str = "lightweight"
    # Min confidence to keep a detected person.
    person_score_thresh: float = 0.4

    # --- Temporal stabilisation --------------------------------------------
    # Track each person over time and only report an activity that PERSISTED.
    # Window = last N processed frames per track; an activity must appear in at
    # least `hold_min_frames` of them to be reported (kills 1-2 frame flicker).
    # Larger = steadier but slower to react. Tune down if the model runs slowly.
    smooth_window: int = 9
    hold_min_frames: int = 3

    # --- Object confirmation ----------------------------------------------
    # A second YOLO (plain object/COCO) checks that the object the activity
    # implies is actually present near the hands, so an empty-handed pose does
    # NOT trigger an activity. "using phone" needs a phone; "taking picture"
    # and "placing object" need a phone / camera / any held object.
    # STRICT mode. False (default) = geometry drives detection and the object
    # box is drawn only as a non-blocking confirmation (works at any distance).
    # True = an activity is vetoed unless its object is detected near the hands
    # (fewer false positives, but needs the object to be visible / close range).
    require_object: bool = False
    # Object (COCO) checkpoint; auto-downloads. yolov8n.pt = fast.
    object_weights: str = "yolov8n.pt"
    # Min confidence to keep a detected object. Low, because a phone in-hand is
    # small / partly occluded and often only weakly detected.
    object_score_thresh: float = 0.12
    # Object detection runs zoomed-in on each person's region (ROI), so a small
    # / distant object like a phone is enlarged and becomes detectable. This is
    # the input resolution for that per-person crop. Larger = better small-object
    # recall but slower.
    object_imgsz: int = 640
    # Run the object detector every Nth processed frame and reuse its result in
    # between, so per-person detection does not tank the frame rate.
    object_every: int = 2
    # Remember objects seen in the last N object-detection passes, so a phone
    # that flickers in and out of detection still keeps its box / confirmation.
    object_memory_frames: int = 12

    # --- Speed -------------------------------------------------------------
    # Run pose every Nth frame and reuse the result in between (Nx faster).
    process_every: int = 1

    # --- Activity geometry (tunable rules) ---------------------------------
    # Min confidence to trust a body keypoint (wrist / ear / etc.). LOW so that
    # distant / CCTV poses, whose keypoints are weaker, are still used — this is
    # what lets a far-away "hand to the ear" register as using phone.
    kpt_conf_thresh: float = 0.15
    # Debug only: overlay the detected skeleton. OFF for users; set True to
    # visualise what the pose model is seeing.
    draw_keypoints: bool = False

    # --- Alert layer (does NOT affect detection) ---------------------------
    # Draw the red on-screen alert (border + banner) when an alert activity is
    # present (using phone / using laptop / taking picture / pick and drop).
    alert_visuals: bool = True
    # Embed a siren into the output video during alert windows (needs ffmpeg).
    alarm_audio: bool = True
    # Keep the alert (visual + siren) active this long after the last detection.
    alert_hold_sec: float = 1.0

