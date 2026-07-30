#!/usr/bin/env python
"""SENITAL AI — pose-based activity detection (CLI).

Detects three activities from body pose (RTMPose keypoints + geometric rules,
stabilised over time): using phone, taking picture, placing object.

    python main.py --input data/input/clip.mp4
"""

import argparse
from pathlib import Path

from src.config import DEFAULT_OUTPUT_DIR, PipelineConfig
from src.pipeline import ActivityPosePipeline

_MODELS = {"balanced": "rtmpose", "accurate": "rtmpose-acc"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pose-based activity detection (RTMPose).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True, help="Path to input video.")
    p.add_argument("--output", "-o", default=None, help="Annotated output path.")
    p.add_argument("--model", choices=["balanced", "accurate"], default="balanced",
                   help="Pose model: balanced (fast) or accurate.")
    p.add_argument("--person-thresh", type=float, default=0.4,
                   help="Min confidence to keep a detected person.")
    p.add_argument("--process-every", type=int, default=1,
                   help="Run detection every Nth frame, reuse in between (Nx faster).")
    p.add_argument("--require-object", action="store_true",
                   help="Strict: only report an activity if a held object is also "
                        "detected in the hand.")
    p.add_argument("--object-thresh", type=float, default=0.12,
                   help="Min confidence for the object detector (strict mode).")
    p.add_argument("--debug-skeleton", action="store_true",
                   help="Overlay the detected skeleton (debugging).")

    # --- Alert layer ---
    p.add_argument("--no-alarm", action="store_true",
                   help="Do NOT embed the siren sound into the output video.")
    p.add_argument("--no-alert-visuals", action="store_true",
                   help="Do NOT draw the red on-screen alert border/banner.")
    p.add_argument("--alert-hold", type=float, default=1.0,
                   help="Keep the alert active this many seconds after the last "
                        "detection (prevents flicker).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{input_path.stem}_activity.mp4"

    cfg = PipelineConfig(
        input_video=str(input_path),
        output_video=str(output_path),
        device="cpu",
        yolo_weights=_MODELS[args.model],
        person_score_thresh=args.person_thresh,
        object_score_thresh=args.object_thresh,
        require_object=args.require_object,
        process_every=args.process_every,
        draw_keypoints=args.debug_skeleton,
        alarm_audio=not args.no_alarm,
        alert_visuals=not args.no_alert_visuals,
        alert_hold_sec=args.alert_hold,
    )
    out = ActivityPosePipeline(cfg).run()
    print(f"\nAnnotated video saved to: {out}")


if __name__ == "__main__":
    main()
