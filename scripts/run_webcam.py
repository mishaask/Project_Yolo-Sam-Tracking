from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from screening_ai.pipeline import ScreeningPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Screening AI on webcam/live camera.")
    parser.add_argument("--source", default="0", help="Camera index, RTSP URL, or video path. Default: 0.")
    parser.add_argument("--weights", default="yolo11n.pt", help="YOLO weights path.")
    parser.add_argument("--tracker", default="configs/botsort_reid.yaml", help="Tracker YAML config.")
    parser.add_argument("--classes", default="configs/classes.yaml", help="Class groups YAML.")
    parser.add_argument("--risk", default="configs/risk_config.yaml", help="Risk config YAML.")
    parser.add_argument("--memory", default="configs/tracking_memory.yaml", help="Project-level memory/reID config YAML.")
    parser.add_argument("--output-video", default="outputs/webcam_annotated.mp4", help="Output annotated video path.")
    parser.add_argument("--output-json", default="outputs/webcam_events.json", help="Output JSON event report path.")
    parser.add_argument("--output-tracks", default="outputs/webcam_tracks.csv", help="Per-frame track CSV path.")
    parser.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size. Default is 640 for better person crops; lower to 320 only if CPU is too slow.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. 'cpu', '0', 'cuda:0'.")
    parser.add_argument("--sam-weights", default="sam2_b.pt", help="SAM/SAM2/MobileSAM weights path.")
    parser.add_argument("--sam-every-n", type=int, default=20, help="Run SAM once every N frames.")
    parser.add_argument("--sam-max-objects", type=int, default=2, help="Maximum detections to segment on each SAM pass.")
    parser.add_argument("--sam-classes", default="backpack,handbag,suitcase", help="Comma-separated class names allowed for SAM; use 'all' for all detections.")
    parser.add_argument("--prefer-sam-masks", action="store_true", help="Use SAM mask geometry/appearance for configured object classes. Person tracking stays YOLO-box based unless you include person in --sam-tracking-classes.")
    parser.add_argument("--sam-tracking-classes", default="", help="Classes whose SAM masks may affect tracking geometry/appearance. Empty = use config default; 'all' = all SAM classes; 'none' = never use SAM for tracking.")
    parser.add_argument("--no-reuse-masks", action="store_true", help="Do not reuse last SAM masks between SAM passes.")
    parser.add_argument("--disable-sam", action="store_true", help="Disable SAM/SAM2 segmentation.")
    parser.add_argument("--display", action="store_true", help="Show live preview window. Press q to stop.")
    parser.add_argument("--no-save-video", action="store_true", help="Do not save annotated video.")
    parser.add_argument("--no-save-tracks", action="store_true", help="Do not save per-frame track CSV.")
    parser.add_argument("--blur-faces", action="store_true", help="Blur detected face-like regions in the saved/displayed output for privacy. This is not recognition.")
    parser.add_argument("--no-trails", action="store_true", help="Disable drawing movement trails.")
    parser.add_argument("--no-owner-links", action="store_true", help="Disable drawing person-bag owner links.")
    parser.add_argument("--pause-recording-on-face", action="store_true", help="Privacy option: do not write video frames when a face-like region is visible. This is face detection only, not recognition.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick testing.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    sam_tracking_classes = None
    if args.sam_tracking_classes.lower() == "all":
        sam_tracking_classes = None if args.sam_classes.lower() == "all" else {x.strip() for x in args.sam_classes.split(",") if x.strip()}
    elif args.sam_tracking_classes.lower() == "none":
        sam_tracking_classes = set()
    elif args.sam_tracking_classes.strip():
        sam_tracking_classes = {x.strip() for x in args.sam_tracking_classes.split(",") if x.strip()}

    pipeline = ScreeningPipeline(
        yolo_weights=args.weights,
        tracker_config=args.tracker,
        classes_config=args.classes,
        risk_config=args.risk,
        memory_config=args.memory,
        sam_weights=args.sam_weights,
        enable_sam=not args.disable_sam,
        sam_every_n_frames=args.sam_every_n,
        sam_max_objects=args.sam_max_objects,
        sam_classes=None if args.sam_classes.lower() == "all" else {x.strip() for x in args.sam_classes.split(",") if x.strip()},
        reuse_last_masks=not args.no_reuse_masks,
        prefer_sam_masks=args.prefer_sam_masks,
        sam_tracking_classes=sam_tracking_classes,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        draw_trails=not args.no_trails,
        draw_links=not args.no_owner_links,
        blur_faces=args.blur_faces,
        pause_recording_on_face=args.pause_recording_on_face,
    )

    pipeline.run(
        source=args.source,
        output_video=None if args.no_save_video else args.output_video,
        output_json=args.output_json,
        output_tracks_csv=None if args.no_save_tracks else args.output_tracks,
        display=args.display,
        max_frames=args.max_frames,
    )

    print("Done.")
    if not args.no_save_video:
        print(f"Video: {args.output_video}")
    print(f"Events: {args.output_json}")
    if not args.no_save_tracks:
        print(f"Tracks: {args.output_tracks}")


if __name__ == "__main__":
    main()
