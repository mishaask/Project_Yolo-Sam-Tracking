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
    parser.add_argument("--target-classes", default=None, help="Optional comma-separated whitelist of YOLO class names to keep. Default = use configs/tracking_memory.yaml target_classes. Use all to disable whitelist.")
    parser.add_argument("--secondary-weights", default=None, help="Optional second YOLO weights file for risk-object detection, e.g. best_v2.pt. This runs with predict(), not track().")
    parser.add_argument("--secondary-target-classes", default="knife,gun", help="Comma-separated class names to keep from the secondary YOLO model. Use all to keep every secondary class.")
    parser.add_argument("--secondary-conf", type=float, default=None, help="Secondary YOLO confidence threshold. Default comes from configs/risk_config.yaml.")
    parser.add_argument("--secondary-imgsz", type=int, default=None, help="Secondary YOLO image size. Default comes from configs/risk_config.yaml or --imgsz.")
    parser.add_argument("--secondary-iou", type=float, default=None, help="Secondary YOLO NMS IoU threshold. Lower values reduce duplicate boxes.")
    parser.add_argument("--secondary-max-det", type=int, default=None, help="Maximum secondary detections per frame.")
    parser.add_argument("--disable-risk-smoothing", action="store_true", help="Disable 3-of-5 style temporal confirmation for risk-class detections.")
    parser.add_argument("--weapon-confirm-window", type=int, default=None, help="Risk confirmation window in frames. Default: risk_config.yaml.")
    parser.add_argument("--weapon-confirm-min-hits", type=int, default=None, help="Minimum matching risk detections inside the confirmation window. Default: risk_config.yaml.")
    parser.add_argument("--weapon-confirm-match-iou", type=float, default=None, help="IoU needed to match risk boxes across frames for confirmation.")
    parser.add_argument("--weapon-confirm-match-center-px", type=float, default=None, help="Fallback center-distance match threshold for risk boxes across frames.")
    parser.add_argument("--disable-risk-clusters", action="store_true", help="Disable risk-object cluster warnings/confirmed-event logic.")
    parser.add_argument("--risk-cluster-match-iou", type=float, default=None, help="IoU threshold for grouping confirmed risk detections into the same temporary R# cluster.")
    parser.add_argument("--risk-cluster-match-center-px", type=float, default=None, help="Center-distance threshold for grouping confirmed risk detections into the same temporary R# cluster.")
    parser.add_argument("--risk-cluster-ttl-frames", type=int, default=None, help="How long an unseen R# risk cluster stays alive.")
    parser.add_argument("--risk-confirm-linked-frames", type=int, default=None, help="Frames the same person must stay linked to the same R# risk cluster before an ALERT event is created.")
    parser.add_argument("--risk-warning-cooldown-frames", type=int, default=None, help="Minimum frame gap between WARNING events from the same R# risk cluster.")
    parser.add_argument("--risk-repeat-warning-window-frames", type=int, default=None, help="Window used for counting repeated warnings per person.")
    parser.add_argument("--risk-repeat-warning-count", type=int, default=None, help="Warnings in the repeat window needed to escalate a person to repeated-warning ALERT.")
    parser.add_argument("--output-video", default="outputs/webcam_annotated.mp4", help="Output annotated video path.")
    parser.add_argument("--output-json", default="outputs/webcam_events.json", help="Output JSON event report path.")
    parser.add_argument("--output-tracks", default="outputs/webcam_tracks.csv", help="Per-frame track CSV path.")
    parser.add_argument("--conf", type=float, default=0.25, help="Base YOLO confidence threshold. Keep low for weak bag detections; class-specific filters in tracking_memory.yaml keep person stricter.")
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
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale the live preview window only. Example: 1.5 makes the on-screen window larger without changing saved video size.")
    parser.add_argument("--display-width", type=int, default=None, help="Optional explicit live preview window width in pixels.")
    parser.add_argument("--display-height", type=int, default=None, help="Optional explicit live preview window height in pixels.")
    parser.add_argument("--label-scale", type=float, default=0.48, help="Text scale for bbox labels, owner-link labels, and FPS. Smaller values reduce clutter.")
    parser.add_argument("--label-thickness", type=int, default=1, help="Text thickness for bbox labels, owner-link labels, FPS, and event overlay.")
    parser.add_argument("--event-overlay", dest="event_overlay", action="store_true", default=True, help="Show a blue/white event notification feed on the annotated frame.")
    parser.add_argument("--no-event-overlay", dest="event_overlay", action="store_false", help="Disable the event notification feed overlay.")
    parser.add_argument("--event-overlay-position", choices=["bottom-right", "bottom-left", "top-right", "top-left"], default="bottom-right", help="Corner used for the event notification feed.")
    parser.add_argument("--event-overlay-max-lines", type=int, default=5, help="Maximum visible event notification lines.")
    parser.add_argument("--event-overlay-ttl-frames", type=int, default=120, help="How long event notifications remain visible, measured in frames.")
    parser.add_argument("--event-overlay-scale", type=float, default=0.46, help="Text scale for the event notification feed.")
    parser.add_argument("--debug-overlay", dest="debug_overlay", action="store_true", default=True, help="Show tracking/ReID debug messages in a separate blue/white feed.")
    parser.add_argument("--no-debug-overlay", dest="debug_overlay", action="store_false", help="Disable the tracking/ReID debug feed.")
    parser.add_argument("--debug-overlay-position", choices=["bottom-right", "bottom-left", "top-right", "top-left"], default="bottom-left", help="Corner used for tracking/ReID debug notifications.")
    parser.add_argument("--debug-overlay-max-lines", type=int, default=4, help="Maximum visible tracking/ReID debug lines.")
    parser.add_argument("--debug-overlay-ttl-frames", type=int, default=150, help="How long tracking/ReID debug notifications remain visible, measured in frames.")
    parser.add_argument("--debug-overlay-scale", type=float, default=0.42, help="Text scale for the tracking/ReID debug feed.")
    parser.add_argument("--no-save-video", action="store_true", help="Do not save annotated video.")
    parser.add_argument("--no-save-tracks", action="store_true", help="Do not save per-frame track CSV.")
    parser.add_argument("--blur-faces", action="store_true", help="Blur detected face-like regions in the saved/displayed output for privacy. This is not recognition.")
    parser.add_argument("--no-trails", action="store_true", help="Disable drawing movement trails.")
    parser.add_argument("--no-owner-links", action="store_true", help="Disable drawing person-bag owner links.")
    parser.add_argument("--pause-recording-on-face", action="store_true", help="Privacy option: do not write video frames when a face-like region is visible. This is face detection only, not recognition.")
    parser.add_argument("--disable-roi-search", action="store_true", help="Disable nested YOLO ROI search inside person/bag boxes.")
    parser.add_argument("--roi-every-n", type=int, default=None, help="Override nested ROI search frequency. Example: 10 means every 10 frames.")
    parser.add_argument("--roi-conf", type=float, default=None, help="Override nested ROI YOLO confidence threshold.")
    parser.add_argument("--roi-imgsz", type=int, default=None, help="Override nested ROI YOLO image size.")
    parser.add_argument("--roi-max-parent-rois", type=int, default=None, help="Maximum parent person/bag boxes searched per ROI pass.")
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

    target_classes_arg = None
    if args.target_classes is not None and args.target_classes.lower() != "all":
        target_classes_arg = {x.strip() for x in args.target_classes.split(",") if x.strip()}
    elif args.target_classes is not None and args.target_classes.lower() == "all":
        target_classes_arg = {"all"}

    secondary_target_classes_arg = None
    if args.secondary_target_classes is not None and args.secondary_target_classes.lower() != "all":
        secondary_target_classes_arg = {x.strip() for x in args.secondary_target_classes.split(",") if x.strip()}
    elif args.secondary_target_classes is not None and args.secondary_target_classes.lower() == "all":
        secondary_target_classes_arg = {"all"}

    pipeline = ScreeningPipeline(
        yolo_weights=args.weights,
        tracker_config=args.tracker,
        classes_config=args.classes,
        risk_config=args.risk,
        memory_config=args.memory,
        target_classes=target_classes_arg,
        secondary_weights=args.secondary_weights,
        secondary_target_classes=secondary_target_classes_arg,
        secondary_conf=args.secondary_conf,
        secondary_imgsz=args.secondary_imgsz,
        secondary_iou=args.secondary_iou,
        secondary_max_det=args.secondary_max_det,
        risk_smoothing_enabled=None if not args.disable_risk_smoothing else False,
        weapon_confirm_window=args.weapon_confirm_window,
        weapon_confirm_min_hits=args.weapon_confirm_min_hits,
        weapon_confirm_match_iou=args.weapon_confirm_match_iou,
        weapon_confirm_match_center_px=args.weapon_confirm_match_center_px,
        risk_cluster_enabled=None if not args.disable_risk_clusters else False,
        risk_cluster_match_iou=args.risk_cluster_match_iou,
        risk_cluster_match_center_px=args.risk_cluster_match_center_px,
        risk_cluster_ttl_frames=args.risk_cluster_ttl_frames,
        risk_confirm_linked_frames=args.risk_confirm_linked_frames,
        risk_warning_cooldown_frames=args.risk_warning_cooldown_frames,
        risk_repeat_warning_window_frames=args.risk_repeat_warning_window_frames,
        risk_repeat_warning_count=args.risk_repeat_warning_count,
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
        enable_roi_search=not args.disable_roi_search,
        roi_every_n_frames=args.roi_every_n,
        roi_confidence=args.roi_conf,
        roi_imgsz=args.roi_imgsz,
        roi_max_parent_rois=args.roi_max_parent_rois,
        label_scale=args.label_scale,
        label_thickness=args.label_thickness,
        event_overlay_enabled=args.event_overlay,
        event_overlay_position=args.event_overlay_position,
        event_overlay_max_lines=args.event_overlay_max_lines,
        event_overlay_ttl_frames=args.event_overlay_ttl_frames,
        event_overlay_scale=args.event_overlay_scale,
        debug_overlay_enabled=args.debug_overlay,
        debug_overlay_position=args.debug_overlay_position,
        debug_overlay_max_lines=args.debug_overlay_max_lines,
        debug_overlay_ttl_frames=args.debug_overlay_ttl_frames,
        debug_overlay_scale=args.debug_overlay_scale,
    )

    pipeline.run(
        source=args.source,
        output_video=None if args.no_save_video else args.output_video,
        output_json=args.output_json,
        output_tracks_csv=None if args.no_save_tracks else args.output_tracks,
        display=args.display,
        max_frames=args.max_frames,
        display_scale=args.display_scale,
        display_width=args.display_width,
        display_height=args.display_height,
    )

    print("Done.")
    if not args.no_save_video:
        print(f"Video: {args.output_video}")
    print(f"Events: {args.output_json}")
    if not args.no_save_tracks:
        print(f"Tracks: {args.output_tracks}")


if __name__ == "__main__":
    main()
