from __future__ import annotations

"""
CPU-friendly live demo with SAM-style masks.

This is not true SAM-on-every-frame real time. It runs YOLO tracking every frame,
runs a lightweight SAM-compatible model only every N frames, limits the number of
segmented objects, and reuses the latest masks between SAM passes. Person ID
tracking remains YOLO/BoT-SORT-box based by default; SAM masks are used for
bags/objects where precise shape is more helpful.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from screening_ai.pipeline import ScreeningPipeline


def main() -> None:
    pipeline = ScreeningPipeline(
        yolo_weights="yolo11n.pt",
        tracker_config="configs/botsort_reid.yaml",
        classes_config="configs/classes.yaml",
        risk_config="configs/risk_config.yaml",
        memory_config="configs/tracking_memory.yaml",
        sam_weights="FastSAM-s.pt",
        enable_sam=True,
        sam_every_n_frames=20,
        sam_max_objects=2,
        sam_classes={"backpack", "handbag", "suitcase"},
        reuse_last_masks=True,
        prefer_sam_masks=True,
        sam_tracking_classes={"backpack", "handbag", "suitcase"},
        conf=0.35,
        imgsz=320,
        device="cpu",
        draw_trails=True,
        draw_links=True,
        blur_faces=False,
    )

    pipeline.run(
        source="0",
        output_video="outputs/webcam_annotated.mp4",
        output_json="outputs/webcam_events.json",
        output_tracks_csv="outputs/webcam_tracks.csv",
        display=True,
        max_frames=None,
    )

    print("Done.")
    print("Video: outputs/webcam_annotated.mp4")
    print("Events: outputs/webcam_events.json")
    print("Tracks: outputs/webcam_tracks.csv")


if __name__ == "__main__":
    main()
