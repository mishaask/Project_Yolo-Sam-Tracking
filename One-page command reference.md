## One-page command reference

**Fresh Windows setup:**

py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
python scripts\smoke_test.py

Best OSNet result:

Using Torchreid/OSNet backend: osnet_x0_25 on cpu
OSNet is available.

**Tracking-only debug:**

python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 480 --device cpu --display


**Dual YOLO live demo: people/bags + trained risk-object model:**

python scripts\run_webcam.py --weights yolo11n.pt --secondary-weights best_v2.pt --secondary-target-classes knife,gun --secondary-conf 0.35 --secondary-iou 0.45 --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 10 --sam-max-objects 3 --sam-classes backpack,handbag,suitcase,knife,gun --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase,knife,gun --imgsz 640 --device cpu --display --output-video outputs\webcam_dual_yolo.mp4 --output-json outputs\webcam_dual_yolo_events.json --output-tracks outputs\webcam_dual_yolo_tracks.csv

**Dual YOLO faster CPU demo:**

python scripts\run_webcam.py --weights yolo11n.pt --secondary-weights best_v2.pt --secondary-target-classes knife,gun --secondary-conf 0.40 --secondary-iou 0.45 --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,knife,gun --prefer-sam-masks --imgsz 320 --device cpu --display --output-video outputs\webcam_dual_yolo_fast.mp4 --output-json outputs\webcam_dual_yolo_fast_events.json --output-tracks outputs\webcam_dual_yolo_fast_tracks.csv

**Disable risk smoothing for debugging only:**

Add --disable-risk-smoothing

**Recommended live demo:**

python scripts\run_webcam.py --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display

**If bags flicker:**

python scripts\run_webcam.py --weights yolo11n.pt --conf 0.22 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display

**Prerecorded video:**

python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv

**Disable SAM:**

python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 320 --device cpu --display --no-save-video

**Disable ROI search:**

python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display --disable-roi-search

**Do not forget:**

--disable-sam = no SAM masks/cropping.
--no-owner-links = no visible bag-person lines.
FastSAM-s.pt is preferred for CPU demos.
imgsz 320 is faster.
imgsz 480/640 gives better boxes but costs FPS.
Use OSNet check before judging ReID quality.
Use tracks CSV and events JSON to debug IDs and owner links.

Dual-YOLO notes:
--weights is the tracked general model for person/bag classes.
--secondary-weights is the trained risk-object model and runs with YOLO predict(), not BoT-SORT track().
--secondary-target-classes controls which secondary classes are merged into the main pipeline.
Risk smoothing is enabled by default: a risk-class object must appear near the same location 3 times in a 5-frame window before being drawn/logged/linked.

## Larger live UI + blue/white event overlay

```powershell
python scripts\run_webcam.py --weights yolo11n.pt --secondary-weights best_v2.pt --secondary-target-classes knife,gun --secondary-conf 0.55 --secondary-iou 0.30 --weapon-confirm-window 7 --weapon-confirm-min-hits 4 --weapon-confirm-match-center-px 80 --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 15 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,knife,gun --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase,knife,gun --imgsz 640 --device cpu --display --display-scale 1.5 --label-scale 0.42 --label-thickness 1 --event-overlay-position bottom-right --event-overlay-max-lines 5 --event-overlay-ttl-frames 150 --output-video outputs\webcam_dual_yolo_ui.mp4 --output-json outputs\webcam_dual_yolo_ui_events.json --output-tracks outputs\webcam_dual_yolo_ui_tracks.csv
```

Display/UI flags: `--display-scale`, `--display-width`, `--display-height`, `--label-scale`, `--label-thickness`, `--event-overlay-position`, `--event-overlay-max-lines`, `--event-overlay-ttl-frames`, `--no-event-overlay`.

## Dual YOLO + risk-cluster demo

```bat
python scripts\run_webcam.py --weights yolo11n.pt --secondary-weights best_v2.pt --secondary-target-classes knife,gun --secondary-conf 0.55 --secondary-iou 0.30 --weapon-confirm-window 7 --weapon-confirm-min-hits 4 --weapon-confirm-match-center-px 80 --risk-confirm-linked-frames 30 --risk-repeat-warning-count 10 --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 15 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,knife,gun --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase,knife,gun --imgsz 640 --device cpu --display --display-scale 1.5 --label-scale 0.42 --event-overlay-position bottom-right --debug-overlay-position bottom-left --output-video outputs\webcam_dual_yolo_cluster.mp4 --output-json outputs\webcam_dual_yolo_cluster_events.json --output-tracks outputs\webcam_dual_yolo_cluster_tracks.csv
```

Right overlay = risk warnings/alerts. Left overlay = tracking/ReID debug messages. Risk objects are displayed as temporary `R#` clusters instead of unstable weapon `G#` global IDs.
