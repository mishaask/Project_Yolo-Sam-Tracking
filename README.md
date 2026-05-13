# Screening AI Project — YOLO + FastSAM + OSNet ReID Tracking

This repository is a Python prototype for detecting people and visible carried objects, tracking anonymous IDs over time, segmenting baggage, linking bags to likely owners, and exporting annotated video, JSON events, and CSV track logs.

## Current pipeline

Video / webcam
  -> YOLO detection
  -> BoT-SORT short-term tracking
  -> relevant-class whitelist and clutter filtering
  -> Nested ROI Search inside selected person/bag boxes
  -> duplicate cleanup
  -> OSNet person ReID + MemoryBank global IDs
  -> FastSAM/SAM masks for selected baggage/object classes
  -> person-bag relationship memory
  -> risk/event logic
  -> annotated MP4 + events JSON + tracks CSV

**Main idea:**

  YOLO finds visible objects.
  BoT-SORT tracks them locally for short time spans.
  OSNet ReID helps reconnect people after track loss.
  MemoryBank assigns stable project-level IDs like G1, G2, G3.
  FastSAM segments selected objects, mainly bags and weapons.
  Relationship memory links bags to likely nearby people.
  Risk logic writes possible unattended-bag and tracking events.

## Current status

**Implemented:**

[OK] Webcam/live-camera processing
[OK] Prerecorded video processing
[OK] YOLO object detection
[OK] BoT-SORT local tracking
[OK] Project global IDs: G1, G2, G3...
[OK] Torchreid/OSNet person ReID backend
[OK] Bad-crop guard for person ReID memory
[OK] Multiple good ReID snapshots per person
[OK] Entry/exit side continuity boost
[OK] Group-safe ID assignment so visible people do not collapse into one ID
[OK] Relevant-class whitelist
[OK] Nested ROI Search for visible objects inside person/bag boxes
[OK] FastSAM/SAM segmentation for selected object classes
[OK] Person-bag relationship memory
[OK] Owner-link display TTL to prevent stale lines staying on screen
[OK] Possible unattended-bag event logic
[OK] Annotated MP4 recording
[OK] Events JSON output
[OK] Per-frame tracks CSV output
[OK] Manual offline CSV/JSON merge helper


**Still in progress:**

[TODO] Integrate any trained weapon/suspicious-object detector cleanly into the main pipeline
[TODO] Tune thresholds on real team videos
[TODO] Improve event reports and dashboard/report view

## 4. Repository structure

Expected layout:


screening_ai_project_deep_reid/
|
|-- configs/
|   |-- data.yaml                 # YOLO training dataset config
|   |-- weapon_data.yaml          # Optional detector dataset config, if present
|   |-- classes.yaml              # Class groups: people, bags, suspicious objects
|   |-- botsort_reid.yaml         # BoT-SORT tracker settings
|   |-- risk_config.yaml          # Event/risk thresholds
|   |-- tracking_memory.yaml      # Global memory, ReID, filters, ROI search
|
|-- scripts/
|   |-- run_webcam.py             # Run live camera/webcam mode
|   |-- run_video.py              # Run prerecorded video mode
|   |-- run_realtime_sam_cpu.py   # CPU-friendly fixed demo launcher
|   |-- train_yolo.py             # Train custom YOLO detector
|   |-- create_dataset_folders.py # Create dataset/input/output folders
|   |-- check_reid_backend.py     # Verify OSNet/Torchreid backend
|   |-- smoke_test.py             # Quick functional test
|   |-- offline_merge_tracks.py   # Manual post-run ID merge helper
|   |-- import_zip_dataset.py     # Optional dataset import helper
|   |-- download_weapon_dataset.py # Optional dataset download helper, if used
|
|-- src/screening_ai/
|   |-- detector.py
|   |-- segmenter.py
|   |-- deep_reid.py
|   |-- appearance.py
|   |-- memory.py
|   |-- association.py
|   |-- risk.py
|   |-- visualization.py
|   |-- privacy.py                # Legacy/optional privacy utilities, if still present
|   |-- pipeline.py
|   |-- utils.py
|
|-- datasets/
|   |-- screening_dataset/
|   |   |-- images/train/
|   |   |-- images/val/
|   |   |-- labels/train/
|   |   |-- labels/val/
|
|-- input/                        # Put input videos here
|-- outputs/                      # Annotated video, JSON, CSV outputs
|-- requirements.txt
|-- requirements_reid_optional.txt
|-- requirements_windows_stable_reid.txt
|-- README.md


Always run commands from the project root, the folder that contains `scripts`, `src`, `configs`, and `README.md`.

### 5.1 Recommended Windows setup with real OSNet ReID

**note:** python3.12 is crucial.

py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
python scripts\smoke_test.py


Expected successful ReID check:


Checking Torchreid/OSNet backend...
Using Torchreid/OSNet backend: osnet_x0_25 on cpu
OSNet is available.


### 5.2 Check Python versions

py -0p

Recommended:

Python 3.12 64-bit

Avoid Python 3.14 for this project right now. The base packages may install, but Torchreid/OSNet is more reliable on Python 3.10-3.12.

### 5.3 Basic setup without guaranteed OSNet

This can work for general YOLO/SAM experiments, but it is not the safest path for real OSNet:

py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements_reid_optional.txt
python scripts\check_reid_backend.py

Use the stable file if OSNet says unavailable.

## 6. First validation checks

### 6.1 Create folders

python scripts\create_dataset_folders.py

This creates the dataset, input, and output folders.

### 6.2 Confirm OSNet

python scripts\check_reid_backend.py

Do not continue debugging person tracking quality until this says:

OSNet is available.

Normal first run:

Downloading...
From: https://drive.google.com/...
To: C:\Users\...\.cache\torch\checkpoints\osnet_x0_25_imagenet.pth

This is expected. Torchreid downloads OSNet weights the first time.

Harmless warnings:

UserWarning: Cython evaluation ... is unavailable
FutureWarning: You are using torch.load with weights_only=False

If the final line says OSNet is available, continue.

### 6.3 Smoke test

python scripts\smoke_test.py

Expected:

Smoke tests passed: association, reconnect, SAM mask geometry, and group-safe ReID work.

`smoke_test.py` checks project logic. It does not prove OSNet is active. Use `check_reid_backend.py` for that.

---

## 7. Recommended run commands

### 7.1  Recommended live demo command

Use this for the current full webcam demo:


python scripts\run_webcam.py --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display


This command is tuned for CPU and webcam testing. It keeps bag detections more easily while relying on class-specific filters to reduce bad person detections.

### 7.3 If bags still flicker too much

Lower global YOLO confidence slightly:

python scripts\run_webcam.py --weights yolo11n.pt --conf 0.22 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display

If this creates fake people, raise the person-specific confidence in `configs/tracking_memory.yaml`:

min_conf_by_class:
  person: 0.60

or:

min_conf_by_class:
  person: 0.65

### 7.4 Faster webcam command if CPU is too slow

```bat
python scripts\run_webcam.py --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 30 --sam-max-objects 1 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display --roi-every-n 20 --roi-max-parent-rois 1
```

If your current branch does not include `--roi-every-n` or `--roi-max-parent-rois`, tune the same values inside `configs/tracking_memory.yaml` instead.

### 7.5 Higher-quality but slower webcam command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --display
```

Use this when you want better boxes/crops and the computer can handle it.

### 7.6 Prerecorded video command

Put a video here:

input\test_video.mp4

Run:

python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv

For faster debugging without SAM:

python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --output-video outputs\test_no_sam_annotated.mp4 --output-json outputs\test_no_sam_events.json --output-tracks outputs\test_no_sam_tracks.csv

### 7.7 Use a full video path directly

python scripts\run_video.py --source "C:\Users\Misha\Desktop\my_video.mp4" --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --output-video outputs\my_video_annotated.mp4 --output-json outputs\my_video_events.json --output-tracks outputs\my_video_tracks.csv

## 8. Important command flags

### Input/output flags

--source
    Webcam camera index, video path, RTSP URL, etc.
    Webcam default: 0
    Video default: input/test_video.mp4

--weights
    YOLO model weights.
    Use yolo11n.pt before custom training.
    Use runs\screening\yolo_screening_detector\weights\best.pt after training.

--tracker
    BoT-SORT tracker config path.
    Default: configs/botsort_reid.yaml

--classes
    Class grouping config path.
    Default: configs/classes.yaml

--risk
    Risk/event threshold config path.
    Default: configs/risk_config.yaml

--memory
    Project MemoryBank/ReID config path.
    Default: configs/tracking_memory.yaml

--output-video
    Output annotated MP4 path.

--output-json
    Output event JSON path.

--output-tracks
    Output per-frame tracks CSV path.

### Detection/performance flags

--conf
    YOLO global confidence threshold.
    Lower values keep weak bag detections but can create more false positives.
    Recommended live tuning: 0.25 or 0.22.

--imgsz
    YOLO image size.
    320 = faster, less detail.
    480 = good webcam compromise.
    640 = better boxes/crops, slower on CPU.

--device cpu
    Force CPU.

--device 0
    Use CUDA GPU 0 if available.

--max-frames
    Stop after N frames. Useful for quick tests.

### SAM/FastSAM flags

--sam-weights
    SAM/FastSAM weights path.
    For CPU demos, pass FastSAM-s.pt explicitly.

--sam-every-n
    Run SAM once every N frames.
    Smaller = smoother masks but slower.
    Larger = faster but masks update less often.

--sam-max-objects
    Maximum detections to segment per SAM pass.
    Use 1 or 2 on CPU.

--sam-classes
    Comma-separated class names that SAM may segment.
    Example: backpack,handbag,suitcase

--prefer-sam-masks
    Use SAM mask geometry/appearance for configured object classes.
    Good for bags if YOLO boxes are rough.

--sam-tracking-classes
    Classes whose SAM masks may affect tracking geometry/appearance.
    Recommended: backpack,handbag,suitcase

--no-reuse-masks
    Do not reuse old SAM masks between SAM passes.
    Use if masks look stale or wrong.

--disable-sam
    Turn off SAM completely.
    Use for ReID/debug/FPS tests.
    Do not use if you expect masks/cropping.

### Visualization flags

--display
    Show preview window. Press q to stop.

--no-save-video
    Do not write annotated MP4.

--no-save-tracks
    Do not write tracks CSV.

--no-trails
    Do not draw movement trails.

--no-owner-links
    Do not draw person-bag owner links.
    Do not use this if the demo goal is to show bag-person connections.

Some branches may still include legacy privacy flags such as `--blur-faces` and `--pause-recording-on-face`. These are face detection/blurring utilities, not face recognition, and they are will not be in the final solution.

---

## 9. Main features explained

### 9.1 Relevant-class whitelist

Instead of blocking wrong classes one by one, the pipeline keeps only project-relevant classes.

Example config in `configs/tracking_memory.yaml`:

target_classes:
  - person
  - backpack
  - handbag
  - suitcase
  - trolley_bag
  - cell phone
  - phone
  - laptop
  - bottle
  - suspicious_object
  - dangerous_object
  - knife
  - gun
  - weapon

### 9.2 Nested ROI Search for overlapping boxes

Large boxes can hide smaller objects. For example:

- a backpack overlaps a person box,
- a phone or laptop is inside a person box,
- a visible object is attached to or on top of a bag,
- full-frame YOLO misses it because it is small or visually dominated by the parent object.

Nested ROI Search fixes this by running a second YOLO pass inside selected parent boxes.

Pipeline section:

full-frame YOLO + tracking
  -> choose selected parent boxes
  -> crop person/bag ROI with padding
  -> run YOLO predict() inside ROI
  -> map detections back to full-frame coordinates
  -> remove duplicates
  -> continue with SAM/ReID/memory/event logic

Important: ROI search uses YOLO `predict()`, not `track()`, so it should not corrupt the main BoT-SORT tracker state.

Typical config:

```yaml
roi_inner_search:
  enabled: true
  every_n_frames: 10
  parent_classes: ["person", "backpack", "handbag", "suitcase", "trolley_bag"]
  max_parent_rois_per_frame: 2
  max_inner_detections_per_roi: 5
  min_parent_confidence: 0.25
  roi_confidence: 0.18
  roi_imgsz: 320
  padding_ratio: 0.12
```

Exclusion rules:

Inside a person box:
  do not search for another person.

Inside a backpack/handbag/suitcase/trolley_bag:
  do not search for person or other bag classes.

Useful inner classes:
  phone, cell phone, laptop, bottle,
  suspicious_object, dangerous_object, knife, gun, weapon.

### 9.3 OSNet Gallery ReID

YOLO/BoT-SORT local IDs are useful but fragile. They can change when a person leaves the frame, becomes occluded, or is missed for several frames.

The project adds a MemoryBank layer:

BoT-SORT local ID: L4
Project global ID: G2
Displayed label: person G2 L4

For people, OSNet extracts an embedding from person crops:

person crop -> OSNet -> embedding vector -> compare to MemoryBank

Current ReID improvements:

- strict Torchreid/OSNet backend,
- bad-crop guard,
- multiple good snapshots per person,
- best-snapshot matching instead of simple averaging,
- entry/exit side continuity boost,
- group-safe assignment so visible people do not collapse into one ID,
- offline merge helper for post-run cleanup.

Bad-crop guard avoids updating memory from:

- side-clipped people,
- top-clipped people,
- tiny boxes,
- strange aspect ratios,
- partial bodies at frame edges.

This protects long-term memory but can make close webcam demos stricter. Stand farther from the camera so the full body is visible.

### 9.4 FastSAM/SAM segmentation

SAM/FastSAM is mainly used for selected object masks, especially:

backpack
handbag
suitcase
trolley_bag
weapons

It is not the main person identity mechanism. Person identity is handled by YOLO boxes + BoT-SORT + OSNet ReID.

### 9.5 Person-bag relationship memory

Bag ownership is estimated using visual relationship history.

A bag/object track stores fields such as:

owner_scores
owner_contact_frames
owner_separation_frames
owner_last_near_frame
owner_link_strength
owner_last_distance_px

Example logic:

bag G4 was near or overlapping person G2 for several seconds
bag G4 later became stationary
person G2 moved away
=> possible unattended bag event

This is an estimate.

## 10. Output files

### 10.1 Annotated video

Examples:

outputs\webcam_annotated.mp4
outputs\test_annotated.mp4
outputs\annotated_video.mp4

Shows:

bounding boxes
G global IDs
L local tracker IDs
confidence values
movement trails
person-bag owner links
SAM masks for selected objects
event messages

### 10.2 Events JSON

Examples:

outputs\webcam_events.json
outputs\test_events.json
outputs\events.json

Contains events such as:

track_reidentified
possible_unattended_bag
offline_track_merge
relationship/risk events

### 10.3 Tracks CSV

Examples:

outputs\webcam_tracks.csv
outputs\test_tracks.csv
outputs\tracks.csv

Useful columns:

frame
global_id
raw_global_id
offline_merged_into
local_tracker_id
class_name
confidence
bbox_x1, bbox_y1, bbox_x2, bbox_y2
center_x, center_y
owner_id
owner_link_strength
owner_contact_frames
owner_separation_frames
owner_last_distance_px
mask_area
used_mask_geometry
crop_quality
crop_quality_reason
last_observed_side
exit_side
entry_side
snapshot_count
reidentified_count
detection_source
parent_class_name
roi_level

ROI-related examples:

detection_source = main          # normal full-frame YOLO/BoT-SORT detection
detection_source = roi:person    # object found inside a person ROI
detection_source = roi:backpack  # object found inside a backpack ROI

---

## 11. Configuration files

### 11.1 `configs/botsort_reid.yaml`

Controls BoT-SORT short-term tracking.

Useful parameters:

track_buffer: 180
match_thresh: 0.72
track_high_thresh: 0.25
new_track_thresh: 0.32
with_reid: true

If the tracker loses people too quickly:

increase track_buffer
slightly lower match_thresh

If the tracker merges different people:

increase match_thresh
increase appearance/ReID thresholds

### 11.2 `configs/tracking_memory.yaml`

Controls project-level global memory, ReID, filters, class whitelist, SAM tracking classes, ROI search, and offline merge.

Important sections:

target_classes:
  # whitelist of allowed classes

min_conf_by_class:
  # per-class confidence thresholds

min_area_ratio_by_class:
  # per-class minimum bbox area

max_area_ratio_by_class:
  # per-class maximum bbox area

sam_tracking_classes:
  # classes whose SAM masks may affect tracking/memory geometry

person_reid:
  enabled: true
  backend: torchreid
  model_name: osnet_x0_25
  allow_torchvision_fallback: false
  require_backend: true

group_safe_assignment:
  enabled: true

roi_inner_search:
  enabled: true

offline_merge:
  enabled: true

### 11.3 `configs/classes.yaml`

Controls class groups.

Example:

person_classes:
  - person

bag_classes:
  - backpack
  - handbag
  - suitcase
  - trolley_bag

suspicious_classes:
  - suspicious_object
  - dangerous_object

### 11.4 `configs/risk_config.yaml`

Controls risk and event thresholds, especially person-bag ownership and unattended-bag logic.

For live demos, relationship thresholds may need to be more forgiving than final evaluation thresholds.

Example demo-style association tuning:

association:
  max_owner_distance_px: 260.0
  min_owner_score: 0.25
  motion_window_frames: 12
  score_decay: 0.98
  score_gain: 0.22
  min_contact_frames: 8
  contact_distance_px: 240.0
  separation_distance_px: 320.0

### 11.5 `configs/data.yaml`

YOLO training dataset config.

Example:

path: datasets/screening_dataset

train: images/train
val: images/val

names:
  0: person
  1: backpack
  2: handbag
  3: suitcase
  4: trolley_bag
  5: suspicious_object
  6: dangerous_object
  7: phone
  8: laptop
  9: bottle

Adjust this before training if the class list changes.

---

## 12. Tuning guide

### 12.1 Bags flicker or disappear

Try:

```bat
--conf 0.25
```

If still too strict:

```bat
--conf 0.22
```

Also lower per-class bag thresholds only if needed:

```yaml
min_conf_by_class:
  backpack: 0.22
  handbag: 0.22
  suitcase: 0.25
```

### 12.2 Fake people appear

Raise person-specific confidence:

```yaml
min_conf_by_class:
  person: 0.60
```

If still bad:

```yaml
min_conf_by_class:
  person: 0.65
```

Also use bbox shape/area filters and avoid static clutter zones.

### 12.3 Owner lines remain too long

Check owner-link TTL/display settings in the visualization or memory configuration. Relationship memory should persist internally, but visible lines should only be drawn when both tracks were seen recently.

Also check:

```text
- Did the bag/person actually disappear from tracking?
- Is FPS very low, causing frame-based TTL to feel too long?
- Is the owner-link drawing using last-seen tracks instead of current/recent tracks?
```

### 12.4 No person-bag connection appears

Check:

```text
- Did you use --no-owner-links?
- Did YOLO detect the bag as backpack/handbag/suitcase?
- Was the bag close to the person long enough?
- Are risk_config.yaml association thresholds too strict?
- Is FPS very low?
```

### 12.5 SAM masks look stale or wrong

Try one of these:

```bat
--sam-every-n 10
```

```bat
--no-reuse-masks
```

or remove:

```bat
--prefer-sam-masks
```

For tracking stability, it is often better to let YOLO boxes drive tracking and use SAM mainly for visualization/mask geometry.

### 12.6 FPS is too low

Fastest command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 320 --device cpu --display --no-save-video
```

Balanced CPU command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 30 --sam-max-objects 1 --sam-classes backpack,handbag,suitcase --imgsz 320 --device cpu --display
```

Check CUDA:

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is available:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device 0 --display
```

---

## 13. Troubleshooting

### 13.1 `OSNet is not available` after installing Torchreid

Symptom:

```text
Torchreid/OSNet backend requested but unavailable
OSNet is not available
```

Most likely cause:

```text
pip installed a too-new PyTorch/Numpy stack for old Torchreid.
```

Fix:

```bat
deactivate
rmdir /s /q .venv
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
```

Expected:

```text
OSNet is available.
```

### 13.2 Python 3.14 environment

Symptom:

```text
Installed packages show cp314
OSNet/Torchreid does not load correctly
```

Fix:

```bat
py -0p
py -3.12 -m venv .venv
```

Use Python 3.12 for this project.

### 13.3 Missing TensorBoard

Symptom:

```text
ModuleNotFoundError: No module named 'tensorboard'
```

Fix:

```bat
python -m pip install tensorboard
```

Better fix: use `requirements_windows_stable_reid.txt`, which includes TensorBoard.

### 13.4 Wrong Torchreid import path

Symptom:

```text
ModuleNotFoundError: No module named 'torchreid.utils'
```

Cause:

```text
Some torchreid installs expose FeatureExtractor at torchreid.reid.utils instead of torchreid.utils.
```

Fix:

Use the patched project files where `check_reid_backend.py` and `deep_reid.py` try both:

```python
from torchreid.utils import FeatureExtractor
# fallback:
from torchreid.reid.utils import FeatureExtractor
```

### 13.5 OSNet weight download fails

Symptom:

```text
Downloading...
From: https://drive.google.com/...
Download failed
```

Fixes:

```text
- Check internet connection.
- Try again later; Google Drive sometimes rate-limits.
- Keep gdown==4.7.3 from the stable requirements.
- Check whether the file already exists in:
  C:\Users\<username>\.cache\torch\checkpoints\osnet_x0_25_imagenet.pth
```

### 13.6 `best.pt` not found

Cause:

```text
You have not trained a custom model yet, or the path is wrong.
```

Fix:

```bat
--weights yolo11n.pt
```

Use `best.pt` only after training creates it or after you place the trained weights in the expected folder.

### 13.7 Input video not found

Cause:

```text
input\test_video.mp4 does not exist.
```

Fix:

```text
Put a file at input\test_video.mp4
```

or pass a full path:

```bat
python scripts\run_video.py --source "C:\Users\Misha\Desktop\my_video.mp4" --weights yolo11n.pt --disable-sam
```

### 13.8 `No module named screening_ai`

Cause:

```text
You are probably not running from the project root, or the src folder is missing.
```

Fix:

```bat
cd C:\Users\Misha\Desktop\screening_ai_project_deep_reid
python scripts\smoke_test.py
```

### 13.9 Webcam does not open

Try camera index 1:

```bat
python scripts\run_webcam.py --source 1 --weights yolo11n.pt --disable-sam --imgsz 480 --device cpu --display
```

Also close other apps using the camera.

### 13.10 Output video missing

Check whether you used:

```bat
--no-save-video
```

Also check that the `outputs` folder exists:

```bat
python scripts\create_dataset_folders.py
```

### 13.11 Same person becomes G2/G5/G8 later

Possible causes:

```text
- Person leaves frame for too long.
- Person re-enters with different pose/scale/lighting.
- Person crop is partial or side-clipped.
- Bad-crop guard refuses to update memory from partial crops.
- OSNet thresholds are strict to avoid wrong merges.
- FPS is low, causing fewer good observations.
```

Fixes:

```text
- Keep the full body visible during the demo.
- Improve lighting.
- Use --imgsz 480 or --imgsz 640.
- Avoid standing too close to the camera.
- Use prerecorded video and the offline merge helper for final reports.
```

### 13.12 Two people merge into one ID

Fixes:

```text
- Increase ReID/appearance thresholds in configs/tracking_memory.yaml.
- Keep group-safe assignment enabled.
- Avoid testing with two people in nearly identical clothing at first.
- Increase YOLO image size.
```

---

## 14. Training custom YOLO models

The pretrained `yolo11n.pt` model is only a baseline. For the final project, train YOLO on project-specific classes and camera angles.

### 14.1 Recommended training workflow

```text
Step 1: Collect safe project videos/images
Step 2: Extract useful frames
Step 3: Label frames in YOLO format
Step 4: Split data into train/val
Step 5: Update configs/data.yaml
Step 6: Train YOLO
Step 7: Validate results
Step 8: Run the pipeline with best.pt
Step 9: Tune tracking/memory/risk thresholds
Step 10: Add SAM for trained object classes
```

### 14.2 Dataset folder layout

```text
datasets/screening_dataset/
|-- images/
|   |-- train/
|   |-- val/
|-- labels/
|   |-- train/
|   |-- val/
```

Each image needs a matching `.txt` label file:

```text
images/train/frame_000123.jpg
labels/train/frame_000123.txt
```

### 14.3 YOLO label format

Each label row:

```text
class_id center_x center_y width height
```

All values are normalized from 0 to 1.

Example:

```text
0 0.512 0.438 0.231 0.604
```

Meaning:

```text
class 0, center x=0.512, center y=0.438, width=0.231, height=0.604
```

### 14.4 Recommended first classes

Start simple:

```text
person
backpack
handbag
suitcase
trolley_bag
phone
laptop
bottle
suspicious_object
dangerous_object
```

If you train or integrate a separate detector, make sure the class names match the names in `configs/data.yaml`, `configs/classes.yaml`, and `target_classes` in `configs/tracking_memory.yaml`.

For restricted/dangerous-object demonstrations, use only safe approved lab props, institution-approved datasets, or synthetic/clearly non-functional examples. Do not collect data with real dangerous items.

### 14.5 Create dataset folders

```bat
python scripts\create_dataset_folders.py
```

### 14.6 Train on CPU

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device cpu
```

CPU training is slow.

### 14.7 Train on GPU

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device 0
```

Expected trained model:

```text
runs\screening\yolo_screening_detector\weights\best.pt
```

### 14.8 Run with trained model

Webcam:

```bat
python scripts\run_webcam.py --weights runs\screening\yolo_screening_detector\weights\best.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --prefer-sam-masks --imgsz 640 --device cpu --display
```

Prerecorded video:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights runs\screening\yolo_screening_detector\weights\best.pt --conf 0.25 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 3 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --prefer-sam-masks --imgsz 640 --device cpu --output-video outputs\trained_annotated.mp4 --output-json outputs\trained_events.json --output-tracks outputs\trained_tracks.csv
```

### 14.9 Optional detector integration note

Earlier notes mention a trained knife/gun detector with strong validation metrics. Before documenting it as part of the final pipeline, verify that the actual weights file exists in the repository or shared drive and that `run_webcam.py` / `run_video.py` can load it directly or through a multi-model integration layer.

Do not claim a trained detector is active in the demo unless the command actually uses its weights.
