# Screening AI Project — YOLO + FastSAM + OSNet ReID Tracking

This project is a Python prototype for a smart visual screening corridor.
It detects people and carried objects, tracks people across time, segments selected baggage objects, links baggage to the likely nearby person, and writes annotated video, JSON events, and CSV track logs.

Current high-level pipeline:

```text
Video / webcam
  -> YOLO detection
  -> BoT-SORT local tracking
  -> OSNet person ReID + MemoryBank global IDs
  -> FastSAM/SAM masks for selected baggage objects
  -> person-bag relationship memory
  -> risk/event logic
  -> annotated MP4 + events JSON + tracks CSV
```

Important privacy note: this project does **not** use face recognition. The system tracks anonymous project IDs such as `G1`, `G2`, etc. Face blur/pause options are only face detection for privacy, not identity recognition.

---

## 0. Read this first: recommended teammate setup

For Windows teammates, use the **stable Windows OSNet setup** below. Do **not** start with plain `requirements.txt` if the goal is to run real OSNet ReID, because pip may install a very new Torch/Numpy stack that breaks old Torchreid.

The setup that worked for us is:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
```

Expected successful ReID check:

```text
Checking Torchreid/OSNet backend...
Using Torchreid/OSNet backend: osnet_x0_25 on cpu
OSNet is available.
```

If you see this, OSNet is active and the project is using the real ReID backend.

---

## 1. Project status

Implemented:

```text
[OK] Webcam/live-camera processing
[OK] Prerecorded video processing
[OK] YOLO object detection
[OK] BoT-SORT short-term tracking
[OK] Project-level global IDs: G1, G2, G3...
[OK] Torchreid/OSNet person ReID backend
[OK] Bad-crop guard for person ReID memory
[OK] Multiple good ReID snapshots per person
[OK] Entry/exit side continuity boost
[OK] Group-safe ID assignment so visible people do not collapse into one ID
[OK] FastSAM/SAM segmentation for selected object classes
[OK] Person-bag relationship memory
[OK] Possible unattended-bag event logic
[OK] Annotated MP4 recording
[OK] Events JSON output
[OK] Per-frame tracks CSV output
[OK] Optional privacy face blur / pause recording on face detection
[OK] Manual offline CSV/JSON merge helper
```

Known limitations:

```text
- RGB video cannot detect hidden objects inside closed bags or under clothing.
- The project is a visual prototype, not a real security scanner.
- CPU-only real-time performance is limited.
- OSNet improves person re-identification but costs FPS on CPU.
- SAM/FastSAM is slower than YOLO and should not run on every frame on CPU.
- Bag ownership is estimated from proximity/motion/contact. It is not proof.
- YOLO pretrained weights are not trained for our exact project classes yet.
```

---

## 2. Repository structure

Expected project layout:

```text
screening_ai_project_deep_reid/
|
|-- configs/
|   |-- data.yaml                 # YOLO training dataset config
|   |-- classes.yaml              # Class groups: people, bags, suspicious objects, etc.
|   |-- botsort_reid.yaml         # BoT-SORT tracker settings
|   |-- risk_config.yaml          # Event/risk thresholds
|   |-- tracking_memory.yaml      # Global memory + ReID + relationship settings
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
|   |-- privacy.py
|   |-- pipeline.py
|
|-- datasets/
|-- input/
|-- outputs/
|-- requirements.txt
|-- requirements_reid_optional.txt
|-- requirements_windows_stable_reid.txt
|-- README.md
```

Always run commands from the project root, the folder that contains `scripts`, `src`, `configs`, and `README.md`.

---

## 3. Setup instructions

### 3.1 Check Python versions

Run:

```bat
py -0p
```

Recommended:

```text
Python 3.12 64-bit
```

Avoid Python 3.14 for this project right now. The base packages may install, but old Torchreid/OSNet is more reliable on Python 3.10-3.12.

---

### 3.2 Recommended Windows setup with real OSNet ReID

Use this for the project team.

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
python scripts\smoke_test.py
```

Why this file is preferred:

```text
requirements_windows_stable_reid.txt pins:
- numpy < 2
- torch == 2.5.1
- torchvision == 0.20.1
- torchreid == 0.2.5
- gdown == 4.7.3
- tensorboard >= 2.16.0
```

These pins avoid the exact OSNet issues we hit during setup.

---

### 3.3 Basic setup without guaranteed OSNet

This may work for general YOLO/SAM tests, but it is not the safest path for real OSNet:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -r requirements_reid_optional.txt
python scripts\check_reid_backend.py
```

Use the stable file if OSNet says unavailable.

---

### 3.4 PowerShell activation issue

If PowerShell blocks activation:

```text
Activate.ps1 cannot be loaded because running scripts is disabled
```

Use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

Or allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### 3.5 Verify active virtual environment

```bat
where python
where pip
```

The first result should be inside the project:

```text
...\screening_ai_project_deep_reid\.venv\Scripts\python.exe
```

---

## 4. First checks after setup

### 4.1 OSNet backend check

```bat
python scripts\check_reid_backend.py
```

Good output:

```text
Checking Torchreid/OSNet backend...
Using Torchreid/OSNet backend: osnet_x0_25 on cpu
OSNet is available.
```

Normal first-run behavior:

```text
Downloading...
From: https://drive.google.com/...
To: C:\Users\...\.cache\torch\checkpoints\osnet_x0_25_imagenet.pth
```

This is expected. Torchreid downloads the OSNet weights the first time.

Harmless warnings:

```text
UserWarning: Cython evaluation ... is unavailable
FutureWarning: You are using torch.load with weights_only=False
```

These warnings do not stop the project.

---

### 4.2 Smoke test

```bat
python scripts\smoke_test.py
```

Expected:

```text
Smoke tests passed: association, reconnect, SAM mask geometry, and group-safe ReID work.
```

Important: `smoke_test.py` checks project logic, but it does not prove OSNet is active. Use `check_reid_backend.py` for that.

---

## 5. Recommended execution plan

Use this order for a new teammate.

### Step 1 — Create folders

```bat
python scripts\create_dataset_folders.py
```

This creates:

```text
datasets/screening_dataset/images/train
datasets/screening_dataset/images/val
datasets/screening_dataset/labels/train
datasets/screening_dataset/labels/val
input
outputs
```

---

### Step 2 — Confirm OSNet

```bat
python scripts\check_reid_backend.py
```

Do not continue debugging tracking quality until this says:

```text
OSNet is available.
```

---

### Step 3 — Run tracking-only debug mode

Use this first to check person IDs and FPS without SAM overhead:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 480 --device cpu --display
```

What this does:

```text
- YOLO detects objects.
- BoT-SORT tracks short-term local IDs.
- OSNet + MemoryBank handles person ReID/global IDs.
- SAM is disabled, so no SAM masks/cropping appears.
- Person-bag links may still use YOLO boxes, but no SAM mask geometry is used.
```

Use this to answer: “Does person ReID work at all?”

---

### Step 4 — Run full SAM + bag-link demo

Use this when you want bag masks and person-bag links visible:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

What this does:

```text
- YOLO detects people and bags.
- BoT-SORT tracks local IDs.
- OSNet tracks anonymous person global IDs.
- FastSAM segments only backpack/handbag/suitcase.
- SAM masks can affect geometry for bag classes only.
- Person tracking stays YOLO/ReID-based.
- Owner links are drawn unless --no-owner-links is used.
```

Do not add `--disable-sam` if you expect SAM masks.

Do not add `--no-owner-links` if you expect visible person-bag lines.

---

### Step 5 — Run prerecorded video

Put a video here:

```text
input\test_video.mp4
```

Then run:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv
```

For faster video debugging without SAM:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --output-video outputs\test_no_sam_annotated.mp4 --output-json outputs\test_no_sam_events.json --output-tracks outputs\test_no_sam_tracks.csv
```

---

### Step 6 — Inspect outputs

Default webcam outputs:

```text
outputs\webcam_annotated.mp4
outputs\webcam_events.json
outputs\webcam_tracks.csv
```

Default video outputs:

```text
outputs\annotated_video.mp4
outputs\events.json
outputs\tracks.csv
```

Look for these CSV fields when debugging:

```text
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
```

Some columns may appear only in upgraded versions or after the offline merge stage.

---

## 6. Main run commands

### 6.1 Fastest person-ReID debug command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 320 --device cpu --display --no-save-video
```

Use when FPS is bad and you only want to see whether IDs are stable.

---

### 6.2 Balanced webcam demo command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 30 --sam-max-objects 1 --sam-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

Use when the full demo is too slow.

---

### 6.3 Full webcam demo command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --display
```

This gives better masks and larger detections, but it is slower on CPU.

---

### 6.4 Fixed CPU-friendly launcher

```bat
python scripts\run_realtime_sam_cpu.py
```

This uses a predefined CPU-friendly configuration from the script:

```text
weights: yolo11n.pt
SAM weights: FastSAM-s.pt
SAM every 20 frames
SAM max objects: 2
SAM classes: backpack, handbag, suitcase
imgsz: 320
device: cpu
```

---

### 6.5 Full video path directly

```bat
python scripts\run_video.py --source "C:\Users\Misha\Desktop\my_video.mp4" --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --output-video outputs\my_video_annotated.mp4 --output-json outputs\my_video_events.json --output-tracks outputs\my_video_tracks.csv
```

---

## 7. Command flags explained

The same core flags exist in `run_webcam.py` and `run_video.py`.

### Input/output flags

```text
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
```

### Detection/performance flags

```text
--conf
    YOLO confidence threshold.
    Default: 0.35.
    Lower means more detections but more false positives.
    Higher means fewer false positives but more missed bags/people.

--imgsz
    YOLO image size.
    320 = faster, less detail.
    480 = good webcam compromise.
    640 = better detection/crops, slower on CPU.

--device cpu
    Force CPU.

--device 0
    Use CUDA GPU 0 if available.

--max-frames
    Stop after N frames. Useful for quick tests.
```

### SAM/FastSAM flags

```text
--sam-weights
    SAM/FastSAM weights path.
    The scripts default to sam2_b.pt, but for CPU demos we recommend explicitly passing FastSAM-s.pt.

--sam-every-n
    Run SAM once every N frames.
    Smaller = smoother masks but slower.
    Larger = faster but masks update less often.

--sam-max-objects
    Maximum number of detections to segment per SAM pass.
    Use 1 or 2 on CPU.

--sam-classes
    Comma-separated class names that SAM may segment.
    Example: backpack,handbag,suitcase
    Use all only for debugging; it is slower.

--prefer-sam-masks
    Use SAM mask geometry/appearance for configured object classes.
    Good for bags if YOLO boxes are rough.
    If masks look stale or hurt tracking, remove this flag.

--sam-tracking-classes
    Classes whose SAM masks may affect tracking geometry/appearance.
    Recommended: backpack,handbag,suitcase
    Use none if SAM masks should be visual-only.
    Use all only for debugging.

--no-reuse-masks
    Do not reuse old SAM masks between SAM passes.
    Use if stale masks are visibly wrong.

--disable-sam
    Turn off SAM completely.
    Use for ReID/debug/FPS tests.
    Do not use if you expect SAM masks/cropping.
```

### Visualization/privacy flags

```text
--display
    Show preview window. Press q to stop.

--no-save-video
    Do not write annotated MP4.

--no-save-tracks
    Do not write CSV.

--blur-faces
    Blur face-like regions for privacy. This is not recognition.

--pause-recording-on-face
    Do not write video frames when a face-like region is visible.
    This is not recognition.

--no-trails
    Do not draw movement trails.

--no-owner-links
    Do not draw person-bag owner links.
    Do not use this if the demo goal is to show bag-person connection.
```

---

## 8. OSNet/ReID troubleshooting guide

This section documents the exact issues we hit so teammates do not repeat them.

### 8.1 `OSNet is not available` after installing `torchreid`

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

---

### 8.2 Python 3.14 environment

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

---

### 8.3 Missing TensorBoard

Symptom:

```text
ModuleNotFoundError: No module named 'tensorboard'
```

Fix:

```bat
python -m pip install tensorboard
```

Better fix: use `requirements_windows_stable_reid.txt`, which includes TensorBoard.

---

### 8.4 Wrong Torchreid import path

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

Do not use the old checker that only imports `torchreid.utils`.

---

### 8.5 Broken checker script syntax error

Symptom:

```text
SyntaxError: unterminated string literal (detected at line 27)
```

Cause:

```text
An old generated check_reid_backend.py had a broken multiline print string.
```

Fix:

Replace `scripts\check_reid_backend.py` with the fixed version from this project.

---

### 8.6 OSNet weight download fails

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

---

### 8.7 Harmless warnings

These are not blockers:

```text
Cython evaluation is unavailable, now use python evaluation
```

```text
FutureWarning: You are using torch.load with weights_only=False
```

If the final output says `OSNet is available`, continue.

---

## 9. SAM / bag-person relationship troubleshooting

### 9.1 “There is no SAM cropping/mask anymore”

Most common cause:

```text
You ran with --disable-sam.
```

Fix: use a SAM command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

Also check:

```text
- Do not use --disable-sam.
- Use --sam-weights FastSAM-s.pt for CPU.
- Make sure YOLO is detecting backpack/handbag/suitcase first.
- SAM segments YOLO detections; if YOLO misses the bag, SAM has nothing useful to segment.
```

---

### 9.2 “There is no connection between bag and person”

Check these first:

```text
- Did you accidentally use --no-owner-links?
- Did YOLO actually detect the bag as backpack/handbag/suitcase?
- Was the bag close to the person long enough?
- Is FPS very low? Frame-based contact thresholds take longer at low FPS.
- Are thresholds in configs/risk_config.yaml too strict for the demo?
```

For a live demo, you may need more forgiving relationship thresholds in `configs/risk_config.yaml`, for example:

```yaml
association:
  max_owner_distance_px: 260.0
  min_owner_score: 0.25
  motion_window_frames: 12
  score_decay: 0.98
  score_gain: 0.22
  min_contact_frames: 8
  contact_distance_px: 240.0
  separation_distance_px: 320.0
```

Use stricter values again for serious evaluation.

---

### 9.3 SAM masks look stale or wrong

Cause:

```text
SAM does not run every frame. Masks may be reused between SAM passes.
```

Fix options:

```bat
--sam-every-n 10
```

or:

```bat
--no-reuse-masks
```

or remove:

```bat
--prefer-sam-masks
```

For tracking stability, it is often better to let YOLO boxes drive tracking and use SAM masks mainly for visualization/segmentation.

---

## 10. FPS / performance troubleshooting

### 10.1 Why FPS is low after OSNet

Expected CPU cost:

```text
YOLO tracking
+ OSNet crop preprocessing
+ OSNet embedding inference
+ snapshot gallery matching
+ optional SAM segmentation
+ video writing
+ display rendering
```

This is heavier than the old color/HSV fallback.

---

### 10.2 Fastest FPS settings

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 320 --device cpu --display --no-save-video
```

---

### 10.3 Balanced CPU settings

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 30 --sam-max-objects 1 --sam-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

---

### 10.4 Higher quality but slower settings

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --display
```

---

### 10.5 GPU check

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If CUDA is available:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device 0 --display
```

---

## 11. Tracking/ReID quality troubleshooting

### 11.1 Same person becomes G2/G5/G8 later

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

---

### 11.2 Two people merge into one ID

Fixes:

```text
- Increase ReID/appearance thresholds in configs/tracking_memory.yaml.
- Keep group_safe_assignment enabled.
- Avoid testing with two people in nearly identical clothing at first.
- Increase YOLO image size.
```

---

### 11.3 Bad person crop / worse crop after ReID upgrade

The bad-crop guard prevents memory from being updated by:

```text
- side-clipped people
- top-clipped people
- tiny boxes
- strange aspect ratios
- partial bodies at frame edges
```

This protects long-term memory but can make a close webcam demo look stricter. Stand farther from the camera so the full body crop is visible.

---

## 12. Offline track merge helper

Use this when the video output split one real person into multiple global IDs and you want a corrected CSV/JSON after processing.

Create a mapping file, for example:

```json
{
  "2": 1,
  "5": 1,
  "6": 1
}
```

Save it as:

```text
outputs\merge_mapping.json
```

Run:

```bat
python scripts\offline_merge_tracks.py --tracks outputs\tracks.csv --events outputs\events.json --mapping outputs\merge_mapping.json --output-tracks outputs\tracks_merged.csv --output-events outputs\events_merged.json
```

Meaning:

```text
G2 -> G1
G5 -> G1
G6 -> G1
```

The script keeps `raw_global_id` and writes the merged `global_id`.

---

## 13. Training custom YOLO model

The pretrained model is only a baseline. For the final project, train YOLO on project-specific classes.

### 13.1 Create dataset folders

```bat
python scripts\create_dataset_folders.py
```

Expected layout:

```text
datasets/screening_dataset/
|-- images/train/
|-- images/val/
|-- labels/train/
|-- labels/val/
```

### 13.2 YOLO label format

Each `.txt` label row:

```text
class_id center_x center_y width height
```

All values are normalized from 0 to 1.

Example:

```text
0 0.512 0.438 0.231 0.604
```

### 13.3 Recommended starting classes

Start simple:

```text
person
backpack
handbag
suitcase
trolley_bag
suspicious_object
dangerous_object
phone
laptop
bottle
```

For restricted/dangerous-object demonstrations, use only safe approved lab props, toy/replica props where permitted by your institution, or approved datasets. Do not use real dangerous items for data collection.

### 13.4 Train on CPU

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device cpu
```

CPU training is slow.

### 13.5 Train on GPU

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device 0
```

Expected trained model:

```text
runs\screening\yolo_screening_detector\weights\best.pt
```

### 13.6 Run with trained model

```bat
python scripts\run_webcam.py --weights runs\screening\yolo_screening_detector\weights\best.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --imgsz 640 --device cpu --display
```

---

## 14. Common non-ReID errors

### 14.1 `best.pt` not found

Cause:

```text
You have not trained a custom model yet.
```

Fix:

```bat
--weights yolo11n.pt
```

Use `best.pt` only after training creates it.

---

### 14.2 Input video not found

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

---

### 14.3 `No module named screening_ai`

Cause:

```text
You are probably not running from the project root, or the src folder is missing.
```

Fix:

```bat
cd C:\Users\Misha\Desktop\screening_ai_project_deep_reid2
python scripts\smoke_test.py
```

---

### 14.4 Webcam does not open

Try camera index 1:

```bat
python scripts\run_webcam.py --source 1 --weights yolo11n.pt --disable-sam --imgsz 480 --device cpu --display
```

Close other apps using the camera.

---

### 14.5 Output video missing

Check whether you used:

```bat
--no-save-video
```

Also check that the `outputs` folder exists:

```bat
python scripts\create_dataset_folders.py
```

---

### 14.6 YOLO/FastSAM weights fail to download

If automatic download fails:

```text
- Check internet.
- Manually place yolo11n.pt or FastSAM-s.pt in the project root.
- Run the same command again.
```

---

## 15. Suggested demo script

Use a simple controlled scenario:

```text
1. Start webcam full demo.
2. Person enters with a visible backpack/handbag/suitcase.
3. Keep the bag close to the person for several seconds.
4. Show person G ID and bag G ID.
5. Show SAM mask on the bag.
6. Show owner link line between person and bag.
7. Put the bag down and move away.
8. Inspect events JSON and tracks CSV.
```

Recommended live demo command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

Use `--imgsz 640` if the machine can handle it.

---

## 16. Team workflow

Suggested split:

```text
Teammate 1: dataset collection + labeling
Teammate 2: YOLO training + evaluation
Teammate 3: tracking/ReID/SAM tuning
Teammate 4: event logic + report/dashboard
```

Before committing changes, run:

```bat
python scripts\check_reid_backend.py
python scripts\smoke_test.py
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 320 --device cpu --display --max-frames 100 --no-save-video
```

Git basics:

```bat
git checkout -b feature/my-feature
git status
git add .
git commit -m "Describe the change"
git push origin feature/my-feature
```

---

## 17. One-page quick reference

Fresh setup:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
python scripts\smoke_test.py
```

Tracking-only debug:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 480 --device cpu --display
```

Full SAM + bag-link demo:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 480 --device cpu --display
```

Prerecorded video:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --sam-tracking-classes backpack,handbag,suitcase --imgsz 640 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv
```

Best OSNet result:

```text
Using Torchreid/OSNet backend: osnet_x0_25 on cpu
OSNet is available.
```

Do not forget:

```text
--disable-sam = no SAM masks/cropping.
--no-owner-links = no visible bag-person lines.
FastSAM-s.pt is preferred for CPU demos.
imgsz 640 is better quality but slower.
imgsz 320/480 is better for live FPS.
```
