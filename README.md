# Screening AI Project — OSNet Gallery ReID Upgrade

## Latest changes in this ZIP

This version improves the person memory layer that sits above YOLO/BoT-SORT:

1. **Strict OSNet person ReID**
   - `configs/tracking_memory.yaml` now requests `person_reid.backend: torchreid` with `model_name: osnet_x0_25`.
   - The generic Torchvision fallback is disabled by default, so the app will not silently claim to use OSNet when Torchreid is missing.
   - Install it with:

```bat
pip install -r requirements.txt
pip install -r requirements_reid_optional.txt
python scripts\check_reid_backend.py
```

2. **Bad-crop guard**
   - Person appearance memory is not updated from side-clipped, top-clipped, tiny, or weird-aspect crops.
   - This prevents the gallery from being poisoned when the person is leaving the frame or only partly visible.
   - The CSV now records `crop_quality`, `crop_quality_reason`, `last_observed_side`, `exit_side`, and `entry_side`.

3. **Multiple good snapshots per person**
   - Each person track stores a small OSNet snapshot gallery instead of constantly averaging every crop into one embedding.
   - Matching compares against the best snapshot in the gallery, which handles pose/view changes better.

4. **Entry/exit side logic**
   - If a person exits on the right and later re-enters on the right, the score receives a small continuity bonus.
   - OSNet similarity is still required, so this is a boost, not an automatic merge.

5. **Higher default YOLO image size**
   - `run_video.py` and `run_webcam.py` now default to `--imgsz 640` for better person crops.
   - Lower to `--imgsz 320` only if your CPU is too slow.

6. **Offline merge pass**
   - After video processing, the pipeline compares completed person tracks and rewrites likely split IDs in `tracks.csv` and `events.json`.
   - Example: if `G1`, `G2`, `G5`, and `G6` are probably the same person, the CSV will keep `raw_global_id` but rewrite `global_id` to the canonical ID.
   - Offline merge events are added as `offline_track_merge` in the JSON report.

Recommended webcam run:

```bat
python scripts
un_webcam.py --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --display
```

Recommended video run:

```bat
python scripts
un_video.py --source input	est_video.mp4 --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --output-video outputsnnotated_video.mp4 --output-json outputs\events.json --output-tracks outputs	racks.csv
```

---

# Screening AI Prototype — YOLO + FastSAM + Deep ReID Tracking

This repository is a Python prototype for a smart visual screening pipeline.
It detects people and carried objects, tracks anonymous people across time, segments selected object classes, and records video/CSV/JSON outputs for debugging and project analysis.

The current recommended baseline is:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --display
```

This baseline means:

```text
YOLO detects people/bags/objects every frame.
BoT-SORT gives short-term local track IDs.
Deep anonymous ReID helps reconnect people after ID loss.
FastSAM segments only bags/handbags/suitcases every 20 frames.
The MemoryBank assigns stable project-level global IDs.
Relationship memory links people to carried bags over time.
The system saves annotated video, events JSON, and tracks CSV.
```

Important: this project does **not** perform facial recognition and does **not** identify real people by name. People are tracked only as anonymous IDs such as `G1`, `G2`, etc.

---

## 1. Current project status

Implemented:

```text
[OK] Webcam/live-camera processing
[OK] Prerecorded video processing
[OK] YOLO object detection
[OK] BoT-SORT short-term tracking
[OK] Project-level global ID memory: G1, G2, G3...
[OK] Deep anonymous person ReID embeddings
[OK] Group-safe ID assignment
[OK] FastSAM segmentation for selected object classes
[OK] Person-bag relationship memory
[OK] Possible unattended-bag event logic
[OK] Annotated MP4 recording
[OK] Events JSON output
[OK] Per-frame tracks CSV output
[OK] Optional privacy face blur / pause recording on face detection
```

Not fully solved yet:

```text
[TODO] Train custom YOLO model on our project-specific classes
[TODO] Improve dataset quality and annotation consistency
[TODO] Add offline track stitching for prerecorded videos
[TODO] Tune abandoned-bag event thresholds using real tests
[TODO] Add dashboard/report view
[TODO] Add true multi-sensor fusion later if hardware/data exists
```

Limitations:

```text
- RGB video cannot detect hidden objects inside closed bags or under clothing.
- CPU-only real-time performance is limited.
- FastSAM is used sparingly because segmentation is slower than detection.
- Deep ReID improves identity continuity but is still not perfect in crowds.
- The system estimates ownership; it cannot prove ownership.
```

---

## 2. Repository structure

```text
screening_ai_project_deep_reid/
│
├── configs/
│   ├── data.yaml                 # YOLO training dataset config
│   ├── classes.yaml              # Class groups: people, bags, suspicious objects, etc.
│   ├── botsort_reid.yaml         # BoT-SORT tracker settings
│   ├── risk_config.yaml          # Event/risk thresholds
│   └── tracking_memory.yaml      # Global memory + ReID + relationship settings
│
├── scripts/
│   ├── run_webcam.py             # Run live camera/webcam mode
│   ├── run_video.py              # Run prerecorded video mode
│   ├── train_yolo.py             # Train custom YOLO detector
│   ├── create_dataset_folders.py # Create dataset folder layout
│   ├── check_reid_backend.py     # Check active ReID backend
│   ├── run_realtime_sam_cpu.py   # CPU-friendly demo launcher
│   └── smoke_test.py             # Quick functional test
│
├── src/screening_ai/
│   ├── detector.py               # YOLO detector/tracker wrapper
│   ├── segmenter.py              # SAM/FastSAM wrapper
│   ├── deep_reid.py              # Deep person ReID backend
│   ├── appearance.py             # Appearance descriptors/fallbacks
│   ├── memory.py                 # Global ID MemoryBank
│   ├── association.py            # Person-bag relationship logic
│   ├── risk.py                   # Event/risk decision logic
│   ├── visualization.py          # Drawing boxes, trails, labels, links
│   ├── privacy.py                # Optional face blur/pause recording
│   └── pipeline.py               # Main pipeline orchestration
│
├── datasets/
│   └── screening_dataset/
│       ├── images/train/
│       ├── images/val/
│       ├── labels/train/
│       └── labels/val/
│
├── input/                        # Put input videos here
├── outputs/                      # Annotated video, JSON, CSV outputs
├── requirements.txt
├── requirements_reid_optional.txt
└── README.md
```

---

## 3. Setup instructions

### 3.1 Windows Command Prompt setup

From the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional stronger ReID backend:

```bat
python -m pip install -r requirements_reid_optional.txt
```

If the optional ReID install fails, the project still runs using the built-in Torchvision fallback.

### 3.2 Download SAM weights (required for segmentation)

SAM weights are not included in the repository. Download FastSAM-s (recommended, lightweight):

**macOS / Linux:**
```bash
curl -L https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-s.pt -o FastSAM-s.pt
```

**Windows:**
```bat
curl -L https://github.com/CASIA-IVA-Lab/FastSAM/releases/download/v1.0/FastSAM-s.pt -o FastSAM-s.pt
```

Place the file in the project root folder (next to `requirements.txt`).

If you prefer to skip segmentation, pass `--disable-sam` to any run command.

### 3.3 Windows PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt instead, or allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3.3 Verify that the virtual environment is active

```bat
where python
where pip
```

The first result should point to:

```text
...\screening_ai_project_deep_reid\.venv\Scripts\python.exe
```

---

## 4. Quick tests

### 4.1 Smoke test

```bat
python scripts\smoke_test.py
```

Expected:

```text
Smoke tests passed: association, reconnect, SAM mask geometry, and group-safe ReID work.
```

### 4.2 Check ReID backend

```bat
python scripts\check_reid_backend.py
```

Possible outputs:

```text
[ReID] Using Torchreid/OSNet backend: osnet_x0_25 on cpu
```

or:

```text
[ReID] Using Torchvision MobileNetV3 deep embedding fallback on cpu
```

or, if deep backends are unavailable:

```text
[ReID] Falling back to HSV appearance descriptors
```

Best situation:

```text
Torchreid/OSNet backend is active.
```

Acceptable situation:

```text
Torchvision MobileNetV3 backend is active.
```

Weakest fallback:

```text
HSV only.
```

---

## 5. Main run commands

### 5.1 Current recommended webcam baseline

Use this first:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --display
```

Outputs:

```text
outputs\webcam_annotated.mp4
outputs\webcam_events.json
outputs\webcam_tracks.csv
```

Press `q` in the preview window to stop.

### 5.2 Webcam without SAM

Useful when debugging tracking only:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --display
```

This often gives better raw person tracking because YOLO gets a larger input size and the CPU is not busy with segmentation.

### 5.3 CPU-friendly quick webcam test

```bat
python scripts\run_webcam.py --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --display --no-save-video --max-frames 100
```

### 5.4 Prerecorded video baseline

Put a video here:

```text
input\test_video.mp4
```

Run:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv
```

### 5.5 Prerecorded video without SAM

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --disable-sam --imgsz 640 --device cpu --output-video outputs\test_no_sam_annotated.mp4 --output-json outputs\test_no_sam_events.json --output-tracks outputs\test_no_sam_tracks.csv
```

### 5.6 Use a full video path directly

```bat
python scripts\run_video.py --source "C:\Users\Misha\Desktop\my_video.mp4" --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --output-video outputs\my_video_annotated.mp4 --output-json outputs\my_video_events.json --output-tracks outputs\my_video_tracks.csv
```

---

## 6. Useful command flags

```text
--weights yolo11n.pt
    YOLO model weights. Use yolo11n.pt before custom training.

--weights runs\screening\yolo_screening_detector\weights\best.pt
    Use this only after training creates best.pt.

--disable-sam
    Turn off SAM/FastSAM segmentation.

--sam-weights FastSAM-s.pt
    Use FastSAM small model for faster segmentation.

--sam-every-n 20
    Run SAM once every 20 frames.

--sam-max-objects 2
    Segment only the top 2 selected objects per SAM pass.

--sam-classes backpack,handbag,suitcase
    Only send these classes to SAM.

--prefer-sam-masks
    Use SAM mask geometry for selected object classes.
    Person tracking still stays YOLO/ReID-based unless configured otherwise.

--imgsz 640
    Faster, lower accuracy. Good for CPU webcam.

--imgsz 640
    Slower, better detection. Good for offline video or no-SAM tests.

--device cpu
    Force CPU mode.

--device 0
    Use GPU 0 if CUDA is available.

--display
    Show preview window.

--no-save-video
    Do not write annotated MP4.

--no-save-tracks
    Do not write tracks CSV.

--max-frames 100
    Stop after 100 frames.

--blur-faces
    Blur face-like regions for privacy. This is detection/blurring, not recognition.

--pause-recording-on-face
    Do not write video frames when a face-like region is visible.
```

---

## 7. Pipeline explanation

### 7.1 High-level pipeline

```text
Input video / webcam
    ↓
YOLO detection
    ↓
BoT-SORT local tracking
    ↓
Deep anonymous person ReID
    ↓
Project MemoryBank assigns global IDs
    ↓
FastSAM segmentation for selected object classes
    ↓
Person-bag relationship memory
    ↓
Risk/event logic
    ↓
Annotated MP4 + events JSON + tracks CSV
```

### 7.2 YOLO detection

YOLO detects visible objects in each frame.

Current pretrained `yolo11n.pt` can already detect common COCO classes such as:

```text
person
backpack
handbag
suitcase
bottle
laptop
cell phone
```

For the final project, we need to train YOLO on our own project-specific classes.

### 7.3 BoT-SORT short-term tracking

BoT-SORT gives local tracker IDs:

```text
person L4
backpack L8
```

These local IDs are useful but not fully reliable. They can change when a person leaves the frame, gets occluded, overlaps another person/object, or is missed by YOLO for a few frames.

### 7.4 Project-level global IDs

The MemoryBank maps local IDs to stable global IDs:

```text
YOLO/BoT-SORT local ID: L4
Project global ID:       G2
```

Displayed label example:

```text
person G2 L4
```

Meaning:

```text
BoT-SORT currently calls this local track L4.
Our project memory believes it belongs to anonymous person G2.
```

### 7.5 Deep anonymous person ReID

For person crops, the system extracts a deep embedding vector:

```text
person crop -> ReID model -> embedding vector
```

A memory entry stores:

```text
G2:
  class_name = person
  appearance_embedding = [0.13, -0.22, ...]
  last_bbox
  last_seen_frame
  local_ids_seen = [L4, L12]
  movement history
```

When a new local person track appears, the system compares its embedding to old global IDs.

Reconnect is allowed only if:

```text
1. class matches: person -> person
2. old global ID is not already visible this frame
3. appearance match is strong enough
4. position/motion is plausible
```

This is group-safe: two visible people should not become the same global ID.

### 7.6 FastSAM segmentation

FastSAM is currently used for:

```text
backpack
handbag
suitcase
```

Not for person identity.

Why:

```text
- Person tracking worked better when based on YOLO + ReID.
- SAM/FastSAM is useful for object masks.
- SAM is slower, especially on CPU.
```

### 7.7 Person-bag relationship memory

Bag ownership is not solved by person ReID.

Instead, every bag/object track stores relationship information:

```text
owner_scores
owner_contact_frames
owner_separation_frames
owner_last_near_frame
owner_link_strength
owner_last_distance_px
```

Example logic:

```text
bag G4 was near/overlapping person G2 for several seconds
bag G4 later became stationary
person G2 moved away
=> possible unattended bag event
```

This is an estimate, not proof.

---

## 8. Output files

### 8.1 Annotated video

Example:

```text
outputs\webcam_annotated.mp4
```

Shows:

```text
bounding boxes
G global IDs
L local tracker IDs
confidence values
movement trails
person-bag owner links
SAM masks for selected objects
important event messages
```

### 8.2 Events JSON

Example:

```text
outputs\webcam_events.json
```

Contains events like:

```text
track_reidentified
possible_unattended_bag
object/person relationship events
```

### 8.3 Tracks CSV

Example:

```text
outputs\webcam_tracks.csv
```

Useful columns:

```text
frame
global_id
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
reidentified_count
```

Use the CSV to debug:

```text
Did person G5 become G12?
Did the bag stay linked to the same owner?
How long was the bag stationary?
When did the tracker lose/reconnect a person?
```

---

## 9. Configuration files

### 9.1 `configs/botsort_reid.yaml`

Controls BoT-SORT short-term tracking.

Useful parameters:

```yaml
track_buffer: 180
match_thresh: 0.72
track_high_thresh: 0.25
new_track_thresh: 0.32
with_reid: true
```

If the tracker loses people too quickly:

```text
increase track_buffer
slightly lower match_thresh
```

If the tracker merges different people:

```text
increase match_thresh
increase appearance thresholds
make MemoryBank thresholds stricter
```

### 9.2 `configs/tracking_memory.yaml`

Controls project-level global memory and ReID.

Important sections:

```yaml
person_reid:
  enabled: true
  backend: auto
  model_name: osnet_x0_25

group_safe_assignment:
  enabled: true

person_memory:
  # thresholds for reconnecting old person IDs

object_memory:
  # thresholds for bags/objects

ignored_zones_norm:
  # optional background zones to ignore
```

### 9.3 `configs/classes.yaml`

Controls class groups.

Example groups:

```yaml
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
```

### 9.4 `configs/data.yaml`

YOLO training dataset config.

Example:

```yaml
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
```

Adjust this file before training if the class list changes.

---

## 10. Training the custom YOLO model

Training is the next major step.

Right now, `yolo11n.pt` is a general pretrained model. It is not trained for our exact project environment, camera angle, or custom object classes.

### 10.1 Recommended training workflow

```text
Step 1: Collect safe project videos/images
Step 2: Extract useful frames
Step 3: Label frames in YOLO format
Step 4: Split data into train/val
Step 5: Update configs/data.yaml
Step 6: Train YOLO
Step 7: Validate results
Step 8: Run the pipeline with best.pt
Step 9: Tune tracking/memory thresholds
Step 10: Add SAM for trained object classes
```

### 10.2 Dataset folder layout

```text
datasets/screening_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

Each image needs a matching `.txt` label file.

Example:

```text
images/train/frame_000123.jpg
labels/train/frame_000123.txt
```

### 10.3 YOLO label format

Each label row:

```text
class_id center_x center_y width height
```

All coordinates are normalized to 0-1.

Example:

```text
0 0.512 0.438 0.231 0.604
```

Meaning:

```text
class 0, centered at x=0.512, y=0.438, width=0.231, height=0.604
```

### 10.4 Class design recommendation

Start simple. Do not create too many classes at once.

Recommended first training classes:

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

For the first training round, the most important classes are:

```text
person
backpack
handbag
suitcase
suspicious_object
dangerous_object
```

Use only safe, approved, non-functional lab props or institution-provided data for restricted/dangerous-object demonstrations.

### 10.5 Create dataset folders

```bat
python scripts\create_dataset_folders.py
```

### 10.6 Train YOLO

CPU training is possible but slow. GPU is strongly recommended.

Basic CPU command:

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device cpu
```

GPU command if CUDA is available:

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8 --device 0
```

The trained model should appear here:

```text
runs\screening\yolo_screening_detector\weights\best.pt
```

### 10.7 Run pipeline with trained model

After training:

```bat
python scripts\run_webcam.py --weights runs\screening\yolo_screening_detector\weights\best.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --prefer-sam-masks --imgsz 640 --device cpu --display
```

For prerecorded video:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights runs\screening\yolo_screening_detector\weights\best.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 3 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object --prefer-sam-masks --imgsz 640 --device cpu --output-video outputs\trained_annotated.mp4 --output-json outputs\trained_events.json --output-tracks outputs\trained_tracks.csv
```

---

## 11. What to do after training

### 11.1 Validate detection quality

Check:

```text
Does the model detect people reliably?
Does it detect bags from different angles?
Does it confuse background furniture with bags?
Does it miss small objects?
Does it produce too many false positives?
```

### 11.2 Tune confidence threshold

If too many false positives:

```bat
--conf 0.45
```

If too many missed detections:

```bat
--conf 0.25
```

### 11.3 Tune SAM classes

Before training:

```bat
--sam-classes backpack,handbag,suitcase
```

After training:

```bat
--sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object
```

### 11.4 Tune relationship memory

Open:

```text
configs\tracking_memory.yaml
```

Tune parameters related to:

```text
minimum contact frames
maximum owner distance
stationary threshold
separation threshold
owner link strength threshold
```

### 11.5 Build test scenarios

Create short test videos for:

```text
1. Person enters with bag and leaves with bag.
2. Person enters with bag, puts it down, walks away.
3. Two people cross paths.
4. Two people with similar clothing pass each other.
5. Person leaves frame and returns later.
6. Bag is moved by another person.
7. Background has bag-like clutter.
```

Save outputs for each scenario and compare:

```text
annotated video
tracks CSV
events JSON
```

---

## 12. Troubleshooting

### 12.1 `best.pt` not found

Error:

```text
FileNotFoundError: runs\screening\yolo_screening_detector\weights\best.pt
```

Cause:

```text
You have not trained the custom model yet.
```

Fix:

```bat
--weights yolo11n.pt
```

Only use `best.pt` after training creates it.

### 12.2 Input video not found

Error:

```text
RuntimeError: Could not open source: input/test_video.mp4
```

Fix:

```text
Put a video at input\test_video.mp4
```

or pass full path:

```bat
python scripts\run_video.py --source "C:\Users\Misha\Desktop\my_video.mp4" --weights yolo11n.pt --disable-sam
```

### 12.3 Webcam is slow

Use:

```bat
--imgsz 640 --device cpu --disable-sam
```

or reduce SAM frequency:

```bat
--sam-every-n 40 --sam-max-objects 1
```

### 12.4 CUDA is not available

Check:

```bat
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If it prints:

```text
CUDA: False
CPU only
```

then use CPU-friendly commands.

### 12.5 The preview opens but output video is missing

Make sure you did not pass:

```bat
--no-save-video
```

Default webcam output:

```text
outputs\webcam_annotated.mp4
```

Default video output:

```text
outputs\annotated_video.mp4
```

### 12.6 People merge into one ID

This should be much harder now because of group-safe assignment.

Try:

```text
- Increase image size to 640.
- Install optional Torchreid backend.
- Make ReID thresholds stricter in configs/tracking_memory.yaml.
- Check if two people are visually too similar and heavily overlapping.
```

### 12.7 Same person becomes G12/G15 later

Try:

```text
- Install optional Torchreid backend.
- Use better lighting.
- Use higher imgsz if CPU allows.
- Reduce occlusion by testing with clearer movement.
- Later add offline track stitching for prerecorded videos.
```

---

## 13. Recommended next development tasks

### Task A — Dataset frame extraction mode

Add a script that saves frames for labeling:

```text
- every N frames
- frames with low confidence detections
- frames where person IDs switch
- frames where bag ownership changes
- frames with false positives
```

Suggested script name:

```text
scripts/extract_training_frames.py
```

Output:

```text
datasets/raw_frames/session_001/frame_000120.jpg
```

### Task B — Label the first dataset

Use any YOLO-compatible annotation tool.

Label consistently:

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

Keep labels simple for the first version.

### Task C — Train first custom YOLO model

Train with:

```bat
python scripts\train_yolo.py --data configs\data.yaml --base yolo11n.pt --epochs 80 --imgsz 640 --batch 8
```

Then run with:

```bat
--weights runs\screening\yolo_screening_detector\weights\best.pt
```

### Task D — Add offline track stitching

For prerecorded video, we can do a two-pass cleanup:

```text
Pass 1: run detection/tracking and save tracks.csv
Pass 2: analyze broken IDs and merge likely same-person tracks
Pass 3: render a corrected video/report
```

This can fix cases where a person briefly becomes `G12` and later should be merged back into `G2`.

### Task E — Improve abandoned-bag event report

Add a clean event summary:

```text
Event: possible unattended bag
Bag: G4
Likely owner: G2
First linked: frame 120
Separated: frame 820
Stationary duration: 12.4 seconds
Owner distance: 340 px
Confidence: medium/high
```

### Task F — Optional dashboard

Create a simple dashboard later:

```text
- Video preview
- Timeline of events
- Table of tracks
- Bag-owner graph
- Export report button
```

---

## 14. Suggested team workflow

Recommended split:

```text
Person 1: dataset collection + labeling
Person 2: YOLO training + evaluation
Person 3: tracking/ReID tuning
Person 4: event logic + report/dashboard
```

Use Git branches:

```text
main
feature/dataset-extraction
feature/yolo-training
feature/offline-track-stitching
feature/dashboard
```

Before merging, run:

```bat
python scripts\smoke_test.py
```

And test at least one webcam/video command.

---

## 15. Git basics for teammates

Clone:

```bat
git clone <repo-url>
cd screening_ai_project_deep_reid
```

Create branch:

```bat
git checkout -b feature/my-feature
```

Check changes:

```bat
git status
```

Commit:

```bat
git add .
git commit -m "Add my feature"
```

Push:

```bat
git push origin feature/my-feature
```

---

## 16. Recommended demo script

For a project demo, show:

```text
1. Run webcam baseline.
2. Show person G IDs.
3. Show bag segmentation mask.
4. Show owner link between person and bag.
5. Put bag down and move away.
6. Show event in webcam_events.json.
7. Open webcam_tracks.csv to show structured data.
```

Recommended command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --display
```

After custom training, replace:

```bat
--weights yolo11n.pt
```

with:

```bat
--weights runs\screening\yolo_screening_detector\weights\best.pt
```

---

## 17. Safety and privacy notes

```text
- The system tracks anonymous IDs, not real identities.
- It does not use facial recognition.
- Face blur is available for privacy.
- It is a visual prototype and should not be presented as a complete security scanner.
- Real-world screening would require careful validation, safe sensors, privacy review, and human oversight.
```

---

## 18. One-page summary

```text
Current best command:
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 640 --device cpu --display

Current pipeline:
YOLO -> BoT-SORT -> Deep ReID -> MemoryBank -> FastSAM for bags -> Person-bag relationship -> Video/JSON/CSV

Next major task:
Build dataset -> label -> train YOLO -> run with best.pt -> tune tracking/events.
```


## Windows OSNet troubleshooting: Torchreid installed but OSNet unavailable

If `python scripts\check_reid_backend.py` says OSNet is unavailable even after `torchreid` is installed, the usual cause is that pip installed a very new PyTorch stack that old Torchreid does not fully support.

Use the stable Windows ReID stack:

```bat
deactivate
rmdir /s /q .venv
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements_windows_stable_reid.txt
python scripts\check_reid_backend.py
```

Expected result:

```text
OSNet is available.
```

This version of `check_reid_backend.py` prints the real Torchreid exception if OSNet still fails, instead of only showing the generic unavailable message.
