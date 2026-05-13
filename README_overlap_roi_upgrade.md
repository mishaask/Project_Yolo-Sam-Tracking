# Upgrade: Nested ROI Search for Overlapping Bounding Boxes

## Why this was added

The previous pipeline worked best with:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

However, it still had an important weakness: large boxes can hide smaller objects. For example:

- a backpack can overlap a person box,
- a visible object can be inside the person box,
- a visible weapon/item can be on top of or inside a bag box,
- full-frame YOLO may miss the small object because it is too small or visually dominated by the parent box.

The new version adds a second-pass detector called **Nested ROI Search**.

## New pipeline logic

The pipeline is now:

```text
Frame
  -> full-frame YOLO + BoT-SORT tracking
  -> clutter filtering
  -> nested ROI search inside selected person/bag boxes
  -> duplicate cleanup
  -> FastSAM/SAM masks for configured bag/object classes
  -> MemoryBank global IDs + Deep ReID for people
  -> person-bag ownership memory
  -> risk/event logic
  -> annotated video + events.json + tracks.csv
```

## What Nested ROI Search does

For selected parent classes, usually:

```yaml
person, backpack, handbag, suitcase, trolley_bag
```

it crops the parent bounding box with a small padding margin and runs plain YOLO `predict()` inside that crop.

Important: it uses `predict()`, not `track()`, so it does **not** corrupt the main BoT-SORT tracker state.

The nested detections are then mapped back to full-frame coordinates and added to the normal detection list.

## Exclusion rules

The nested search is class-aware.

Inside a `person` box, the system does **not** search for another `person`.

Inside a `backpack`, `handbag`, `suitcase`, or `trolley_bag`, the system does **not** search for:

```yaml
person, backpack, handbag, suitcase, trolley_bag
```

Instead, it searches for useful inner classes such as:

```yaml
phone, cell phone, laptop, bottle,
suspicious_object, dangerous_object, knife, gun, weapon
```

Before custom training, `yolo11n.pt` can only find its built-in COCO classes. After custom training, the same ROI logic will become more useful for your custom weapon/suspicious-object classes.

## Configuration

The settings are in:

```text
configs/tracking_memory.yaml
```

Section:

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

## Recommended webcam command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

Nested ROI Search is enabled by default in this version.

## Faster webcam command if CPU is too slow

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 30 --sam-max-objects 1 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display --roi-every-n 20 --roi-max-parent-rois 1
```

## Disable nested ROI search

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display --disable-roi-search
```

## Prerecorded video command

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --output-video outputs\test_annotated.mp4 --output-json outputs\test_events.json --output-tracks outputs\test_tracks.csv
```

## New output columns in tracks.csv

The CSV now includes:

```text
detection_source
parent_class_name
roi_level
```

Examples:

```text
main          -> normal full-frame YOLO/BoT-SORT detection
roi:person    -> object found inside a person ROI
roi:backpack  -> object found inside a backpack ROI
```

This helps you prove whether an object came from the normal detector or the nested ROI recovery pass.

## Important limitation

This does not detect hidden objects inside a closed bag or under clothing. It only helps with **visible** objects that were missed because of scale/overlap. True hidden-object detection requires additional non-ionizing sensors or sensor fusion.

## Recommended next step

Use this ROI version for data collection and testing, then train the YOLO model on your real project classes:

```yaml
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
knife
gun
weapon
```

After training, run the same pipeline with:

```bat
python scripts\run_video.py --source input\test_video.mp4 --weights runs\screening\yolo_screening_detector\weights\best.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase,suspicious_object,dangerous_object,knife,gun,weapon --prefer-sam-masks --imgsz 320 --device cpu
```
