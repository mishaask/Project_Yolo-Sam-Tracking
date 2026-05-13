# Dangerous-object visualization and person-link upgrade

This version adds project behavior for detected risk classes such as `knife`, `gun`, `weapon`, `suspicious_object`, and `dangerous_object`.

## Important limitation

The system can only log a knife/gun if YOLO actually predicts that class. With `yolo11n.pt`, the model is a generic COCO model. It may classify a visible knife as `cell phone`, especially when the object is small, shiny, held in a hand, or partially occluded. In that case the code cannot safely know it is a knife. The real fix is to train the custom model on your own knife/gun/suspicious-object classes.

## What changed

1. Risk-class masks are drawn reddish/red instead of yellow.
2. Risk-class bounding boxes/labels are red and include `RISK`.
3. Risk classes are automatically prioritized for SAM when SAM is enabled. This works even if the command only says `--sam-classes backpack,handbag,suitcase`.
4. Detected risk objects are linked quickly to the nearest plausible person using the same relationship-memory fields as bags, but with faster thresholds.
5. `webcam_events.json` now logs `risk_object_detected` events with `owner_id` when an owner/person link exists.
6. `weapon` was added to `configs/classes.yaml` risk classes.

## Current baseline command

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

After training, use your custom model:

```bat
python scripts\run_webcam.py --weights runs\screening\yolo_screening_detector\weights\best.pt --sam-weights FastSAM-s.pt --sam-every-n 10 --sam-max-objects 3 --sam-classes backpack,handbag,suitcase,knife,gun,weapon,suspicious_object,dangerous_object --prefer-sam-masks --imgsz 640 --device cpu --display
```

## Config files

Risk classes are defined in:

```text
configs/classes.yaml
```

Fast person-link behavior for risk objects is defined in:

```text
configs/risk_config.yaml
```

Look for:

```yaml
risk_object_link:
  max_owner_distance_px: 230.0
  min_owner_score: 0.16
  score_gain: 0.70
  min_contact_frames: 1
```

Bags still use slower ownership confirmation because abandoned-bag logic requires stronger evidence over time.
