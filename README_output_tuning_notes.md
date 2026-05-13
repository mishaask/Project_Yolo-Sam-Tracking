# Output tuning notes

This version adds three fixes based on the latest webcam output:

1. **Relevant-class whitelist** remains enabled. Random COCO classes should not enter tracking/CSV.
2. **Lower global YOLO confidence, stricter person filter**: `--conf` now defaults to `0.25` so weak bag detections are not lost, while `person: 0.60` plus bbox-shape gates remove many chair/clothes false-person detections.
3. **Owner-link display TTL**: ownership memory still persists for abandoned-bag logic, but magenta Gbag->Gperson lines are drawn only while both tracks were visible very recently. This prevents stale links remaining on-screen after a box disappears.

Recommended live command:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

If bags are still flickering too much, use:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --conf 0.22 --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

If fake people appear again, raise in `configs/tracking_memory.yaml`:

```yaml
min_conf_by_class:
  person: 0.65
```
