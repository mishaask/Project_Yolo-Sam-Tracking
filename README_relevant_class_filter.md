# Relevant Class Whitelist Upgrade

This upgrade replaces the old idea of blocking random wrong classes one by one, such as:

```yaml
cat: 1.00
carrot: 1.00
tie: 1.00
teddy bear: 1.00
```

with a positive whitelist. The pipeline now keeps only project-relevant classes.

## Where it is configured

Open:

```text
configs/tracking_memory.yaml
```

The important section is:

```yaml
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
```

Any YOLO class outside this list is ignored before it reaches memory, SAM, CSV, events, or visualization.

## Why this is better

A blocklist is endless. If YOLO mistakes a chair for `cat` today, tomorrow it might say `tie`, `carrot`, or `teddy bear`. A whitelist says: only classes relevant to our screening pipeline are allowed.

## Important limitation

A whitelist does not solve false detections when YOLO uses a relevant label incorrectly. For example, if clothes on a chair are detected as `person`, that still passes the whitelist because `person` is relevant. That case is handled with class-specific confidence and area thresholds:

```yaml
min_conf_by_class:
  person: 0.50
  backpack: 0.30
  handbag: 0.30
  suitcase: 0.35
```

If false people still appear, raise `person` to `0.55` or `0.60`.

## Run normally

```bat
python scripts\run_webcam.py --weights yolo11n.pt --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

The config whitelist is used automatically.

## Temporarily override the whitelist from command line

Example: keep only people and bags:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --target-classes person,backpack,handbag,suitcase --sam-weights FastSAM-s.pt --sam-every-n 20 --sam-max-objects 2 --sam-classes backpack,handbag,suitcase --prefer-sam-masks --imgsz 320 --device cpu --display
```

Disable whitelist for debugging:

```bat
python scripts\run_webcam.py --weights yolo11n.pt --target-classes all --disable-sam --imgsz 640 --device cpu --display
```

## After training

After training the custom model, add any new class names that the model outputs to `target_classes`, for example:

```yaml
  - pistol
  - rifle
  - explosive_object
  - suspicious_package
```

The class names must match the names in the trained YOLO model/data YAML.
