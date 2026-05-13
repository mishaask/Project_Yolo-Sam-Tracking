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
