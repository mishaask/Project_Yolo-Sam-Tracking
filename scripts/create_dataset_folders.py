from pathlib import Path

folders = [
    "datasets/screening_dataset/images/train",
    "datasets/screening_dataset/images/val",
    "datasets/screening_dataset/labels/train",
    "datasets/screening_dataset/labels/val",
    "input",
    "outputs",
]

for folder in folders:
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"OK: {folder}")
