"""
Download and merge knife + gun datasets from Roboflow Universe.

Requirements:
    pip install roboflow

Usage:
    python scripts/download_weapon_dataset.py --api-key YOUR_KEY

    # Or override with specific datasets you found on Roboflow:
    python scripts/download_weapon_dataset.py --api-key YOUR_KEY \\
        --knife-workspace my-workspace --knife-project knife-detection --knife-version 1 \\
        --gun-workspace   my-workspace --gun-project   gun-detection   --gun-version 1

How to find datasets on Roboflow Universe (free, no annotation needed):
    1. Go to https://universe.roboflow.com
    2. Search "knife detection" → pick a dataset → click "Download Dataset"
       → choose "YOLOv8" format → get the Python snippet
       → note the workspace, project name, and version number
    3. Repeat for "gun detection" or "pistol detection"
    4. Pass those values as --knife-* and --gun-* arguments

What this script does:
    - Downloads the knife dataset (YOLOv8 format)
    - Downloads the gun/pistol dataset (YOLOv8 format)
    - Remaps all labels to:   0 = knife   1 = gun
    - Merges everything into  datasets/weapon_dataset/
    - Prints a summary of how many images were collected
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def download_and_remap(
    rf,
    workspace: str,
    project: str,
    version: int,
    target_class_id: int,
    dest_root: Path,
) -> dict[str, int]:
    """Download one Roboflow dataset and merge images/labels into dest_root.

    All class IDs are rewritten to target_class_id (0=knife, 1=gun).
    """
    print(f"\n[download] {workspace}/{project} v{version}  →  class_id={target_class_id}")
    proj = rf.workspace(workspace).project(project)
    ver  = proj.version(version)
    ds   = ver.download("yolov8", location=str(dest_root / "_tmp" / project), overwrite=True)
    tmp_root = Path(ds.location)

    counts = {"train": 0, "val": 0}
    for split in ("train", "valid", "val"):
        img_src = tmp_root / split / "images"
        lbl_src = tmp_root / split / "labels"
        if not img_src.exists():
            continue

        split_key = "val" if split in ("valid", "val") else "train"
        img_dst = dest_root / "images" / split_key
        lbl_dst = dest_root / "labels" / split_key
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        for img_file in img_src.glob("*"):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            dst_name = f"{project}_{img_file.name}"
            shutil.copy2(img_file, img_dst / dst_name)

            lbl_file = lbl_src / img_file.with_suffix(".txt").name
            dst_lbl  = lbl_dst / Path(dst_name).with_suffix(".txt")
            if lbl_file.exists():
                lines = lbl_file.read_text().strip().splitlines()
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        parts[0] = str(target_class_id)
                        new_lines.append(" ".join(parts))
                dst_lbl.write_text("\n".join(new_lines) + "\n")
            else:
                dst_lbl.write_text("")

            counts[split_key] += 1

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"  → {counts['train']} train  /  {counts['val']} val images copied")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Download knife + gun datasets from Roboflow.")
    parser.add_argument("--api-key", required=True, help="Your Roboflow API key.")
    parser.add_argument("--dest", default="datasets/weapon_dataset")

    # Knife dataset args
    parser.add_argument("--knife-workspace", default="", help="Roboflow workspace for knife dataset.")
    parser.add_argument("--knife-project",   default="", help="Roboflow project slug for knife dataset.")
    parser.add_argument("--knife-version",   type=int, default=1)

    # Gun dataset args
    parser.add_argument("--gun-workspace", default="", help="Roboflow workspace for gun dataset.")
    parser.add_argument("--gun-project",   default="", help="Roboflow project slug for gun dataset.")
    parser.add_argument("--gun-version",   type=int, default=1)

    args = parser.parse_args()

    # Validate that workspace/project were provided
    missing = []
    for name, val in [
        ("--knife-workspace", args.knife_workspace),
        ("--knife-project",   args.knife_project),
        ("--gun-workspace",   args.gun_workspace),
        ("--gun-project",     args.gun_project),
    ]:
        if not val:
            missing.append(name)
    if missing:
        print("\n[ERROR] Missing required arguments:", ", ".join(missing))
        print("\nHow to find them:")
        print("  1. Go to https://universe.roboflow.com")
        print("  2. Search 'knife detection' → pick a dataset → Download Dataset → YOLOv8")
        print("     The Python snippet shows:  rf.workspace('WORKSPACE').project('PROJECT').version(N)")
        print("  3. Same for 'gun detection' or 'pistol detection'")
        print("\nExample:")
        print("  python scripts/download_weapon_dataset.py --api-key KEY \\")
        print("    --knife-workspace myworkspace --knife-project knife-det --knife-version 2 \\")
        print("    --gun-workspace   myworkspace --gun-project   gun-det   --gun-version 1")
        raise SystemExit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit("\n[ERROR] roboflow not installed. Run:  pip install roboflow\n")

    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=args.api_key)

    datasets = [
        (args.knife_workspace, args.knife_project, args.knife_version, 0),
        (args.gun_workspace,   args.gun_project,   args.gun_version,   1),
    ]
    total = {"train": 0, "val": 0}
    for ws, proj, ver, class_id in datasets:
        counts = download_and_remap(rf, ws, proj, ver, class_id, dest_root)
        total["train"] += counts["train"]
        total["val"]   += counts["val"]

    print("\n[done] Final dataset:")
    print(f"  train: {total['train']} images")
    print(f"  val:   {total['val']} images")
    print(f"\nNext step:")
    print(f"  python scripts/train_yolo.py --data configs/weapon_data.yaml --base yolo11n.pt --epochs 30 --batch 16 --device mps")


if __name__ == "__main__":
    main()
