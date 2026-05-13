"""
Import a Roboflow ZIP dataset into datasets/weapon_dataset/.

Usage:
    # For the knife dataset:
    python scripts/import_zip_dataset.py --zip ~/Downloads/knife-cqnb0.zip --class-id 0

    # For the gun dataset (once downloaded):
    python scripts/import_zip_dataset.py --zip ~/Downloads/gun-xxx.zip --class-id 1

class-id:  0 = knife   1 = gun
All existing class IDs in the labels are remapped to the given class-id.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def remap_and_copy_labels(src_dir: Path, dst_dir: Path, class_id: int) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for lbl in src_dir.glob("*.txt"):
        lines = lbl.read_text().strip().splitlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                parts[0] = str(class_id)
                new_lines.append(" ".join(parts))
        (dst_dir / lbl.name).write_text("\n".join(new_lines) + "\n" if new_lines else "")
        count += 1
    return count


def copy_images(src_dir: Path, dst_dir: Path, max_images: int | None = None) -> int:
    import random
    dst_dir.mkdir(parents=True, exist_ok=True)
    imgs = [f for f in src_dir.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    if max_images is not None:
        random.seed(42)
        random.shuffle(imgs)
        imgs = imgs[:max_images]
    for img in imgs:
        shutil.copy2(img, dst_dir / img.name)
    return len(imgs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip",      required=True, help="Path to the downloaded ZIP file.")
    parser.add_argument("--class-id", required=True, type=int, help="0=knife  1=gun")
    parser.add_argument("--dest",      default="datasets/weapon_dataset")
    parser.add_argument("--max-images", type=int, default=None, help="Limite le nombre d'images importées (pour équilibrer les classes).")
    args = parser.parse_args()

    zip_path  = Path(args.zip).expanduser()
    dest_root = Path(args.dest)
    class_name = {0: "knife", 1: "gun"}.get(args.class_id, str(args.class_id))
    max_images = args.max_images

    print(f"Extracting {zip_path.name}  →  class {args.class_id} ({class_name})")

    tmp = dest_root / "_tmp_extract"
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # Roboflow zips use train / valid (or val) / test splits
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    totals = {"images": 0, "labels": 0}

    for src_split, dst_split in split_map.items():
        img_src = tmp / src_split / "images"
        lbl_src = tmp / src_split / "labels"
        if not img_src.exists():
            continue
        n_img = copy_images(img_src, dest_root / "images" / dst_split, max_images=max_images)
        n_lbl = remap_and_copy_labels(lbl_src, dest_root / "labels" / dst_split, args.class_id)
        print(f"  {src_split} → {dst_split}: {n_img} images, {n_lbl} labels")
        totals["images"] += n_img
        totals["labels"] += n_lbl

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nDone. {totals['images']} images, {totals['labels']} labels added to {dest_root}")
    print(f"\nSi tu as encore un dataset gun à importer, relance avec --class-id 1")
    print(f"Sinon, lance l'entraînement :")
    print(f"  python scripts/train_yolo.py --data configs/weapon_data.yaml --base yolo11n.pt --epochs 30 --batch 16 --device mps")


if __name__ == "__main__":
    main()
