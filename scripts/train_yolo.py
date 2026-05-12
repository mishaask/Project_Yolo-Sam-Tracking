from __future__ import annotations

import argparse
from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train custom YOLO detector for the screening prototype.")
    parser.add_argument("--data", default="configs/data.yaml", help="YOLO data YAML.")
    parser.add_argument("--base", default="yolo11n.pt", help="Base YOLO model.")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. 'cpu', '0', 'cuda:0'.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model = YOLO(args.base)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": "runs/screening",
        "name": "yolo_screening_detector",
        "patience": 20,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device

    model.train(**train_kwargs)
    model.val(data=args.data)


if __name__ == "__main__":
    main()
