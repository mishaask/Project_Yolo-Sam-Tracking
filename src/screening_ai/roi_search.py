from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from screening_ai.detector import Detection, YoloDetector
from screening_ai.memory import bbox_iou_xyxy


def _as_float_box(box: np.ndarray) -> np.ndarray:
    return np.asarray(box, dtype=float).reshape(4)


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = _as_float_box(box)
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return np.array([x1, y1, x2, y2], dtype=float)


def _expand_box(box: np.ndarray, width: int, height: int, padding_ratio: float) -> np.ndarray:
    x1, y1, x2, y2 = _as_float_box(box)
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = bw * float(padding_ratio)
    pad_y = bh * float(padding_ratio)
    return _clip_box(np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=float), width, height)


def _center_inside(child_box: np.ndarray, parent_box: np.ndarray, margin_px: float = 0.0) -> bool:
    cx = (float(child_box[0]) + float(child_box[2])) / 2.0
    cy = (float(child_box[1]) + float(child_box[3])) / 2.0
    x1, y1, x2, y2 = _as_float_box(parent_box)
    return (x1 - margin_px) <= cx <= (x2 + margin_px) and (y1 - margin_px) <= cy <= (y2 + margin_px)


def _area(box: np.ndarray) -> float:
    x1, y1, x2, y2 = _as_float_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


@dataclass(slots=True)
class RoiSearchConfig:
    enabled: bool = True
    every_n_frames: int = 10
    parent_classes: set[str] | None = None
    max_parent_rois_per_frame: int = 2
    max_inner_detections_per_roi: int = 5
    min_parent_confidence: float = 0.25
    roi_confidence: float = 0.18
    roi_imgsz: int = 320
    padding_ratio: float = 0.12
    duplicate_same_class_iou: float = 0.55
    require_center_inside_parent: bool = True
    inside_parent_margin_px: float = 8.0
    min_child_area_ratio_of_frame: float = 0.00025
    max_child_area_ratio_of_parent: float = 0.65
    include_classes_by_parent: dict[str, set[str]] | None = None
    exclude_classes_by_parent: dict[str, set[str]] | None = None

    @classmethod
    def from_mapping(cls, cfg: dict[str, Any]) -> "RoiSearchConfig":
        def _set_or_none(value: Any) -> set[str] | None:
            if value is None:
                return None
            if isinstance(value, str):
                if value.lower() == "all":
                    return None
                return {x.strip() for x in value.split(",") if x.strip()}
            return {str(x).strip() for x in value if str(x).strip()}

        def _class_map(value: Any) -> dict[str, set[str]]:
            if not isinstance(value, dict):
                return {}
            out: dict[str, set[str]] = {}
            for key, raw_classes in value.items():
                classes = _set_or_none(raw_classes)
                if classes is not None:
                    out[str(key)] = classes
            return out

        return cls(
            enabled=bool(cfg.get("enabled", True)),
            every_n_frames=max(1, int(cfg.get("every_n_frames", 10))),
            parent_classes=_set_or_none(cfg.get("parent_classes", ["person", "backpack", "handbag", "suitcase", "trolley_bag"])),
            max_parent_rois_per_frame=max(0, int(cfg.get("max_parent_rois_per_frame", 2))),
            max_inner_detections_per_roi=max(0, int(cfg.get("max_inner_detections_per_roi", 5))),
            min_parent_confidence=float(cfg.get("min_parent_confidence", 0.25)),
            roi_confidence=float(cfg.get("roi_confidence", 0.18)),
            roi_imgsz=max(96, int(cfg.get("roi_imgsz", 320))),
            padding_ratio=float(cfg.get("padding_ratio", 0.12)),
            duplicate_same_class_iou=float(cfg.get("duplicate_same_class_iou", 0.55)),
            require_center_inside_parent=bool(cfg.get("require_center_inside_parent", True)),
            inside_parent_margin_px=float(cfg.get("inside_parent_margin_px", 8.0)),
            min_child_area_ratio_of_frame=float(cfg.get("min_child_area_ratio_of_frame", 0.00025)),
            max_child_area_ratio_of_parent=float(cfg.get("max_child_area_ratio_of_parent", 0.65)),
            include_classes_by_parent=_class_map(cfg.get("include_classes_by_parent", {})),
            exclude_classes_by_parent=_class_map(cfg.get("exclude_classes_by_parent", {})),
        )


class NestedRoiObjectSearch:
    """Second-pass detector for objects hidden by overlapping parent boxes.

    Main YOLO/BoT-SORT detections remain the source of people/bag tracks. This
    module adds supplemental untracked detections inside selected parent ROIs
    such as a person box or bag box. It is intentionally conservative: it
    excludes the parent class, rejects duplicate same-class boxes, and does not
    call tracker() on crops.
    """

    def __init__(self, detector: YoloDetector, config: RoiSearchConfig) -> None:
        self.detector = detector
        self.config = config

    def find_inner_detections(
        self,
        frame: np.ndarray,
        frame_idx: int,
        main_detections: list[Detection],
    ) -> list[Detection]:
        cfg = self.config
        if not cfg.enabled or cfg.max_parent_rois_per_frame <= 0 or cfg.max_inner_detections_per_roi <= 0:
            return []
        if frame_idx % cfg.every_n_frames != 0:
            return []
        if not main_detections:
            return []

        height, width = frame.shape[:2]
        frame_area = float(max(1, width * height))

        parents = [
            det for det in main_detections
            if det.confidence >= cfg.min_parent_confidence
            and (cfg.parent_classes is None or det.class_name in cfg.parent_classes)
        ]
        parents.sort(key=lambda d: d.confidence, reverse=True)
        parents = parents[: cfg.max_parent_rois_per_frame]

        extra: list[Detection] = []
        existing = list(main_detections)

        for parent in parents:
            parent_box = _clip_box(parent.bbox_xyxy, width, height)
            parent_area = max(1.0, _area(parent_box))
            roi_box = _expand_box(parent_box, width, height, cfg.padding_ratio)
            x1, y1, x2, y2 = roi_box.astype(int)
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_detections = self.detector.detect_frame(
                crop,
                conf=cfg.roi_confidence,
                imgsz=cfg.roi_imgsz,
            )
            crop_detections.sort(key=lambda d: d.confidence, reverse=True)

            accepted_for_parent = 0
            for child in crop_detections:
                if accepted_for_parent >= cfg.max_inner_detections_per_roi:
                    break
                if not self._class_allowed(parent.class_name, child.class_name):
                    continue

                full_box = child.bbox_xyxy.astype(float).copy()
                full_box[[0, 2]] += float(x1)
                full_box[[1, 3]] += float(y1)
                full_box = _clip_box(full_box, width, height)

                if cfg.require_center_inside_parent and not _center_inside(
                    full_box,
                    parent_box,
                    margin_px=cfg.inside_parent_margin_px,
                ):
                    continue

                child_area = _area(full_box)
                if child_area / frame_area < cfg.min_child_area_ratio_of_frame:
                    continue
                if child_area / parent_area > cfg.max_child_area_ratio_of_parent:
                    continue

                if self._is_duplicate(child.class_name, full_box, existing, cfg.duplicate_same_class_iou):
                    continue
                if self._is_duplicate(child.class_name, full_box, extra, cfg.duplicate_same_class_iou):
                    continue

                extra.append(
                    Detection(
                        bbox_xyxy=full_box,
                        class_id=child.class_id,
                        class_name=child.class_name,
                        confidence=child.confidence,
                        track_id=None,
                        source=f"roi:{parent.class_name}",
                        parent_class_name=parent.class_name,
                        parent_bbox_xyxy=parent_box.copy(),
                        roi_level=parent.roi_level + 1,
                    )
                )
                accepted_for_parent += 1

        return extra

    def _class_allowed(self, parent_class: str, child_class: str) -> bool:
        cfg = self.config
        include_map = cfg.include_classes_by_parent or {}
        exclude_map = cfg.exclude_classes_by_parent or {}

        include = include_map.get(parent_class)
        if include is not None and child_class not in include:
            return False

        exclude = set(exclude_map.get(parent_class, set()))
        # Always avoid finding the same parent class inside itself unless the
        # config explicitly removes it from the exclude list.
        exclude.add(parent_class)
        return child_class not in exclude

    @staticmethod
    def _is_duplicate(
        class_name: str,
        bbox_xyxy: np.ndarray,
        detections: list[Detection],
        iou_threshold: float,
    ) -> bool:
        for det in detections:
            if det.class_name != class_name:
                continue
            if bbox_iou_xyxy(det.bbox_xyxy, bbox_xyxy) >= iou_threshold:
                return True
        return False
