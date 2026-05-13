from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ultralytics import YOLO


@dataclass(slots=True)
class Detection:
    bbox_xyxy: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    track_id: Optional[int]
    source: str = "main"
    parent_class_name: Optional[str] = None
    parent_bbox_xyxy: Optional[np.ndarray] = None
    roi_level: int = 0


class YoloDetector:
    """Small wrapper around Ultralytics YOLO detection/tracking."""

    def __init__(
        self,
        weights: str,
        tracker_config: str,
        conf: float = 0.35,
        imgsz: int = 640,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(weights)
        self.tracker_config = tracker_config
        self.conf = conf
        self.imgsz = imgsz
        self.device = device

    def track_frame(self, frame: np.ndarray) -> list[Detection]:
        kwargs: dict[str, object] = {
            "source": frame,
            "persist": True,
            "tracker": self.tracker_config,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.track(**kwargs)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        track_ids: list[Optional[int]]
        if result.boxes.id is None:
            track_ids = [None for _ in range(len(boxes))]
        else:
            track_ids = [int(x) for x in result.boxes.id.cpu().numpy().astype(int)]

        detections: list[Detection] = []
        for bbox, class_id, confidence, track_id in zip(boxes, class_ids, confidences, track_ids):
            class_name = str(result.names.get(int(class_id), class_id))
            detections.append(
                Detection(
                    bbox_xyxy=bbox.astype(float),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    track_id=track_id,
                    source="main",
                )
            )

        return detections

    def detect_frame(
        self,
        frame: np.ndarray,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> list[Detection]:
        """Run plain YOLO detection without updating the tracker state.

        This is used for nested ROI searches. We intentionally use predict()
        instead of track() so a small crop does not corrupt BoT-SORT's main
        full-frame tracker. Returned detections have track_id=None and are
        later stitched by the project-level MemoryBank if needed.
        """
        kwargs: dict[str, object] = {
            "source": frame,
            "conf": self.conf if conf is None else float(conf),
            "imgsz": self.imgsz if imgsz is None else int(imgsz),
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model.predict(**kwargs)
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        detections: list[Detection] = []
        for bbox, class_id, confidence in zip(boxes, class_ids, confidences):
            class_name = str(result.names.get(int(class_id), class_id))
            detections.append(
                Detection(
                    bbox_xyxy=bbox.astype(float),
                    class_id=int(class_id),
                    class_name=class_name,
                    confidence=float(confidence),
                    track_id=None,
                    source="predict",
                )
            )
        return detections
