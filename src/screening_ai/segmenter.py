from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass(slots=True)
class SegmentationResult:
    mask: Optional[np.ndarray]


class SamBoxSegmenter:
    """
    Optional SAM/SAM2/MobileSAM/FastSAM box-prompt segmenter.

    YOLO gives bounding boxes. The segmenter receives those boxes as prompts and
    returns masks. The class is lazy-loaded so YOLO-only tracking remains fast.
    """

    def __init__(self, weights: str = "sam2_b.pt", enabled: bool = False, device: Optional[str] = None) -> None:
        self.weights = weights
        self.enabled = enabled
        self.device = device
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return

        lower = self.weights.lower()
        if "fastsam" in lower:
            from ultralytics import FastSAM

            self._model = FastSAM(self.weights)
        else:
            from ultralytics import SAM

            self._model = SAM(self.weights)

    def segment_boxes(self, frame: np.ndarray, boxes_xyxy: list[np.ndarray]) -> list[SegmentationResult]:
        if not self.enabled or not boxes_xyxy:
            return [SegmentationResult(mask=None) for _ in boxes_xyxy]

        self._load()
        assert self._model is not None

        boxes_list = [box.astype(float).tolist() for box in boxes_xyxy]

        kwargs: dict[str, object] = {
            "source": frame,
            "bboxes": boxes_list,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        try:
            results = self._model(**kwargs)
        except Exception:
            # Keep the main pipeline alive if SAM is unavailable or too heavy for this system.
            return [SegmentationResult(mask=None) for _ in boxes_xyxy]

        if not results:
            return [SegmentationResult(mask=None) for _ in boxes_xyxy]

        result = results[0]
        if getattr(result, "masks", None) is None or result.masks is None:
            return [SegmentationResult(mask=None) for _ in boxes_xyxy]

        masks = result.masks.data.cpu().numpy()
        out: list[SegmentationResult] = []

        for i in range(len(boxes_xyxy)):
            if i >= len(masks):
                out.append(SegmentationResult(mask=None))
                continue

            mask = masks[i].astype(np.uint8)
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            out.append(SegmentationResult(mask=mask))

        return out
