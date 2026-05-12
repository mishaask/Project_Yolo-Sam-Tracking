from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from screening_ai.deep_reid import DeepPersonReID


_DEEP_PERSON_REID: Optional[DeepPersonReID] = None


@dataclass(slots=True)
class PersonCropQuality:
    score: float
    usable_for_match: bool
    usable_for_memory_update: bool
    touches_left: bool
    touches_right: bool
    touches_top: bool
    touches_bottom: bool
    too_small: bool
    bad_aspect: bool
    reason: str


def configure_deep_person_reid(extractor: Optional[DeepPersonReID]) -> None:
    """Configure the process-wide anonymous full-body ReID extractor."""
    global _DEEP_PERSON_REID
    _DEEP_PERSON_REID = extractor


def get_deep_person_reid_status() -> str:
    if _DEEP_PERSON_REID is None:
        return "Deep person ReID not configured"
    return _DEEP_PERSON_REID.describe()


def clamp_bbox_xyxy(bbox_xyxy: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    """Clamp an xyxy box to image bounds."""
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    x1 = max(0, min(int(round(x1)), width - 1))
    y1 = max(0, min(int(round(y1)), height - 1))
    x2 = max(0, min(int(round(x2)), width - 1))
    y2 = max(0, min(int(round(y2)), height - 1))

    if x2 <= x1:
        x2 = min(width - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(height - 1, y1 + 1)

    return x1, y1, x2, y2


def crop_from_bbox(frame: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Return a safe crop for the given box, or None if it is invalid."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox_xyxy, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def assess_person_crop_quality(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    bbox_xyxy: np.ndarray,
    edge_margin_ratio: float = 0.025,
    min_height_ratio: float = 0.28,
    min_area_ratio: float = 0.018,
    min_aspect_ratio: float = 0.16,
    max_aspect_ratio: float = 0.85,
    allow_bottom_edge: bool = True,
) -> PersonCropQuality:
    """Score whether a person crop is safe to store as identity memory.

    Matching can use medium-quality crops, but memory updates are stricter.
    The goal is to avoid poisoning a person's gallery with partial exit crops,
    side-clipped bodies, tiny detections, or obvious non-body boxes.
    """
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    frame_area = float(max(1, width * height))

    margin_x = max(2.0, float(width) * float(edge_margin_ratio))
    margin_y = max(2.0, float(height) * float(edge_margin_ratio))

    touches_left = x1 <= margin_x
    touches_right = x2 >= float(width) - margin_x
    touches_top = y1 <= margin_y
    touches_bottom = y2 >= float(height) - margin_y

    height_ratio = box_h / float(max(1, height))
    area_ratio = (box_w * box_h) / frame_area
    aspect = box_w / box_h

    too_small = height_ratio < float(min_height_ratio) or area_ratio < float(min_area_ratio)
    bad_aspect = aspect < float(min_aspect_ratio) or aspect > float(max_aspect_ratio)

    update_edge_fail = touches_left or touches_right or touches_top or (touches_bottom and not allow_bottom_edge)
    match_edge_fail = touches_left and touches_right

    score = 1.0
    if update_edge_fail:
        score -= 0.35
    if too_small:
        score -= 0.35
    if bad_aspect:
        score -= 0.25
    if touches_bottom and allow_bottom_edge:
        score -= 0.05
    score = max(0.0, min(1.0, score))

    reasons: list[str] = []
    if touches_left:
        reasons.append("left-edge")
    if touches_right:
        reasons.append("right-edge")
    if touches_top:
        reasons.append("top-edge")
    if touches_bottom:
        reasons.append("bottom-edge")
    if too_small:
        reasons.append("too-small/partial")
    if bad_aspect:
        reasons.append("bad-aspect")
    reason = ",".join(reasons) if reasons else "good"

    usable_for_memory_update = score >= 0.70 and not update_edge_fail and not too_small and not bad_aspect
    usable_for_match = score >= 0.35 and not match_edge_fail and not bad_aspect

    return PersonCropQuality(
        score=score,
        usable_for_match=usable_for_match,
        usable_for_memory_update=usable_for_memory_update,
        touches_left=touches_left,
        touches_right=touches_right,
        touches_top=touches_top,
        touches_bottom=touches_bottom,
        too_small=too_small,
        bad_aspect=bad_aspect,
        reason=reason,
    )


def edge_side_from_bbox(
    frame_shape: tuple[int, int] | tuple[int, int, int],
    bbox_xyxy: np.ndarray,
    edge_ratio: float = 0.055,
) -> Optional[str]:
    """Return the closest frame edge touched by a bbox, prioritizing left/right."""
    height = int(frame_shape[0])
    width = int(frame_shape[1])
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    mx = max(2.0, float(width) * float(edge_ratio))
    my = max(2.0, float(height) * float(edge_ratio))

    distances: list[tuple[float, str]] = []
    if x1 <= mx:
        distances.append((max(0.0, x1), "left"))
    if x2 >= float(width) - mx:
        distances.append((max(0.0, float(width) - x2), "right"))
    if y1 <= my:
        distances.append((max(0.0, y1), "top"))
    if y2 >= float(height) - my:
        distances.append((max(0.0, float(height) - y2), "bottom"))

    if not distances:
        return None

    horizontal = [item for item in distances if item[1] in {"left", "right"}]
    if horizontal:
        horizontal.sort(key=lambda x: x[0])
        return horizontal[0][1]

    distances.sort(key=lambda x: x[0])
    return distances[0][1]


def mask_area(mask: Optional[np.ndarray]) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 0))


def bbox_from_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return xyxy bbox from a binary mask, or None if the mask is empty."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=float)


def center_from_mask(mask: Optional[np.ndarray]) -> Optional[tuple[float, float]]:
    """Return centroid from a binary mask, or None if the mask is empty."""
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def _crop_mask(mask: Optional[np.ndarray], bbox_xyxy: np.ndarray, frame_shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    h, w = frame_shape
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    x1, y1, x2, y2 = clamp_bbox_xyxy(bbox_xyxy, w, h)
    cropped = mask[y1:y2, x1:x2]
    if cropped.size == 0 or np.count_nonzero(cropped) == 0:
        return None
    return cropped.astype(np.uint8)


def _hist_for_region(hsv_region: np.ndarray, mask_region: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if hsv_region.size == 0:
        return None
    if min(hsv_region.shape[:2]) < 4:
        return None

    hist = cv2.calcHist(
        [hsv_region],
        channels=[0, 1, 2],
        mask=mask_region,
        histSize=[16, 8, 8],
        ranges=[0, 180, 0, 256, 0, 256],
    )
    hist = hist.astype(np.float32).flatten()
    norm = float(np.linalg.norm(hist))
    if norm < 1e-8:
        return None
    return hist / norm


def _embedding_from_crop(crop: np.ndarray, mask_crop: Optional[np.ndarray], bands: int = 3) -> Optional[np.ndarray]:
    if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
        return None

    crop = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_AREA)
    if mask_crop is not None:
        mask_crop = cv2.resize(mask_crop, (64, 128), interpolation=cv2.INTER_NEAREST)
        mask_crop = (mask_crop > 0).astype(np.uint8) * 255

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    descriptors: list[np.ndarray] = []
    bands = max(1, int(bands))
    h = hsv.shape[0]
    for i in range(bands):
        y1 = int(round(i * h / bands))
        y2 = int(round((i + 1) * h / bands))
        hsv_region = hsv[y1:y2, :]
        mask_region = mask_crop[y1:y2, :] if mask_crop is not None else None
        hist = _hist_for_region(hsv_region, mask_region)
        if hist is None:
            hist = _hist_for_region(hsv_region, None)
        if hist is not None:
            descriptors.append(hist)

    if not descriptors:
        return None

    embedding = np.concatenate(descriptors).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-8:
        return None
    return embedding / norm


def hsv_histogram_embedding(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bands: int = 3,
) -> Optional[np.ndarray]:
    """Lightweight color descriptor used for bags and as a fallback/debug signal."""
    crop = crop_from_bbox(frame, bbox_xyxy)
    if crop is None:
        return None
    mask_crop = _crop_mask(mask, bbox_xyxy, frame.shape[:2])
    return _embedding_from_crop(crop, mask_crop, bands=bands)


def _relative_subbox(bbox_xyxy: np.ndarray, rx1: float, ry1: float, rx2: float, ry2: float) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return np.array([
        x1 + rx1 * w,
        y1 + ry1 * h,
        x1 + rx2 * w,
        y1 + ry2 * h,
    ], dtype=float)


def person_clothing_histogram(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Secondary anonymous color cue retained only as a tie-breaker/debug feature."""
    full = hsv_histogram_embedding(frame, bbox_xyxy, mask=mask, bands=4)
    torso_box = _relative_subbox(bbox_xyxy, 0.18, 0.22, 0.82, 0.72)
    torso = hsv_histogram_embedding(frame, torso_box, mask=mask, bands=2)
    upper_box = _relative_subbox(bbox_xyxy, 0.12, 0.10, 0.88, 0.58)
    upper = hsv_histogram_embedding(frame, upper_box, mask=mask, bands=2)

    parts = [p for p in (torso, upper, full) if p is not None]
    if not parts:
        return None

    weighted: list[np.ndarray] = []
    weights = [1.20, 1.00, 0.70]
    for part, weight in zip(parts, weights[: len(parts)]):
        weighted.append((part * float(weight)).astype(np.float32))

    embedding = np.concatenate(weighted).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm < 1e-8:
        return None
    return embedding / norm


def person_deep_reid_embedding(frame: np.ndarray, bbox_xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Primary anonymous person descriptor: full-body deep ReID embedding."""
    if _DEEP_PERSON_REID is None or not _DEEP_PERSON_REID.is_available():
        return None
    crop = crop_from_bbox(frame, bbox_xyxy)
    return _DEEP_PERSON_REID.extract(crop)


def appearance_embedding_for_class(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    class_name: str,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Class-aware primary appearance embedding used by MemoryBank."""
    if class_name == "person":
        deep = person_deep_reid_embedding(frame, bbox_xyxy)
        if deep is not None:
            return deep
        return person_clothing_histogram(frame, bbox_xyxy, mask=mask)
    return hsv_histogram_embedding(frame, bbox_xyxy, mask=mask, bands=3)


def secondary_embedding_for_class(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    class_name: str,
    mask: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """Secondary descriptor. For people this is clothing color; for bags it is unused."""
    if class_name == "person":
        return person_clothing_histogram(frame, bbox_xyxy, mask=mask)
    return None


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0
    value = float(np.dot(a, b) / denom)
    return max(0.0, min(1.0, value))
