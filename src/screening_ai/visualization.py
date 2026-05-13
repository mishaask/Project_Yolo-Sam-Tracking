from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from screening_ai.memory import MemoryBank


def draw_detection(
    frame: np.ndarray,
    bbox_xyxy: np.ndarray,
    class_name: str,
    track_id: int,
    confidence: float,
    mask: Optional[np.ndarray] = None,
    owner_id: Optional[int] = None,
    local_tracker_id: Optional[int] = None,
    reidentified_count: int = 0,
    is_risk: bool = False,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy.astype(int)

    # Normal objects stay green/yellow. Risk-class objects get a red bbox and
    # a reddish SAM mask so knives/guns/custom dangerous classes are visually
    # separated from bags. OpenCV uses BGR colors.
    box_color = (0, 0, 255) if is_risk else (60, 220, 60)
    mask_color = np.array([35, 35, 255]) if is_risk else np.array([0, 220, 220])

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    owner_text = f" owner:G{owner_id}" if owner_id is not None else ""
    local_text = f" L{local_tracker_id}" if local_tracker_id is not None else ""
    reid_text = f" reid:{reidentified_count}" if reidentified_count > 0 else ""
    risk_text = " RISK" if is_risk else ""
    label = f"{class_name} G{track_id}{local_text} {confidence:.2f}{owner_text}{reid_text}{risk_text}"

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
        cv2.LINE_AA,
    )

    if mask is not None:
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = frame.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * mask_color).astype(np.uint8)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def draw_track_trails(frame: np.ndarray, memory_bank: MemoryBank, trail_length: int = 40) -> None:
    for track in memory_bank.active_tracks():
        points = [point for _, point in track.history[-trail_length:]]
        if len(points) < 2:
            continue
        int_points = [(int(round(x)), int(round(y))) for x, y in points]
        for p1, p2 in zip(int_points[:-1], int_points[1:]):
            cv2.line(frame, p1, p2, (255, 255, 0), 2)


def draw_owner_links(
    frame: np.ndarray,
    links: list[tuple],
) -> None:
    for link in links:
        # Backward compatible: old links are (object_point, owner_point, object_id, owner_id).
        # New links may include (object_point, owner_point, object_id, owner_id, is_risk).
        object_point, owner_point, object_id, owner_id = link[:4]
        is_risk = bool(link[4]) if len(link) >= 5 else False
        color = (0, 0, 255) if is_risk else (255, 0, 255)
        cv2.line(frame, object_point, owner_point, color, 2)
        mid = ((object_point[0] + owner_point[0]) // 2, (object_point[1] + owner_point[1]) // 2)
        cv2.putText(
            frame,
            f"G{object_id}->G{owner_id}",
            mid,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_event_banner(frame: np.ndarray, messages: list[str]) -> None:
    if not messages:
        return

    max_messages = messages[-3:]
    y = 30
    for message in max_messages:
        cv2.putText(
            frame,
            message[:110],
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28


def draw_fps(frame: np.ndarray, fps_value: float) -> None:
    text = f"FPS: {fps_value:.1f}"
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
