from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from screening_ai.memory import MemoryBank


EventMessage = str | tuple[int, str, str] | tuple[int, str]


def _clip_text_to_width(text: str, font, scale: float, thickness: int, max_width: int) -> str:
    """Return text shortened to fit max_width pixels."""
    if max_width <= 20:
        return text[:12]
    if cv2.getTextSize(text, font, scale, thickness)[0][0] <= max_width:
        return text
    ellipsis = "..."
    low, high = 0, len(text)
    best = ellipsis
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid].rstrip() + ellipsis
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if width <= max_width:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _event_type_prefix(event_type: str) -> str:
    normalized = event_type.lower().strip()
    if "confirmed" in normalized or "repeated" in normalized or "alert" in normalized:
        return "ALERT"
    if "warning" in normalized:
        return "WARN"
    if "risk" in normalized:
        return "RISK"
    if "unattended" in normalized:
        return "BAG"
    if "reid" in normalized or "merge" in normalized or "track" in normalized:
        return "REID"
    return "INFO"


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
    label_scale: float = 0.48,
    label_thickness: int = 1,
    box_thickness: int = 2,
    display_label: Optional[str] = None,
) -> None:
    x1, y1, x2, y2 = bbox_xyxy.astype(int)

    # Normal objects stay green/yellow. Risk-class objects get a red bbox and
    # a reddish SAM mask so knives/guns/custom dangerous classes are visually
    # separated from bags. OpenCV uses BGR colors.
    box_color = (0, 0, 255) if is_risk else (60, 220, 60)
    mask_color = np.array([35, 35, 255]) if is_risk else np.array([0, 220, 220])

    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, max(1, int(box_thickness)))

    owner_text = f" ->G{owner_id}" if owner_id is not None else ""
    local_text = f" L{local_tracker_id}" if local_tracker_id is not None else ""
    reid_text = f" reid:{reidentified_count}" if reidentified_count > 0 else ""
    risk_text = " RISK" if is_risk else ""
    label = display_label or f"{class_name} G{track_id}{local_text} {confidence:.2f}{owner_text}{reid_text}{risk_text}"

    cv2.putText(
        frame,
        label,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(label_scale),
        box_color,
        max(1, int(label_thickness)),
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
    label_scale: float = 0.42,
    label_thickness: int = 1,
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
            float(label_scale),
            color,
            max(1, int(label_thickness)),
            cv2.LINE_AA,
        )


def draw_event_banner(
    frame: np.ndarray,
    messages: Sequence[EventMessage],
    position: str = "bottom-right",
    max_lines: int = 5,
    scale: float = 0.46,
    thickness: int = 1,
    title: str = "SECURITY EVENTS",
) -> None:
    """Draw a compact blue/white event feed in a screen corner."""
    if not messages or max_lines <= 0:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = frame.shape[:2]
    margin = 12
    line_h = max(17, int(25 * scale + 9))
    header_h = line_h + 6
    panel_w = min(max(310, int(w * 0.42)), w - 2 * margin)
    text_w = panel_w - 22

    normalized: list[tuple[str, str]] = []
    for item in messages[-max_lines:]:
        if isinstance(item, tuple):
            if len(item) >= 3:
                _, event_type, message = item[:3]
            elif len(item) == 2:
                event_type, message = "info", str(item[1])
            else:
                event_type, message = "info", str(item)
        else:
            event_type, message = "info", str(item)
        prefix = _event_type_prefix(str(event_type))
        normalized.append((prefix, f"{prefix}: {message}"))

    panel_h = header_h + len(normalized) * line_h + 12
    pos = position.lower().replace("_", "-")
    x1 = margin if "left" in pos else w - panel_w - margin
    y1 = margin if "top" in pos else h - panel_h - margin
    x2 = x1 + panel_w
    y2 = y1 + panel_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    overlay = frame.copy()
    # Dark blue translucent panel with a brighter blue header.
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (82, 42, 12), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, min(y2, y1 + header_h)), (170, 88, 22), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (245, 245, 245), 1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    cv2.putText(
        frame,
        title,
        (x1 + 10, y1 + line_h),
        font,
        float(scale),
        (255, 255, 255),
        max(1, int(thickness)),
        cv2.LINE_AA,
    )

    y = y1 + header_h + line_h - 4
    for prefix, message in normalized:
        color = (255, 255, 255)
        if prefix == "ALERT":
            color = (255, 255, 255)
        elif prefix == "WARN":
            color = (235, 248, 255)
        elif prefix == "RISK":
            color = (245, 250, 255)
        elif prefix == "REID":
            color = (220, 240, 255)
        elif prefix == "BAG":
            color = (235, 245, 255)
        text = _clip_text_to_width(message, font, float(scale), max(1, int(thickness)), text_w)
        cv2.putText(
            frame,
            text,
            (x1 + 10, y),
            font,
            float(scale),
            color,
            max(1, int(thickness)),
            cv2.LINE_AA,
        )
        y += line_h


def draw_fps(frame: np.ndarray, fps_value: float, scale: float = 0.52, thickness: int = 1) -> None:
    text = f"FPS: {fps_value:.1f}"
    cv2.putText(
        frame,
        text,
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        (255, 255, 255),
        max(1, int(thickness)),
        cv2.LINE_AA,
    )
