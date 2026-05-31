from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from screening_ai.association import center_distance
from screening_ai.memory import MemoryBank, bbox_iou_xyxy

if TYPE_CHECKING:
    from screening_ai.detector import Detection


@dataclass(slots=True)
class Event:
    frame: int
    type: str
    message: str
    track_id: int | None = None
    class_name: str | None = None
    owner_id: int | None = None
    confidence: float | None = None
    distance_px: float | None = None
    cluster_id: str | None = None
    severity: str | None = None
    person_warning_count: int | None = None

    def to_json(self) -> dict:
        return asdict(self)


class RiskConfirmationFilter:
    """Confirm risk detections only after repeated evidence across nearby frames.

    YOLO can flicker on small objects. This filter keeps a short per-class history
    and only allows a current risk detection to pass if it appeared near the same
    location in enough recent frames, for example 3 hits in the last 5 frames.
    """

    def __init__(
        self,
        classes: set[str],
        enabled: bool = True,
        window_frames: int = 5,
        min_hits: int = 3,
        match_iou: float = 0.10,
        match_center_distance_px: float = 140.0,
    ) -> None:
        self.classes = set(classes)
        self.enabled = bool(enabled)
        self.window_frames = max(1, int(window_frames))
        self.min_hits = max(1, int(min_hits))
        self.match_iou = float(match_iou)
        self.match_center_distance_px = float(match_center_distance_px)
        self._history: dict[str, deque[tuple[int, list[np.ndarray]]]] = {}

    def update(self, frame_idx: int, detections: list[Detection]) -> None:
        if not self.enabled:
            return

        by_class: dict[str, list[np.ndarray]] = {}
        for det in detections:
            if det.class_name not in self.classes:
                continue
            by_class.setdefault(det.class_name, []).append(det.bbox_xyxy.astype(float).copy())

        oldest_allowed = int(frame_idx) - self.window_frames + 1
        for class_name in self.classes:
            history = self._history.setdefault(class_name, deque())
            while history and history[0][0] < oldest_allowed:
                history.popleft()
            if class_name in by_class:
                history.append((int(frame_idx), by_class[class_name]))

    def is_confirmed(self, detection: Detection, frame_idx: int) -> bool:
        if not self.enabled or detection.class_name not in self.classes:
            return True

        history = self._history.get(detection.class_name)
        if not history:
            return False

        oldest_allowed = int(frame_idx) - self.window_frames + 1
        hits = 0
        for hist_frame, boxes in history:
            if hist_frame < oldest_allowed:
                continue
            if any(self._boxes_match(detection.bbox_xyxy, box) for box in boxes):
                hits += 1
        return hits >= self.min_hits

    def _boxes_match(self, left: np.ndarray, right: np.ndarray) -> bool:
        if bbox_iou_xyxy(left, right) >= self.match_iou:
            return True
        return self._center_distance(left, right) <= self.match_center_distance_px

    @staticmethod
    def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
        lx1, ly1, lx2, ly2 = left.astype(float)
        rx1, ry1, rx2, ry2 = right.astype(float)
        left_center = np.array([(lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0], dtype=float)
        right_center = np.array([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0], dtype=float)
        return float(np.linalg.norm(left_center - right_center))


@dataclass(slots=True)
class RiskCluster:
    """Short-lived cluster for risk-object detections.

    People/bags deserve stable project IDs. Small risk objects often flicker and
    move by a few pixels every frame, so they are grouped into temporary R#
    clusters instead of being displayed as many different global G# IDs.
    """

    cluster_number: int
    class_name: str
    first_frame: int
    last_frame: int
    bbox_xyxy: np.ndarray
    last_confidence: float
    max_confidence: float
    hit_frames: deque[int] = field(default_factory=deque)
    total_hits: int = 0
    owner_id: int | None = None
    owner_linked_frames: dict[int, int] = field(default_factory=dict)
    owner_last_seen_frame: dict[int, int] = field(default_factory=dict)
    warned_owner_ids: set[int] = field(default_factory=set)
    confirmed_owner_ids: set[int] = field(default_factory=set)
    last_warning_frame: int = -10**9
    last_confirmed_frame: int = -10**9
    display_state: str = "possible"

    @property
    def cluster_id(self) -> str:
        return f"R{self.cluster_number}"

    def update_geometry(self, frame_idx: int, bbox_xyxy: np.ndarray, confidence: float, ema_alpha: float = 0.55) -> None:
        bbox = bbox_xyxy.astype(float)
        self.bbox_xyxy = ema_alpha * bbox + (1.0 - ema_alpha) * self.bbox_xyxy
        self.last_frame = int(frame_idx)
        self.last_confidence = float(confidence)
        self.max_confidence = max(float(self.max_confidence), float(confidence))
        self.total_hits += 1
        if not self.hit_frames or self.hit_frames[-1] != int(frame_idx):
            self.hit_frames.append(int(frame_idx))

    def mark_owner_link(self, frame_idx: int, owner_id: int | None, max_gap_frames: int = 2) -> None:
        if owner_id is None:
            self.owner_id = None
            return

        owner_id = int(owner_id)
        last_owner_frame = self.owner_last_seen_frame.get(owner_id)
        if last_owner_frame is not None and int(frame_idx) - int(last_owner_frame) <= max_gap_frames:
            # Count real frame span, not just detection count. This makes a
            # 30-frame threshold mean roughly one second at 30 FPS.
            self.owner_linked_frames[owner_id] = self.owner_linked_frames.get(owner_id, 0) + max(1, int(frame_idx) - int(last_owner_frame))
        else:
            self.owner_linked_frames[owner_id] = max(1, self.owner_linked_frames.get(owner_id, 0))

        self.owner_last_seen_frame[owner_id] = int(frame_idx)
        self.owner_id = owner_id

    def linked_frames_for_owner(self, owner_id: int | None) -> int:
        if owner_id is None:
            return 0
        return int(self.owner_linked_frames.get(int(owner_id), 0))


class RiskClusterManager:
    """Aggregate confirmed weapon/risk detections into warnings and events."""

    def __init__(
        self,
        classes: set[str],
        enabled: bool = True,
        match_iou: float = 0.08,
        match_center_distance_px: float = 95.0,
        ttl_frames: int = 45,
        owner_link_gap_frames: int = 3,
        warning_cooldown_frames: int = 30,
        confirmed_linked_frames: int = 30,
        confirmed_cooldown_frames: int = 120,
        repeated_warning_window_frames: int = 300,
        repeated_warning_count: int = 10,
        repeated_warning_cooldown_frames: int = 300,
        high_confidence_warning: float = 0.85,
    ) -> None:
        self.classes = set(classes)
        self.enabled = bool(enabled)
        self.match_iou = float(match_iou)
        self.match_center_distance_px = float(match_center_distance_px)
        self.ttl_frames = max(1, int(ttl_frames))
        self.owner_link_gap_frames = max(1, int(owner_link_gap_frames))
        self.warning_cooldown_frames = max(1, int(warning_cooldown_frames))
        self.confirmed_linked_frames = max(1, int(confirmed_linked_frames))
        self.confirmed_cooldown_frames = max(1, int(confirmed_cooldown_frames))
        self.repeated_warning_window_frames = max(1, int(repeated_warning_window_frames))
        self.repeated_warning_count = max(1, int(repeated_warning_count))
        self.repeated_warning_cooldown_frames = max(1, int(repeated_warning_cooldown_frames))
        self.high_confidence_warning = float(high_confidence_warning)
        self._next_cluster_number = 1
        self.clusters: list[RiskCluster] = []
        self.person_warning_frames: dict[int, deque[int]] = {}
        self.last_repeated_warning_frame: dict[int, int] = {}

    def update_detection(
        self,
        frame_idx: int,
        class_name: str,
        bbox_xyxy: np.ndarray,
        confidence: float,
        owner_id: int | None = None,
        source_track_id: int | None = None,
    ) -> tuple[RiskCluster | None, list[Event]]:
        if not self.enabled or class_name not in self.classes:
            return None, []

        self._expire_old_clusters(frame_idx)
        cluster = self._find_or_create_cluster(frame_idx, class_name, bbox_xyxy, confidence)
        cluster.update_geometry(frame_idx, bbox_xyxy, confidence)
        cluster.mark_owner_link(frame_idx, owner_id, max_gap_frames=self.owner_link_gap_frames)

        events: list[Event] = []
        warning_allowed = owner_id is not None or float(confidence) >= self.high_confidence_warning
        if warning_allowed and frame_idx - cluster.last_warning_frame >= self.warning_cooldown_frames:
            cluster.last_warning_frame = int(frame_idx)
            cluster.display_state = "warning"
            warning_count = None
            if owner_id is not None:
                warning_count = self._record_person_warning(int(owner_id), int(frame_idx))
            owner_text = f" near person G{owner_id}" if owner_id is not None else ""
            events.append(
                Event(
                    frame=int(frame_idx),
                    type="risk_warning",
                    message=f"WARNING: possible {class_name} {cluster.cluster_id}{owner_text}",
                    track_id=source_track_id,
                    class_name=class_name,
                    owner_id=owner_id,
                    confidence=float(confidence),
                    cluster_id=cluster.cluster_id,
                    severity="warning",
                    person_warning_count=warning_count,
                )
            )
            if owner_id is not None:
                repeated = self._maybe_repeated_warning_event(int(owner_id), int(frame_idx), class_name, cluster, float(confidence))
                if repeated is not None:
                    events.append(repeated)

        if owner_id is not None:
            linked_frames = cluster.linked_frames_for_owner(int(owner_id))
            owner_already_confirmed = int(owner_id) in cluster.confirmed_owner_ids
            confirm_cooldown_ok = frame_idx - cluster.last_confirmed_frame >= self.confirmed_cooldown_frames
            if linked_frames >= self.confirmed_linked_frames and (not owner_already_confirmed or confirm_cooldown_ok):
                cluster.confirmed_owner_ids.add(int(owner_id))
                cluster.last_confirmed_frame = int(frame_idx)
                cluster.display_state = "confirmed"
                events.append(
                    Event(
                        frame=int(frame_idx),
                        type="risk_object_confirmed",
                        message=(
                            f"ALERT: confirmed {class_name} {cluster.cluster_id} "
                            f"linked to person G{owner_id} for {linked_frames} frames"
                        ),
                        track_id=source_track_id,
                        class_name=class_name,
                        owner_id=int(owner_id),
                        confidence=float(confidence),
                        cluster_id=cluster.cluster_id,
                        severity="confirmed",
                    )
                )

        return cluster, events

    def _find_or_create_cluster(self, frame_idx: int, class_name: str, bbox_xyxy: np.ndarray, confidence: float) -> RiskCluster:
        match = self._find_cluster(frame_idx, class_name, bbox_xyxy)
        if match is not None:
            return match

        cluster = RiskCluster(
            cluster_number=self._next_cluster_number,
            class_name=str(class_name),
            first_frame=int(frame_idx),
            last_frame=int(frame_idx),
            bbox_xyxy=bbox_xyxy.astype(float).copy(),
            last_confidence=float(confidence),
            max_confidence=float(confidence),
        )
        cluster.hit_frames.append(int(frame_idx))
        cluster.total_hits = 1
        self._next_cluster_number += 1
        self.clusters.append(cluster)
        return cluster

    def _find_cluster(self, frame_idx: int, class_name: str, bbox_xyxy: np.ndarray) -> RiskCluster | None:
        best_cluster: RiskCluster | None = None
        best_score = -1.0
        for cluster in self.clusters:
            if cluster.class_name != class_name:
                continue
            if int(frame_idx) - int(cluster.last_frame) > self.ttl_frames:
                continue
            iou = bbox_iou_xyxy(cluster.bbox_xyxy, bbox_xyxy)
            center_dist = self._center_distance(cluster.bbox_xyxy, bbox_xyxy)
            if iou < self.match_iou and center_dist > self.match_center_distance_px:
                continue
            score = iou + max(0.0, 1.0 - center_dist / max(self.match_center_distance_px, 1.0))
            if score > best_score:
                best_score = score
                best_cluster = cluster
        return best_cluster

    def _record_person_warning(self, owner_id: int, frame_idx: int) -> int:
        history = self.person_warning_frames.setdefault(int(owner_id), deque())
        oldest_allowed = int(frame_idx) - self.repeated_warning_window_frames + 1
        while history and history[0] < oldest_allowed:
            history.popleft()
        history.append(int(frame_idx))
        return len(history)

    def _maybe_repeated_warning_event(
        self,
        owner_id: int,
        frame_idx: int,
        class_name: str,
        cluster: RiskCluster,
        confidence: float,
    ) -> Event | None:
        history = self.person_warning_frames.get(int(owner_id), deque())
        if len(history) < self.repeated_warning_count:
            return None
        last_frame = self.last_repeated_warning_frame.get(int(owner_id), -10**9)
        if int(frame_idx) - last_frame < self.repeated_warning_cooldown_frames:
            return None
        self.last_repeated_warning_frame[int(owner_id)] = int(frame_idx)
        return Event(
            frame=int(frame_idx),
            type="risk_repeated_warning",
            message=(
                f"ALERT: person G{owner_id} reached {len(history)} risk warnings "
                f"within {self.repeated_warning_window_frames} frames"
            ),
            class_name=class_name,
            owner_id=int(owner_id),
            confidence=float(confidence),
            cluster_id=cluster.cluster_id,
            severity="escalated",
            person_warning_count=len(history),
        )

    def _expire_old_clusters(self, frame_idx: int) -> None:
        self.clusters = [
            cluster for cluster in self.clusters
            if int(frame_idx) - int(cluster.last_frame) <= self.ttl_frames
        ]

    @staticmethod
    def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
        lx1, ly1, lx2, ly2 = left.astype(float)
        rx1, ry1, rx2, ry2 = right.astype(float)
        left_center = np.array([(lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0], dtype=float)
        right_center = np.array([(rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0], dtype=float)
        return float(np.linalg.norm(left_center - right_center))


class RiskEngine:
    def __init__(
        self,
        risk_classes: set[str],
        bag_classes: set[str],
        stationary_threshold_frames: int = 90,
        owner_distance_threshold_px: float = 250.0,
        unattended_cooldown_frames: int = 60,
        separation_threshold_frames: int = 90,
        min_owner_contact_frames: int = 45,
        risk_detection_cooldown_frames: int = 30,
    ) -> None:
        self.risk_classes = risk_classes
        self.bag_classes = bag_classes
        self.stationary_threshold_frames = stationary_threshold_frames
        self.owner_distance_threshold_px = owner_distance_threshold_px
        self.unattended_cooldown_frames = unattended_cooldown_frames
        self.separation_threshold_frames = separation_threshold_frames
        self.min_owner_contact_frames = min_owner_contact_frames
        self._last_unattended_alert_frame: dict[int, int] = {}
        self._last_risk_alert_frame: dict[int, int] = {}
        self.risk_detection_cooldown_frames = int(risk_detection_cooldown_frames)

    def detection_events(
        self,
        frame_idx: int,
        track_id: int,
        class_name: str,
        confidence: float,
        owner_id: int | None = None,
    ) -> list[Event]:
        if class_name not in self.risk_classes:
            return []

        last_alert = self._last_risk_alert_frame.get(track_id, -10**9)
        if frame_idx - last_alert < self.risk_detection_cooldown_frames:
            return []
        self._last_risk_alert_frame[track_id] = frame_idx

        owner_text = f" linked to person G{owner_id}" if owner_id is not None else ""
        return [
            Event(
                frame=frame_idx,
                type="risk_object_detected",
                message=f"Risk-class object detected: {class_name} G{track_id}{owner_text}",
                track_id=track_id,
                class_name=class_name,
                owner_id=owner_id,
                confidence=confidence,
            )
        ]

    def unattended_bag_events(self, frame_idx: int, memory_bank: MemoryBank) -> list[Event]:
        events: list[Event] = []

        for bag in memory_bank.active_tracks():
            if bag.class_name not in self.bag_classes:
                continue
            if bag.owner_id is None:
                continue
            if bag.stationary_frames < self.stationary_threshold_frames:
                continue
            if bag.owner_contact_frames.get(bag.owner_id, 0) < self.min_owner_contact_frames:
                continue
            if bag.owner_separation_frames.get(bag.owner_id, 0) < self.separation_threshold_frames:
                continue

            owner = memory_bank.tracks.get(bag.owner_id)
            if owner is None:
                continue

            dist = center_distance(owner, bag)
            if dist <= self.owner_distance_threshold_px:
                continue

            last_alert = self._last_unattended_alert_frame.get(bag.global_id, -10**9)
            if frame_idx - last_alert < self.unattended_cooldown_frames:
                continue

            self._last_unattended_alert_frame[bag.global_id] = frame_idx
            events.append(
                Event(
                    frame=frame_idx,
                    type="unattended_bag",
                    message=(
                        f"Bag track G{bag.global_id} was linked to person G{bag.owner_id}, "
                        f"then stayed still while the person moved away"
                    ),
                    track_id=bag.global_id,
                    class_name=bag.class_name,
                    owner_id=bag.owner_id,
                    distance_px=dist,
                    severity="warning",
                )
            )

        return events
