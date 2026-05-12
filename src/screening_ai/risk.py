from __future__ import annotations

from dataclasses import asdict, dataclass

from screening_ai.association import center_distance
from screening_ai.memory import MemoryBank


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

    def to_json(self) -> dict:
        return asdict(self)


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
    ) -> None:
        self.risk_classes = risk_classes
        self.bag_classes = bag_classes
        self.stationary_threshold_frames = stationary_threshold_frames
        self.owner_distance_threshold_px = owner_distance_threshold_px
        self.unattended_cooldown_frames = unattended_cooldown_frames
        self.separation_threshold_frames = separation_threshold_frames
        self.min_owner_contact_frames = min_owner_contact_frames
        self._last_unattended_alert_frame: dict[int, int] = {}

    def detection_events(self, frame_idx: int, track_id: int, class_name: str, confidence: float) -> list[Event]:
        if class_name not in self.risk_classes:
            return []

        return [
            Event(
                frame=frame_idx,
                type="risk_object_detected",
                message=f"Risk-class object detected: {class_name} on track {track_id}",
                track_id=track_id,
                class_name=class_name,
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
                )
            )

        return events
