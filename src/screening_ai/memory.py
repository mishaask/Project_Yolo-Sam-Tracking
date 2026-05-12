from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from screening_ai.appearance import (
    PersonCropQuality,
    assess_person_crop_quality,
    bbox_from_mask,
    center_from_mask,
    cosine_similarity,
    appearance_embedding_for_class,
    secondary_embedding_for_class,
    edge_side_from_bbox,
    mask_area,
)


def bbox_center_xy(bbox_xyxy: np.ndarray) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy.astype(float)
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a.astype(float)
    bx1, by1, bx2, by2 = b.astype(float)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection

    if union <= 1e-8:
        return 0.0
    return float(intersection / union)




def _normalize_embedding(vector: np.ndarray) -> Optional[np.ndarray]:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return None
    return vector / norm


def _gallery_best_similarity(gallery: list[np.ndarray], candidate: Optional[np.ndarray]) -> float:
    if candidate is None or not gallery:
        return 0.0
    best = 0.0
    for item in gallery:
        best = max(best, cosine_similarity(item, candidate))
    return best


def _gallery_pair_similarity(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    if not a or not b:
        return 0.0
    best = 0.0
    for left in a:
        for right in b:
            best = max(best, cosine_similarity(left, right))
    return best


def _gallery_mean(gallery: list[np.ndarray]) -> Optional[np.ndarray]:
    if not gallery:
        return None
    valid = [np.asarray(item, dtype=np.float32).reshape(-1) for item in gallery if item is not None]
    if not valid:
        return None
    shapes = {item.shape for item in valid}
    if len(shapes) != 1:
        return None
    return _normalize_embedding(np.mean(valid, axis=0))


def predicted_center_from_history(history: list[tuple[int, tuple[float, float]]], target_frame: int) -> tuple[float, float]:
    """Simple constant-velocity prediction used during re-identification."""
    if not history:
        return (0.0, 0.0)
    last_frame, last_center = history[-1]
    if len(history) < 3:
        return last_center

    # Use the earliest point from a short window to estimate stable velocity.
    ref_frame, ref_center = history[max(0, len(history) - 8)]
    frame_delta = max(1, last_frame - ref_frame)
    velocity = (np.array(last_center, dtype=float) - np.array(ref_center, dtype=float)) / float(frame_delta)
    gap = max(0, target_frame - last_frame)
    predicted = np.array(last_center, dtype=float) + velocity * float(min(gap, 45))
    return (float(predicted[0]), float(predicted[1]))

def choose_geometry_bbox(bbox_xyxy: np.ndarray, mask: Optional[np.ndarray], prefer_mask: bool) -> np.ndarray:
    if prefer_mask:
        mask_bbox = bbox_from_mask(mask)
        if mask_bbox is not None:
            return mask_bbox.astype(float)
    return bbox_xyxy.astype(float)


def choose_geometry_center(bbox_xyxy: np.ndarray, mask: Optional[np.ndarray], prefer_mask: bool) -> tuple[float, float]:
    if prefer_mask:
        mask_center = center_from_mask(mask)
        if mask_center is not None:
            return mask_center
    return bbox_center_xy(bbox_xyxy)


@dataclass(slots=True)
class TrackResolution:
    global_id: int
    local_tracker_id: Optional[int]
    was_reidentified: bool = False
    previous_local_tracker_id: Optional[int] = None
    match_score: float = 0.0
    merged_from_global_id: Optional[int] = None


@dataclass(slots=True)
class TrackMemory:
    """
    Persistent track memory.

    global_id is our project-level stable ID.
    local_tracker_id is the temporary ID returned by YOLO/BoT-SORT.
    last_mask is optional and is used only as a segmentation cue inside this run.
    """

    global_id: int
    class_name: str
    first_frame: int
    last_frame: int
    last_bbox_xyxy: np.ndarray
    smoothed_bbox_xyxy: np.ndarray
    last_center: tuple[float, float]
    last_confidence: float
    local_tracker_id: Optional[int] = None
    previous_local_tracker_ids: list[int] = field(default_factory=list)
    owner_id: Optional[int] = None
    owner_scores: dict[int, float] = field(default_factory=dict)
    owner_contact_frames: dict[int, int] = field(default_factory=dict)
    owner_separation_frames: dict[int, int] = field(default_factory=dict)
    owner_last_near_frame: dict[int, int] = field(default_factory=dict)
    owner_first_link_frame: Optional[int] = None
    owner_separated_from_frame: Optional[int] = None
    owner_last_distance_px: float = 0.0
    owner_link_strength: float = 0.0
    history: list[tuple[int, tuple[float, float]]] = field(default_factory=list)
    stationary_frames: int = 0
    missing_frames: int = 0
    appearance: Optional[np.ndarray] = None
    secondary_appearance: Optional[np.ndarray] = None
    appearance_gallery: list[np.ndarray] = field(default_factory=list)
    secondary_appearance_gallery: list[np.ndarray] = field(default_factory=list)
    last_crop_quality: float = 0.0
    last_crop_quality_reason: str = "unknown"
    good_snapshot_count: int = 0
    last_observed_side: Optional[str] = None
    exit_side: Optional[str] = None
    exit_frame: Optional[int] = None
    entry_side: Optional[str] = None
    reidentified_count: int = 0
    last_mask: Optional[np.ndarray] = None
    last_mask_area: int = 0
    used_mask_geometry: bool = False


class MemoryBank:
    """
    Stores current and historical state of tracked persons, bags, and objects.

    Ultralytics/BoT-SORT tracks local IDs, but those IDs may change after
    occlusion, fast motion, or re-entry. MemoryBank adds a stable global_id layer
    and tries to stitch new local tracker IDs back to older global tracks.

    When SAM masks are available, the memory can use mask-derived appearance and
    mask-derived geometry instead of only YOLO boxes. That improves association
    when boxes include background or nearby people.
    """

    def __init__(
        self,
        max_missing_frames: int = 180,
        max_reid_gap_frames: int = 180,
        ema_alpha: float = 0.65,
        stationary_movement_px: float = 3.0,
        min_stitch_score: float = 0.58,
        max_center_distance_px: float = 260.0,
        min_appearance_similarity: float = 0.18,
        appearance_weight: float = 0.55,
        iou_weight: float = 0.20,
        proximity_weight: float = 0.25,
        person_min_stitch_score: float = 0.72,
        person_min_appearance_similarity: float = 0.52,
        person_min_secondary_similarity: float = 0.18,
        person_deep_appearance_weight: float = 0.74,
        person_secondary_appearance_weight: float = 0.08,
        person_iou_weight: float = 0.06,
        person_proximity_weight: float = 0.12,
        person_gallery_size: int = 8,
        person_gallery_duplicate_similarity: float = 0.985,
        person_crop_edge_margin_ratio: float = 0.025,
        person_crop_min_height_ratio: float = 0.28,
        person_crop_min_area_ratio: float = 0.018,
        person_crop_min_aspect_ratio: float = 0.16,
        person_crop_max_aspect_ratio: float = 0.85,
        person_crop_allow_bottom_edge: bool = True,
        entry_exit_enabled: bool = True,
        entry_exit_edge_ratio: float = 0.055,
        entry_exit_window_frames: int = 240,
        entry_exit_same_side_bonus: float = 0.08,
    ) -> None:
        self.tracks: dict[int, TrackMemory] = {}
        # Class-aware mapping prevents a local tracker ID from one class
        # (for example suitcase L1) from being accidentally reused as another
        # class (for example person L1). This was one cause of unstable IDs and
        # appearance-vector shape crashes.
        self.local_to_global: dict[tuple[str, int], int] = {}
        self._next_global_id = 1

        self.max_missing_frames = int(max_missing_frames)
        self.max_reid_gap_frames = int(max_reid_gap_frames)
        self.ema_alpha = float(ema_alpha)
        self.stationary_movement_px = float(stationary_movement_px)
        self.min_stitch_score = float(min_stitch_score)
        self.max_center_distance_px = float(max_center_distance_px)
        self.min_appearance_similarity = float(min_appearance_similarity)
        self.appearance_weight = float(appearance_weight)
        self.iou_weight = float(iou_weight)
        self.proximity_weight = float(proximity_weight)

        # Group-safe person re-identification thresholds. This deliberately does
        # NOT force all people into one ID. A person can reconnect to an old ID
        # only when the older local tracker is not also visible in the current
        # frame and the clothing/motion score is strong enough.
        self.person_min_stitch_score = float(person_min_stitch_score)
        self.person_min_appearance_similarity = float(person_min_appearance_similarity)
        self.person_min_secondary_similarity = float(person_min_secondary_similarity)
        self.person_deep_appearance_weight = float(person_deep_appearance_weight)
        self.person_secondary_appearance_weight = float(person_secondary_appearance_weight)
        self.person_iou_weight = float(person_iou_weight)
        self.person_proximity_weight = float(person_proximity_weight)
        self.person_gallery_size = int(person_gallery_size)
        self.person_gallery_duplicate_similarity = float(person_gallery_duplicate_similarity)
        self.person_crop_edge_margin_ratio = float(person_crop_edge_margin_ratio)
        self.person_crop_min_height_ratio = float(person_crop_min_height_ratio)
        self.person_crop_min_area_ratio = float(person_crop_min_area_ratio)
        self.person_crop_min_aspect_ratio = float(person_crop_min_aspect_ratio)
        self.person_crop_max_aspect_ratio = float(person_crop_max_aspect_ratio)
        self.person_crop_allow_bottom_edge = bool(person_crop_allow_bottom_edge)
        self.entry_exit_enabled = bool(entry_exit_enabled)
        self.entry_exit_edge_ratio = float(entry_exit_edge_ratio)
        self.entry_exit_window_frames = int(entry_exit_window_frames)
        self.entry_exit_same_side_bonus = float(entry_exit_same_side_bonus)

    def make_track_id(self, track_id: Optional[int]) -> int:
        """Backward-compatible helper retained for older scripts/tests."""
        if track_id is not None:
            return int(track_id)
        global_id = self._next_global_id
        self._next_global_id += 1
        return global_id

    def _allocate_global_id(self) -> int:
        global_id = self._next_global_id
        self._next_global_id += 1
        return global_id

    def resolve_global_id(
        self,
        local_tracker_id: Optional[int],
        class_name: str,
        frame_idx: int,
        bbox_xyxy: np.ndarray,
        confidence: float,
        frame: np.ndarray,
        excluded_global_ids: Optional[set[int]] = None,
        active_local_ids_by_class: Optional[dict[str, set[int]]] = None,
        mask: Optional[np.ndarray] = None,
        prefer_mask_geometry: bool = False,
    ) -> TrackResolution:
        """
        Convert YOLO's local tracker ID into our persistent project global ID.

        If the local ID is new, we try to match it to a recently lost global track
        using class, mask-aware appearance, proximity, IoU, and entry/exit side logic.
        """
        del confidence  # kept in signature for compatibility and future scoring
        excluded_global_ids = excluded_global_ids or set()
        active_local_ids_by_class = active_local_ids_by_class or {}
        local_id = int(local_tracker_id) if local_tracker_id is not None else None

        local_key = (class_name, local_id) if local_id is not None else None

        effective_bbox = choose_geometry_bbox(bbox_xyxy, mask, prefer_mask_geometry)
        candidate_quality: Optional[PersonCropQuality] = None
        candidate_entry_side: Optional[str] = None
        can_use_candidate_appearance = True

        if class_name == "person":
            candidate_quality = self._person_crop_quality(frame.shape, effective_bbox)
            candidate_entry_side = edge_side_from_bbox(frame.shape, effective_bbox, self.entry_exit_edge_ratio)
            can_use_candidate_appearance = candidate_quality.usable_for_match

        appearance = None
        secondary = None
        if can_use_candidate_appearance:
            appearance = appearance_embedding_for_class(frame, effective_bbox, class_name, mask=mask)
            secondary = secondary_embedding_for_class(frame, effective_bbox, class_name, mask=mask)

        if local_key is not None and local_key in self.local_to_global:
            global_id = self.local_to_global[local_key]
            if global_id not in excluded_global_ids and global_id in self.tracks:
                memory = self.tracks[global_id]
                if memory.class_name == class_name:
                    return TrackResolution(global_id=global_id, local_tracker_id=local_id)
        match = self._find_reid_match(
            class_name=class_name,
            frame_idx=frame_idx,
            bbox_xyxy=effective_bbox,
            appearance=appearance,
            secondary_appearance=secondary,
            excluded_global_ids=excluded_global_ids,
            current_local_id=local_id,
            active_local_ids_by_class=active_local_ids_by_class,
            candidate_entry_side=candidate_entry_side,
            candidate_quality=candidate_quality,
        )

        if match is not None:
            global_id, score = match
            memory = self.tracks[global_id]
            previous_local = memory.local_tracker_id
            if local_key is not None:
                self.local_to_global[local_key] = global_id
                if previous_local is not None and previous_local != local_id:
                    memory.previous_local_tracker_ids.append(previous_local)
                memory.local_tracker_id = local_id
            memory.reidentified_count += 1
            if candidate_entry_side is not None:
                memory.entry_side = candidate_entry_side
            return TrackResolution(
                global_id=global_id,
                local_tracker_id=local_id,
                was_reidentified=True,
                previous_local_tracker_id=previous_local,
                match_score=score,
            )

        global_id = self._allocate_global_id()
        if local_key is not None:
            self.local_to_global[local_key] = global_id
        return TrackResolution(global_id=global_id, local_tracker_id=local_id)


    def _find_reid_match(
        self,
        class_name: str,
        frame_idx: int,
        bbox_xyxy: np.ndarray,
        appearance: Optional[np.ndarray],
        secondary_appearance: Optional[np.ndarray],
        excluded_global_ids: set[int],
        current_local_id: Optional[int] = None,
        active_local_ids_by_class: Optional[dict[str, set[int]]] = None,
        candidate_entry_side: Optional[str] = None,
        candidate_quality: Optional[PersonCropQuality] = None,
    ) -> Optional[tuple[int, float]]:
        best_global_id: Optional[int] = None
        best_score = 0.0
        center = np.array(bbox_center_xy(bbox_xyxy), dtype=float)
        active_local_ids_by_class = active_local_ids_by_class or {}
        active_ids_for_class = active_local_ids_by_class.get(class_name, set())

        for global_id, memory in self.tracks.items():
            if global_id in excluded_global_ids:
                continue
            if memory.class_name != class_name:
                continue

            # Group-safety guard: never reconnect a new local tracker ID to an
            # older global track whose previous local tracker is still visible
            # in this same frame. Without this, two different people in a group
            # can collapse into one global ID.
            if (
                memory.local_tracker_id is not None
                and current_local_id is not None
                and memory.local_tracker_id != current_local_id
                and memory.local_tracker_id in active_ids_for_class
            ):
                continue

            gap = frame_idx - memory.last_frame
            if gap < 0 or gap > self.max_reid_gap_frames:
                continue

            same_entry_exit_side = (
                class_name == "person"
                and self.entry_exit_enabled
                and candidate_entry_side is not None
                and memory.exit_side is not None
                and candidate_entry_side == memory.exit_side
                and gap <= self.entry_exit_window_frames
            )

            predicted_center = np.array(predicted_center_from_history(memory.history, frame_idx), dtype=float)
            raw_prev_center = np.array(memory.last_center, dtype=float)
            predicted_distance = float(np.linalg.norm(center - predicted_center))
            last_distance = float(np.linalg.norm(center - raw_prev_center))
            distance = min(predicted_distance, last_distance)

            class_distance_limit = self.max_center_distance_px * (1.35 if class_name == "person" else 1.0)
            if same_entry_exit_side:
                class_distance_limit *= 1.55
            if distance > class_distance_limit and gap > 5:
                continue

            proximity_score = 1.0 - min(distance / max(class_distance_limit, 1.0), 1.0)
            iou_score = bbox_iou_xyxy(memory.smoothed_bbox_xyxy, bbox_xyxy)

            app_score = cosine_similarity(memory.appearance, appearance)
            secondary_score = cosine_similarity(memory.secondary_appearance, secondary_appearance)
            if class_name == "person":
                app_score = max(app_score, _gallery_best_similarity(memory.appearance_gallery, appearance))
                secondary_score = max(
                    secondary_score,
                    _gallery_best_similarity(memory.secondary_appearance_gallery, secondary_appearance),
                )

            required_app = self.person_min_appearance_similarity if class_name == "person" else self.min_appearance_similarity
            required_proximity = 0.30 if class_name == "person" else 0.35
            if class_name == "person":
                if app_score < required_app:
                    continue
                if secondary_score < self.person_min_secondary_similarity and proximity_score < required_proximity and iou_score < 0.05 and not same_entry_exit_side:
                    continue
                quality_multiplier = 1.0
                if candidate_quality is not None:
                    quality_multiplier = 0.88 + 0.12 * candidate_quality.score
                score = (
                    self.person_deep_appearance_weight * app_score
                    + self.person_secondary_appearance_weight * secondary_score
                    + self.person_iou_weight * iou_score
                    + self.person_proximity_weight * proximity_score
                ) * quality_multiplier
                if same_entry_exit_side:
                    score += self.entry_exit_same_side_bonus
            else:
                if app_score < required_app and iou_score < 0.05 and proximity_score < required_proximity:
                    continue
                score = (
                    self.appearance_weight * app_score
                    + self.iou_weight * iou_score
                    + self.proximity_weight * proximity_score
                )

            if score > best_score:
                best_score = score
                best_global_id = global_id

        class_threshold = self.person_min_stitch_score if class_name == "person" else self.min_stitch_score
        if best_global_id is None or best_score < class_threshold:
            return None
        return best_global_id, best_score


    def _person_crop_quality(self, frame_shape, bbox_xyxy: np.ndarray) -> PersonCropQuality:
        return assess_person_crop_quality(
            frame_shape=frame_shape,
            bbox_xyxy=bbox_xyxy,
            edge_margin_ratio=self.person_crop_edge_margin_ratio,
            min_height_ratio=self.person_crop_min_height_ratio,
            min_area_ratio=self.person_crop_min_area_ratio,
            min_aspect_ratio=self.person_crop_min_aspect_ratio,
            max_aspect_ratio=self.person_crop_max_aspect_ratio,
            allow_bottom_edge=self.person_crop_allow_bottom_edge,
        )

    def _add_embedding_snapshot(self, gallery: list[np.ndarray], embedding: Optional[np.ndarray]) -> bool:
        if embedding is None:
            return False
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        normalized = _normalize_embedding(embedding)
        if normalized is None:
            return False
        if gallery and _gallery_best_similarity(gallery, normalized) >= self.person_gallery_duplicate_similarity:
            return False
        gallery.append(normalized)
        while len(gallery) > self.person_gallery_size:
            gallery.pop(0)
        return True

    def track_gallery_similarity(self, left: TrackMemory, right: TrackMemory) -> tuple[float, float]:
        """Return best primary and secondary gallery similarities between two tracks."""
        left_gallery = left.appearance_gallery or ([left.appearance] if left.appearance is not None else [])
        right_gallery = right.appearance_gallery or ([right.appearance] if right.appearance is not None else [])
        left_secondary = left.secondary_appearance_gallery or ([left.secondary_appearance] if left.secondary_appearance is not None else [])
        right_secondary = right.secondary_appearance_gallery or ([right.secondary_appearance] if right.secondary_appearance is not None else [])
        primary = _gallery_pair_similarity(left_gallery, right_gallery)
        secondary = _gallery_pair_similarity(left_secondary, right_secondary)
        return primary, secondary

    def merge_tracks(
        self,
        source_global_id: int,
        target_global_id: int,
        new_local_id: Optional[int] = None,
    ) -> None:
        """Merge source track into target track. Used by optional offline cleanup tools only."""
        if source_global_id == target_global_id:
            return
        source = self.tracks.get(source_global_id)
        target = self.tracks.get(target_global_id)
        if source is None or target is None:
            return
        if source.class_name != target.class_name:
            return

        # Redirect all local IDs that pointed to the source.
        for key, value in list(self.local_to_global.items()):
            if value == source_global_id:
                self.local_to_global[key] = target_global_id

        if source.local_tracker_id is not None:
            target.previous_local_tracker_ids.append(source.local_tracker_id)
        target.previous_local_tracker_ids.extend(source.previous_local_tracker_ids)

        if new_local_id is not None:
            target.local_tracker_id = int(new_local_id)

        # Preserve owner scores/history and keep the most recent state. The next
        # update() call in this frame will overwrite geometry with the current box.
        for owner_id, score in source.owner_scores.items():
            target.owner_scores[owner_id] = max(target.owner_scores.get(owner_id, 0.0), score)
        for owner_id, frames in source.owner_contact_frames.items():
            target.owner_contact_frames[owner_id] = max(target.owner_contact_frames.get(owner_id, 0), frames)
        for owner_id, frames in source.owner_separation_frames.items():
            target.owner_separation_frames[owner_id] = max(target.owner_separation_frames.get(owner_id, 0), frames)
        for owner_id, frame in source.owner_last_near_frame.items():
            target.owner_last_near_frame[owner_id] = max(target.owner_last_near_frame.get(owner_id, -1), frame)
        if source.owner_id is not None and target.owner_id is None:
            target.owner_id = source.owner_id
        if target.owner_first_link_frame is None:
            target.owner_first_link_frame = source.owner_first_link_frame
        if source.owner_link_strength > target.owner_link_strength:
            target.owner_link_strength = source.owner_link_strength

        for embedding in source.appearance_gallery:
            self._add_embedding_snapshot(target.appearance_gallery, embedding)
        for embedding in source.secondary_appearance_gallery:
            self._add_embedding_snapshot(target.secondary_appearance_gallery, embedding)
        merged_primary = _gallery_mean(target.appearance_gallery)
        if merged_primary is not None:
            target.appearance = merged_primary
        merged_secondary = _gallery_mean(target.secondary_appearance_gallery)
        if merged_secondary is not None:
            target.secondary_appearance = merged_secondary
        target.good_snapshot_count = max(target.good_snapshot_count, len(target.appearance_gallery))
        if source.exit_side is not None:
            target.exit_side = source.exit_side
            target.exit_frame = source.exit_frame

        target.reidentified_count += source.reidentified_count + 1
        target.missing_frames = min(target.missing_frames, source.missing_frames)

        # Keep the source out of future matching; previous CSV/video frames may
        # already contain it, but the pipeline will rewrite CSV rows when possible.
        del self.tracks[source_global_id]

    def update(
        self,
        track_id: int,
        class_name: str,
        frame_idx: int,
        bbox_xyxy: np.ndarray,
        confidence: float,
        frame: Optional[np.ndarray] = None,
        local_tracker_id: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        prefer_mask_geometry: bool = False,
    ) -> TrackMemory:
        raw_bbox = bbox_xyxy.astype(float)
        bbox = choose_geometry_bbox(raw_bbox, mask, prefer_mask_geometry)
        center = choose_geometry_center(raw_bbox, mask, prefer_mask_geometry)
        used_mask = prefer_mask_geometry and mask is not None and bbox_from_mask(mask) is not None

        crop_quality: Optional[PersonCropQuality] = None
        observed_side: Optional[str] = None
        can_update_appearance = frame is not None
        if frame is not None and class_name == "person":
            crop_quality = self._person_crop_quality(frame.shape, bbox)
            observed_side = edge_side_from_bbox(frame.shape, bbox, self.entry_exit_edge_ratio)
            can_update_appearance = crop_quality.usable_for_memory_update

        appearance = appearance_embedding_for_class(frame, bbox, class_name, mask=mask) if can_update_appearance and frame is not None else None
        secondary_appearance = secondary_embedding_for_class(frame, bbox, class_name, mask=mask) if can_update_appearance and frame is not None else None

        if track_id not in self.tracks:
            appearance_gallery: list[np.ndarray] = []
            secondary_gallery: list[np.ndarray] = []
            if class_name == "person" and appearance is not None:
                normalized = _normalize_embedding(appearance)
                if normalized is not None:
                    appearance_gallery.append(normalized)
                    appearance = normalized
            if class_name == "person" and secondary_appearance is not None:
                normalized_secondary = _normalize_embedding(secondary_appearance)
                if normalized_secondary is not None:
                    secondary_gallery.append(normalized_secondary)
                    secondary_appearance = normalized_secondary

            memory = TrackMemory(
                global_id=track_id,
                class_name=class_name,
                first_frame=frame_idx,
                last_frame=frame_idx,
                last_bbox_xyxy=bbox,
                smoothed_bbox_xyxy=bbox.copy(),
                last_center=center,
                last_confidence=float(confidence),
                local_tracker_id=local_tracker_id,
                history=[(frame_idx, center)],
                appearance=appearance,
                secondary_appearance=secondary_appearance,
                appearance_gallery=appearance_gallery,
                secondary_appearance_gallery=secondary_gallery,
                last_crop_quality=0.0 if crop_quality is None else float(crop_quality.score),
                last_crop_quality_reason="n/a" if crop_quality is None else crop_quality.reason,
                good_snapshot_count=len(appearance_gallery),
                last_observed_side=observed_side,
                entry_side=observed_side,
                last_mask=mask,
                last_mask_area=mask_area(mask),
                used_mask_geometry=used_mask,
            )
            self.tracks[track_id] = memory
            return memory

        memory = self.tracks[track_id]
        class_changed = memory.class_name != class_name
        if class_changed:
            memory.appearance_gallery.clear()
            memory.secondary_appearance_gallery.clear()
            memory.good_snapshot_count = 0

        alpha = self.ema_alpha
        smoothed_bbox = alpha * memory.smoothed_bbox_xyxy + (1.0 - alpha) * bbox
        smoothed_center = bbox_center_xy(smoothed_bbox)
        if used_mask:
            smoothed_center = center

        movement_px = float(np.linalg.norm(np.array(smoothed_center) - np.array(memory.last_center)))

        if movement_px < self.stationary_movement_px:
            memory.stationary_frames += 1
        else:
            memory.stationary_frames = 0

        memory.class_name = class_name
        memory.last_frame = frame_idx
        memory.last_bbox_xyxy = bbox
        memory.smoothed_bbox_xyxy = smoothed_bbox
        memory.last_center = smoothed_center
        memory.last_confidence = float(confidence)
        memory.local_tracker_id = local_tracker_id
        memory.history.append((frame_idx, smoothed_center))
        memory.missing_frames = 0
        memory.used_mask_geometry = used_mask

        if crop_quality is not None:
            memory.last_crop_quality = float(crop_quality.score)
            memory.last_crop_quality_reason = crop_quality.reason
        if observed_side is not None:
            memory.last_observed_side = observed_side
            memory.entry_side = observed_side

        if mask is not None:
            memory.last_mask = mask
            memory.last_mask_area = mask_area(mask)

        if class_name == "person":
            if appearance is not None:
                self._add_embedding_snapshot(memory.appearance_gallery, appearance)
                memory.good_snapshot_count = len(memory.appearance_gallery)
                gallery_mean = _gallery_mean(memory.appearance_gallery)
                if gallery_mean is not None:
                    memory.appearance = gallery_mean
                elif memory.appearance is None or class_changed or memory.appearance.shape != appearance.shape:
                    memory.appearance = appearance
            if secondary_appearance is not None:
                self._add_embedding_snapshot(memory.secondary_appearance_gallery, secondary_appearance)
                secondary_mean = _gallery_mean(memory.secondary_appearance_gallery)
                if secondary_mean is not None:
                    memory.secondary_appearance = secondary_mean
                elif memory.secondary_appearance is None or class_changed or memory.secondary_appearance.shape != secondary_appearance.shape:
                    memory.secondary_appearance = secondary_appearance
        else:
            if appearance is not None:
                if memory.appearance is None or class_changed or memory.appearance.shape != appearance.shape:
                    memory.appearance = appearance
                else:
                    updated = 0.90 * memory.appearance + 0.10 * appearance
                    norm = float(np.linalg.norm(updated))
                    memory.appearance = updated / norm if norm > 1e-8 else appearance

            if secondary_appearance is not None:
                if memory.secondary_appearance is None or class_changed or memory.secondary_appearance.shape != secondary_appearance.shape:
                    memory.secondary_appearance = secondary_appearance
                else:
                    updated_secondary = 0.85 * memory.secondary_appearance + 0.15 * secondary_appearance
                    norm_secondary = float(np.linalg.norm(updated_secondary))
                    memory.secondary_appearance = updated_secondary / norm_secondary if norm_secondary > 1e-8 else secondary_appearance

        return memory

    def mark_missing(self, seen_ids: set[int]) -> None:
        for track_id, memory in self.tracks.items():
            if track_id not in seen_ids:
                if memory.missing_frames == 0 and memory.last_observed_side is not None:
                    memory.exit_side = memory.last_observed_side
                    memory.exit_frame = memory.last_frame
                memory.missing_frames += 1

    def active_tracks(self, max_missing_frames: Optional[int] = None) -> list[TrackMemory]:
        max_missing = self.max_missing_frames if max_missing_frames is None else max_missing_frames
        return [t for t in self.tracks.values() if t.missing_frames <= max_missing]

    def recently_seen_tracks(self, max_gap_frames: Optional[int] = None) -> list[TrackMemory]:
        max_gap = self.max_reid_gap_frames if max_gap_frames is None else max_gap_frames
        return [t for t in self.tracks.values() if t.missing_frames <= max_gap]
