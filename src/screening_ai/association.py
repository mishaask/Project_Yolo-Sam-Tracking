from __future__ import annotations

import numpy as np

from screening_ai.memory import MemoryBank, TrackMemory, bbox_iou_xyxy


def center_distance(a: TrackMemory, b: TrackMemory) -> float:
    return float(np.linalg.norm(np.array(a.last_center) - np.array(b.last_center)))


def movement_similarity(a: TrackMemory, b: TrackMemory, window: int = 20) -> float:
    if len(a.history) < 2 or len(b.history) < 2:
        return 0.0

    a_points = np.array([point for _, point in a.history[-window:]], dtype=float)
    b_points = np.array([point for _, point in b.history[-window:]], dtype=float)

    n = min(len(a_points), len(b_points))
    if n < 2:
        return 0.0

    a_points = a_points[-n:]
    b_points = b_points[-n:]

    a_vec = a_points[-1] - a_points[0]
    b_vec = b_points[-1] - b_points[0]

    a_norm = float(np.linalg.norm(a_vec))
    b_norm = float(np.linalg.norm(b_vec))

    if a_norm < 1e-6 or b_norm < 1e-6:
        return 0.0

    cosine = float(np.dot(a_vec, b_vec) / (a_norm * b_norm))
    return max(0.0, min(1.0, cosine))


def assign_bag_owners(
    memory_bank: MemoryBank,
    person_classes: set[str],
    bag_classes: set[str],
    frame_idx: int = 0,
    max_distance_px: float = 180.0,
    min_owner_score: float = 0.45,
    motion_window_frames: int = 20,
    score_decay: float = 0.98,
    score_gain: float = 0.12,
    min_contact_frames: int = 45,
    contact_distance_px: float = 150.0,
    separation_distance_px: float = 260.0,
) -> None:
    """
    Assign likely bag ownership using a separate relationship memory.

    Person ReID answers: "which anonymous person track is this?"
    This function answers: "which anonymous person was this bag moving/standing with?"

    Ownership is not decided from one frame. The bag stores per-person evidence:
      - owner_scores: accumulated proximity/motion score
      - owner_contact_frames: how many frames the bag was near/overlapping a person
      - owner_separation_frames: how long the confirmed owner has been far away
      - owner_last_near_frame: last frame where bag and person were near/contacting
    """
    people = [t for t in memory_bank.active_tracks() if t.class_name in person_classes]
    bags = [t for t in memory_bank.active_tracks() if t.class_name in bag_classes]

    for bag in bags:
        for person_id in list(bag.owner_scores.keys()):
            bag.owner_scores[person_id] *= score_decay
            if bag.owner_scores[person_id] < 0.02:
                del bag.owner_scores[person_id]

        seen_people_this_frame: set[int] = set()

        for person in people:
            dist = center_distance(person, bag)
            iou = bbox_iou_xyxy(person.smoothed_bbox_xyxy, bag.smoothed_bbox_xyxy)
            is_contact = dist <= contact_distance_px or iou >= 0.015
            is_candidate = dist <= max_distance_px or iou >= 0.015

            if is_contact:
                bag.owner_contact_frames[person.global_id] = bag.owner_contact_frames.get(person.global_id, 0) + 1
                bag.owner_last_near_frame[person.global_id] = int(frame_idx)
                bag.owner_separation_frames[person.global_id] = 0
                seen_people_this_frame.add(person.global_id)

            if not is_candidate:
                continue

            proximity_score = 1.0 - min(dist / max(max_distance_px, 1.0), 1.0)
            motion_score = movement_similarity(person, bag, window=motion_window_frames)
            overlap_score = min(iou * 6.0, 1.0)
            contact_bonus = 0.20 if is_contact else 0.0
            instant_score = 0.48 * proximity_score + 0.30 * motion_score + 0.22 * overlap_score + contact_bonus

            if instant_score <= 0.0:
                continue

            current = bag.owner_scores.get(person.global_id, 0.0)
            bag.owner_scores[person.global_id] = min(1.0, current + score_gain * instant_score)

        # Update separation counters for previously considered people that were not near this frame.
        for person_id in list(bag.owner_contact_frames.keys()):
            if person_id in seen_people_this_frame:
                continue
            person = memory_bank.tracks.get(person_id)
            if person is None:
                continue
            if center_distance(person, bag) >= separation_distance_px:
                bag.owner_separation_frames[person_id] = bag.owner_separation_frames.get(person_id, 0) + 1
            else:
                bag.owner_separation_frames[person_id] = 0

        if not bag.owner_scores:
            continue

        # Only confirmed relationships can become ownership links.
        confirmed_candidates = [
            (person_id, score)
            for person_id, score in bag.owner_scores.items()
            if bag.owner_contact_frames.get(person_id, 0) >= min_contact_frames and score >= min_owner_score
        ]
        if not confirmed_candidates:
            continue

        best_person_id, best_score = max(confirmed_candidates, key=lambda item: item[1])
        if bag.owner_id is None:
            bag.owner_first_link_frame = int(frame_idx)
        elif bag.owner_id != best_person_id:
            # Switch only when the new owner evidence is clearly stronger.
            old_score = bag.owner_scores.get(bag.owner_id, 0.0)
            if best_score < old_score + 0.18:
                best_person_id = bag.owner_id
                best_score = old_score

        bag.owner_id = best_person_id
        bag.owner_link_strength = float(best_score)

        owner = memory_bank.tracks.get(best_person_id)
        if owner is not None:
            dist = center_distance(owner, bag)
            bag.owner_last_distance_px = float(dist)
            if dist >= separation_distance_px:
                if bag.owner_separated_from_frame is None:
                    bag.owner_separated_from_frame = int(frame_idx)
            else:
                bag.owner_separated_from_frame = None


def owner_link_lines(
    memory_bank: MemoryBank,
    bag_classes: set[str],
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    """Return lines that can be drawn between a bag and its likely owner."""
    lines = []
    for bag in memory_bank.active_tracks():
        if bag.class_name not in bag_classes or bag.owner_id is None:
            continue
        owner = memory_bank.tracks.get(bag.owner_id)
        if owner is None:
            continue
        p1 = (int(round(bag.last_center[0])), int(round(bag.last_center[1])))
        p2 = (int(round(owner.last_center[0])), int(round(owner.last_center[1])))
        lines.append((p1, p2, bag.global_id, owner.global_id))
    return lines
