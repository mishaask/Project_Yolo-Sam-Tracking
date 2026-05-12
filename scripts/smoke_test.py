from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from screening_ai.association import assign_bag_owners
from screening_ai.memory import MemoryBank


def test_association() -> None:
    bank = MemoryBank()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Person and bag move together for several frames.
    for f in range(20):
        person_box = np.array([100 + f * 3, 100, 180 + f * 3, 300], dtype=float)
        bag_box = np.array([185 + f * 3, 180, 230 + f * 3, 270], dtype=float)
        bank.update(1, "person", f, person_box, 0.9, frame=frame, local_tracker_id=1)
        bank.update(2, "backpack", f, bag_box, 0.9, frame=frame, local_tracker_id=2)
        assign_bag_owners(bank, {"person"}, {"backpack"}, frame_idx=f, max_distance_px=160.0, min_contact_frames=10)

    assert bank.tracks[2].owner_id == 1, "Bag should be associated with the moving person."


def test_reconnect_with_local_id_change() -> None:
    bank = MemoryBank(min_stitch_score=0.3, max_center_distance_px=300.0)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:260, 100:180] = (0, 0, 255)

    first_box = np.array([100, 100, 180, 260], dtype=float)
    first = bank.resolve_global_id(7, "person", 0, first_box, 0.9, frame)
    bank.update(first.global_id, "person", 0, first_box, 0.9, frame=frame, local_tracker_id=7)
    bank.mark_missing({first.global_id})

    # Same object reappears nearby with a new YOLO local ID.
    second_box = np.array([108, 102, 188, 262], dtype=float)
    second = bank.resolve_global_id(99, "person", 8, second_box, 0.9, frame)

    assert second.global_id == first.global_id, "New local ID should reconnect to old global ID."


def test_mask_geometry() -> None:
    bank = MemoryBank()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = np.array([80, 80, 220, 320], dtype=float)
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[110:300, 120:180] = 1

    memory = bank.update(
        1,
        "person",
        0,
        bbox,
        0.9,
        frame=frame,
        local_tracker_id=1,
        mask=mask,
        prefer_mask_geometry=True,
    )
    assert memory.used_mask_geometry is True
    assert memory.last_mask_area > 0
    assert 120 <= memory.last_center[0] <= 180


def test_group_safe_no_visible_person_collapse() -> None:
    bank = MemoryBank(person_min_stitch_score=0.45, person_min_appearance_similarity=0.05)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[100:300, 100:220] = (20, 80, 200)
    frame[100:300, 260:380] = (20, 80, 200)

    # Person A already has local ID 1 and is still visible in this frame.
    box_a = np.array([100, 100, 220, 320], dtype=float)
    a = bank.resolve_global_id(1, "person", 0, box_a, 0.9, frame)
    bank.update(a.global_id, "person", 0, box_a, 0.9, frame=frame, local_tracker_id=1)

    # Person B has a similar appearance and appears with local ID 2 in the same frame.
    # Group-safe ReID must not collapse B into A while A's local tracker is visible.
    box_b = np.array([260, 100, 380, 320], dtype=float)
    b = bank.resolve_global_id(
        2, "person", 1, box_b, 0.9, frame, active_local_ids_by_class={"person": {1, 2}}
    )

    assert b.global_id != a.global_id, "Two visible people must not collapse into one global ID."


def main() -> None:
    test_association()
    test_reconnect_with_local_id_change()
    test_mask_geometry()
    test_group_safe_no_visible_person_collapse()
    print("Smoke tests passed: association, reconnect, SAM mask geometry, and group-safe ReID work.")


if __name__ == "__main__":
    main()
