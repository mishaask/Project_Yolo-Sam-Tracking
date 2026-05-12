from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite already-produced track CSV IDs using a simple mapping JSON.")
    parser.add_argument("--tracks", default="outputs/tracks.csv", help="Input tracks CSV.")
    parser.add_argument("--events", default="outputs/events.json", help="Input events JSON.")
    parser.add_argument("--mapping", required=True, help="JSON mapping, e.g. {\"2\": 1, \"5\": 1, \"6\": 1}.")
    parser.add_argument("--output-tracks", default="outputs/tracks_merged.csv", help="Output merged CSV.")
    parser.add_argument("--output-events", default="outputs/events_merged.json", help="Output merged JSON.")
    return parser


def canonical(value: int, mapping: dict[int, int]) -> int:
    seen: set[int] = set()
    while value in mapping and value not in seen:
        seen.add(value)
        value = int(mapping[value])
    return int(value)


def main() -> None:
    args = build_parser().parse_args()
    mapping_raw = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    mapping = {int(k): int(v) for k, v in mapping_raw.items()}

    with open(args.tracks, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for field in ["raw_global_id", "offline_merged_into"]:
        if field not in fieldnames:
            fieldnames.insert(2, field)

    for row in rows:
        raw = int(row.get("raw_global_id") or row.get("global_id") or -1)
        new_id = canonical(raw, mapping)
        row["raw_global_id"] = raw
        row["global_id"] = new_id
        row["offline_merged_into"] = "" if new_id == raw else new_id
        if row.get("owner_id") not in {None, ""}:
            row["owner_id"] = canonical(int(row["owner_id"]), mapping)

    Path(args.output_tracks).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_tracks, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    for event in events:
        if event.get("track_id") is not None:
            event["track_id"] = canonical(int(event["track_id"]), mapping)
        if event.get("owner_id") is not None:
            event["owner_id"] = canonical(int(event["owner_id"]), mapping)
        msg = event.get("message", "")
        for old_id, new_id in mapping.items():
            msg = msg.replace(f"G{old_id}", f"G{canonical(new_id, mapping)}")
            msg = msg.replace(f"track {old_id}", f"track {canonical(new_id, mapping)}")
        event["message"] = msg

    Path(args.output_events).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_events).write_text(json.dumps(events, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {args.output_tracks}")
    print(f"Wrote {args.output_events}")


if __name__ == "__main__":
    main()
