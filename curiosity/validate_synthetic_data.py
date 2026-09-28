#!/usr/bin/env python3
"""Validate synthetic office dataset against consistency and quality rules.

Usage:
    python validate_synthetic_data.py
    python validate_synthetic_data.py --fix-stats
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "synthetic_data"
DAYS = ["2026-05-26", "2026-05-27", "2026-05-28"]
ROOMS = [
    "Dock Hall", "Pod A", "Pod B", "The Lab",
    "The Kitchen", "The Yard", "The Stage", "The Cellar",
    "The Booth", "The Deck",
]

# Distance matrix (symmetric)
_distances = {
    ("Dock Hall", "Pod A"): 2, ("Dock Hall", "Pod B"): 2, ("Dock Hall", "The Lab"): 3,
    ("Dock Hall", "The Kitchen"): 1, ("Dock Hall", "The Yard"): 1, ("Dock Hall", "The Stage"): 2,
    ("Dock Hall", "The Cellar"): 3, ("Dock Hall", "The Booth"): 1, ("Dock Hall", "The Deck"): 2,
    ("Pod A", "Pod B"): 2, ("Pod A", "The Lab"): 3, ("Pod A", "The Kitchen"): 2,
    ("Pod A", "The Yard"): 2, ("Pod A", "The Stage"): 2, ("Pod A", "The Cellar"): 3,
    ("Pod A", "The Booth"): 2, ("Pod A", "The Deck"): 2, ("Pod B", "The Lab"): 3,
    ("Pod B", "The Kitchen"): 2, ("Pod B", "The Yard"): 2, ("Pod B", "The Stage"): 2,
    ("Pod B", "The Cellar"): 3, ("Pod B", "The Booth"): 2, ("Pod B", "The Deck"): 2,
    ("The Lab", "The Kitchen"): 3, ("The Lab", "The Yard"): 3, ("The Lab", "The Stage"): 3,
    ("The Lab", "The Cellar"): 2, ("The Lab", "The Booth"): 3, ("The Lab", "The Deck"): 3,
    ("The Kitchen", "The Yard"): 1, ("The Kitchen", "The Stage"): 2,
    ("The Kitchen", "The Cellar"): 3, ("The Kitchen", "The Booth"): 1,
    ("The Kitchen", "The Deck"): 2, ("The Yard", "The Stage"): 2,
    ("The Yard", "The Cellar"): 3, ("The Yard", "The Booth"): 1,
    ("The Yard", "The Deck"): 2, ("The Stage", "The Cellar"): 3,
    ("The Stage", "The Booth"): 2, ("The Stage", "The Deck"): 1,
    ("The Cellar", "The Booth"): 3, ("The Cellar", "The Deck"): 3,
    ("The Booth", "The Deck"): 2,
}
distances = {}
for (r1, r2), d in _distances.items():
    distances[(r1, r2)] = d
    distances[(r2, r1)] = d
for r in ROOMS:
    distances[(r, r)] = 0


def parse_time(t: str) -> int:
    h, m = map(int, t.split(":"))
    return h * 60 + m


def validate_day(date_str: str) -> dict:
    """Run all validation checks for a single day."""
    results = {"passed": 0, "failed": 0, "warnings": 0, "details": []}

    def log(status: str, msg: str):
        results["details"].append((status, msg))
        if status == "PASS":
            results["passed"] += 1
        elif status == "FAIL":
            results["failed"] += 1
        else:
            results["warnings"] += 1

    # ------------------------------------------------------------------
    # 1. File existence and JSON validity
    # ------------------------------------------------------------------
    time_log_path = BASE_DIR / f"office_time_log_{date_str}.json"
    interactions_path = BASE_DIR / f"interactions_{date_str}.json"
    person_locs_path = BASE_DIR / f"person_locations_{date_str}.json"

    for path in [time_log_path, interactions_path, person_locs_path]:
        if not path.exists():
            log("FAIL", f"Missing file: {path.name}")
            return results

    try:
        with open(time_log_path, "r", encoding="utf-8") as f:
            time_log = json.load(f)
    except json.JSONDecodeError as exc:
        log("FAIL", f"Invalid JSON in time log: {exc}")
        return results

    try:
        with open(interactions_path, "r", encoding="utf-8") as f:
            interactions = json.load(f)
    except json.JSONDecodeError as exc:
        log("FAIL", f"Invalid JSON in interactions: {exc}")
        return results

    try:
        with open(person_locs_path, "r", encoding="utf-8") as f:
            person_locs = json.load(f)
    except json.JSONDecodeError as exc:
        log("FAIL", f"Invalid JSON in person locations: {exc}")
        return results

    log("PASS", f"All files present and valid JSON")

    # ------------------------------------------------------------------
    # 2. Entry counts
    # ------------------------------------------------------------------
    expected_entries = len(ROOMS) * 661  # 08:00 to 19:00 inclusive = 661 minutes
    if len(time_log) != expected_entries:
        log("FAIL", f"Time log has {len(time_log)} entries, expected {expected_entries}")
    else:
        log("PASS", f"Time log has exactly {expected_entries} entries")

    # ------------------------------------------------------------------
    # 3. Room coverage
    # ------------------------------------------------------------------
    room_counts = Counter(e["room"] for e in time_log)
    missing_rooms = [r for r in ROOMS if room_counts.get(r, 0) != 661]
    if missing_rooms:
        log("FAIL", f"Rooms with wrong entry count: {missing_rooms}")
    else:
        log("PASS", f"All {len(ROOMS)} rooms have exactly 661 entries")

    # ------------------------------------------------------------------
    # 4. Time continuity
    # ------------------------------------------------------------------
    for room in ROOMS:
        room_entries = [e for e in time_log if e["room"] == room]
        times = [e["time"] for e in room_entries]
        expected_times = []
        t = parse_time("08:00")
        while t <= parse_time("19:00"):
            h, m = divmod(t, 60)
            expected_times.append(f"{h:02d}:{m:02d}")
            t += 1
        if times != expected_times:
            log("FAIL", f"Time continuity broken in {room}")
            break
    else:
        log("PASS", "Time continuity OK for all rooms")

    # ------------------------------------------------------------------
    # 5. Person consistency (no one in two rooms at once)
    # ------------------------------------------------------------------
    person_at_time = defaultdict(lambda: defaultdict(list))
    for entry in time_log:
        t = entry["time"]
        r = entry["room"]
        for p in entry.get("people_present", []):
            person_at_time[t][p].append(r)

    violations = 0
    for t, people in person_at_time.items():
        for p, rooms in people.items():
            if len(rooms) > 1:
                violations += 1
                if violations <= 3:
                    log("FAIL", f"{p} in multiple rooms at {t}: {rooms}")
    if violations == 0:
        log("PASS", "No person consistency violations")
    elif violations > 3:
        log("FAIL", f"Total person consistency violations: {violations}")

    # ------------------------------------------------------------------
    # 6. Travel time constraints
    # ------------------------------------------------------------------
    person_trace = defaultdict(list)
    for entry in time_log:
        t = entry["time"]
        for p in entry.get("people_present", []):
            person_trace[p].append((t, entry["room"]))

    travel_violations = 0
    for p, trace in person_trace.items():
        trace.sort(key=lambda x: parse_time(x[0]))
        for i in range(1, len(trace)):
            t1, r1 = trace[i - 1]
            t2, r2 = trace[i]
            diff = parse_time(t2) - parse_time(t1)
            if r1 != r2 and diff < distances.get((r1, r2), 1):
                travel_violations += 1
                if travel_violations <= 3:
                    log("FAIL", f"{p} moved {r1} -> {r2} in {diff} min (need {distances.get((r1, r2), 1)})")
    if travel_violations == 0:
        log("PASS", "No travel time violations")
    elif travel_violations > 3:
        log("FAIL", f"Total travel time violations: {travel_violations}")

    # ------------------------------------------------------------------
    # 7. Interaction ID uniqueness
    # ------------------------------------------------------------------
    int_ids = [i["interaction_id"] for i in interactions if "interaction_id" in i]
    if len(int_ids) != len(set(int_ids)):
        duplicates = [item for item, count in Counter(int_ids).items() if count > 1]
        log("FAIL", f"Duplicate interaction IDs: {duplicates[:5]}")
    else:
        log("PASS", f"All {len(int_ids)} interaction IDs are unique")

    # ------------------------------------------------------------------
    # 8. Interaction ↔ time log cross-reference
    # ------------------------------------------------------------------
    tl_ids = {e.get("interaction_id") for e in time_log if e.get("interaction_id")}
    orphan_interactions = [i for i in interactions if i["interaction_id"] not in tl_ids]
    if orphan_interactions:
        log("FAIL", f"{len(orphan_interactions)} interactions not referenced in time log")
    else:
        log("PASS", "All interactions referenced in time log")

    # ------------------------------------------------------------------
    # 9. Person locations consistency
    # ------------------------------------------------------------------
    pl_counts = Counter((pl["time"], pl["person"], pl["room"]) for pl in person_locs)
    duplicates = [k for k, v in pl_counts.items() if v > 1]
    if duplicates:
        log("FAIL", f"Duplicate person locations: {duplicates[:3]}")
    else:
        log("PASS", "No duplicate person locations")

    # Check person locations match time log
    tl_people = defaultdict(lambda: defaultdict(set))
    for entry in time_log:
        for p in entry.get("people_present", []):
            tl_people[entry["time"]][entry["room"]].add(p)

    pl_mismatches = 0
    for pl in person_locs:
        t, p, r = pl["time"], pl["person"], pl["room"]
        if p not in tl_people[t][r]:
            pl_mismatches += 1
            if pl_mismatches <= 3:
                log("FAIL", f"Person location mismatch: {p} at {t} in {r} not in time log")
    if pl_mismatches == 0:
        log("PASS", "All person locations match time log")
    elif pl_mismatches > 3:
        log("FAIL", f"Total person location mismatches: {pl_mismatches}")

    # ------------------------------------------------------------------
    # 10. Scene text quality (no reasoning words)
    # ------------------------------------------------------------------
    reasoning_words = ["because", "since", "to ", "in order to", "so that", "therefore"]
    bad_scenes = 0
    for entry in time_log:
        scene = entry.get("scene", "").lower()
        for word in reasoning_words:
            if word in scene:
                bad_scenes += 1
                if bad_scenes <= 3:
                    log("WARN", f"Scene at {entry['time']} {entry['room']} may contain reasoning: '{entry['scene'][:60]}'")
                break
    if bad_scenes == 0:
        log("PASS", "No scenes with reasoning words detected")
    else:
        log("WARN", f"{bad_scenes} scenes contain potential reasoning words (sampled above)")

    # ------------------------------------------------------------------
    # 11. Event count sanity
    # ------------------------------------------------------------------
    event_count = sum(1 for e in time_log if e.get("event") == 1)
    if event_count < 500:
        log("WARN", f"Very few events: {event_count} (expected > 500 for a full day)")
    elif event_count > 3000:
        log("WARN", f"Very many events: {event_count} (expected < 3000)")
    else:
        log("PASS", f"Event count {event_count} is in expected range")

    # ------------------------------------------------------------------
    # 12. Event=1 entries must have people_present OR be first minute
    # ------------------------------------------------------------------
    empty_events = [e for e in time_log if e.get("event") == 1 and not e.get("people_present") and e["time"] != "08:00"]
    if empty_events:
        log("WARN", f"{len(empty_events)} event=1 entries have no people present (excluding 08:00)")
    else:
        log("PASS", "All event=1 entries (except 08:00) have people present")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate synthetic office dataset")
    parser.add_argument("--fix-stats", action="store_true", help="Print per-room event stats for manual review")
    args = parser.parse_args()

    print("=" * 70)
    print("SYNTHETIC DATA VALIDATION REPORT")
    print("=" * 70)

    all_ok = True
    for day in DAYS:
        print(f"\n--- {day} ---")
        results = validate_day(day)
        for status, msg in results["details"]:
            marker = "✅" if status == "PASS" else ("❌" if status == "FAIL" else "⚠️")
            print(f"  {marker} {msg}")
        print(f"  Results: {results['passed']} passed, {results['failed']} failed, {results['warnings']} warnings")
        if results["failed"] > 0:
            all_ok = False

        if args.fix_stats:
            time_log_path = BASE_DIR / f"office_time_log_{day}.json"
            with open(time_log_path, "r", encoding="utf-8") as f:
                time_log = json.load(f)
            events = [e for e in time_log if e.get("event") == 1]
            print(f"  Per-room event stats:")
            room_event_counts = Counter(e["room"] for e in events)
            for r in ROOMS:
                print(f"    {r:15s}: {room_event_counts.get(r, 0)}")

    print("\n" + "=" * 70)
    if all_ok:
        print("OVERALL: ALL CHECKS PASSED")
    else:
        print("OVERALL: SOME CHECKS FAILED")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
