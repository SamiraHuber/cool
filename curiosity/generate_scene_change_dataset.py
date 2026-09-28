#!/usr/bin/env python3
"""Generate synthetic 3-day office dataset for scene-change-aware navigation.

Outputs:
    - scene_log_2026-05-2{6,7,8}.json   — one entry per room per 5-min interval
    - person_locations_*.json           — per-person location trail
    - object_locations_*.json           — per-object location trail

Dataset properties:
    • 6 rooms, 8 people, 12 objects + implied activity objects
    • 5-minute intervals, 09:00–14:00 (60 intervals/day)
    • Objects have default owners; they move with their owner
    • Activities have realistic durations (coffee = 5-10 min, sitting = 20-200 min, etc.)
    • Activity-implied objects (cup, plate, etc.) appear consistently in scenes
    • Scheduled events + random micro-movements create realistic changes
    • Scene-change labels are computed post-hoc by the change tracker
"""

import json
import random
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOMS = {
    "kitchen":      {"type": "social",   "capacity": 8,  "fixtures": ["coffee_machine", "fridge", "microwave"]},
    "storage":      {"type": "utility",  "capacity": 4,  "fixtures": ["shelf", "toolbox"]},
    "office 1":     {"type": "office",   "capacity": 3,  "fixtures": ["desk_1", "desk_2", "desk_3"]},
    "office 2":     {"type": "office",   "capacity": 3,  "fixtures": ["desk_1", "desk_2", "desk_3"]},
    "office 3":     {"type": "office",   "capacity": 3,  "fixtures": ["desk_1", "desk_2", "desk_3"]},
    "meeting room": {"type": "meeting",  "capacity": 10, "fixtures": ["projector", "whiteboard", "conference_table"]},
}

# Distance matrix (walking minutes, symmetric)
_distances = {
    ("kitchen", "meeting room"): 2,
    ("kitchen", "office 1"):     2,
    ("kitchen", "office 2"):     3,
    ("kitchen", "office 3"):     4,
    ("kitchen", "storage"):      3,
    ("meeting room", "office 1"): 3,
    ("meeting room", "office 2"): 2,
    ("meeting room", "office 3"): 3,
    ("meeting room", "storage"):  2,
    ("office 1", "office 2"):     1,
    ("office 1", "office 3"):     2,
    ("office 1", "storage"):      4,
    ("office 2", "office 3"):     1,
    ("office 2", "storage"):      3,
    ("office 3", "storage"):      2,
}

DISTANCES: dict[str, dict[str, int]] = defaultdict(dict)
for r in ROOMS:
    DISTANCES[r][r] = 0
for (r1, r2), d in _distances.items():
    DISTANCES[r1][r2] = d
    DISTANCES[r2][r1] = d

# People with home offices
PEOPLE = {
    "Alex":  {"home": "office 1", "role": "dev"},
    "Blake": {"home": "office 1", "role": "dev"},
    "Casey": {"home": "office 2", "role": "designer"},
    "Dana":  {"home": "office 2", "role": "pm"},
    "Evan":  {"home": "office 3", "role": "dev"},
    "Frank": {"home": "office 3", "role": "analyst"},
    "Grace": {"home": "office 1", "role": "intern"},
    "Heidi": {"home": "office 2", "role": "writer"},
}

# Objects with default owners
OBJECTS = {
    # Personal objects (move with owner)
    "laptop":       {"owner": "Alex",  "type": "personal"},
    "red_cup":      {"owner": "Alex",  "type": "personal"},
    "backpack":     {"owner": "Blake", "type": "personal"},
    "water_bottle": {"owner": "Blake", "type": "personal"},
    "notebook":     {"owner": "Casey", "type": "personal"},
    "pen":          {"owner": "Casey", "type": "personal"},
    "headphones":   {"owner": "Dana",  "type": "personal"},
    "coffee_mug":   {"owner": "Evan",  "type": "personal"},
    "tablet":       {"owner": "Frank", "type": "personal"},
    "jacket":       {"owner": "Frank", "type": "personal"},
    "yoga_mat":     {"owner": "Grace", "type": "personal"},
    "camera":       {"owner": "Heidi", "type": "personal"},
}

# Activity templates per room type
ACTIVITIES = {
    "office": [
        "works at their desk", "types on their laptop", "reads a document",
        "joins a video call", "writes notes", "checks their phone",
    ],
    "meeting": [
        "sits at the table", "talks with a colleague", "points at the screen",
        "writes on the whiteboard", "listens attentively", "takes notes",
    ],
    "social": [
        "makes coffee", "prepares a snack", "washes a mug",
        "chats near the counter", "fills a water bottle", "waits by the microwave",
    ],
    "utility": [
        "organizes boxes", "looks for a tool", "checks inventory",
        "carries a box", "sweeps the floor",
    ],
}

# Realistic activity durations in minutes (min, max)
# 5-minute intervals mean 1 step = 5 min.
ACTIVITY_DURATIONS = {
    # Quick tasks (5–15 min)
    "makes coffee":         (5, 10),
    "prepares a snack":     (5, 15),
    "washes a mug":         (5, 10),
    "fills a water bottle": (5, 5),
    "waits by the microwave": (5, 10),
    "checks their phone":   (5, 5),
    "points at the screen": (5, 10),
    "chats near the counter": (5, 20),
    "looks for a tool":     (5, 15),
    "checks inventory":     (5, 15),
    "carries a box":        (5, 15),

    # Medium tasks (10–60 min)
    "reads a document":     (10, 40),
    "writes notes":         (10, 30),
    "takes notes":          (10, 30),
    "types on their laptop": (15, 120),
    "joins a video call":   (15, 120),
    "sits at the table":    (20, 200),
    "listens attentively":  (10, 120),
    "organizes boxes":      (10, 30),
    "sweeps the floor":     (10, 20),
    "talks with a colleague": (5, 30),
    "writes on the whiteboard": (5, 20),

    # Long tasks (20–200 min)
    "works at their desk":  (20, 200),
}

# Objects that should appear when a person is doing a specific activity
# These are added to objects_present for consistency.
ACTIVITY_IMPLICIT_OBJECTS = {
    "makes coffee":         ["cup"],
    "prepares a snack":     ["plate", "bowl"],
    "washes a mug":         ["mug"],
    "reads a document":     ["document"],
    "checks their phone":   ["phone"],
    "organizes boxes":      ["box"],
    "checks inventory":     ["clipboard"],
    "carries a box":        ["box"],
    "sweeps the floor":     ["broom"],
}

# Days
DAYS = ["2026-05-26", "2026-05-27", "2026-05-28"]
START_TIME = "09:00"
END_TIME = "14:00"
INTERVAL_MINUTES = 5

# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def time_str(dt: datetime) -> str:
    return dt.strftime("%H:%M")


def parse_time(t: str) -> datetime:
    return datetime.strptime(t, "%H:%M")


def add_minutes(t: str, minutes: int) -> str:
    return time_str(parse_time(t) + timedelta(minutes=minutes))


def generate_times() -> list[str]:
    """Generate all time slots from START_TIME to END_TIME inclusive."""
    times = []
    t = START_TIME
    while t <= END_TIME:
        times.append(t)
        t = add_minutes(t, INTERVAL_MINUTES)
    return times


# ---------------------------------------------------------------------------
# State classes
# ---------------------------------------------------------------------------

class PersonState:
    def __init__(self, name: str, home: str):
        self.name = name
        self.home = home
        self.room = home
        self.activity = "works at their desk"
        self.activity_end_time: str | None = None
        self.present = True

    def move_to(self, room: str, current_time: str | None = None):
        self.room = room
        room_type = ROOMS[room]["type"]
        self.activity = random.choice(ACTIVITIES.get(room_type, ACTIVITIES["office"]))
        self._set_activity_duration(current_time)

    def _set_activity_duration(self, current_time: str | None = None):
        duration_min, duration_max = ACTIVITY_DURATIONS.get(
            self.activity, (10, 30)
        )
        duration = random.randint(duration_min, duration_max)
        # Round up to nearest 5-minute interval for cleaner boundaries
        duration = max(INTERVAL_MINUTES, ((duration + INTERVAL_MINUTES - 1) // INTERVAL_MINUTES) * INTERVAL_MINUTES)
        if current_time is not None:
            self.activity_end_time = add_minutes(current_time, duration)
        else:
            self.activity_end_time = None

    def maybe_update_activity(self, current_time: str):
        """If the current activity has expired, pick a new one in the same room."""
        if self.activity_end_time is None or current_time >= self.activity_end_time:
            room_type = ROOMS[self.room]["type"]
            self.activity = random.choice(ACTIVITIES.get(room_type, ACTIVITIES["office"]))
            self._set_activity_duration(current_time)


class ObjectState:
    def __init__(self, name: str, owner: str, obj_type: str):
        self.name = name
        self.owner = owner
        self.obj_type = obj_type
        # Initial location: owner's home office
        self.room = PEOPLE[owner]["home"]
        self.carried = False

    def update_location(self, owner_room: str):
        """Object follows its owner with high probability."""
        if self.obj_type == "personal":
            # 90% chance object is with owner, 10% left behind
            if random.random() < 0.9:
                self.room = owner_room
                self.carried = True
            # else: stays in previous room


def describe_scene(room: str, people: list[str], person_activities: dict[str, str], objects: list[str]) -> str:
    """Generate a natural-language scene caption."""
    parts = []

    # Describe people
    if not people:
        parts.append("No one is here")
    else:
        for p in sorted(people):
            parts.append(f"{p} {person_activities.get(p, 'is here')}")

    # Describe objects (only non-fixtures)
    fixture_names = set()
    for f in ROOMS[room].get("fixtures", []):
        fixture_names.add(f)
        fixture_names.add(f.replace("_", " "))

    movable_objects = [o.replace("_", " ") for o in objects if o not in fixture_names and o.replace("_", " ") not in fixture_names]

    if movable_objects:
        if len(movable_objects) == 1:
            parts.append(f"a {movable_objects[0]} is on the table")
        elif len(movable_objects) == 2:
            parts.append(f"a {movable_objects[0]} and a {movable_objects[1]} are on the table")
        else:
            obj_str = ", ".join(f"a {o}" for o in movable_objects[:-1])
            parts.append(f"{obj_str}, and a {movable_objects[-1]} are on the table")

    return ". ".join(parts) + "."


# ---------------------------------------------------------------------------
# Schedule / event generator
# ---------------------------------------------------------------------------

def get_day_schedule(day_idx: int, times: list[str]) -> dict[str, list[dict]]:
    """Return scheduled events for the day.
    
    Each event: {"start": time_str, "end": time_str, "room": str, "people": list, "description": str}
    """
    events = []
    all_people = list(PEOPLE.keys())

    if day_idx == 0:
        # Day 1 — Normal
        events.append({"start": "09:05", "end": "09:15", "room": "meeting room", "people": all_people, "description": "morning stand-up"})
        events.append({"start": "12:00", "end": "12:30", "room": "kitchen", "people": random.sample(all_people, 5), "description": "lunch"})
        events.append({"start": "13:00", "end": "13:30", "room": "meeting room", "people": random.sample(all_people, 4), "description": "project review"})
    elif day_idx == 1:
        # Day 2 — Demo Day
        events.append({"start": "09:05", "end": "09:15", "room": "meeting room", "people": all_people, "description": "all-hands demo"})
        events.append({"start": "11:00", "end": "11:30", "room": "meeting room", "people": random.sample(all_people, 6), "description": "client presentation"})
        events.append({"start": "12:00", "end": "12:45", "room": "kitchen", "people": random.sample(all_people, 6), "description": "team lunch"})
    else:
        # Day 3 — Crunch
        events.append({"start": "09:05", "end": "09:10", "room": "meeting room", "people": all_people, "description": "daily sync"})
        events.append({"start": "10:00", "end": "10:30", "room": "meeting room", "people": random.sample(all_people, 3), "description": "architecture discussion"})
        events.append({"start": "12:00", "end": "12:20", "room": "kitchen", "people": random.sample(all_people, 3), "description": "quick lunch"})

    return events


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def simulate_day(date_str: str, day_idx: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Simulate one day and return scene log, person locations, object locations."""
    times = generate_times()
    events = get_day_schedule(day_idx, times)

    # Initialize people and objects
    people_states = {name: PersonState(name, info["home"]) for name, info in PEOPLE.items()}
    object_states = {name: ObjectState(name, info["owner"], info["type"]) for name, info in OBJECTS.items()}

    # Precompute event coverage: for each time slot, which scheduled event is active?
    event_coverage = defaultdict(list)
    for ev in events:
        ev_start_idx = times.index(ev["start"])
        ev_end_idx = times.index(ev["end"])
        for idx in range(ev_start_idx, ev_end_idx):
            event_coverage[times[idx]].append(ev)

    scene_log = []
    person_locations = []
    object_locations = []

    for t in times:
        # --- Apply scheduled events ---
        active_events = event_coverage.get(t, [])
        for ev in active_events:
            for p in ev["people"]:
                people_states[p].move_to(ev["room"], t)

        # --- Random micro-movements ---
        for name, ps in people_states.items():
            if not ps.present:
                continue

            # If in scheduled event, stay
            in_scheduled = any(name in ev["people"] and ev["start"] <= t < ev["end"] for ev in events)
            if in_scheduled:
                continue

            # Random movement probability
            # Storage room and office 1 are made less active
            is_office_1_person = ps.home == "office 1"
            r = random.random()
            if r < 0.05:
                # Go to kitchen
                ps.move_to("kitchen", t)
            elif r < 0.08:
                # Go to meeting room
                ps.move_to("meeting room", t)
            elif r < 0.085:
                # Go to storage (reduced from 0.10 to 0.085 → ~0.5% chance)
                ps.move_to("storage", t)
            elif r < (0.20 if is_office_1_person else 0.30):
                # Return to home office (office 1 people less likely to return)
                ps.move_to(ps.home, t)
            # else: stay where they are

        # --- Update activities (if not in a scheduled event) ---
        for name, ps in people_states.items():
            in_scheduled = any(name in ev["people"] and ev["start"] <= t < ev["end"] for ev in events)
            if not in_scheduled:
                ps.maybe_update_activity(t)

        # --- Update object locations ---
        for obj_name, obj in object_states.items():
            owner_room = people_states[obj.owner].room
            obj.update_location(owner_room)

        # --- Record room states ---
        for room in ROOMS:
            people_in_room = [p.name for p in people_states.values() if p.room == room]
            person_activities = {p.name: p.activity for p in people_states.values() if p.room == room}

            # Base objects from object states
            objects_in_room = [o.name for o in object_states.values() if o.room == room]

            # Add activity-implied objects for consistency
            implied = set()
            for p_name in people_in_room:
                activity = person_activities.get(p_name, "")
                implied.update(ACTIVITY_IMPLICIT_OBJECTS.get(activity, []))
            objects_in_room = sorted(set(objects_in_room) | implied)

            scene_entry = {
                "date": date_str,
                "time": t,
                "room": room,
                "scene": describe_scene(room, people_in_room, person_activities, objects_in_room),
                "people_present": sorted(people_in_room),
                "objects_present": objects_in_room,
                "person_activities": person_activities,
            }
            scene_log.append(scene_entry)

        # --- Record person locations ---
        for name, ps in people_states.items():
            person_locations.append({
                "time": t,
                "person": name,
                "room": ps.room,
                "activity": ps.activity,
            })

        # --- Record object locations ---
        for name, os in object_states.items():
            object_locations.append({
                "time": t,
                "object": name,
                "room": os.room,
                "with_owner": os.carried,
            })

    return scene_log, person_locations, object_locations


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "datasets" / "default"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scene_logs = []
    all_person_locations = []
    all_object_locations = []

    for day_idx, date_str in enumerate(DAYS):
        print(f"Simulating {date_str} (day {day_idx + 1})...")
        scene_log, person_locs, object_locs = simulate_day(date_str, day_idx)
        all_scene_logs.extend(scene_log)
        all_person_locations.extend(person_locs)
        all_object_locations.extend(object_locs)

        # Save daily files
        (output_dir / f"scene_log_{date_str}.json").write_text(
            json.dumps(scene_log, indent=2), encoding="utf-8"
        )
        (output_dir / f"person_locations_{date_str}.json").write_text(
            json.dumps(person_locs, indent=2), encoding="utf-8"
        )
        (output_dir / f"object_locations_{date_str}.json").write_text(
            json.dumps(object_locs, indent=2), encoding="utf-8"
        )
        print(f"  {len(scene_log)} scene entries, {len(person_locs)} person locations, {len(object_locs)} object locations")

    # Save combined
    (output_dir / "scene_log_all.json").write_text(
        json.dumps(all_scene_logs, indent=2), encoding="utf-8"
    )
    (output_dir / "person_locations_all.json").write_text(
        json.dumps(all_person_locations, indent=2), encoding="utf-8"
    )
    (output_dir / "object_locations_all.json").write_text(
        json.dumps(all_object_locations, indent=2), encoding="utf-8"
    )

    print(f"\nDone. Output in {output_dir}")
    print(f"Total: {len(all_scene_logs)} scene entries across {len(DAYS)} days")


if __name__ == "__main__":
    main()
