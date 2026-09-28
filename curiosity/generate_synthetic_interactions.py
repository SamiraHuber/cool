#!/usr/bin/env python3
"""Generate structured synthetic interactions from existing scene logs.

Reads scene_log_*.json files and produces synthetic_interactions_*.json
with person-object and person-person interactions using the 10+ verb ontology.

Verbs: sitting, drinking, cutting, holding, preparing, stands next to,
       carrying, talking to, puts, uses, places onto, picks up

Output format per interaction:
    {
        "date": "2026-05-26",
        "time": "09:00",
        "room": "kitchen",
        "subject": "Dana",
        "target": "coffee_machine",
        "action": "uses",
        "caption": "Dana uses the coffee_machine to make coffee",
        "verb_category": "active",   # "active" or "passive"
        "ownership_bearing": false   # true if strongly suggests ownership
    }
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Configuration (mirrors generate_scene_change_dataset.py)
# ---------------------------------------------------------------------------

ROOMS = {
    "kitchen":      {"type": "social",   "fixtures": ["coffee_machine", "fridge", "microwave", "sink", "counter"]},
    "storage":      {"type": "utility",  "fixtures": ["shelf", "toolbox", "box"]},
    "office 1":     {"type": "office",   "fixtures": ["desk", "chair", "laptop_stand"]},
    "office 2":     {"type": "office",   "fixtures": ["desk", "chair", "laptop_stand"]},
    "office 3":     {"type": "office",   "fixtures": ["desk", "chair", "laptop_stand"]},
    "meeting room": {"type": "meeting",  "fixtures": ["projector", "whiteboard", "conference_table", "chair"]},
}

PEOPLE = {
    "Alex":  {"home": "office 1"},
    "Blake": {"home": "office 1"},
    "Casey": {"home": "office 2"},
    "Dana":  {"home": "office 2"},
    "Evan":  {"home": "office 3"},
    "Frank": {"home": "office 3"},
    "Grace": {"home": "office 1"},
    "Heidi": {"home": "office 2"},
}

OBJECTS = {
    "laptop":       {"owner": "Alex"},
    "red_cup":      {"owner": "Alex"},
    "backpack":     {"owner": "Blake"},
    "water_bottle": {"owner": "Blake"},
    "notebook":     {"owner": "Casey"},
    "pen":          {"owner": "Casey"},
    "headphones":   {"owner": "Dana"},
    "coffee_mug":   {"owner": "Evan"},
    "tablet":       {"owner": "Frank"},
    "jacket":       {"owner": "Frank"},
    "yoga_mat":     {"owner": "Grace"},
    "camera":       {"owner": "Heidi"},
}

# Verb ontology with categories
# ACTIVE = ongoing/changing action → robot should stay
# PASSIVE = static state → robot may move on
VERB_ONTOLOGY = {
    "sitting":          {"category": "passive", "ownership_bearing": False},
    "drinking":         {"category": "active",  "ownership_bearing": True},
    "cutting":          {"category": "active",  "ownership_bearing": False},
    "holding":          {"category": "active",  "ownership_bearing": True},
    "preparing":        {"category": "active",  "ownership_bearing": False},
    "stands next to":   {"category": "passive", "ownership_bearing": False},
    "carrying":         {"category": "active",  "ownership_bearing": True},
    "talking to":       {"category": "active",  "ownership_bearing": False},
    "puts":             {"category": "active",  "ownership_bearing": True},
    "uses":             {"category": "passive", "ownership_bearing": True},
    "places onto":      {"category": "active",  "ownership_bearing": True},
    "picks up":         {"category": "active",  "ownership_bearing": True},
}

# Activity → interaction mappings
# Each activity maps to one or more (action, target_template, caption_template) tuples
ACTIVITY_INTERACTIONS = {
    "works at their desk": [
        ("uses", "{obj}", "{subject} uses their {obj} while working at the desk"),
        ("sitting", "desk", "{subject} is sitting at their desk"),
    ],
    "types on their laptop": [
        ("uses", "laptop", "{subject} uses their laptop, typing quickly"),
        ("sitting", "desk", "{subject} is sitting at their desk"),
    ],
    "reads a document": [
        ("holding", "document", "{subject} is holding and reading a document"),
        ("sitting", "desk", "{subject} is sitting at their desk"),
    ],
    "joins a video call": [
        ("uses", "laptop", "{subject} uses their laptop for a video call"),
        ("sitting", "desk", "{subject} is sitting at their desk"),
    ],
    "writes notes": [
        ("uses", "notebook", "{subject} uses their notebook to write notes"),
        ("holding", "pen", "{subject} is holding a pen"),
    ],
    "checks their phone": [
        ("holding", "phone", "{subject} is holding their phone and checking messages"),
    ],
    "sits at the table": [
        ("sitting", "conference_table", "{subject} is sitting at the conference table"),
    ],
    "talks with a colleague": [
        ("talking to", "{colleague}", "{subject} is talking to {colleague}"),
        ("stands next to", "conference_table", "{subject} stands next to the conference table"),
    ],
    "points at the screen": [
        ("stands next to", "projector", "{subject} stands next to the projector screen"),
    ],
    "writes on the whiteboard": [
        ("uses", "whiteboard", "{subject} uses the whiteboard to write notes"),
        ("stands next to", "whiteboard", "{subject} stands next to the whiteboard"),
    ],
    "listens attentively": [
        ("sitting", "conference_table", "{subject} is sitting at the conference table, listening"),
    ],
    "takes notes": [
        ("uses", "notebook", "{subject} uses their notebook to take notes"),
        ("holding", "pen", "{subject} is holding a pen"),
    ],
    "makes coffee": [
        ("preparing", "coffee", "{subject} is preparing coffee"),
        ("uses", "coffee_machine", "{subject} uses the coffee_machine"),
    ],
    "prepares a snack": [
        ("preparing", "snack", "{subject} is preparing a snack"),
        ("uses", "fridge", "{subject} uses the fridge to get ingredients"),
    ],
    "washes a mug": [
        ("holding", "mug", "{subject} is holding a mug and washing it"),
        ("uses", "sink", "{subject} uses the sink"),
    ],
    "chats near the counter": [
        ("talking to", "{colleague}", "{subject} is talking to {colleague} near the counter"),
        ("stands next to", "counter", "{subject} stands next to the counter"),
    ],
    "fills a water bottle": [
        ("holding", "water_bottle", "{subject} is holding their water_bottle and filling it"),
        ("uses", "sink", "{subject} uses the sink"),
    ],
    "waits by the microwave": [
        ("stands next to", "microwave", "{subject} stands next to the microwave, waiting"),
    ],
    "organizes boxes": [
        ("puts", "box", "{subject} puts a box on the shelf"),
        ("places onto", "shelf", "{subject} places a box onto the shelf"),
    ],
    "looks for a tool": [
        ("holding", "toolbox", "{subject} is holding the toolbox, looking for a tool"),
    ],
    "checks inventory": [
        ("holding", "clipboard", "{subject} is holding a clipboard and checking inventory"),
    ],
    "carries a box": [
        ("carrying", "box", "{subject} is carrying a box"),
    ],
    "sweeps the floor": [
        ("holding", "broom", "{subject} is holding a broom and sweeping the floor"),
    ],
}


def _pick_personal_object(person: str, objects_present: list[str]) -> str | None:
    """Return a personal object belonging to person that is present, or None."""
    personal = [o for o, info in OBJECTS.items() if info["owner"] == person and o in objects_present]
    if personal:
        return random.choice(personal)
    return None


def _pick_colleague(person: str, people_present: list[str]) -> str | None:
    """Return another person present in the room, or None."""
    others = [p for p in people_present if p != person]
    if others:
        return random.choice(others)
    return None


def generate_interactions_for_scene(scene_entry: dict) -> list[dict]:
    """Generate structured interactions for a single scene entry."""
    room = scene_entry["room"]
    people_present = scene_entry.get("people_present", [])
    objects_present = scene_entry.get("objects_present", [])
    activities = scene_entry.get("person_activities", {})
    fixtures = ROOMS.get(room, {}).get("fixtures", [])
    date = scene_entry["date"]
    time = scene_entry["time"]

    interactions = []

    for person, activity in activities.items():
        mappings = ACTIVITY_INTERACTIONS.get(activity, [])
        if not mappings:
            continue

        colleague = _pick_colleague(person, people_present)
        personal_obj = _pick_personal_object(person, objects_present)

        for action_template, target_template, caption_template in mappings:
            # Skip colleague-based interactions if no colleague present
            if "{colleague}" in target_template and colleague is None:
                continue

            # Determine target
            target = target_template
            if target == "{obj}":
                if personal_obj:
                    target = personal_obj
                elif objects_present:
                    target = random.choice(objects_present)
                else:
                    target = random.choice(fixtures) if fixtures else "table"
            elif target == "{colleague}":
                target = colleague
            elif target == "laptop" and "laptop" in objects_present:
                target = "laptop"
            elif target == "notebook" and "notebook" in objects_present:
                target = "notebook"
            elif target == "water_bottle" and "water_bottle" in objects_present:
                target = "water_bottle"
            elif target == "coffee_mug" and "coffee_mug" in objects_present:
                target = "coffee_mug"
            elif target == "pen" and "pen" in objects_present:
                target = "pen"
            elif target == "backpack" and "backpack" in objects_present:
                target = "backpack"

            # If target is a personal object not present, skip or substitute
            if target in OBJECTS and target not in objects_present:
                if fixtures:
                    target = random.choice(fixtures)
                else:
                    continue

            # Build caption
            caption = caption_template.format(subject=person, colleague=colleague or "someone", obj=target)

            # Get verb metadata
            verb_info = VERB_ONTOLOGY.get(action_template, {"category": "passive", "ownership_bearing": False})

            interactions.append({
                "date": date,
                "time": time,
                "room": room,
                "subject": person,
                "target": target,
                "action": action_template,
                "caption": caption,
                "verb_category": verb_info["category"],
                "ownership_bearing": verb_info["ownership_bearing"],
            })

    # Add a few ambient/fixture-only interactions if room has people but few interactions
    if people_present and len(interactions) < len(people_present):
        for person in people_present:
            if person not in activities:
                continue
            # Add a generic fixture interaction
            if fixtures and random.random() < 0.3:
                fixture = random.choice(fixtures)
                interactions.append({
                    "date": date,
                    "time": time,
                    "room": room,
                    "subject": person,
                    "target": fixture,
                    "action": "stands next to",
                    "caption": f"{person} stands next to the {fixture}",
                    "verb_category": "passive",
                    "ownership_bearing": False,
                })

    return interactions


def process_day(scene_log_path: Path) -> list[dict]:
    """Read a daily scene log and generate all interactions."""
    with scene_log_path.open("r", encoding="utf-8") as f:
        scene_log = json.load(f)

    all_interactions = []
    for scene_entry in scene_log:
        interactions = generate_interactions_for_scene(scene_entry)
        all_interactions.extend(interactions)

    return all_interactions


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "datasets" / "default"
    days = ["2026-05-26", "2026-05-27", "2026-05-28"]

    all_interactions = []
    stats = defaultdict(lambda: defaultdict(int))

    for day in days:
        scene_log_path = data_dir / f"scene_log_{day}.json"
        if not scene_log_path.exists():
            print(f"Warning: {scene_log_path} not found, skipping")
            continue

        interactions = process_day(scene_log_path)
        all_interactions.extend(interactions)

        # Save daily file
        out_path = data_dir / f"synthetic_interactions_{day}.json"
        out_path.write_text(json.dumps(interactions, indent=2), encoding="utf-8")
        print(f"{day}: {len(interactions)} interactions")

        # Stats
        for inter in interactions:
            stats[day][inter["action"]] += 1

    # Save combined
    combined_path = data_dir / "synthetic_interactions_all.json"
    combined_path.write_text(json.dumps(all_interactions, indent=2), encoding="utf-8")
    print(f"\nTotal: {len(all_interactions)} interactions across {len(days)} days")
    print(f"Combined saved to {combined_path}")

    # Print verb distribution
    print("\n--- Verb distribution ---")
    total_by_verb = defaultdict(int)
    for day in days:
        for verb, count in sorted(stats[day].items()):
            total_by_verb[verb] += count
    for verb, count in sorted(total_by_verb.items(), key=lambda x: -x[1]):
        cat = VERB_ONTOLOGY.get(verb, {}).get("category", "?")
        print(f"  {count:4d} | {verb:15s} ({cat})")

    active = sum(c for v, c in total_by_verb.items() if VERB_ONTOLOGY.get(v, {}).get("category") == "active")
    passive = sum(c for v, c in total_by_verb.items() if VERB_ONTOLOGY.get(v, {}).get("category") == "passive")
    print(f"\nActive: {active}, Passive: {passive}")


if __name__ == "__main__":
    main()
