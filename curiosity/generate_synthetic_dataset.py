#!/usr/bin/env python3
"""Generate synthetic 3-day office dataset for tool-calling navigation agent testing.

Outputs:
    - office_time_log_2026-05-2{6,7,8}.json
    - interactions_2026-05-2{6,7,8}.json
    - person_locations_2026-05-2{6,7,8}.json
"""

import json
import random
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Office Map
# ---------------------------------------------------------------------------
ROOMS = {
    "Dock Hall": {"type": "open", "capacity": 12, "distance": {}},
    "Pod A": {"type": "quiet", "capacity": 4, "distance": {}},
    "Pod B": {"type": "quiet", "capacity": 4, "distance": {}},
    "The Lab": {"type": "lab", "capacity": 4, "distance": {}},
    "The Kitchen": {"type": "social", "capacity": 10, "distance": {}},
    "The Yard": {"type": "social", "capacity": 8, "distance": {}},
    "The Stage": {"type": "presentation", "capacity": 30, "distance": {}},
    "The Cellar": {"type": "utility", "capacity": 4, "distance": {}},
    "The Booth": {"type": "meeting", "capacity": 4, "distance": {}},
    "The Deck": {"type": "outdoor", "capacity": 8, "distance": {}},
}

# Distance matrix (symmetric)
_distances = {
    ("Dock Hall", "Pod A"): 2,
    ("Dock Hall", "Pod B"): 2,
    ("Dock Hall", "The Lab"): 3,
    ("Dock Hall", "The Kitchen"): 1,
    ("Dock Hall", "The Yard"): 1,
    ("Dock Hall", "The Stage"): 2,
    ("Dock Hall", "The Cellar"): 3,
    ("Dock Hall", "The Booth"): 1,
    ("Dock Hall", "The Deck"): 2,
    ("Pod A", "Pod B"): 2,
    ("Pod A", "The Lab"): 3,
    ("Pod A", "The Kitchen"): 2,
    ("Pod A", "The Yard"): 2,
    ("Pod A", "The Stage"): 2,
    ("Pod A", "The Cellar"): 3,
    ("Pod A", "The Booth"): 2,
    ("Pod A", "The Deck"): 2,
    ("Pod B", "The Lab"): 3,
    ("Pod B", "The Kitchen"): 2,
    ("Pod B", "The Yard"): 2,
    ("Pod B", "The Stage"): 2,
    ("Pod B", "The Cellar"): 3,
    ("Pod B", "The Booth"): 2,
    ("Pod B", "The Deck"): 2,
    ("The Lab", "The Kitchen"): 3,
    ("The Lab", "The Yard"): 3,
    ("The Lab", "The Stage"): 3,
    ("The Lab", "The Cellar"): 2,
    ("The Lab", "The Booth"): 3,
    ("The Lab", "The Deck"): 3,
    ("The Kitchen", "The Yard"): 1,
    ("The Kitchen", "The Stage"): 2,
    ("The Kitchen", "The Cellar"): 3,
    ("The Kitchen", "The Booth"): 1,
    ("The Kitchen", "The Deck"): 2,
    ("The Yard", "The Stage"): 2,
    ("The Yard", "The Cellar"): 3,
    ("The Yard", "The Booth"): 1,
    ("The Yard", "The Deck"): 2,
    ("The Stage", "The Cellar"): 3,
    ("The Stage", "The Booth"): 2,
    ("The Stage", "The Deck"): 1,
    ("The Cellar", "The Booth"): 3,
    ("The Cellar", "The Deck"): 3,
    ("The Booth", "The Deck"): 2,
}

for (r1, r2), d in _distances.items():
    ROOMS[r1]["distance"][r2] = d
    ROOMS[r2]["distance"][r1] = d

for r in ROOMS:
    ROOMS[r]["distance"][r] = 0

ALL_ROOM_NAMES = list(ROOMS.keys())

# ---------------------------------------------------------------------------
# People
# ---------------------------------------------------------------------------
PEOPLE = {
    "Alex": {"team": "Dock Hall", "role": "dev"},
    "Blake": {"team": "Dock Hall", "role": "dev"},
    "Casey": {"team": "Dock Hall", "role": "dev"},
    "Dana": {"team": "Dock Hall", "role": "dev"},
    "Evan": {"team": "Dock Hall", "role": "dev"},
    "Frank": {"team": "Dock Hall", "role": "dev"},
    "Grace": {"team": "Pod A", "role": "writer"},
    "Heidi": {"team": "Pod A", "role": "writer"},
    "Ivan": {"team": "Pod B", "role": "analyst"},
    "Judy": {"team": "Pod B", "role": "analyst"},
    "Karl": {"team": "The Lab", "role": "hardware"},
    "Luna": {"team": "The Lab", "role": "hardware"},
    "Morgan": {"team": "Dock Hall", "role": "pm"},
    "Nina": {"team": "Dock Hall", "role": "designer"},
    "Oscar": {"team": "Dock Hall", "role": "ops"},
    "Paula": {"team": "Dock Hall", "role": "intern"},
}

ALL_PEOPLE = list(PEOPLE.keys())

# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------
def time_str(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def parse_time(t: str) -> datetime:
    return datetime.strptime(t, "%H:%M")

def add_minutes(t: str, minutes: int) -> str:
    return time_str(parse_time(t) + timedelta(minutes=minutes))

# ---------------------------------------------------------------------------
# Scene description builders
# ---------------------------------------------------------------------------
def describe_scene(room: str, people: list[str], activities: dict[str, str]) -> str:
    """Generate a plain description of a room scene."""
    if not people:
        return "No one is here."
    
    parts = []
    for p in sorted(people):
        act = activities.get(p, "is here")
        parts.append(f"{p} {act}")
    
    if len(parts) == 1:
        return parts[0] + "."
    elif len(parts) == 2:
        return f"{parts[0]}. {parts[1]}."
    else:
        return ". ".join(parts) + "."

# ---------------------------------------------------------------------------
# Activity templates per room
# ---------------------------------------------------------------------------
ACTIVITY_TEMPLATES = {
    "Dock Hall": [
        "works at their desk", "types on their laptop", "reads a document",
        "discusses with a colleague", "stands up and stretches", "writes on the whiteboard",
        "debugs code on the screen", "joins a video call with headphones",
    ],
    "Pod A": [
        "works quietly at their desk", "reads a book", "writes notes",
        "listens to music with headphones", "focuses on a document",
    ],
    "Pod B": [
        "works quietly at their desk", "analyzes data on the screen", "writes a report",
        "listens to music with headphones", "focuses on a spreadsheet",
    ],
    "The Lab": [
        "soldiers a circuit board", "checks a 3D printer", "tests a prototype",
        "organizes tools on the bench", "examines a device with a magnifier",
        "writes measurements in a notebook",
    ],
    "The Kitchen": [
        "makes coffee at the machine", "prepares a sandwich", "washes a mug",
        "grabs a snack from the fridge", "chats near the counter", "fills a water bottle",
        "cuts vegetables on the board", "waits by the microwave",
    ],
    "The Yard": [
        "plays foosball", "sits on the couch", "chats with a colleague",
        "reads a magazine", "checks their phone", "laughs at a joke",
        "drinks coffee on the sofa",
    ],
    "The Stage": [
        "sets up a presentation", "gives a talk at the podium", "points at the screen",
        "answers questions from the audience", "adjusts the projector",
        "writes on the flip chart", "waits for the session to start",
    ],
    "The Cellar": [
        "checks server lights", "replaces a cable", "reads a monitor display",
        "organizes boxes on the shelf", "sweeps the floor", "inspects equipment",
    ],
    "The Booth": [
        "sits at the meeting table", "talks with a colleague", "shares a laptop screen",
        "writes on the whiteboard", "listens attentively", "takes notes",
    ],
    "The Deck": [
        "enjoys the sun on a chair", "has a call with headphones", "chats with a colleague",
        "looks at the view", "reads a tablet", "drinks a cold beverage",
    ],
}

ARRIVALS = [
    "comes into the door", "walks in", "enters the room",
]

DEPARTURES = [
    "walks out", "leaves the room", "heads out",
]

# ---------------------------------------------------------------------------
# State machine for each person
# ---------------------------------------------------------------------------
class PersonState:
    def __init__(self, name):
        self.name = name
        self.room = None  # None = not in office
        self.activity = None
        self.arrival_time = None
        self.departure_time = None
        self.travel_end = None  # time string when travel finishes
        self.travel_destination = None
    
    def is_present(self, t: str) -> bool:
        return self.room is not None and (self.arrival_time is None or t >= self.arrival_time) and (self.departure_time is None or t < self.departure_time) and self.travel_end is None
    
    def is_traveling(self, t: str) -> bool:
        return self.travel_end is not None and t < self.travel_end

# ---------------------------------------------------------------------------
# Day schedule definitions
# ---------------------------------------------------------------------------
def get_day_schedule(day: int):
    """Return a list of scheduled events for the day.
    
    Each event: (start_time, end_time, room, people, activity_type, description)
    """
    schedule = []
    
    if day == 1:
        # Day 1 — Normal Day (2026-05-26)
        # Arrivals
        for i, p in enumerate(ALL_PEOPLE):
            t = add_minutes("08:00", i % 40)
            home_room = PEOPLE[p]["team"] if PEOPLE[p]["team"] != "mobile" else "Dock Hall"
            schedule.append((t, t, home_room, [p], "arrival", f"{p} arrives at work"))
        
        # Stand-up 09:30
        schedule.append(("09:30", "09:45", "Dock Hall", ALL_PEOPLE, "meeting", "Morning stand-up"))
        
        # Deep work blocks
        for p in ALL_PEOPLE:
            if PEOPLE[p]["team"] != "mobile":
                schedule.append(("10:00", "12:00", PEOPLE[p]["team"], [p], "work", f"{p} deep work"))
            else:
                schedule.append(("10:00", "12:00", "Dock Hall", [p], "work", f"{p} works"))
        
        # Lunch 12:00
        lunch_groups = [
            (["Alex", "Blake", "Casey", "Morgan"], "The Kitchen"),
            (["Dana", "Evan", "Frank", "Nina"], "The Yard"),
            (["Grace", "Heidi", "Ivan", "Judy"], "The Kitchen"),
            (["Karl", "Luna", "Oscar", "Paula"], "The Yard"),
        ]
        for people, room in lunch_groups:
            schedule.append(("12:00", "12:45", room, people, "lunch", "Lunch"))
        
        # Post-lunch yard time
        schedule.append(("12:45", "13:15", "The Yard", ["Alex", "Blake", "Casey", "Dana"], "social", "Post-lunch chat"))
        
        # Afternoon meetings
        schedule.append(("14:00", "14:30", "The Booth", ["Morgan", "Nina", "Oscar", "Paula"], "meeting", "PM sync"))
        schedule.append(("14:30", "15:00", "The Deck", ["Grace", "Heidi", "Ivan", "Judy"], "meeting", "Writer sync"))
        schedule.append(("15:00", "15:30", "The Stage", ["Alex", "Blake", "Casey", "Dana", "Evan", "Frank"], "meeting", "Dev retro"))
        
        # Departures
        for i, p in enumerate(ALL_PEOPLE):
            t = add_minutes("16:00", i % 40)
            schedule.append((t, t, PEOPLE[p]["team"], [p], "departure", f"{p} leaves"))
    
    elif day == 2:
        # Day 2 — Demo Day (2026-05-27)
        # Early arrivals for setup
        for i, p in enumerate(["Morgan", "Nina", "Alex", "Karl"]):
            t = add_minutes("07:30", i * 5)
            schedule.append((t, t, "The Stage", [p], "arrival", f"{p} arrives early"))
        
        # Normal arrivals
        for i, p in enumerate([p for p in ALL_PEOPLE if p not in ["Morgan", "Nina", "Alex", "Karl"]]):
            t = add_minutes("08:00", i * 3)
            schedule.append((t, t, "Dock Hall", [p], "arrival", f"{p} arrives"))
        
        # Demo workshop morning
        schedule.append(("09:00", "10:30", "The Stage", ALL_PEOPLE, "presentation", "Demo presentations"))
        schedule.append(("10:30", "10:45", "The Kitchen", ALL_PEOPLE, "break", "Coffee break"))
        schedule.append(("10:45", "12:00", "The Stage", ALL_PEOPLE, "presentation", "Demo continues"))
        
        # Catered lunch
        schedule.append(("12:00", "13:30", "The Kitchen", ALL_PEOPLE[:8], "lunch", "Catered lunch"))
        schedule.append(("12:00", "13:30", "The Yard", ALL_PEOPLE[8:], "lunch", "Catered lunch"))
        schedule.append(("12:30", "13:30", "The Deck", ["Morgan", "Nina", "Oscar", "Paula"], "lunch", "PM lunch"))
        
        # Client 1:1s
        schedule.append(("13:30", "14:00", "The Booth", ["Morgan", "Alex"], "meeting", "Client 1:1"))
        schedule.append(("14:00", "14:30", "The Deck", ["Nina", "Blake"], "meeting", "Client design review"))
        schedule.append(("14:30", "15:00", "The Booth", ["Oscar", "Casey"], "meeting", "Client tech sync"))
        
        # Team retrospective
        schedule.append(("15:00", "16:30", "The Stage", ALL_PEOPLE, "meeting", "Team retrospective"))
        
        # Cleanup
        schedule.append(("16:30", "17:00", "The Stage", ["Karl", "Luna", "Oscar"], "work", "Cleanup"))
        
        # Departures
        for i, p in enumerate(ALL_PEOPLE):
            t = add_minutes("16:30", i % 50)
            schedule.append((t, t, "Dock Hall", [p], "departure", f"{p} leaves"))
    
    elif day == 3:
        # Day 3 — Crunch + Maintenance (2026-05-28)
        # Late arrivals
        for i, p in enumerate(ALL_PEOPLE):
            t = add_minutes("08:30", i % 20)
            schedule.append((t, t, PEOPLE[p]["team"], [p], "arrival", f"{p} arrives late"))
        
        # Intense work all day in Dock Hall + Pods
        for p in ALL_PEOPLE:
            if PEOPLE[p]["team"] == "The Lab":
                schedule.append(("09:00", "12:00", "The Lab", [p], "work", f"{p} lab work"))
            elif PEOPLE[p]["team"] == "mobile":
                schedule.append(("09:00", "12:00", "Dock Hall", [p], "work", f"{p} works"))
            else:
                schedule.append(("09:00", "12:00", PEOPLE[p]["team"], [p], "work", f"{p} crunch work"))
        
        # Quick lunch at desks (brief kitchen visits)
        schedule.append(("12:00", "12:15", "The Kitchen", ["Alex", "Blake", "Casey"], "lunch", "Quick lunch"))
        schedule.append(("12:15", "12:30", "The Kitchen", ["Dana", "Evan", "Frank"], "lunch", "Quick lunch"))
        schedule.append(("12:00", "12:30", "Dock Hall", [p for p in ALL_PEOPLE if p not in ["Alex", "Blake", "Casey", "Dana", "Evan", "Frank"]], "work", "Work through lunch"))
        
        # Cellar maintenance
        schedule.append(("12:30", "14:00", "The Cellar", ["Karl", "Oscar"], "maintenance", "Server maintenance"))
        
        # Afternoon crunch
        for p in ALL_PEOPLE:
            if p in ["Karl", "Oscar"]:
                schedule.append(("14:00", "18:00", "The Lab", [p], "work", f"{p} continues work"))
            elif PEOPLE[p]["team"] == "mobile":
                schedule.append(("14:00", "18:00", "Dock Hall", [p], "work", f"{p} afternoon crunch"))
            else:
                schedule.append(("14:00", "18:00", PEOPLE[p]["team"], [p], "work", f"{p} afternoon crunch"))
        
        # Late departures
        for i, p in enumerate(ALL_PEOPLE):
            t = add_minutes("18:00", i % 40)
            schedule.append((t, t, PEOPLE[p]["team"], [p], "departure", f"{p} leaves late"))
    
    return schedule


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class DaySimulator:
    def __init__(self, date: str, day_num: int):
        self.date = date
        self.day_num = day_num
        self.people = {p: PersonState(p) for p in ALL_PEOPLE}
        self.time_log = []  # List of dicts per room per minute
        self.interactions = []
        self.person_locations = []
        self.minute_index = defaultdict(lambda: defaultdict(list))  # time -> room -> [people]
        self.prev_scenes = {}  # room -> scene text
        self.interaction_counter = 0
    
    def interaction_id(self, t: str, room: str) -> str:
        self.interaction_counter += 1
        room_short = room.lower().replace(" ", "_").replace("the_", "")
        return f"int_{self.date.replace('-', '')}_{t.replace(':', '')}_{room_short}_{self.interaction_counter:04d}"
    
    def run(self):
        schedule = get_day_schedule(self.day_num)
        
        # Convert schedule to minute-by-minute assignments
        assignments = defaultdict(lambda: defaultdict(list))  # time -> room -> [(person, activity_type)]
        for start, end, room, people, act_type, desc in schedule:
            t = start
            while t <= end:
                for p in people:
                    assignments[t][room].append((p, act_type))
                t = add_minutes(t, 1)
        
        # Generate minute-by-minute trace
        t = "08:00"
        end_t = "19:00"
        while t <= end_t:
            # Determine who is where at this minute
            room_people = defaultdict(list)
            room_activities = defaultdict(dict)
            
            for p_name, p_state in self.people.items():
                # Check scheduled assignments
                assigned_room = None
                assigned_act = None
                for room, people_acts in assignments.get(t, {}).items():
                    for pname, act_type in people_acts:
                        if pname == p_name:
                            assigned_room = room
                            assigned_act = act_type
                            break
                
                if assigned_act == "arrival":
                    if p_state.room is None:
                        p_state.room = assigned_room
                        p_state.activity = "arriving"
                        p_state.arrival_time = t
                elif assigned_act == "departure":
                    if p_state.room is not None:
                        p_state.room = None
                        p_state.activity = None
                        p_state.departure_time = t
                elif assigned_room:
                    if p_state.room != assigned_room and p_state.room is not None:
                        # Need to travel
                        dist = ROOMS[p_state.room]["distance"].get(assigned_room, 1)
                        p_state.travel_end = add_minutes(t, dist)
                        p_state.travel_destination = assigned_room
                        p_state.room = None  # traveling
                    elif p_state.room is None and p_state.travel_end is None:
                        p_state.room = assigned_room
                    
                    if p_state.room == assigned_room:
                        p_state.activity = assigned_act
                
                # Handle ongoing travel
                if p_state.travel_end is not None and t >= p_state.travel_end:
                    p_state.room = p_state.travel_destination
                    p_state.travel_end = None
                    p_state.travel_destination = None
                
                # Record if present
                if p_state.room is not None and p_state.travel_end is None:
                    room_people[p_state.room].append(p_name)
                    if p_state.activity in ["work", "deep work", "crunch work", "lab work", "focus"]:
                        act = random.choice(ACTIVITY_TEMPLATES.get(p_state.room, ["is here"]))
                    elif p_state.activity == "arrival":
                        act = random.choice(ARRIVALS)
                    elif p_state.activity == "lunch":
                        act = random.choice(["eats lunch", "prepares food", "eats a sandwich", "drinks soup"])
                    elif p_state.activity == "social":
                        act = random.choice(["chats with a colleague", "laughs", "relaxes"])
                    elif p_state.activity == "meeting":
                        act = random.choice(["sits at the table", "talks with the group", "listens attentively"])
                    elif p_state.activity == "presentation":
                        act = random.choice(["gives a talk", "points at the screen", "listens to the speaker"])
                    elif p_state.activity == "maintenance":
                        act = random.choice(["checks server lights", "replaces a cable", "reads a monitor"])
                    else:
                        act = random.choice(ACTIVITY_TEMPLATES.get(p_state.room, ["is here"]))
                    room_activities[p_state.room][p_name] = act
            
            # Generate entries for all rooms
            for room in ALL_ROOM_NAMES:
                people_here = sorted(room_people[room])
                scene = describe_scene(room, people_here, room_activities.get(room, {}))
                
                # Determine if event=1 (scene changed vs previous minute)
                prev_scene = self.prev_scenes.get(room, "")
                event = 1 if scene != prev_scene else 0
                
                int_id = self.interaction_id(t, room) if event == 1 else None
                
                entry = {
                    "date": self.date,
                    "time": t,
                    "room": room,
                    "scene": scene,
                    "event": event,
                    "people_present": people_here,
                }
                if int_id:
                    entry["interaction_id"] = int_id
                
                self.time_log.append(entry)
                self.prev_scenes[room] = scene
                
                # Record interactions for event=1
                if event == 1 and people_here:
                    self.interactions.append({
                        "interaction_id": int_id,
                        "local_created_at": f"{self.date}T{t}:00+02:00",
                        "room_name": room,
                        "caption": scene,
                        "action": room_activities.get(room, {}).get(people_here[0], "presence") if people_here else "empty",
                        "participants": people_here,
                        "related_objects": [],
                        "scene_text": scene,
                    })
                
                # Record person locations
                for p in people_here:
                    self.person_locations.append({
                        "time": t,
                        "person": p,
                        "room": room,
                        "activity": room_activities.get(room, {}).get(p, "present"),
                        "interaction_id": int_id if int_id else "",
                    })
            
            t = add_minutes(t, 1)


def main():
    output_dir = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "synthetic_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    days = [
        ("2026-05-26", 1),
        ("2026-05-27", 2),
        ("2026-05-28", 3),
    ]
    
    for date_str, day_num in days:
        print(f"Generating Day {day_num} ({date_str})...")
        sim = DaySimulator(date_str, day_num)
        sim.run()
        
        # Write time log
        tl_path = output_dir / f"office_time_log_{date_str}.json"
        with open(tl_path, "w", encoding="utf-8") as f:
            json.dump(sim.time_log, f, indent=2, ensure_ascii=False)
        print(f"  Time log: {len(sim.time_log)} entries -> {tl_path}")
        
        # Write interactions
        int_path = output_dir / f"interactions_{date_str}.json"
        with open(int_path, "w", encoding="utf-8") as f:
            json.dump(sim.interactions, f, indent=2, ensure_ascii=False)
        print(f"  Interactions: {len(sim.interactions)} entries -> {int_path}")
        
        # Write person locations
        pl_path = output_dir / f"person_locations_{date_str}.json"
        with open(pl_path, "w", encoding="utf-8") as f:
            json.dump(sim.person_locations, f, indent=2, ensure_ascii=False)
        print(f"  Person locations: {len(sim.person_locations)} entries -> {pl_path}")
        
        # Stats
        event_count = sum(1 for e in sim.time_log if e["event"] == 1)
        print(f"  Event minutes: {event_count}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
