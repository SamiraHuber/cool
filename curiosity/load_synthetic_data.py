#!/usr/bin/env python3
"""Load synthetic 3-day office dataset into PostgreSQL for tool-calling agent testing.

Usage:
    python load_synthetic_data.py [--clean]
"""

import argparse
import json
import random
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psycopg2

DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:15432/bordsupr")
BASE_DIR = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "synthetic_data"
DAYS = ["2026-05-26", "2026-05-27", "2026-05-28"]

MAP_NAME = "the_dock"

# Room bounding boxes (x1, y1, x2, y2) — simple rectangles
ROOM_BOUNDS = {
    "Dock Hall": (0.0, 0.0, 10.0, 10.0),
    "Pod A": (-8.0, 5.0, -4.0, 9.0),
    "Pod B": (-8.0, -5.0, -4.0, -1.0),
    "The Lab": (12.0, -5.0, 16.0, -1.0),
    "The Kitchen": (5.0, 8.0, 9.0, 12.0),
    "The Yard": (-5.0, -10.0, -1.0, -6.0),
    "The Stage": (5.0, -8.0, 15.0, -4.0),
    "The Cellar": (12.0, 5.0, 16.0, 9.0),
    "The Booth": (-2.0, 8.0, 2.0, 12.0),
    "The Deck": (8.0, 8.0, 12.0, 12.0),
}

ROOM_ORDER = list(ROOM_BOUNDS.keys())


def room_center(room: str) -> tuple[float, float]:
    x1, y1, x2, y2 = ROOM_BOUNDS[room]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def get_conn():
    return psycopg2.connect(DB_URL)


def ensure_map(cur) -> int:
    """Create or get the map ID."""
    cur.execute("SELECT id FROM maps WHERE name = %s", (MAP_NAME,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO maps (name) VALUES (%s) RETURNING id", (MAP_NAME,))
    return cur.fetchone()[0]


def ensure_rooms(cur, map_id: int):
    """Create rooms with bounding boxes if they don't exist."""
    cur.execute("ALTER TABLE rooms ADD COLUMN IF NOT EXISTS map_id BIGINT")
    cur.execute("ALTER TABLE rooms ADD COLUMN IF NOT EXISTS map_name TEXT")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rooms_map_id ON rooms(map_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rooms_map_name ON rooms(map_name)")

    for room_name, (x1, y1, x2, y2) in ROOM_BOUNDS.items():
        cur.execute("SELECT id FROM rooms WHERE name = %s LIMIT 1", (room_name,))
        row = cur.fetchone()
        if row:
            cur.execute(
                """
                UPDATE rooms SET x1=%s, y1=%s, x2=%s, y2=%s, map_id=%s, map_name=%s
                WHERE id=%s
                """,
                (x1, y1, x2, y2, map_id, MAP_NAME, row[0]),
            )
        else:
            cur.execute(
                """
                INSERT INTO rooms (name, x1, y1, x2, y2, map_id, map_name)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (room_name, x1, y1, x2, y2, map_id, MAP_NAME),
            )


def clean_existing_data(cur, map_id: int):
    """Remove previously loaded synthetic data for this map."""
    # Delete interactions for this map
    cur.execute("DELETE FROM interactions WHERE map_id = %s", (map_id,))
    # Delete scenes for this map
    cur.execute("DELETE FROM scenes WHERE map_id = %s", (map_id,))
    # Delete robot visits for this map
    cur.execute("DELETE FROM robot_visits WHERE map_id = %s", (map_id,))
    print(f"Cleaned existing data for map '{MAP_NAME}' (id={map_id})")


def load_day(cur, map_id: int, date_str: str):
    """Load one day of synthetic data into the database."""
    print(f"\nLoading {date_str}...")

    time_log_path = BASE_DIR / f"office_time_log_{date_str}.json"
    interactions_path = BASE_DIR / f"interactions_{date_str}.json"

    with open(time_log_path, "r", encoding="utf-8") as f:
        time_log = json.load(f)
    with open(interactions_path, "r", encoding="utf-8") as f:
        interactions_raw = json.load(f)

    # Build interaction lookup by ID
    interactions_by_id = {i["interaction_id"]: i for i in interactions_raw}

    # --- Load scenes ---
    # One scene per event=1 entry (and maybe some event=0 for continuity)
    # We'll create a scene for every time_log entry to keep it simple,
    # but only link interactions to event=1 entries.
    scene_id_map = {}  # (time, room) -> scene_id
    scene_count = 0

    for entry in time_log:
        room = entry["room"]
        time_val = entry["time"]
        cx, cy = room_center(room)
        # Add tiny jitter so scenes in same room aren't identical
        x = cx + random.uniform(-0.3, 0.3)
        y = cy + random.uniform(-0.3, 0.3)

        ts = datetime.strptime(f"{date_str} {time_val}", "%Y-%m-%d %H:%M")
        ts = ts.replace(tzinfo=timezone(timedelta(hours=2)))  # +02:00

        cur.execute(
            """
            INSERT INTO scenes (caption, x, y, timestamp, map_id, source_frame)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (entry["scene"], x, y, ts, map_id, f"synth_{date_str}_{time_val}"),
        )
        scene_id = cur.fetchone()[0]
        scene_id_map[(time_val, room)] = scene_id
        scene_count += 1

    print(f"  Inserted {scene_count} scenes")

    # --- Load interactions ---
    interaction_count = 0
    for entry in time_log:
        if entry.get("event") != 1:
            continue
        int_id = entry.get("interaction_id")
        if not int_id:
            continue

        int_data = interactions_by_id.get(int_id)
        if not int_data:
            continue

        scene_id = scene_id_map.get((entry["time"], entry["room"]))
        if not scene_id:
            continue

        ts = datetime.strptime(f"{date_str} {entry['time']}", "%Y-%m-%d %H:%M")
        ts = ts.replace(tzinfo=timezone(timedelta(hours=2)))

        cur.execute(
            """
            INSERT INTO interactions (action, caption, created_at, scene_id, map_id, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                int_data.get("action", "presence"),
                int_data.get("caption", entry["scene"]),
                ts,
                scene_id,
                map_id,
                0.9,
            ),
        )
        interaction_count += 1

    print(f"  Inserted {interaction_count} interactions")

    # --- Load robot visits ---
    # Simulate a robot doing fixed 5-minute rotations through all rooms
    visit_count = 0
    t = "08:00"
    room_idx = 0
    while t <= "19:00":
        room = ROOM_ORDER[room_idx % len(ROOM_ORDER)]
        arrived = datetime.strptime(f"{date_str} {t}", "%Y-%m-%d %H:%M")
        arrived = arrived.replace(tzinfo=timezone(timedelta(hours=2)))

        # Stay 5 minutes
        end_t = datetime.strptime(f"{date_str} {t}", "%Y-%m-%d %H:%M") + timedelta(minutes=4)
        if end_t.hour >= 19 and end_t.minute > 0:
            end_t = datetime.strptime(f"{date_str} 19:00", "%Y-%m-%d %H:%M")
        end_t = end_t.replace(tzinfo=timezone(timedelta(hours=2)))

        # Count scenes and interactions during this visit
        cur.execute(
            """
            SELECT COUNT(*), COUNT(i.id)
            FROM scenes s
            LEFT JOIN interactions i ON i.scene_id = s.id
            WHERE s.map_id = %s
              AND s.timestamp >= %s AND s.timestamp <= %s
              AND s.x BETWEEN %s AND %s
              AND s.y BETWEEN %s AND %s
            """,
            (
                map_id,
                arrived,
                end_t,
                ROOM_BOUNDS[room][0], ROOM_BOUNDS[room][2],
                ROOM_BOUNDS[room][1], ROOM_BOUNDS[room][3],
            ),
        )
        scene_c, int_c = cur.fetchone()

        cur.execute(
            """
            INSERT INTO robot_visits (room_name, map_id, arrived_at, departed_at, scene_count, interaction_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (room, map_id, arrived, end_t, scene_c or 0, int_c or 0),
        )
        visit_count += 1

        # Advance 5 minutes
        next_dt = datetime.strptime(f"{date_str} {t}", "%Y-%m-%d %H:%M") + timedelta(minutes=5)
        t = next_dt.strftime("%H:%M")
        room_idx += 1

    print(f"  Inserted {visit_count} robot visits")


def main():
    parser = argparse.ArgumentParser(description="Load synthetic data into PostgreSQL")
    parser.add_argument("--clean", action="store_true", help="Remove existing synthetic data for this map before loading")
    args = parser.parse_args()

    print("=" * 60)
    print("SYNTHETIC DATA LOADER")
    print("=" * 60)

    conn = get_conn()
    cur = conn.cursor()

    # Ensure map exists
    map_id = ensure_map(cur)
    print(f"Using map '{MAP_NAME}' (id={map_id})")

    # Ensure rooms exist
    ensure_rooms(cur, map_id)
    print(f"Ensured {len(ROOM_BOUNDS)} rooms exist")

    if args.clean:
        clean_existing_data(cur, map_id)

    for date_str in DAYS:
        load_day(cur, map_id, date_str)

    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "=" * 60)
    print("LOAD COMPLETE")
    print("=" * 60)
    print("\nYou can now query the data with agent tools:")
    print(f"  get_room_event_density(hours=24)  -> scoped to map '{MAP_NAME}'")
    print(f"  get_room_visit_history()          -> scoped to map '{MAP_NAME}'")
    print(f"  get_stale_rooms(threshold_minutes=30)")
    print(f"  get_recent_interactions(minutes=60)")
    print("\nTo scope the agent to this map, set the active map:")
    print(f"  export TOOLBOX_MAP_ACTIVE_PATH=/shared/maps/toolbox_saved/active.json")
    print(f"  echo '{{\"mode\":\"frozen\",\"name\":\"{MAP_NAME}\"}}' > /shared/maps/toolbox_saved/active.json")
    print("\nOr run unscoped (all maps) by ensuring no active map file exists.")


if __name__ == "__main__":
    main()
