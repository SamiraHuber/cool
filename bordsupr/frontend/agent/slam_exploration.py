"""SLAM tab autonomous exploration logic.

Drives room-to-room exploration using scene captions from the database,
similar to the robot-pipeline v4 strategy but using the robot's actual
SLAM pose to determine the current room.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg2

from .pipeline_strategy import run_v4_decision, run_v5_decision, run_v6_decision

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _resolve_map_id(map_name: str) -> int | None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM maps WHERE name = %s", (map_name,))
            row = cur.fetchone()
            return int(row[0]) if row else None


def fetch_rooms_for_map(map_name: str) -> list[dict]:
    """Return rooms for the given map name, ordered by name."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.name, r.x1, r.y1, r.x2, r.y2, r.x3, r.y3, r.x4, r.y4, r.entry_x, r.entry_y
                FROM rooms r
                JOIN maps m ON m.id = r.map_id
                WHERE m.name = %s
                ORDER BY r.name
                """,
                (map_name,),
            )
            rows = cur.fetchall()
    return [
        {
            "name": r[0],
            "x1": r[1],
            "y1": r[2],
            "x2": r[3],
            "y2": r[4],
            "x3": r[5],
            "y3": r[6],
            "x4": r[7],
            "y4": r[8],
            "entry_x": r[9],
            "entry_y": r[10],
        }
        for r in rows
    ]


_SHARED_STITCHED_CAPTION_PATH = Path("/shared/stitched_caption.json")
_STITCHED_CAPTION_MAX_AGE_SEC = 15.0


def _read_stitched_caption() -> dict | None:
    """Read the latest stitched-caption file if it exists and is fresh."""
    try:
        if not _SHARED_STITCHED_CAPTION_PATH.exists():
            return None
        stat = _SHARED_STITCHED_CAPTION_PATH.stat()
        age = time.time() - stat.st_mtime
        if age > _STITCHED_CAPTION_MAX_AGE_SEC:
            return None
        data = json.loads(_SHARED_STITCHED_CAPTION_PATH.read_text())
        if data.get("caption"):
            return {"caption": data["caption"], "timestamp": data.get("timestamp"), "age_sec": age}
    except Exception:
        pass
    return None


def get_last_scene_for_map(map_id: int) -> dict | None:
    """Return the most recent scene row for the given map_id.

    Also checks /shared/stitched_caption.json for a fresher caption
    from the stitched front-middle camera.
    """
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, caption, timestamp, source_frame
                FROM scenes
                WHERE map_id = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (map_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            scene = {
                "id": row[0],
                "caption": row[1],
                "timestamp": row[2],
                "source_frame": row[3],
            }
            stitched = _read_stitched_caption()
            if stitched:
                scene["stitched_caption"] = stitched["caption"]
                logger.info(
                    f"[STITCHED_CAPTION] using stitched caption (age={stitched.get('age_sec', 0):.1f}s)"
                )
            return scene


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def determine_current_room(robot_pose: dict, rooms: list[dict]) -> str | None:
    """Return the room name the robot is currently inside, or None."""
    x = robot_pose.get("x")
    y = robot_pose.get("y")
    if x is None or y is None:
        return None
    for room in rooms:
        poly = [
            (room.get("x1"), room.get("y1")),
            (room.get("x2"), room.get("y2")),
            (room.get("x3"), room.get("y3")),
            (room.get("x4"), room.get("y4")),
        ]
        poly = [(px, py) for px, py in poly if px is not None and py is not None]
        if len(poly) < 3:
            continue
        if _point_in_polygon(float(x), float(y), poly):
            return room["name"]
    return None


def room_centroid(room: dict) -> tuple[float, float] | None:
    """Compute centroid from up to 4 corner points."""
    xs = [room.get(f"x{i}") for i in range(1, 5)]
    ys = [room.get(f"y{i}") for i in range(1, 5)]
    xs = [v for v in xs if v is not None]
    ys = [v for v in ys if v is not None]
    if not xs or not ys:
        return None
    return sum(xs) / len(xs), sum(ys) / len(ys)


def room_approach_yaw(room: dict, entry_x: float, entry_y: float) -> float:
    """Return a yaw angle (radians) that faces the room interior from the entry point."""
    centroid = room_centroid(room)
    if centroid is None:
        return 0.0
    cx, cy = centroid
    return math.atan2(cy - entry_y, cx - entry_x)


def _closest_point_on_segment(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Return the closest point on segment (x1,y1)-(x2,y2) to point (px,py)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return x1, y1
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    return x1 + t * dx, y1 + t * dy


class _CostmapCache:
    """Lazy-loaded global costmap for free-cell entry-point detection."""

    _path = "/shared/nav2_global_costmap_snapshot.json"
    _last_mtime: float | None = None
    data: list[int] | None = None
    width: int = 0
    height: int = 0
    resolution: float = 0.05
    origin_x: float = 0.0
    origin_y: float = 0.0

    @classmethod
    def refresh(cls) -> bool:
        try:
            st = os.stat(cls._path)
        except FileNotFoundError:
            cls.data = None
            return False
        if cls.data is not None and cls._last_mtime is not None and st.st_mtime <= cls._last_mtime:
            return True
        with open(cls._path, "r") as f:
            payload = json.load(f)
        m = payload.get("map", {})
        cls.data = m.get("data")
        cls.width = int(m.get("width", 0))
        cls.height = int(m.get("height", 0))
        cls.resolution = float(m.get("resolution", 0.05))
        origin = m.get("origin", {})
        if isinstance(origin, dict):
            if "x" in origin and "y" in origin:
                cls.origin_x = float(origin.get("x", 0.0))
                cls.origin_y = float(origin.get("y", 0.0))
            else:
                pos = origin.get("position", {})
                cls.origin_x = float(pos.get("x", 0.0)) if isinstance(pos, dict) else 0.0
                cls.origin_y = float(pos.get("y", 0.0)) if isinstance(pos, dict) else 0.0
        else:
            cls.origin_x = 0.0
            cls.origin_y = 0.0
        cls._last_mtime = st.st_mtime
        return cls.data is not None and cls.width > 0 and cls.height > 0

    @classmethod
    def _cell_value(cls, wx: float, wy: float) -> int:
        """Return costmap cell value at world coords, or 255 if out of bounds."""
        if cls.data is None:
            return 255
        cx = int((wx - cls.origin_x) / cls.resolution)
        cy = int((wy - cls.origin_y) / cls.resolution)
        if 0 <= cx < cls.width and 0 <= cy < cls.height:
            return int(cls.data[cy * cls.width + cx])
        return 255

    @classmethod
    def is_free(cls, wx: float, wy: float, neighbor_radius: int = 1) -> bool:
        if not cls.refresh() or cls.data is None:
            return False
        # Center cell must be known and below lethal threshold
        center_val = cls._cell_value(wx, wy)
        if center_val < 0 or center_val >= 100:
            return False
        # At least one cell in the neighborhood must be completely free (0)
        cx = int((wx - cls.origin_x) / cls.resolution)
        cy = int((wy - cls.origin_y) / cls.resolution)
        for dy in range(-neighbor_radius, neighbor_radius + 1):
            for dx in range(-neighbor_radius, neighbor_radius + 1):
                ix = cx + dx
                iy = cy + dy
                if 0 <= ix < cls.width and 0 <= iy < cls.height:
                    val = cls.data[iy * cls.width + ix]
                    if val == 0:
                        return True
        return False

    @classmethod
    def snap_to_free(cls, wx: float, wy: float, max_dist_m: float = 1.0, step_m: float = 0.05) -> tuple[float, float] | None:
        """Search outward in a spiral for the nearest free cell."""
        if not cls.refresh() or cls.data is None:
            return None
        # Check original point first
        if cls.is_free(wx, wy):
            return wx, wy
        # Spiral search in grid cells
        cx = int((wx - cls.origin_x) / cls.resolution)
        cy = int((wy - cls.origin_y) / cls.resolution)
        max_radius = max(1, int(math.ceil(max_dist_m / cls.resolution)))
        for radius in range(1, max_radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in (-radius, radius):
                    ix, iy = cx + dx, cy + dy
                    if 0 <= ix < cls.width and 0 <= iy < cls.height:
                        val = cls.data[iy * cls.width + ix]
                        if 0 <= val < 100:
                            return cls.origin_x + (ix + 0.5) * cls.resolution, cls.origin_y + (iy + 0.5) * cls.resolution
            for dx in range(-radius + 1, radius):
                for dy in (-radius, radius):
                    ix, iy = cx + dx, cy + dy
                    if 0 <= ix < cls.width and 0 <= iy < cls.height:
                        val = cls.data[iy * cls.width + ix]
                        if 0 <= val < 100:
                            return cls.origin_x + (ix + 0.5) * cls.resolution, cls.origin_y + (iy + 0.5) * cls.resolution
        return None


def _sample_edge(x1: float, y1: float, x2: float, y2: float, step: float = 0.1):
    """Yield points along edge (x1,y1)-(x2,y2) at given step interval."""
    dx = x2 - x1
    dy = y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        yield x1, y1
        return
    n = max(1, int(length / step))
    for i in range(n + 1):
        t = i / n
        yield x1 + t * dx, y1 + t * dy


def room_entry_point(
    room: dict, robot_pose: dict | None, interior_offset_m: float = 0.0
) -> tuple[float, float] | None:
    """Return the entry point for the room.

    Priority:
    1. Custom entry_x/entry_y if set in the database.
    2. A free cell on the room perimeter (using global costmap).
    3. The closest point on the perimeter (no costmap or all occupied).
    4. The room centroid if robot pose is unknown.

    If interior_offset_m > 0, the returned point is shifted that many metres
    toward the room centroid so the robot enters the room instead of stopping
    at the door.
    """
    raw_x, raw_y = None, None

    entry_x = room.get("entry_x")
    entry_y = room.get("entry_y")
    if entry_x is not None and entry_y is not None:
        raw_x, raw_y = float(entry_x), float(entry_y)
    elif robot_pose is None:
        raw_x, raw_y = room_centroid(room)
    else:
        rx = robot_pose.get("x")
        ry = robot_pose.get("y")
        if rx is None or ry is None:
            raw_x, raw_y = room_centroid(room)
        else:
            poly = [
                (room.get("x1"), room.get("y1")),
                (room.get("x2"), room.get("y2")),
                (room.get("x3"), room.get("y3")),
                (room.get("x4"), room.get("y4")),
            ]
            poly = [(px, py) for px, py in poly if px is not None and py is not None]
            # If only two diagonal corners are provided, infer the full rectangle
            if len(poly) == 2:
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                poly = [
                    (min(xs), min(ys)),
                    (max(xs), min(ys)),
                    (max(xs), max(ys)),
                    (min(xs), max(ys)),
                ]
            if len(poly) < 2:
                raw_x, raw_y = room_centroid(room)
            else:
                def _edge_length(x1, y1, x2, y2):
                    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

                # Try costmap-aware free-cell detection first
                costmap_ok = _CostmapCache.refresh()
                best_x, best_y = None, None
                best_dist = float("inf")
                if costmap_ok:
                    # Pass 1: find door candidates — short contiguous free segments on each edge
                    # A door is a free gap that is short relative to the edge length (not an open end)
                    door_candidates = []
                    for i in range(len(poly)):
                        x1, y1 = poly[i]
                        x2, y2 = poly[(i + 1) % len(poly)]
                        edge_len = _edge_length(float(x1), float(y1), float(x2), float(y2))
                        samples = list(_sample_edge(float(x1), float(y1), float(x2), float(y2), step=0.05))
                        free_start = None
                        for idx, (sx, sy) in enumerate(samples):
                            is_free = _CostmapCache._cell_value(sx, sy) == 0
                            if is_free and free_start is None:
                                free_start = idx
                            elif not is_free and free_start is not None:
                                seg_len = _edge_length(
                                    samples[free_start][0], samples[free_start][1],
                                    samples[idx - 1][0], samples[idx - 1][1],
                                )
                                if seg_len <= 1.5 and seg_len / edge_len <= 0.25:
                                    cx, cy = _closest_point_on_segment(
                                        float(rx), float(ry),
                                        samples[free_start][0], samples[free_start][1],
                                        samples[idx - 1][0], samples[idx - 1][1],
                                    )
                                    door_candidates.append((cx, cy))
                                free_start = None
                        if free_start is not None:
                            seg_len = _edge_length(
                                samples[free_start][0], samples[free_start][1],
                                samples[-1][0], samples[-1][1],
                            )
                            if seg_len <= 1.5 and seg_len / edge_len <= 0.25:
                                cx, cy = _closest_point_on_segment(
                                    float(rx), float(ry),
                                    samples[free_start][0], samples[free_start][1],
                                    samples[-1][0], samples[-1][1],
                                )
                                door_candidates.append((cx, cy))
                    for mx, my in door_candidates:
                        d = (mx - float(rx)) ** 2 + (my - float(ry)) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_x, best_y = mx, my

                    if best_x is None:
                        # Pass 2: fall back to any point that passes is_free (inflated but traversable)
                        best_dist = float("inf")
                        for i in range(len(poly)):
                            x1, y1 = poly[i]
                            x2, y2 = poly[(i + 1) % len(poly)]
                            for sx, sy in _sample_edge(float(x1), float(y1), float(x2), float(y2), step=0.1):
                                if _CostmapCache.is_free(sx, sy, neighbor_radius=1):
                                    d = (sx - float(rx)) ** 2 + (sy - float(ry)) ** 2
                                    if d < best_dist:
                                        best_dist = d
                                        best_x, best_y = sx, sy

                if best_x is None:
                    # Fallback: closest point on perimeter, then snap to nearest free cell
                    for i in range(len(poly)):
                        x1, y1 = poly[i]
                        x2, y2 = poly[(i + 1) % len(poly)]
                        cx, cy = _closest_point_on_segment(float(rx), float(ry), float(x1), float(y1), float(x2), float(y2))
                        d = (cx - float(rx)) ** 2 + (cy - float(ry)) ** 2
                        if d < best_dist:
                            best_dist = d
                            best_x, best_y = cx, cy

                    if costmap_ok and best_x is not None:
                        snapped = _CostmapCache.snap_to_free(best_x, best_y)
                        if snapped is not None:
                            best_x, best_y = snapped

                raw_x, raw_y = best_x, best_y

    if interior_offset_m > 0 and raw_x is not None and raw_y is not None:
        centroid = room_centroid(room)
        if centroid is not None:
            cx, cy = centroid
            dx = cx - raw_x
            dy = cy - raw_y
            dist = math.hypot(dx, dy)
            if dist > 0:
                ox = raw_x + (dx / dist) * interior_offset_m
                oy = raw_y + (dy / dist) * interior_offset_m
                if _CostmapCache.is_free(ox, oy, neighbor_radius=1):
                    return ox, oy

    if raw_x is None or raw_y is None:
        return None
    return raw_x, raw_y


def get_dwell_minutes_for_room(map_id: int, room_name: str) -> int | None:
    """Return minutes since the robot arrived in room_name, or None."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            # 1. Try open robot_visits row
            cur.execute(
                """
                SELECT arrived_at FROM robot_visits
                WHERE room_name = %s AND map_id = %s AND departed_at IS NULL
                ORDER BY arrived_at DESC
                LIMIT 1
                """,
                (room_name, map_id),
            )
            row = cur.fetchone()
            if row and row[0]:
                arrived = row[0]
                now = datetime.now(timezone.utc)
                if arrived.tzinfo is None:
                    arrived = arrived.replace(tzinfo=timezone.utc)
                return int((now - arrived).total_seconds() / 60)

            # 2. Fall back to first navigation_decision for this room
            cur.execute(
                """
                SELECT created_at FROM navigation_decisions
                WHERE map_id = %s AND target_room = %s
                ORDER BY created_at ASC
                LIMIT 1
                """,
                (map_id, room_name),
            )
            row = cur.fetchone()
            if row and row[0]:
                first_decision = row[0]
                now = datetime.now(timezone.utc)
                if first_decision.tzinfo is None:
                    first_decision = first_decision.replace(tzinfo=timezone.utc)
                return int((now - first_decision).total_seconds() / 60)

            # 3. Fall back to first scene on this map
            cur.execute(
                "SELECT timestamp FROM scenes WHERE map_id = %s ORDER BY timestamp ASC LIMIT 1",
                (map_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                first_scene = row[0]
                now = datetime.now(timezone.utc)
                if first_scene.tzinfo is None:
                    first_scene = first_scene.replace(tzinfo=timezone.utc)
                return int((now - first_scene).total_seconds() / 60)

    return None


def get_active_room_from_visits(map_id: int) -> str | None:
    """Return the room name of the most recent open visit, or None."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT room_name FROM robot_visits
                WHERE map_id = %s AND departed_at IS NULL
                ORDER BY arrived_at DESC
                LIMIT 1
                """,
                (map_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None


def get_previous_scene_objects_for_room(
    map_id: int,
    room_name: str,
    current_scene_id: int,
    rooms: list[dict],
) -> list[dict] | None:
    """Find the most recent previous scene whose (x,y) lies inside room_name and return its objects."""
    room = next((r for r in rooms if r["name"] == room_name), None)
    if room is None:
        return None

    poly = [
        (room.get("x1"), room.get("y1")),
        (room.get("x2"), room.get("y2")),
        (room.get("x3"), room.get("y3")),
        (room.get("x4"), room.get("y4")),
    ]
    poly = [(px, py) for px, py in poly if px is not None and py is not None]
    if len(poly) < 3:
        return None

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, x, y
                FROM scenes
                WHERE map_id = %s AND id < %s
                ORDER BY timestamp DESC
                LIMIT 50
                """,
                (map_id, current_scene_id),
            )
            for row in cur.fetchall():
                scene_id, sx, sy = row
                if sx is None or sy is None:
                    continue
                if _point_in_polygon(float(sx), float(sy), poly):
                    from .pipeline_strategy import _get_scene_objects
                    return _get_scene_objects(scene_id)
    return None


def get_step_number_for_map(map_id: int) -> int:
    """Count existing slam_exploration decisions for this map."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM navigation_decisions WHERE map_id = %s AND decision_type LIKE %s",
                (map_id, "slam_exploration_%"),
            )
            row = cur.fetchone()
            return int(row[0]) if row else 0


def clear_exploration_history(map_name: str) -> dict:
    """Delete slam_exploration decisions and visits for the given map.

    Returns a dict with keys: ok, deleted_decisions, deleted_visits.
    """
    map_id = _resolve_map_id(map_name)
    if map_id is None:
        return {"ok": False, "error": f"Map '{map_name}' not found."}
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM navigation_decisions WHERE map_id = %s AND decision_type LIKE %s",
                (map_id, "slam_exploration_%"),
            )
            deleted_decisions = cur.rowcount
            cur.execute(
                "DELETE FROM robot_visits WHERE map_id = %s",
                (map_id,),
            )
            deleted_visits = cur.rowcount
        conn.commit()
    return {"ok": True, "deleted_decisions": deleted_decisions, "deleted_visits": deleted_visits}


def save_decision(
    map_id: int,
    scene_id: int | None,
    room_name: str,
    decision: dict,
    step_number: int,
    decision_type: str = "slam_exploration_v4",
) -> None:
    """Persist one exploration decision to the database."""
    tool_calls = decision.get("tool_calls") or []
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO navigation_decisions
                (decision_type, target_room, target_x, target_y,
                 dwell_time_seconds, reasoning, scene_change_prediction,
                 map_id, scene_changed, change_severity, activities_changed,
                 tool_calls_json, step_number)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    decision_type,
                    decision.get("target_room"),
                    None,
                    None,
                    None,
                    decision.get("reasoning"),
                    None,
                    map_id,
                    decision.get("scene_changed"),
                    decision.get("change_severity"),
                    decision.get("activities_changed"),
                    json.dumps(tool_calls) if tool_calls else None,
                    step_number,
                ),
            )
        conn.commit()


def transition_visit(from_room: str | None, to_room: str, map_id: int) -> None:
    """Close visit in from_room and open a new one in to_room."""
    now = datetime.now(timezone.utc)
    with _get_conn() as conn:
        with conn.cursor() as cur:
            if from_room:
                cur.execute(
                    """
                    UPDATE robot_visits
                    SET departed_at = %s
                    WHERE ctid = (
                        SELECT ctid FROM robot_visits
                        WHERE room_name = %s AND map_id = %s AND departed_at IS NULL
                        ORDER BY arrived_at DESC
                        LIMIT 1
                    )
                    """,
                    (now, from_room, map_id),
                )
            # Close any stale open visit for to_room to prevent duplicates
            cur.execute(
                """
                UPDATE robot_visits
                SET departed_at = %s
                WHERE room_name = %s AND map_id = %s AND departed_at IS NULL
                """,
                (now, to_room, map_id),
            )
            cur.execute(
                """
                INSERT INTO robot_visits
                (room_name, map_id, arrived_at, scene_count)
                VALUES (%s, %s, %s, 0)
                """,
                (to_room, map_id, now),
            )
        conn.commit()


def run_slam_exploration_step(
    map_name: str,
    robot_pose: dict | None,
    strategy_type: str = "v4",
    disabled_rooms: list[str] | None = None,
    min_dwell_minutes: int = 10,
) -> dict[str, Any]:
    """Run one exploration step for the SLAM tab.

    Args:
        map_name: Name of the currently selected map.
        robot_pose: Robot pose dict with x, y, z, yaw.
        strategy_type: "v4" or "v5".
        disabled_rooms: List of room names to exclude from exploration targets.
        min_dwell_minutes: Minimum dwell time before agent considers moving.

    Returns a dict with keys:
        action, target_room, target_x, target_y, current_room,
        reasoning, scene_caption, step_number, dwell_minutes,
        scene_changed, change_severity, error
    """
    disabled_rooms = [r.strip() for r in (disabled_rooms or []) if r.strip()]
    disabled_rooms_lower = {r.lower() for r in disabled_rooms}
    map_id = _resolve_map_id(map_name)
    if map_id is None:
        return {"action": "stay", "error": f"Map '{map_name}' not found in database."}

    rooms = fetch_rooms_for_map(map_name)
    if not rooms:
        return {"action": "stay", "error": f"No rooms defined for map '{map_name}'."}

    scene = get_last_scene_for_map(map_id)
    if not scene:
        return {"action": "stay", "error": "No scene captions available yet."}

    current_room = determine_current_room(robot_pose, rooms) if robot_pose else None
    room_list = [r["name"] for r in rooms if r["name"].lower() not in disabled_rooms_lower]

    # Trust DB open-visit state before falling back to geometric detection.
    # This prevents re-issuing MOVE commands when the robot is on a doorway
    # boundary where determine_current_room returns None.
    visit_room = get_active_room_from_visits(map_id)
    active_room = current_room or visit_room or (room_list[0] if room_list else None)
    logger.info(
        f"[SLAM_EXPLORE] map={map_name} geo_room={current_room} visit_room={visit_room} "
        f"active_room={active_room} disabled={disabled_rooms} all_rooms={[r['name'] for r in rooms]} filtered={room_list}"
    )

    if not room_list:
        return {"action": "stay", "error": f"All rooms are disabled for map '{map_name}'.", "current_room": current_room}

    step_number = get_step_number_for_map(map_id) + 1

    dwell_minutes = get_dwell_minutes_for_room(map_id, active_room) if active_room else None

    try:
        # Prefer stitched caption (front-middle camera) when available
        caption_for_decision = scene.get("stitched_caption") or scene.get("caption") or ""

        if strategy_type == "v6":
            prev_objects = None
            if active_room:
                prev_objects = get_previous_scene_objects_for_room(
                    map_id, active_room, scene["id"], rooms
                )
            decision = run_v6_decision(
                scene_id=scene["id"],
                room_name=active_room,
                scene_caption=caption_for_decision,
                map_id=map_id,
                room_list=room_list,
                step_number=step_number,
                dwell_minutes=dwell_minutes,
                min_dwell_minutes=min_dwell_minutes,
                previous_scene_objects=prev_objects,
            )
        elif strategy_type == "v5":
            decision = run_v5_decision(
                scene_id=scene["id"],
                room_name=active_room,
                scene_caption=caption_for_decision,
                map_id=map_id,
                room_list=room_list,
                step_number=step_number,
                dwell_minutes=dwell_minutes,
                min_dwell_minutes=min_dwell_minutes,
            )
        else:
            decision = run_v4_decision(
                scene_id=scene["id"],
                room_name=active_room,
                scene_caption=caption_for_decision,
                map_id=map_id,
                room_list=room_list,
                step_number=step_number,
                dwell_minutes=dwell_minutes,
                min_dwell_minutes=min_dwell_minutes,
            )
    except Exception as exc:
        logger.exception("VLM decision failed")
        return {
            "action": "stay",
            "error": f"Decision engine failed: {exc}",
            "current_room": active_room,
            "scene_caption": scene.get("stitched_caption") or scene.get("caption"),
            "step_number": step_number,
        }

    # Hard min-dwell enforcement: override VLM if not enough time spent
    if (
        dwell_minutes is not None
        and dwell_minutes < min_dwell_minutes
        and decision.get("action") == "move"
    ):
        decision["action"] = "stay"
        decision["target_room"] = active_room
        orig_reasoning = decision.get("reasoning") or ""
        decision["reasoning"] = (
            f"[MIN_DWELL] {orig_reasoning} "
            f"(Forced stay: only {dwell_minutes:.1f} min elapsed, minimum is {min_dwell_minutes} min.)"
        )
        logger.info(
            "[SLAM_EXPLORE] min_dwell enforced: dwell=%.1f < min=%d → forced stay in %s",
            dwell_minutes,
            min_dwell_minutes,
            active_room,
        )

    # Resolve target coordinates if moving
    target_room = decision.get("target_room")
    target_x = None
    target_y = None
    if decision.get("action") == "move" and target_room:
        for r in rooms:
            if r["name"].lower() == target_room.lower() and r["name"].lower() not in disabled_rooms_lower:
                entry = room_entry_point(r, robot_pose, interior_offset_m=1.0)
                if entry:
                    target_x, target_y = entry
                break

    decision_type = f"slam_exploration_{strategy_type}"
    save_decision(map_id, scene["id"], active_room, decision, step_number, decision_type=decision_type)

    if decision.get("action") == "move" and target_room and target_room != active_room:
        transition_visit(active_room, target_room, map_id)

    return {
        "action": decision.get("action", "stay"),
        "target_room": target_room,
        "target_x": target_x,
        "target_y": target_y,
        "current_room": active_room,
        "reasoning": decision.get("reasoning", ""),
        "scene_caption": scene.get("stitched_caption") or scene.get("caption"),
        "step_number": step_number,
        "dwell_minutes": dwell_minutes,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scene_changed": decision.get("scene_changed"),
        "change_severity": decision.get("change_severity"),
        "error": decision.get("error"),
        "prompt": decision.get("prompt"),
    }
