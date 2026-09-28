import base64
import contextvars
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import psycopg2

from .vlm_client import get_vlm_client, get_vlm_model
from .text_embeddings import embed_text, get_text_embedding_dim
from .slam_exploration import room_approach_yaw, room_entry_point

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")
AGENT_QUERY_TIMEZONE = os.getenv("AGENT_QUERY_TIMEZONE", os.getenv("TZ", "Europe/Berlin"))
MIN_OBJECT_OBSERVATIONS = 1
SEARCH_OBJECTS_BY_CLASS_ID_LIMIT = 5
DEFAULT_INTERACTION_EVENT_LIMIT = 40
DEFAULT_OBJECT_OBSERVATION_LIMIT = 20
MAX_HISTORY_RESULT_LIMIT = 200
TOOLBOX_MAP_ACTIVE_PATH = Path(
    os.getenv("TOOLBOX_MAP_ACTIVE_PATH", "/shared/maps/toolbox_saved/active.json")
)
TOOLBOX_MAP_PATH = Path(
    os.getenv("TOOLBOX_MAP_PATH", "/shared/toolbox_map_snapshot.json")
)
SLAM_TAB_ROBOT_POSE_PATH = Path(
    os.getenv("SLAM_TAB_ROBOT_POSE_PATH", "/shared/slam_tab/robot_pose.json")
)
_BUILTIN_EXTENDED_CLASS_NAMES = {
    80: "headphones",
    81: "dishwasher",
    82: "coffee_machine",
    83: "kitchen_counter",
}
NAVIGATION_USABLE_POSITION_SOURCES = {"depth", "dynosam"}


def _extended_class_name_paths() -> list[Path]:
    env_path = os.getenv("EXTENDED_CLASS_NAMES_PATH")
    paths: list[Path] = []
    if env_path:
        paths.append(Path(env_path))

    module_root_path = Path(__file__).resolve().parents[1] / "extended_classes.txt"
    paths.append(module_root_path)

    app_path = Path("/app/extended_classes.txt")
    if app_path != module_root_path:
        paths.append(app_path)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _load_extended_class_names() -> dict[int, str]:
    mapping: dict[int, str] = dict(_BUILTIN_EXTENDED_CLASS_NAMES)
    lines: list[str] | None = None
    for path in _extended_class_name_paths():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            break
        except Exception:
            continue

    if lines is None:
        return mapping

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"(\d+)\s*=\s*([a-zA-Z0-9_ -]+)", line)
        if not match:
            continue
        mapping[int(match.group(1))] = match.group(2).strip()
    return mapping

YOLO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}

YOLO_CLASS_NAMES.update(_load_extended_class_names())


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


_active_map_override: contextvars.ContextVar[str | None] = contextvars.ContextVar("_active_map_override", default=None)


def set_active_map_override(name: str | None) -> None:
    """Override the active map name used by tools (e.g. from the topnav building selector)."""
    _active_map_override.set(name)


def _active_map_name() -> str | None:
    override = _active_map_override.get()
    if override is not None:
        return override
    try:
        if not TOOLBOX_MAP_ACTIVE_PATH.exists():
            return None
        with TOOLBOX_MAP_ACTIVE_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    mode = str(payload.get("mode") or "recording").strip().lower()
    value = payload.get("name") if mode == "frozen" else (payload.get("observation_map_name") or payload.get("name"))
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip()).strip("._-")
    return text or None


def _resolve_map_id(cur, map_name: str | None) -> int | None:
    normalized_name = str(map_name or "").strip()
    if not normalized_name:
        return None
    cur.execute("SELECT id FROM maps WHERE name = %s LIMIT 1", (normalized_name,))
    row = cur.fetchone()
    return int(row[0]) if row and row[0] is not None else None


def _normalize_sort_order(sort_order: str | None) -> str:
    value = str(sort_order or "desc").strip().lower()
    if value in {"asc", "ascending", "oldest", "earliest", "first"}:
        return "asc"
    return "desc"


def _is_navigation_usable_position_source(position_source: str | None) -> bool:
    # Older synthetic/imported datasets do not always populate position_source.
    # Only explicit YOLO-only coordinates should be rejected for navigation.
    normalized = str(position_source or "").strip().lower()
    if not normalized:
        return True
    return normalized in NAVIGATION_USABLE_POSITION_SOURCES


def _parse_time_input(value: str | None) -> tuple[str | None, str | None]:
    """Parse a time string and return (normalized_value, time_type).

    Types:
    - 'hhmm': HH:MM exact time (24-hour format)
    - 'date': YYYY-MM-DD exact date
    - 'iso': full ISO timestamp
    Returns (None, None) for unparseable or empty input.
    """
    normalized = str(value or "").strip()
    if not normalized:
        return None, None

    # HH:MM or H:MM (24h)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", normalized)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}", "hhmm"
        return None, None

    # HH:MM:SS or H:MM:SS (24h) — strip seconds
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})", normalized)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        second = int(m.group(3))
        if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
            return f"{hour:02d}:{minute:02d}", "hhmm"
        return None, None

    # H:MM AM/PM or HH:MM AM/PM (case insensitive, optional space)
    m = re.fullmatch(r"(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)", normalized)
    if m:
        hour = int(m.group(1))
        minute = int(m.group(2))
        ampm = m.group(3).lower().replace(".", "")
        if not (1 <= hour <= 12 and 0 <= minute <= 59):
            return None, None
        if "pm" in ampm and hour != 12:
            hour += 12
        elif "am" in ampm and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}", "hhmm"

    # H AM/PM or HH AM/PM (e.g., "2pm", "11 AM")
    m = re.fullmatch(r"(\d{1,2})\s*([AaPp]\.?[Mm]\.?)", normalized)
    if m:
        hour = int(m.group(1))
        ampm = m.group(2).lower().replace(".", "")
        if not (1 <= hour <= 12):
            return None, None
        if "pm" in ampm and hour != 12:
            hour += 12
        elif "am" in ampm and hour == 12:
            hour = 0
        return f"{hour:02d}:00", "hhmm"

    # YYYY-MM-DD
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", normalized):
        return normalized, "date"

    # ISO timestamp
    try:
        parsed = datetime.fromisoformat(normalized)
        return parsed.isoformat(), "iso"
    except ValueError:
        pass

    return None, None


def _build_timestamp_filter(timestamp_expr: str, time_filter: str | None) -> tuple[str, list[object], str | None]:
    normalized, time_type = _parse_time_input(time_filter)
    if normalized is None:
        return "", [], None

    if time_type == "hhmm":
        target_seconds = int(normalized[:2]) * 3600 + int(normalized[3:5]) * 60
        return (
            f""" AND LEAST(
                ABS((EXTRACT(EPOCH FROM ({timestamp_expr} AT TIME ZONE %s)::time)::int %% 86400) - {target_seconds}),
                86400 - ABS((EXTRACT(EPOCH FROM ({timestamp_expr} AT TIME ZONE %s)::time)::int %% 86400) - {target_seconds})
            ) <= 600""",
            [AGENT_QUERY_TIMEZONE, AGENT_QUERY_TIMEZONE],
            normalized,
        )

    if time_type == "date":
        return (
            f" AND DATE(({timestamp_expr}) AT TIME ZONE %s) = %s::date",
            [AGENT_QUERY_TIMEZONE, normalized],
            normalized,
        )

    # time_type == "iso"
    parsed = datetime.fromisoformat(normalized)
    local_iso = parsed.replace(tzinfo=None).isoformat(sep=" ")
    return (
        f" AND ({timestamp_expr} AT TIME ZONE %s) BETWEEN (%s::timestamp - INTERVAL '1 minute') AND (%s::timestamp + INTERVAL '1 minute')",
        [AGENT_QUERY_TIMEZONE, local_iso, local_iso],
        local_iso,
    )


def _build_time_range_bound(
    timestamp_expr: str,
    bound_value: str | None,
    operator: str,
) -> tuple[str, list[object], str | None]:
    normalized, time_type = _parse_time_input(bound_value)
    if normalized is None:
        return "", [], None

    if time_type == "hhmm":
        return (
            f" AND to_char(({timestamp_expr}) AT TIME ZONE %s, 'HH24:MI') {operator} %s",
            [AGENT_QUERY_TIMEZONE, normalized],
            normalized,
        )

    if time_type == "date":
        return (
            f" AND DATE(({timestamp_expr}) AT TIME ZONE %s) {operator} %s::date",
            [AGENT_QUERY_TIMEZONE, normalized],
            normalized,
        )

    # time_type == "iso"
    parsed = datetime.fromisoformat(normalized)
    local_iso = parsed.replace(tzinfo=None).isoformat(sep=" ")
    return (
        f" AND ({timestamp_expr} AT TIME ZONE %s) {operator} %s::timestamp",
        [AGENT_QUERY_TIMEZONE, local_iso],
        local_iso,
    )


def _build_time_range_filters(
    timestamp_expr: str,
    start_time: str | None,
    end_time: str | None,
) -> tuple[str, list[object], str | None, str | None]:
    start_normalized, start_type = _parse_time_input(start_time)
    end_normalized, end_type = _parse_time_input(end_time)

    # Exact-time query (start == end): use fuzzy matching so small timezone
    # shifts or seconds-level offsets don't cause misses.
    if start_normalized is not None and start_normalized == end_normalized:
        if start_type == "hhmm":
            target_seconds = int(start_normalized[:2]) * 3600 + int(start_normalized[3:5]) * 60
            return (
                f""" AND LEAST(
                    ABS((EXTRACT(EPOCH FROM ({timestamp_expr} AT TIME ZONE %s)::time)::int %% 86400) - {target_seconds}),
                    86400 - ABS((EXTRACT(EPOCH FROM ({timestamp_expr} AT TIME ZONE %s)::time)::int %% 86400) - {target_seconds})
                ) <= 600""",
                [AGENT_QUERY_TIMEZONE, AGENT_QUERY_TIMEZONE],
                start_normalized,
                start_normalized,
            )
        if start_type == "iso":
            parsed = datetime.fromisoformat(start_normalized)
            local_iso = parsed.replace(tzinfo=None).isoformat(sep=" ")
            return (
                f" AND ({timestamp_expr} AT TIME ZONE %s) BETWEEN (%s::timestamp - INTERVAL '1 minute') AND (%s::timestamp + INTERVAL '1 minute')",
                [AGENT_QUERY_TIMEZONE, local_iso, local_iso],
                local_iso,
                local_iso,
            )

    start_clause, start_params, normalized_start_time = _build_time_range_bound(timestamp_expr, start_time, ">=")
    end_clause, end_params, normalized_end_time = _build_time_range_bound(timestamp_expr, end_time, "<=")
    return (
        f"{start_clause}{end_clause}",
        [*start_params, *end_params],
        normalized_start_time,
        normalized_end_time,
    )


def _to_local_timestamp(timestamp_value) -> str | None:
    if timestamp_value is None:
        return None
    if not isinstance(timestamp_value, datetime):
        return str(timestamp_value)
    try:
        return timestamp_value.astimezone(ZoneInfo(AGENT_QUERY_TIMEZONE)).isoformat(sep=" ", timespec="seconds")
    except Exception:
        return str(timestamp_value)


def _get_active_map_scope(cur) -> tuple[str | None, int | None, bool]:
    active_map_name = _active_map_name()
    scope_requested = active_map_name is not None
    active_map_id = _resolve_map_id(cur, active_map_name) if scope_requested else None
    if scope_requested and active_map_id is None:
        # The UI may point at a transient toolbox map alias that is not present in the DB-backed maps table.
        # In that case, fall back to an unscoped query rather than hiding all historical results.
        return None, None, False
    return active_map_name, active_map_id, scope_requested


def _load_json_path(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _ensure_rooms_schema() -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE rooms ADD COLUMN IF NOT EXISTS map_id BIGINT")
            cur.execute("ALTER TABLE rooms ADD COLUMN IF NOT EXISTS map_name TEXT")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rooms_map_id ON rooms(map_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rooms_map_name ON rooms(map_name)")
        conn.commit()


def _ensure_objects_schema() -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("ALTER TABLE objects ADD COLUMN IF NOT EXISTS name TEXT")
        conn.commit()


def _class_name(class_id):
    if class_id is None:
        return None
    return YOLO_CLASS_NAMES.get(int(class_id), f"class_{class_id}")


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _fetch_object_snapshot(cur, object_id: int, active_map_id: int | None) -> tuple | None:
    map_filter_sql = ""
    latest_filter_sql = ""
    params: list[object] = []
    if active_map_id is not None:
        map_filter_sql = """
            AND (
                oo.map_id = %s
                OR EXISTS (
                    SELECT 1 FROM scenes s2
                    WHERE s2.id = oo.scene_id AND s2.map_id = %s
                )
            )
        """
        latest_filter_sql = """
            AND (
                oo2.map_id = %s
                OR EXISTS (
                    SELECT 1 FROM scenes s2
                    WHERE s2.id = oo2.scene_id AND s2.map_id = %s
                )
            )
        """
        params.extend([active_map_id, active_map_id])
        params.extend([active_map_id, active_map_id])
    params.append(object_id)
    cur.execute(
        f"""
        SELECT
            o.id,
            o.class_id,
            o.name,
            COUNT(oo.id) AS obs_count,
            MAX(oo.created_at) AS last_seen,
            latest.x,
            latest.y,
            latest.latest_obs_id
        FROM objects o
        LEFT JOIN object_observations oo
          ON oo.object_id = o.id
         {map_filter_sql}
        LEFT JOIN LATERAL (
            SELECT oo2.x, oo2.y, oo2.id AS latest_obs_id
            FROM object_observations oo2
            WHERE oo2.object_id = o.id {latest_filter_sql}
            ORDER BY oo2.created_at DESC
            LIMIT 1
        ) latest ON TRUE
        WHERE o.id = %s
        GROUP BY o.id, o.class_id, o.name, latest.x, latest.y, latest.latest_obs_id
        """,
        tuple(params),
    )
    return cur.fetchone()


def _room_scope_clause(alias: str, active_map_name: str | None, active_map_id: int | None) -> tuple[str, list[object]]:
    if active_map_id is not None and active_map_name:
        return f"(({alias}.map_id = %s) OR ({alias}.map_id IS NULL AND {alias}.map_name = %s))", [active_map_id, active_map_name]
    if active_map_id is not None:
        return f"{alias}.map_id = %s", [active_map_id]
    if active_map_name:
        return f"{alias}.map_name = %s", [active_map_name]
    return "", []


def _room_bounds_from_row(row, start_idx=2):
    """Compute bounding box min/max from x1..x4, y1..y4 in a DB row, handling NULLs."""
    xs = []
    ys = []
    for i in range(4):
        x = row[start_idx + i * 2]
        y = row[start_idx + i * 2 + 1]
        if x is not None and y is not None:
            xs.append(float(x))
            ys.append(float(y))
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _resolve_room_name(cur, active_map_name: str | None, active_map_id: int | None, x, y) -> str | None:
    if x is None or y is None:
        return None
    room_scope_clause, room_scope_params = _room_scope_clause("rooms", active_map_name, active_map_id)
    if room_scope_clause:
        cur.execute(
            f"""SELECT name FROM rooms WHERE {room_scope_clause}
            AND %s BETWEEN LEAST(x1, x2, x3, x4) AND GREATEST(x1, x2, x3, x4)
            AND %s BETWEEN LEAST(y1, y2, y3, y4) AND GREATEST(y1, y2, y3, y4)
            LIMIT 1""",
            tuple(room_scope_params + [x, y]),
        )
    else:
        cur.execute(
            """SELECT name FROM rooms
            WHERE %s BETWEEN LEAST(x1, x2, x3, x4) AND GREATEST(x1, x2, x3, x4)
            AND %s BETWEEN LEAST(y1, y2, y3, y4) AND GREATEST(y1, y2, y3, y4)
            LIMIT 1""",
            (x, y),
        )
    room_row = cur.fetchone()
    if room_row and room_row[0]:
        return room_row[0]

    if room_scope_clause:
        cur.execute(
            f"""
            SELECT name, x1, y1, x2, y2, x3, y3, x4, y4
            FROM rooms
            WHERE {room_scope_clause}
            """,
            tuple(room_scope_params),
        )
        room_rows = cur.fetchall()
        if not room_rows:
            cur.execute(
                """
                SELECT name, x1, y1, x2, y2, x3, y3, x4, y4
                FROM rooms
                """
            )
            room_rows = cur.fetchall()
    else:
        cur.execute(
            """
            SELECT name, x1, y1, x2, y2, x3, y3, x4, y4
            FROM rooms
            """
        )
        room_rows = cur.fetchall()
    best_name = None
    best_distance = None
    for row in room_rows:
        name = row[0]
        min_x, min_y, max_x, max_y = _room_bounds_from_row(row, start_idx=1)
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        distance = math.hypot(float(x) - center_x, float(y) - center_y)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_name = name
    return best_name or "unknown"


def _lookup_named_object_ids(cur, object_name: str) -> list[tuple]:
    normalized_name = str(object_name or "").strip()
    if not normalized_name:
        return []
    normalized_spaced_name = " ".join(re.findall(r"[a-z0-9]+", normalized_name.lower().replace("_", " ")))
    cur.execute(
        """
        SELECT id, name, class_id
        FROM objects
        WHERE lower(coalesce(name, '')) = lower(%s)
           OR replace(lower(coalesce(name, '')), '_', ' ') = %s
        ORDER BY class_id ASC, id ASC
        """,
        (normalized_name, normalized_spaced_name),
    )
    return cur.fetchall()


def search_objects_by_class_id(
    class_id: int | None = None,
    object_name: str | None = None,
    min_observations: int = MIN_OBJECT_OBSERVATIONS,
    limit: int = SEARCH_OBJECTS_BY_CLASS_ID_LIMIT,
) -> dict:
    """Find objects by YOLO class id and/or tracked object name.

    At least one of class_id or object_name should be provided.
    When object_name is given without a class_id, the search falls back to
    name-based matching across all classes.
    """
    safe_class_id = int(class_id) if class_id is not None else None
    requested_name = str(object_name or "").strip()
    normalized_name = " ".join(re.findall(r"[a-z0-9]+", requested_name.lower().replace("_", " ")))
    min_count = max(1, int(min_observations))
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    candidate_limit = max(6, safe_limit * 3)

    _ensure_rooms_schema()
    _ensure_objects_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {
                    "class_id": safe_class_id,
                    "class_name": _class_name(safe_class_id),
                    "object_name": requested_name or None,
                    "min_observations": min_count,
                    "limit": safe_limit,
                    "results": [],
                }

            params: list[object] = []
            map_filter = ""
            if active_map_id is not None:
                map_filter = """
                    AND (
                        oo.map_id = %s
                        OR EXISTS (
                            SELECT 1 FROM scenes s2
                            WHERE s2.id = oo.scene_id AND s2.map_id = %s
                        )
                    )
                """
                params.extend([active_map_id, active_map_id])
            query_params = list(params)
            filters = []
            if safe_class_id is not None:
                filters.append("o.class_id = %s")
                query_params.append(safe_class_id)
            if requested_name:
                normalized_like = f"%{normalized_name}%"
                # If the requested name is numeric, also match against object id
                # so users can refer to objects like "object 462"
                if requested_name.isdigit():
                    filters.append("(o.name ILIKE %s OR replace(lower(coalesce(o.name, '')), '_', ' ') LIKE %s OR o.id = %s)")
                    query_params.extend([f"%{requested_name}%", normalized_like, int(requested_name)])
                else:
                    filters.append("(o.name ILIKE %s OR replace(lower(coalesce(o.name, '')), '_', ' ') LIKE %s)")
                    query_params.extend([f"%{requested_name}%", normalized_like])
            if not filters:
                return {
                    "class_id": safe_class_id,
                    "class_name": _class_name(safe_class_id),
                    "object_name": requested_name or None,
                    "min_observations": min_count,
                    "limit": safe_limit,
                    "results": [],
                }
            where_clause = "WHERE " + " AND ".join(filters)

            cur.execute(
                f"""
                SELECT o.id, MAX(oo.created_at) as last_seen
                FROM objects o
                LEFT JOIN object_observations oo ON oo.object_id = o.id{map_filter}
                {where_clause}
                GROUP BY o.id
                HAVING COUNT(oo.id) >= %s
                ORDER BY last_seen DESC NULLS LAST
                LIMIT %s
                """,
                tuple(query_params + [min_count, candidate_limit]),
            )
            ordered_ids = [int(row[0]) for row in cur.fetchall()]

            # If scoped search returns nothing, fall back to unscoped search
            # so the agent can find objects even when the active map doesn't contain them.
            if not ordered_ids and active_map_id is not None:
                cur.execute(
                    f"""
                    SELECT o.id, MAX(oo.created_at) as last_seen
                    FROM objects o
                    LEFT JOIN object_observations oo ON oo.object_id = o.id
                    {where_clause}
                    GROUP BY o.id
                    HAVING COUNT(oo.id) >= %s
                    ORDER BY last_seen DESC NULLS LAST
                    LIMIT %s
                    """,
                    tuple(query_params[len(params):] + [min_count, candidate_limit]),
                )
                ordered_ids = [int(row[0]) for row in cur.fetchall()]
                active_map_id = None  # unscoped snapshot fetch

            rows = []
            for object_id in ordered_ids[:candidate_limit]:
                snapshot = _fetch_object_snapshot(cur, object_id, active_map_id)
                if snapshot is not None and int(snapshot[3] or 0) >= min_count:
                    rows.append(snapshot)

    results = []
    with _get_conn() as conn:
        with conn.cursor() as cur:
            for r in rows:
                x, y, obs_id = r[5], r[6], r[7]
                room = None
                if x is not None and y is not None:
                    room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)

                results.append(
                    {
                        "object_id": r[0],
                        "class_id": r[1],
                        "class_name": _class_name(r[1]),
                        "name": r[2],
                        "observation_count": r[3],
                        "last_seen_at": str(r[4]) if r[4] else None,
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                        "room": room,
                        "observation_id": obs_id,
                    }
                )
                if len(results) >= safe_limit:
                    break

    return {
        "class_id": safe_class_id,
        "class_name": _class_name(safe_class_id) if safe_class_id is not None else None,
        "object_name": requested_name or None,
        "min_observations": min_count,
        "limit": safe_limit,
        "results": results[:safe_limit],
    }


def get_object_interactions(
    object_id: str,
    limit: int | None = None,
    sort_order: str = "desc",
    action: str | None = None,
) -> dict:
    """Return all interactions an object participated in, with co-participants."""
    normalized_sort_order = _normalize_sort_order(sort_order)
    order_direction = "ASC" if normalized_sort_order == "asc" else "DESC"
    safe_limit = None if limit is None else max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    action_filter = str(action or "").strip().lower() or None
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {
                    "object_id": object_id,
                    "interaction_count": 0,
                    "returned_count": 0,
                    "limit": safe_limit,
                    "sort_order": normalized_sort_order,
                    "latest_interaction_at": None,
                    "oldest_interaction_at": None,
                    "action_breakdown": [],
                    "top_co_participants": [],
                    "interactions": [],
                }

            action_clause = " AND lower(i.action) = %s" if action_filter is not None else ""
            limit_clause = " LIMIT %s" if safe_limit is not None else ""

            def _count_and_fetch(with_map: bool) -> tuple[int, list]:
                map_clause = " AND i.map_id = %s" if with_map else ""
                params: list[object] = [object_id, object_id]
                if with_map:
                    params.append(active_map_id)
                if action_filter is not None:
                    params.append(action_filter)
                cur.execute(
                    f"""
                    SELECT COUNT(DISTINCT i.id)
                    FROM interactions i
                    LEFT JOIN object_observations subject_obs ON subject_obs.id = i.subject_id
                    LEFT JOIN object_observations object_obs ON object_obs.id = i.object_id
                    WHERE (subject_obs.object_id = %s OR object_obs.object_id = %s)
                    {map_clause}{action_clause}
                    """,
                    tuple(params),
                )
                count = int(cur.fetchone()[0] or 0)
                query_params = list(params)
                if safe_limit is not None:
                    query_params.append(safe_limit)
                cur.execute(
                    f"""
                    SELECT
                        i.id,
                        i.action,
                        i.caption,
                        i.created_at,
                        subject_obs.object_id AS subject_object_id,
                        subject_obj.class_id AS subject_class_id,
                        subject_obj.name AS subject_name,
                        subject_obs.x AS subject_x,
                        subject_obs.y AS subject_y,
                        object_obs.object_id AS object_object_id,
                        object_obj.class_id AS object_class_id,
                        object_obj.name AS object_name,
                        object_obs.x AS object_x,
                        object_obs.y AS object_y,
                        COALESCE(subject_obs.scene_id, object_obs.scene_id) AS scene_id,
                        s.x AS scene_x,
                        s.y AS scene_y
                    FROM interactions i
                    LEFT JOIN object_observations subject_obs ON subject_obs.id = i.subject_id
                    LEFT JOIN object_observations object_obs ON object_obs.id = i.object_id
                    LEFT JOIN objects subject_obj ON subject_obj.id = subject_obs.object_id
                    LEFT JOIN objects object_obj ON object_obj.id = object_obs.object_id
                    LEFT JOIN scenes s ON s.id = COALESCE(subject_obs.scene_id, object_obs.scene_id)
                    WHERE (subject_obs.object_id = %s OR object_obs.object_id = %s)
                    {map_clause}{action_clause}
                    ORDER BY i.created_at {order_direction}, i.id {order_direction}
                    {limit_clause}
                    """,
                    tuple(query_params),
                )
                return count, cur.fetchall()

            total_count, rows = _count_and_fetch(active_map_id is not None)
            if total_count == 0 and active_map_id is not None:
                # Same fallback as search_objects_by_class_id: when the active map
                # contains no interactions for this object, retry unscoped so
                # cross-map history stays visible to the agent.
                total_count, rows = _count_and_fetch(False)
                if total_count > 0:
                    active_map_id = None

            grouped: dict = {}
            for r in rows:
                iid = r[0]
                if iid not in grouped:
                    room = _resolve_room_name(cur, active_map_name_value, active_map_id, r[15], r[16])
                    grouped[iid] = {
                        "interaction_id": iid,
                        "action": r[1],
                        "caption": r[2],
                        "created_at": str(r[3]),
                        "local_created_at": _to_local_timestamp(r[3]),
                        "scene_id": r[14],
                        "x": float(r[15]) if r[15] is not None else None,
                        "y": float(r[16]) if r[16] is not None else None,
                        "room": room,
                        "co_participants": [],
                    }
                queried_is_subject = str(r[4]) == str(object_id)
                co_id = r[9] if queried_is_subject else r[4]
                co_class_id = r[10] if queried_is_subject else r[5]
                co_name = r[11] if queried_is_subject else r[6]
                co_x = r[12] if queried_is_subject else r[7]
                co_y = r[13] if queried_is_subject else r[8]
                if co_id is not None and str(co_id) != str(object_id):
                    grouped[iid]["co_participants"].append(
                        {
                            "object_id": co_id,
                            "class_id": co_class_id,
                            "class_name": _class_name(co_class_id),
                            "name": co_name,
                            "x": float(co_x) if co_x is not None else None,
                            "y": float(co_y) if co_y is not None else None,
                        }
                    )

    interactions = list(grouped.values())
    action_counts: Counter[str] = Counter()
    participant_counts: Counter[str] = Counter()
    participant_labels: dict[str, dict[str, str | int | None]] = {}

    people_by_action: dict[str, dict[str, dict]] = {}
    for interaction in interactions:
        action = str(interaction.get("action") or "unknown")
        action_counts[action] += 1
        for participant in interaction.get("co_participants") or []:
            participant_key = str(participant.get("object_id") or participant.get("name") or participant.get("class_name") or "unknown")
            participant_counts[participant_key] += 1
            participant_labels.setdefault(
                participant_key,
                {
                    "object_id": participant.get("object_id"),
                    "name": participant.get("name"),
                    "class_name": participant.get("class_name"),
                },
            )
            people_by_action.setdefault(action, {})
            people_by_action[action].setdefault(
                participant_key,
                {
                    "object_id": participant.get("object_id"),
                    "name": participant.get("name"),
                    "class_name": participant.get("class_name"),
                    "count": 0,
                },
            )
            people_by_action[action][participant_key]["count"] += 1

    # If action-filtered query returned nothing, fetch unfiltered action breakdown
    # so the caller knows which actions are actually available.
    if action_filter and total_count == 0:
        unfiltered_params = [object_id, object_id]
        if active_map_id is not None:
            unfiltered_params.append(active_map_id)
        cur.execute(
            f"""
            SELECT i.action, COUNT(DISTINCT i.id)
            FROM interactions i
            LEFT JOIN object_observations subject_obs ON subject_obs.id = i.subject_id
            LEFT JOIN object_observations object_obs ON object_obs.id = i.object_id
            WHERE (subject_obs.object_id = %s OR object_obs.object_id = %s)
            {where_map_clause}
            GROUP BY i.action
            ORDER BY COUNT(DISTINCT i.id) DESC
            """,
            tuple(unfiltered_params),
        )
        available_actions = [row[0] for row in cur.fetchall()]
    else:
        available_actions = None

    result = {
        "object_id": object_id,
        "interaction_count": total_count,
        "returned_count": len(interactions),
        "limit": safe_limit,
        "sort_order": normalized_sort_order,
        "latest_interaction_at": max((item.get("created_at") for item in interactions), default=None),
        "oldest_interaction_at": min((item.get("created_at") for item in interactions), default=None),
        "action_breakdown": [
            {"action": action, "count": count}
            for action, count in action_counts.most_common(5)
        ],
        "top_co_participants": [
            {
                **participant_labels[key],
                "count": count,
            }
            for key, count in participant_counts.most_common(5)
        ],
        "people_by_action": {
            action: [
                {
                    "name": info.get("name"),
                    "class_name": info.get("class_name"),
                    "count": info["count"],
                }
                for _key, info in sorted(participants.items(), key=lambda x: x[1]["count"], reverse=True)[:3]
            ]
            for action, participants in people_by_action.items()
        },
        "interactions": interactions,
    }
    if available_actions:
        result["available_actions"] = available_actions
        result["message"] = f"No interactions with action '{action_filter}'. Available actions: {', '.join(available_actions)}."
    return result


def get_object_person_interactions(
    object_id: str,
    action: str | None = None,
) -> dict:
    """Return all people who interacted with one object, grouped by person and action."""
    history = get_object_interactions(object_id, action=action)
    interactions = history.get("interactions") or []

    people: dict[str, dict] = {}
    action_counts_by_person: dict[str, Counter[str]] = {}

    for interaction in interactions:
        interaction_id = interaction.get("interaction_id")
        interaction_action = str(interaction.get("action") or "unknown")
        created_at = interaction.get("created_at")

        for participant in interaction.get("co_participants") or []:
            if participant.get("class_name") != "person":
                continue

            person_key = str(
                participant.get("object_id")
                or participant.get("name")
                or participant.get("class_name")
                or "unknown"
            )
            if person_key not in people:
                people[person_key] = {
                    "object_id": participant.get("object_id"),
                    "name": participant.get("name"),
                    "class_name": participant.get("class_name"),
                    "interaction_count": 0,
                    "interaction_ids": [],
                    "latest_interaction_at": created_at,
                    "oldest_interaction_at": created_at,
                    "latest_relevant_interaction_id": interaction_id,
                    "latest_relevant_scene_id": interaction.get("scene_id"),
                    "latest_relevant_x": participant.get("x"),
                    "latest_relevant_y": participant.get("y"),
                    "latest_relevant_room": interaction.get("room"),
                }
                action_counts_by_person[person_key] = Counter()

            person_summary = people[person_key]
            person_summary["interaction_count"] += 1
            if interaction_id is not None:
                person_summary["interaction_ids"].append(interaction_id)

            latest_interaction_at = person_summary.get("latest_interaction_at")
            oldest_interaction_at = person_summary.get("oldest_interaction_at")
            if created_at is not None and (latest_interaction_at is None or created_at > latest_interaction_at):
                person_summary["latest_interaction_at"] = created_at
                person_summary["latest_relevant_interaction_id"] = interaction_id
                person_summary["latest_relevant_scene_id"] = interaction.get("scene_id")
                person_summary["latest_relevant_x"] = participant.get("x")
                person_summary["latest_relevant_y"] = participant.get("y")
                person_summary["latest_relevant_room"] = interaction.get("room")
            if created_at is not None and (oldest_interaction_at is None or created_at < oldest_interaction_at):
                person_summary["oldest_interaction_at"] = created_at

            action_counts_by_person[person_key][interaction_action] += 1

    person_interactions = []
    for person_key, person_summary in people.items():
        person_interactions.append(
            {
                **person_summary,
                "actions": [
                    {"action": action, "count": count}
                    for action, count in action_counts_by_person[person_key].most_common()
                ],
            }
        )

    person_interactions.sort(
        key=lambda item: (int(item.get("interaction_count") or 0), str(item.get("latest_interaction_at") or "")),
        reverse=True,
    )

    result = {
        "object_id": object_id,
        "action_filter": action,
        "found": bool(person_interactions),
        "person_count": len(person_interactions),
        "interaction_count": history.get("interaction_count", len(interactions)),
        "latest_interaction_at": history.get("latest_interaction_at"),
        "oldest_interaction_at": history.get("oldest_interaction_at"),
        "person_interactions": person_interactions,
    }
    if history.get("available_actions"):
        result["available_actions"] = history["available_actions"]
        result["message"] = history.get("message")
    return result


def _resolve_class_ids_by_name(class_name_query: str) -> list[int]:
    """Return YOLO class ids whose class_name matches or contains the query."""
    normalized = str(class_name_query or "").strip().lower()
    if not normalized:
        return []
    # Exact match first
    for cid, cname in YOLO_CLASS_NAMES.items():
        if cname.lower() == normalized:
            return [cid]
    # Substring match
    matches = []
    for cid, cname in YOLO_CLASS_NAMES.items():
        cname_lower = cname.lower()
        if normalized in cname_lower or cname_lower in normalized:
            matches.append(cid)
    return matches


def get_interaction_events(
    object_id: str | None = None,
    person_name: str | None = None,
    other_person_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: int = DEFAULT_INTERACTION_EVENT_LIMIT,
    sort_order: str = "desc",
    room_name: str | None = None,
    action: str | None = None,
    co_participant_class_name: str | None = None,
) -> dict:
    """Return enriched interaction events filtered by object ids and/or tracked object names.
    
    co_participant_class_name: If provided, only return events where a participant (other than
    the primary queried entity) has a matching class_name. Supports partial matching against
    the YOLO class list (e.g., "ball" matches "sports ball").
    """
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    action_filter = str(action or "").strip().lower() or None
    timestamp_filter_clause, timestamp_filter_params, normalized_start_time, normalized_end_time = (
        _build_time_range_filters("i.created_at", start_time, end_time)
    )
    _ensure_rooms_schema()
    _ensure_objects_schema()
    active_map_name = _active_map_name()
    co_participant_class_ids = _resolve_class_ids_by_name(co_participant_class_name) if co_participant_class_name else []

    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {
                    "found": False,
                    "object_id": object_id,
                    "person_name": person_name,
                    "other_person_name": other_person_name,
                    "start_time": normalized_start_time,
                    "end_time": normalized_end_time,
                    "limit": safe_limit,
                    "events": [],
                }

            person_rows = _lookup_named_object_ids(cur, person_name) if person_name else []
            other_person_rows = _lookup_named_object_ids(cur, other_person_name) if other_person_name else []
            person_ids = [row[0] for row in person_rows]
            other_person_ids = [row[0] for row in other_person_rows]

            if person_name and not person_ids:
                return {"found": False, "error": f"Unknown tracked object/person '{person_name}'.", "events": []}
            if other_person_name and not other_person_ids:
                return {"found": False, "error": f"Unknown tracked object/person '{other_person_name}'.", "events": []}

            resolved_object_name = None
            if object_id is not None:
                object_id_text = str(object_id).strip()
                if not re.fullmatch(r"\d+", object_id_text):
                    object_rows = _lookup_named_object_ids(cur, object_id_text)
                    if not object_rows:
                        return {
                            "found": False,
                            "object_id": object_id,
                            "error": (
                                f"Unknown tracked object/person '{object_id_text}'. "
                                "Call search_objects_by_class_id with object_name first."
                            ),
                            "events": [],
                        }
                    object_id = str(object_rows[0][0])
                    resolved_object_name = object_rows[0][1]

            where_clauses: list[str] = []
            params: list = []

            if object_id is not None:
                where_clauses.append("(subject_obs.object_id = %s OR object_obs.object_id = %s)")
                params.extend([object_id, object_id])

            if person_ids:
                where_clauses.append("(subject_obs.object_id = ANY(%s) OR object_obs.object_id = ANY(%s))")
                params.extend([person_ids, person_ids])

            if other_person_ids:
                where_clauses.append("(subject_obs.object_id = ANY(%s) OR object_obs.object_id = ANY(%s))")
                params.extend([other_person_ids, other_person_ids])
            if active_map_id is not None:
                where_clauses.append("i.map_id = %s")
                params.append(active_map_id)

            query = """
                SELECT
                    i.id,
                    i.action,
                    i.caption,
                    i.created_at,
                    subject_obs.object_id AS subject_object_id,
                    subject_obj.class_id AS subject_class_id,
                    subject_obj.name AS subject_name,
                    subject_obs.x AS subject_x,
                    subject_obs.y AS subject_y,
                    object_obs.object_id AS object_object_id,
                    object_obj.class_id AS object_class_id,
                    object_obj.name AS object_name,
                    object_obs.x AS object_x,
                    object_obs.y AS object_y,
                    COALESCE(subject_obs.scene_id, object_obs.scene_id) AS scene_id,
                    s.x,
                    s.y
                FROM interactions i
                LEFT JOIN object_observations subject_obs ON subject_obs.id = i.subject_id
                LEFT JOIN object_observations object_obs ON object_obs.id = i.object_id
                LEFT JOIN objects subject_obj ON subject_obj.id = subject_obs.object_id
                LEFT JOIN objects object_obj ON object_obj.id = object_obs.object_id
                LEFT JOIN scenes s ON s.id = COALESCE(subject_obs.scene_id, object_obs.scene_id)
            """
            combined_where_clauses = list(where_clauses)
            if timestamp_filter_clause:
                combined_where_clauses.append(timestamp_filter_clause.removeprefix(" AND "))
            if action_filter is not None:
                combined_where_clauses.append("lower(i.action) = %s")

            if combined_where_clauses:
                query += " WHERE " + " AND ".join(combined_where_clauses)
            params.extend(timestamp_filter_params)
            if action_filter is not None:
                params.append(action_filter)

            normalized_sort_order = _normalize_sort_order(sort_order)
            order_direction = "ASC" if normalized_sort_order == "asc" else "DESC"
            # For exact HH:MM queries, fetch extra rows so action-priority post-sort can surface the best match
            is_exact_hhmm = (
                normalized_start_time == normalized_end_time
                and normalized_start_time is not None
                and re.fullmatch(r"\d{2}:\d{2}", normalized_start_time) is not None
            )
            sql_limit = max(safe_limit * 3, 20) if is_exact_hhmm else safe_limit
            # When room_name is specified we post-filter; fetch more rows so the filter
            # does not accidentally discard matching events that sit just beyond the limit.
            if room_name:
                sql_limit = max(sql_limit * 5, 50)
            # Same for co_participant_class_name filtering
            if co_participant_class_name:
                sql_limit = max(sql_limit * 5, 100)
            query += f" ORDER BY i.created_at {order_direction}, i.id {order_direction} LIMIT %s"
            params.append(sql_limit)

            cur.execute(query, params)
            rows = cur.fetchall()

            events = []
            for row in rows:
                # Use scene coordinates if available; fall back to subject coordinates
                scene_x, scene_y = row[15], row[16]
                if scene_x is None and row[7] is not None:
                    scene_x = row[7]
                if scene_y is None and row[8] is not None:
                    scene_y = row[8]
                room = _resolve_room_name(cur, active_map_name, active_map_id, scene_x, scene_y)
                participants = []
                for object_id_value, class_id_value, name_value, pos_x, pos_y, role in (
                    (row[4], row[5], row[6], row[7], row[8], "subject"),
                    (row[9], row[10], row[11], row[12], row[13], "object"),
                ):
                    if object_id_value is None:
                        continue
                    participants.append(
                        {
                            "object_id": object_id_value,
                            "class_id": class_id_value,
                            "class_name": _class_name(class_id_value),
                            "name": name_value,
                            "x": float(pos_x) if pos_x is not None else None,
                            "y": float(pos_y) if pos_y is not None else None,
                            "role": role,
                        }
                    )

                if person_ids:
                    participant_ids = {participant["object_id"] for participant in participants}
                    if not participant_ids.intersection(person_ids):
                        continue
                if other_person_ids:
                    participant_ids = {participant["object_id"] for participant in participants}
                    if not participant_ids.intersection(other_person_ids):
                        continue

                person_participants = [
                    participant for participant in participants if participant.get("class_name") == "person"
                ]
                object_participants = [
                    participant for participant in participants if participant.get("class_name") != "person"
                ]
                events.append(
                    {
                        "interaction_id": row[0],
                        "action": row[1],
                        "caption": row[2],
                        "created_at": str(row[3]) if row[3] else None,
                        "local_created_at": _to_local_timestamp(row[3]),
                        "matched_time_filter": normalized_start_time == normalized_end_time and normalized_start_time is not None,
                        "scene_id": row[14],
                        "room": room,
                        "x": float(scene_x) if scene_x is not None else None,
                        "y": float(scene_y) if scene_y is not None else None,
                        "participants": participants,
                        "person_participants": person_participants,
                        "object_participants": object_participants,
                    }
                )

    if room_name:
        target = str(room_name).strip().lower()
        events = [e for e in events if str(e.get("room") or "").lower() == target]

    if co_participant_class_name:
        target = str(co_participant_class_name).strip().lower()
        primary_ids = set()
        if object_id is not None:
            primary_ids.add(str(object_id))
        if person_ids:
            primary_ids.update(str(pid) for pid in person_ids)
        if other_person_ids:
            primary_ids.update(str(pid) for pid in other_person_ids)
        filtered_events = []
        for event in events:
            for participant in event.get("participants", []):
                if str(participant.get("object_id")) in primary_ids:
                    continue
                p_class = str(participant.get("class_name") or "").lower()
                if target in p_class or p_class in target:
                    filtered_events.append(event)
                    break
        events = filtered_events

    # For exact HH:MM queries, prioritize more specific actions so the first event is the best answer
    if (
        normalized_start_time == normalized_end_time
        and normalized_start_time is not None
        and re.fullmatch(r"\d{2}:\d{2}", normalized_start_time) is not None
    ):
        _action_priority = {
            "holding": 10, "using": 10, "talking_to": 10,
            "putting": 9, "placeing": 9, "loading": 9,
            "opening": 5, "closing": 5, "pick_up": 8, "put_down": 8,
            "next_to": 1, "looking_at": 2,
        }
        events.sort(
            key=lambda e: (-_action_priority.get(str(e.get("action") or "").lower(), 0), -(e.get("interaction_id") or 0)),
        )

    # Trim to requested limit after all filtering and sorting
    events = events[:safe_limit]

    return {
        "found": bool(events),
        "object_id": object_id,
        "resolved_object_name": resolved_object_name,
        "person_name": person_name,
        "other_person_name": other_person_name,
        "co_participant_class_name": co_participant_class_name,
        "start_time": normalized_start_time,
        "end_time": normalized_end_time,
        "action_filter": action,
        "limit": safe_limit,
        "events": events,
    }


def get_object_last_location(object_id: str, require_navigation_usable: bool = False) -> dict:
    """Return the most recent x/y position and room for an object.
    
    If require_navigation_usable is True, only return positions backed by
    depth or DynoSAM (position_source in {'depth', 'dynosam'}).
    """
    _ensure_rooms_schema()
    _ensure_objects_schema()
    active_map_name = _active_map_name()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"object_id": object_id, "found": False}

            map_clause = " AND oo.map_id = %s" if active_map_id is not None else ""
            depth_clause = (
                " AND (oo.position_source IS NULL OR oo.position_source IN ('depth', 'dynosam'))"
                if require_navigation_usable
                else ""
            )
            params = [object_id]
            if active_map_id is not None:
                params.append(active_map_id)
            cur.execute(
                f"""
                SELECT oo.x, oo.y, oo.z, oo.created_at, s.caption, o.name, oo.position_source
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.object_id = %s AND oo.x IS NOT NULL AND oo.y IS NOT NULL {map_clause}{depth_clause}
                ORDER BY oo.created_at DESC
                LIMIT 1
                """,
                tuple(params),
            )
            row = cur.fetchone()
            if not row:
                return {"object_id": object_id, "found": False}

            x, y, z, ts, scene_caption, name, position_source = row

            room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)

    return {
        "object_id": object_id,
        "name": name,
        "found": True,
        "x": float(x),
        "y": float(y),
        "z": float(z) if z is not None else None,
        "last_seen_at": str(ts),
        "room": room or "unknown",
        "scene_caption": scene_caption,
        "position_source": position_source,
        "navigation_usable": _is_navigation_usable_position_source(position_source),
    }


def get_object_first_location(object_id: str) -> dict:
    """Return the earliest recorded x/y position and room for an object."""
    _ensure_rooms_schema()
    _ensure_objects_schema()
    active_map_name = _active_map_name()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"object_id": object_id, "found": False}

            map_clause = " AND oo.map_id = %s" if active_map_id is not None else ""
            params = [object_id]
            if active_map_id is not None:
                params.append(active_map_id)
            cur.execute(
                f"""
                SELECT oo.x, oo.y, oo.z, oo.created_at, s.caption, o.name
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.object_id = %s AND oo.x IS NOT NULL AND oo.y IS NOT NULL {map_clause}
                ORDER BY oo.created_at ASC
                LIMIT 1
                """,
                tuple(params),
            )
            row = cur.fetchone()
            if not row:
                return {"object_id": object_id, "found": False}

            x, y, z, ts, scene_caption, name = row

            room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)

    return {
        "object_id": object_id,
        "name": name,
        "found": True,
        "x": float(x),
        "y": float(y),
        "z": float(z) if z is not None else None,
        "first_seen_at": str(ts),
        "room": room or "unknown",
        "scene_caption": scene_caption,
    }


def _yaw_from_quaternion(q: dict) -> float:
    x = float(q.get("x", 0.0) or 0.0)
    y = float(q.get("y", 0.0) or 0.0)
    z = float(q.get("z", 0.0) or 0.0)
    w = float(q.get("w", 1.0) or 1.0)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def get_robot_location() -> dict:
    """Return the robot's latest live pose from the SLAM pose file.

    Falls back to the toolbox map snapshot if the live pose is unavailable.
    """
    live_pose = _load_json_path(SLAM_TAB_ROBOT_POSE_PATH)
    if isinstance(live_pose, dict) and live_pose.get("position") is not None:
        pos = live_pose.get("position") or {}
        orient = live_pose.get("orientation") or {}
        x = pos.get("x")
        y = pos.get("y")
        if x is not None and y is not None:
            _ensure_rooms_schema()
            room = "unknown"
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    active_map_name, active_map_id, _scope_requested = _get_active_map_scope(cur)
                    resolved_room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)
                    if resolved_room:
                        room = str(resolved_room)
            return {
                "found": True,
                "x": float(x),
                "y": float(y),
                "z": float(pos.get("z", 0.0) or 0.0),
                "yaw": _yaw_from_quaternion(orient),
                "room": room,
                "map_name": active_map_name,
                "source": str(SLAM_TAB_ROBOT_POSE_PATH),
            }

    payload = _load_json_path(TOOLBOX_MAP_PATH)
    if payload is None:
        return {
            "found": False,
            "error": (
                f"Robot location is unavailable. "
                f"Live pose not found at {SLAM_TAB_ROBOT_POSE_PATH} "
                f"and toolbox snapshot not found at {TOOLBOX_MAP_PATH}."
            ),
        }

    robot_pose = payload.get("robot")
    if not isinstance(robot_pose, dict):
        return {
            "found": False,
            "error": "Robot pose is not present in the current toolbox map snapshot.",
        }

    x = robot_pose.get("x")
    y = robot_pose.get("y")
    if x is None or y is None:
        return {
            "found": False,
            "error": "Robot pose does not contain usable x/y coordinates.",
        }

    _ensure_rooms_schema()
    room = "unknown"

    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, _scope_requested = _get_active_map_scope(cur)
            resolved_room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)
            if resolved_room:
                room = str(resolved_room)

    return {
        "found": True,
        "x": float(x),
        "y": float(y),
        "z": float(robot_pose.get("z", 0.0) or 0.0),
        "yaw": float(robot_pose.get("yaw", 0.0) or 0.0),
        "room": room,
        "map_name": active_map_name,
        "source": str(TOOLBOX_MAP_PATH),
    }


def list_objects(
    class_name: str = None,
    min_observations: int = MIN_OBJECT_OBSERVATIONS,
    limit: int = 20,
) -> dict:
    """List tracked objects, optionally filtered by class name."""
    min_count = max(1, int(min_observations))
    safe_limit = max(1, min(int(limit), 50))
    _ensure_objects_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"filter": class_name, "min_observations": min_count, "limit": safe_limit, "objects": []}

            obs_join = "LEFT JOIN object_observations oo ON oo.object_id = o.id"
            params_prefix: list = []
            if active_map_id is not None:
                obs_join += " AND oo.map_id = %s"
                params_prefix.append(active_map_id)

            if class_name:
                matching_ids = [
                    cid for cid, name in YOLO_CLASS_NAMES.items() if name in class_name.lower()
                ]
                params = list(params_prefix)
                params.extend([matching_ids, f"%{class_name}%", min_count])
                cur.execute(
                    f"""
                    SELECT o.id, o.class_id, o.name, COUNT(oo.id), MAX(oo.created_at)
                    FROM objects o
                    {obs_join}
                    WHERE o.class_id = ANY(%s) OR o.name ILIKE %s
                    GROUP BY o.id, o.class_id, o.name
                    HAVING COUNT(oo.id) >= %s
                    ORDER BY MAX(oo.created_at) DESC NULLS LAST
                    LIMIT %s
                    """,
                    tuple(params + [safe_limit]),
                )
            else:
                params = list(params_prefix)
                params.extend([min_count, safe_limit])
                cur.execute(
                    f"""
                    SELECT o.id, o.class_id, o.name, COUNT(oo.id), MAX(oo.created_at)
                    FROM objects o
                    {obs_join}
                    GROUP BY o.id, o.class_id, o.name
                    HAVING COUNT(oo.id) >= %s
                    ORDER BY MAX(oo.created_at) DESC NULLS LAST
                    LIMIT %s
                    """,
                    tuple(params),
                )
            rows = cur.fetchall()

    return {
        "filter": class_name,
        "min_observations": min_count,
        "limit": safe_limit,
        "objects": [
            {
                "object_id": r[0],
                "class_id": r[1],
                "class_name": _class_name(r[1]),
                "name": r[2],
                "observation_count": r[3],
                "last_seen_at": str(r[4]) if r[4] else None,
            }
            for r in rows
        ],
    }


def get_room_exploration_status(limit: int = 10) -> dict:
    """Return rooms ordered by oldest last-seen scene first, so agents can pick stale places to explore."""
    safe_limit = max(1, min(int(limit), 100))
    _ensure_rooms_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"limit": safe_limit, "rooms": []}
            room_scope_clause, room_scope_params = _room_scope_clause("r", active_map_name, active_map_id)
            if room_scope_clause:
                cur.execute(
                    f"""
                SELECT
                    r.id,
                    r.name,
                    r.x1,
                    r.y1,
                    r.x2,
                    r.y2,
                    MAX(s.timestamp) AS last_seen_at,
                    COUNT(s.id) AS scene_count
                FROM rooms r
                LEFT JOIN scenes s
                    ON s.map_id = %s
                   AND s.x BETWEEN r.x1 AND r.x2
                   AND s.y BETWEEN r.y1 AND r.y2
                WHERE {room_scope_clause}
                GROUP BY r.id, r.name, r.x1, r.y1, r.x2, r.y2
                ORDER BY MAX(s.timestamp) ASC NULLS FIRST, r.id ASC
                LIMIT %s
                """,
                    tuple([active_map_id] + room_scope_params + [safe_limit]),
                )
            else:
                cur.execute(
                    """
                SELECT
                    r.id,
                    r.name,
                    r.x1,
                    r.y1,
                    r.x2,
                    r.y2,
                    MAX(s.timestamp) AS last_seen_at,
                    COUNT(s.id) AS scene_count
                FROM rooms r
                LEFT JOIN scenes s
                    ON s.x BETWEEN r.x1 AND r.x2
                   AND s.y BETWEEN r.y1 AND r.y2
                GROUP BY r.id, r.name, r.x1, r.y1, r.x2, r.y2
                ORDER BY MAX(s.timestamp) ASC NULLS FIRST, r.id ASC
                LIMIT %s
                """,
                    (safe_limit,),
                )
            rows = cur.fetchall()

    return {
        "limit": safe_limit,
        "rooms": [
            {
                "room_id": row[0],
                "name": row[1],
                "x1": float(row[2]),
                "y1": float(row[3]),
                "x2": float(row[4]),
                "y2": float(row[5]),
                "last_seen_at": str(row[6]) if row[6] else None,
                "scene_count": int(row[7] or 0),
                "status": "never_seen" if row[6] is None else "seen_before",
            }
            for row in rows
        ],
    }


def get_interaction_map_summary(limit: int = 10) -> dict:
    """Return per-room interaction density so agents can compare hotspots with quieter areas."""
    safe_limit = max(1, min(int(limit), 100))
    _ensure_rooms_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"limit": safe_limit, "rooms": []}
            room_scope_clause, room_scope_params = _room_scope_clause("r", active_map_name, active_map_id)
            if room_scope_clause:
                cur.execute(
                    f"""
                SELECT
                    r.id,
                    r.name,
                    r.x1,
                    r.y1,
                    r.x2,
                    r.y2,
                    COUNT(i.id) AS interaction_count,
                    MAX(i.created_at) AS last_interaction_at
                FROM rooms r
                LEFT JOIN scenes s
                    ON s.map_id = %s
                   AND s.x BETWEEN r.x1 AND r.x2
                   AND s.y BETWEEN r.y1 AND r.y2
                LEFT JOIN interactions i
                    ON i.id IN (
                        SELECT i2.id
                        FROM interactions i2
                        LEFT JOIN object_observations subject_obs ON subject_obs.id = i2.subject_id
                        LEFT JOIN object_observations object_obs ON object_obs.id = i2.object_id
                        WHERE i2.map_id = %s
                          AND COALESCE(subject_obs.scene_id, object_obs.scene_id) = s.id
                    )
                WHERE {room_scope_clause}
                GROUP BY r.id, r.name, r.x1, r.y1, r.x2, r.y2
                ORDER BY COUNT(i.id) DESC, r.id ASC
                LIMIT %s
                """,
                    tuple([active_map_id, active_map_id] + room_scope_params + [safe_limit]),
                )
            else:
                cur.execute(
                    """
                SELECT
                    r.id,
                    r.name,
                    r.x1,
                    r.y1,
                    r.x2,
                    r.y2,
                    COUNT(i.id) AS interaction_count,
                    MAX(i.created_at) AS last_interaction_at
                FROM rooms r
                LEFT JOIN scenes s
                    ON s.x BETWEEN r.x1 AND r.x2
                   AND s.y BETWEEN r.y1 AND r.y2
                LEFT JOIN interactions i
                    ON i.id IN (
                        SELECT i2.id
                        FROM interactions i2
                        LEFT JOIN object_observations subject_obs ON subject_obs.id = i2.subject_id
                        LEFT JOIN object_observations object_obs ON object_obs.id = i2.object_id
                        WHERE COALESCE(subject_obs.scene_id, object_obs.scene_id) = s.id
                    )
                GROUP BY r.id, r.name, r.x1, r.y1, r.x2, r.y2
                ORDER BY COUNT(i.id) DESC, r.id ASC
                LIMIT %s
                """,
                    (safe_limit,),
                )
            rows = cur.fetchall()

    rooms = [
        {
            "room_id": row[0],
            "name": row[1],
            "x1": float(row[2]),
            "y1": float(row[3]),
            "x2": float(row[4]),
            "y2": float(row[5]),
            "interaction_count": int(row[6] or 0),
            "last_interaction_at": str(row[7]) if row[7] else None,
        }
        for row in rows
    ]

    if rooms:
        counts = [room["interaction_count"] for room in rooms]
        max_count = max(counts)
        min_count = min(counts)
        for room in rooms:
            if room["interaction_count"] == max_count:
                room["density_label"] = "high"
            elif room["interaction_count"] == min_count:
                room["density_label"] = "low"
            else:
                room["density_label"] = "medium"

    return {
        "limit": safe_limit,
        "rooms": rooms,
    }


def _is_cell_reachable(data, width, height, free_threshold, start_col, start_row, goal_col, goal_row, max_visited=8000):
    """BFS from start to goal on the occupancy grid. Returns True if goal is reachable."""
    if (start_col, start_row) == (goal_col, goal_row):
        return True
    if not (0 <= start_col < width and 0 <= start_row < height):
        return False
    if not (0 <= goal_col < width and 0 <= goal_row < height):
        return False
    start_v = int(data[start_row * width + start_col])
    if start_v < 0 or start_v > free_threshold:
        return False
    goal_v = int(data[goal_row * width + goal_col])
    if goal_v < 0 or goal_v > free_threshold:
        return False
    frontier = [(start_col, start_row)]
    visited = {(start_col, start_row)}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while frontier:
        col, row = frontier.pop(0)
        for dc, dr in neighbors:
            nc, nr = col + dc, row + dr
            if not (0 <= nc < width and 0 <= nr < height):
                continue
            if (nc, nr) in visited:
                continue
            v = int(data[nr * width + nc])
            if v < 0 or v > free_threshold:
                continue
            if (nc, nr) == (goal_col, goal_row):
                return True
            visited.add((nc, nr))
            frontier.append((nc, nr))
            if len(visited) > max_visited:
                return False
    return False


def _snap_room_target_to_free_cell(cx: float, cy: float, min_x: float, min_y: float, max_x: float, max_y: float) -> tuple[float, float, str]:
    """Read the toolbox occupancy map and find a navigable point inside the room that is reachable from the robot."""
    try:
        payload = _load_json_path(TOOLBOX_MAP_PATH)
        if not payload or not isinstance(payload, dict):
            return cx, cy, "no_map"
        map_data = payload.get("map") or {}
        resolution = float(map_data.get("resolution") or 0.0)
        width = int(map_data.get("width") or 0)
        height = int(map_data.get("height") or 0)
        origin = map_data.get("origin") or {}
        origin_x = float(origin.get("x") or 0.0)
        origin_y = float(origin.get("y") or 0.0)
        data = list(map_data.get("data") or [])
        if resolution <= 0 or width <= 0 or height <= 0 or len(data) != width * height:
            return cx, cy, "invalid_map"

        free_threshold = 35
        max_snap_distance_m = 1.5
        snap_radius_cells = max(1, int(math.ceil(max_snap_distance_m / resolution)))

        # Get robot position
        robot_pose = _load_json_path(SLAM_TAB_ROBOT_POSE_PATH)
        robot_col = robot_row = None
        rx = ry = None
        if robot_pose and isinstance(robot_pose, dict):
            position = robot_pose.get("position") or {}
            rx = position.get("x")
            ry = position.get("y")
            if rx is not None and ry is not None:
                robot_col = int(math.floor((float(rx) - origin_x) / resolution))
                robot_row = int(math.floor((float(ry) - origin_y) / resolution))

        def _grid_xy(x, y):
            return int(math.floor((x - origin_x) / resolution)), int(math.floor((y - origin_y) / resolution))

        def _is_free(x, y):
            c, r = _grid_xy(x, y)
            if 0 <= c < width and 0 <= r < height:
                v = int(data[r * width + c])
                return 0 <= v <= free_threshold
            return False

        def _is_reachable(x, y):
            if robot_col is None or robot_row is None:
                return True  # No robot pose known, assume reachable
            gc, gr = _grid_xy(x, y)
            return _is_cell_reachable(data, width, height, free_threshold, robot_col, robot_row, gc, gr)

        # 0. Try the closest point on the room boundary (entry point)
        if rx is not None and ry is not None:
            rx_f = float(rx)
            ry_f = float(ry)
            # If inside the room, find the nearest edge; if outside, clamp to boundary
            if min_x <= rx_f <= max_x and min_y <= ry_f <= max_y:
                dist_left = rx_f - min_x
                dist_right = max_x - rx_f
                dist_bottom = ry_f - min_y
                dist_top = max_y - ry_f
                min_dist = min(dist_left, dist_right, dist_bottom, dist_top)
                if min_dist == dist_left:
                    ex, ey = min_x, ry_f
                elif min_dist == dist_right:
                    ex, ey = max_x, ry_f
                elif min_dist == dist_bottom:
                    ex, ey = rx_f, min_y
                else:
                    ex, ey = rx_f, max_y
            else:
                ex = max(min_x, min(max_x, rx_f))
                ey = max(min_y, min(max_y, ry_f))
            if _is_free(ex, ey) and _is_reachable(ex, ey):
                return ex, ey, "entry"

        # 1. Try the exact center first
        if _is_free(cx, cy) and _is_reachable(cx, cy):
            return cx, cy, "center"

        # 2. Generate a grid of candidate points inside the room, slightly inset from edges
        inset_x = max(0.15, (max_x - min_x) * 0.1)
        inset_y = max(0.15, (max_y - min_y) * 0.1)
        candidates = []
        for fx in [0.2, 0.35, 0.5, 0.65, 0.8]:
            for fy in [0.2, 0.35, 0.5, 0.65, 0.8]:
                px = min_x + inset_x + fx * (max_x - min_x - 2 * inset_x)
                py = min_y + inset_y + fy * (max_y - min_y - 2 * inset_y)
                px = max(min_x, min(max_x, px))
                py = max(min_y, min(max_y, py))
                dist_to_center = math.hypot(px - cx, py - cy)
                candidates.append((dist_to_center, px, py))

        # Sort by distance to center (prefer center-most candidates)
        candidates.sort(key=lambda t: t[0])

        # 3. Check each candidate; if free AND reachable, return it
        for _, px, py in candidates:
            if _is_free(px, py) and _is_reachable(px, py):
                return px, py, "candidate"

        # 4. Fallback: search around each candidate for nearest free+reachable cell
        best_free = None
        best_free_dist = None
        for _, px, py in candidates:
            pcol, prow = _grid_xy(px, py)
            local_best = None
            local_best_dist = None
            for radius in range(1, snap_radius_cells + 1):
                for r in range(max(0, prow - radius), min(height, prow + radius + 1)):
                    for c in range(max(0, pcol - radius), min(width, pcol + radius + 1)):
                        if max(abs(c - pcol), abs(r - prow)) != radius:
                            continue
                        v = int(data[r * width + c])
                        if v < 0 or v > free_threshold:
                            continue
                        if robot_col is not None and robot_row is not None:
                            if not _is_cell_reachable(data, width, height, free_threshold, robot_col, robot_row, c, r, max_visited=2000):
                                continue
                        d = math.hypot(c - pcol, r - prow)
                        if local_best is None or d < local_best_dist:
                            local_best = (c, r)
                            local_best_dist = d
                if local_best is not None:
                    break
            if local_best is not None:
                wx = origin_x + (local_best[0] + 0.5) * resolution
                wy = origin_y + (local_best[1] + 0.5) * resolution
                d_center = math.hypot(wx - cx, wy - cy)
                if best_free is None or d_center < best_free_dist:
                    best_free = (wx, wy)
                    best_free_dist = d_center

        if best_free is not None:
            return best_free[0], best_free[1], "snapped"
        return cx, cy, "unreachable"
    except Exception:
        return cx, cy, "error"


def get_room_navigation_target(room_name: str) -> dict:
    """Return the entry point of a named room so the robot can navigate there."""
    normalized_name = " ".join(str(room_name or "").split())
    if not normalized_name:
        return {"found": False, "error": "room_name is required"}
    _ensure_rooms_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, _scope_requested = _get_active_map_scope(cur)
            room_scope_clause, room_scope_params = _room_scope_clause("rooms", active_map_name, active_map_id)
            selects = "id, name, x1, y1, x2, y2, x3, y3, x4, y4, entry_x, entry_y"
            if room_scope_clause:
                cur.execute(
                    f"""
                SELECT {selects}
                FROM rooms
                WHERE {room_scope_clause} AND lower(name) = lower(%s)
                ORDER BY id ASC
                LIMIT 1
                """,
                    tuple(room_scope_params + [normalized_name]),
                )
            else:
                cur.execute(
                    f"""
                SELECT {selects}
                FROM rooms
                WHERE lower(name) = lower(%s)
                ORDER BY id ASC
                LIMIT 1
                """,
                    (normalized_name,),
                )
            row = cur.fetchone()

            if row is None:
                if room_scope_clause:
                    cur.execute(
                        f"""
                    SELECT {selects}
                    FROM rooms
                    WHERE {room_scope_clause} AND lower(name) LIKE lower(%s)
                    ORDER BY
                        CASE WHEN lower(name) LIKE lower(%s) THEN 0 ELSE 1 END,
                        id ASC
                    LIMIT 1
                    """,
                        tuple(room_scope_params + [f"%{normalized_name}%", f"{normalized_name}%"]),
                    )
                else:
                    cur.execute(
                        f"""
                    SELECT {selects}
                    FROM rooms
                    WHERE lower(name) LIKE lower(%s)
                    ORDER BY
                        CASE WHEN lower(name) LIKE lower(%s) THEN 0 ELSE 1 END,
                        id ASC
                    LIMIT 1
                    """,
                        (f"%{normalized_name}%", f"{normalized_name}%"),
                    )
                row = cur.fetchone()

            # Fuzzy token fallback: try each significant word as a substring
            if row is None:
                generic_words = {"room", "office", "area", "space", "the", "and"}
                words = [w for w in normalized_name.split() if len(w) >= 3 and w.lower() not in generic_words]
                # Also try input with common suffixes stripped
                stripped = normalized_name
                for suffix in (" room", " office", " area", " space"):
                    if stripped.endswith(suffix):
                        stripped = stripped[: -len(suffix)]
                        break
                if stripped != normalized_name and len(stripped) >= 3:
                    words.insert(0, stripped)
                for word in words:
                    like_word = f"%{word}%"
                    if room_scope_clause:
                        cur.execute(
                            f"""
                        SELECT {selects}
                        FROM rooms
                        WHERE {room_scope_clause} AND lower(name) LIKE lower(%s)
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                            tuple(room_scope_params + [like_word]),
                        )
                    else:
                        cur.execute(
                            f"""
                        SELECT {selects}
                        FROM rooms
                        WHERE lower(name) LIKE lower(%s)
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                            (like_word,),
                        )
                    row = cur.fetchone()
                    if row is not None:
                        break

            # Final fallback: search all rooms regardless of map scope
            if row is None and room_scope_clause:
                cur.execute(
                    f"""
                SELECT {selects}
                FROM rooms
                WHERE lower(name) = lower(%s)
                ORDER BY id ASC
                LIMIT 1
                """,
                    (normalized_name,),
                )
                row = cur.fetchone()
                if row is None:
                    cur.execute(
                        f"""
                    SELECT {selects}
                    FROM rooms
                    WHERE lower(name) LIKE lower(%s)
                    ORDER BY
                        CASE WHEN lower(name) LIKE lower(%s) THEN 0 ELSE 1 END,
                        id ASC
                    LIMIT 1
                        """,
                        (f"%{normalized_name}%", f"{normalized_name}%"),
                    )
                    row = cur.fetchone()
                # Also try fuzzy across all maps
                if row is None:
                    generic_words = {"room", "office", "area", "space", "the", "and"}
                    words = [w for w in normalized_name.split() if len(w) >= 3 and w.lower() not in generic_words]
                    stripped = normalized_name
                    for suffix in (" room", " office", " area", " space"):
                        if stripped.endswith(suffix):
                            stripped = stripped[: -len(suffix)]
                            break
                    if stripped != normalized_name and len(stripped) >= 3:
                        words.insert(0, stripped)
                    for word in words:
                        cur.execute(
                            f"""
                        SELECT {selects}
                        FROM rooms
                        WHERE lower(name) LIKE lower(%s)
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                            (f"%{word}%",),
                        )
                        row = cur.fetchone()
                        if row is not None:
                            break

    if row is None:
        return {
            "found": False,
            "room_name": normalized_name,
            "error": f"Could not find a room named '{normalized_name}'.",
        }

    room = {
        "id": row[0],
        "name": row[1],
        "x1": row[2],
        "y1": row[3],
        "x2": row[4],
        "y2": row[5],
        "x3": row[6],
        "y3": row[7],
        "x4": row[8],
        "y4": row[9],
        "entry_x": row[10],
        "entry_y": row[11],
    }

    robot_pose = _load_json_path(SLAM_TAB_ROBOT_POSE_PATH)
    if robot_pose and isinstance(robot_pose, dict):
        position = robot_pose.get("position") or {}
        robot_pose = {"x": position.get("x"), "y": position.get("y")}

    interior_offset = 0.0 if normalized_name.lower() == "hallway" else 1.0
    nx, ny = room_entry_point(room, robot_pose, interior_offset_m=interior_offset)
    if nx is None or ny is None:
        return {
            "found": False,
            "room_name": normalized_name,
            "error": f"Could not calculate entry point for '{normalized_name}'.",
        }

    yaw = room_approach_yaw(room, nx, ny)
    min_x, min_y, max_x, max_y = _room_bounds_from_row(row, start_idx=2)
    return {
        "found": True,
        "room_id": row[0],
        "room_name": row[1],
        "x": nx,
        "y": ny,
        "yaw": yaw,
        "x1": min_x,
        "y1": min_y,
        "x2": max_x,
        "y2": max_y,
        "selection_strategy": "entry",
    }


def get_objects_in_room(room_name: str, limit: int = 20) -> dict:
    """Return tracked objects and people whose latest known position is inside a named room."""
    normalized_name = " ".join(str(room_name or "").split())
    safe_limit = max(1, min(int(limit), 100))
    if not normalized_name:
        return {"found": False, "error": "room_name is required", "objects": []}
    _ensure_rooms_schema()
    _ensure_objects_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, _scope_requested = _get_active_map_scope(cur)
            room_scope_clause, room_scope_params = _room_scope_clause("rooms", active_map_name, active_map_id)
            room_row = None
            for scoped in [True, False]:
                clause = room_scope_clause if scoped else ""
                params = list(room_scope_params) if scoped else []
                if clause:
                    cur.execute(
                        f"""
                        SELECT id, name, x1, y1, x2, y2
                        FROM rooms
                        WHERE {clause} AND lower(name) = lower(%s)
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                        tuple(params + [normalized_name]),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, name, x1, y1, x2, y2, x3, y3, x4, y4
                        FROM rooms
                        WHERE lower(name) = lower(%s)
                        ORDER BY id ASC
                        LIMIT 1
                        """,
                        (normalized_name,),
                    )
                room_row = cur.fetchone()
                if room_row is None:
                    if clause:
                        cur.execute(
                            f"""
                            SELECT id, name, x1, y1, x2, y2, x3, y3, x4, y4
                            FROM rooms
                            WHERE {clause} AND lower(name) LIKE lower(%s)
                            ORDER BY
                                CASE WHEN lower(name) LIKE lower(%s) THEN 0 ELSE 1 END,
                                id ASC
                            LIMIT 1
                            """,
                            tuple(params + [f"%{normalized_name}%", f"{normalized_name}%"]),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, name, x1, y1, x2, y2, x3, y3, x4, y4
                            FROM rooms
                            WHERE lower(name) LIKE lower(%s)
                            ORDER BY
                                CASE WHEN lower(name) LIKE lower(%s) THEN 0 ELSE 1 END,
                                id ASC
                            LIMIT 1
                            """,
                            (f"%{normalized_name}%", f"{normalized_name}%"),
                        )
                    room_row = cur.fetchone()
                if room_row:
                    break

            if room_row is None:
                return {
                    "found": False,
                    "room_name": normalized_name,
                    "error": f"Could not find a room named '{normalized_name}'.",
                    "objects": [],
                }

            rx1, ry1, rx2, ry2 = _room_bounds_from_row(room_row, start_idx=2)

            map_filter_sql = ""
            map_params: list[object] = []
            if active_map_id is not None:
                map_filter_sql = """
                    AND (
                        oo.map_id = %s
                        OR EXISTS (
                            SELECT 1 FROM scenes s2
                            WHERE s2.id = oo.scene_id AND s2.map_id = %s
                        )
                    )
                """
                map_params.extend([active_map_id, active_map_id])

            cur.execute(
                f"""
                SELECT DISTINCT ON (o.id)
                    o.id,
                    o.class_id,
                    o.name,
                    oo.x,
                    oo.y,
                    oo.created_at
                FROM objects o
                JOIN object_observations oo ON oo.object_id = o.id
                WHERE oo.x BETWEEN %s AND %s
                  AND oo.y BETWEEN %s AND %s
                  {map_filter_sql}
                ORDER BY o.id, oo.created_at DESC
                LIMIT %s
                """,
                tuple([rx1, rx2, ry1, ry2] + map_params + [safe_limit]),
            )
            rows = cur.fetchall()

    objects = []
    for r in rows:
        objects.append({
            "object_id": r[0],
            "class_id": r[1],
            "class_name": _class_name(r[1]),
            "name": r[2],
            "x": float(r[3]) if r[3] is not None else None,
            "y": float(r[4]) if r[4] is not None else None,
            "last_seen_at": str(r[5]) if r[5] else None,
        })

    return {
        "found": True,
        "room_id": room_row[0],
        "room_name": room_row[1],
        "x1": rx1,
        "y1": ry1,
        "x2": rx2,
        "y2": ry2,
        "objects": objects,
    }


def move_to_position(x: float, y: float, reason: str | None = None, standoff_m: float = 0.0, yaw: float | None = None) -> dict:
    """Dummy robot movement tool used for planning and transcript visibility.

    This does not move the robot yet. It only records the requested target position.
    When navigating to an object, set standoff_m=1.0 so the robot stops 1 metre short.
    yaw is the desired final orientation in radians (0 = facing +x, pi/2 = facing +y).
    """
    result = {
        "ok": True,
        "executed": False,
        "action": "move_to_position",
        "x": float(x),
        "y": float(y),
        "reason": reason,
        "standoff_m": float(standoff_m),
        "status": "dummy_only",
        "message": (
            f"Dummy move request recorded for x={float(x):.3f}, y={float(y):.3f}. "
            "Robot motion is not connected yet."
        ),
    }
    if yaw is not None:
        result["yaw"] = float(yaw)
    return result


def get_object_captions(object_id: str) -> dict:
    """Return scene and interaction captions associated with an object's observations."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"object_id": object_id, "scene_captions": [], "interaction_captions": []}

            scene_filter = " AND oo.map_id = %s" if active_map_id is not None else ""
            interaction_filter = " AND i.map_id = %s" if active_map_id is not None else ""
            scene_params = [object_id]
            interaction_params = [object_id, object_id]
            if active_map_id is not None:
                scene_params.append(active_map_id)
                interaction_params.append(active_map_id)
            cur.execute(
                f"""
                SELECT DISTINCT s.caption, s.timestamp
                FROM object_observations oo
                JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.object_id = %s {scene_filter} AND s.caption IS NOT NULL AND btrim(s.caption) <> ''
                ORDER BY s.timestamp DESC
                LIMIT 10
                """,
                tuple(scene_params),
            )
            scene_rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT DISTINCT i.action, i.caption, i.created_at
                FROM interactions i
                LEFT JOIN object_observations subject_obs ON subject_obs.id = i.subject_id
                LEFT JOIN object_observations object_obs ON object_obs.id = i.object_id
                WHERE (subject_obs.object_id = %s OR object_obs.object_id = %s){interaction_filter}
                ORDER BY i.created_at DESC
                LIMIT 10
                """,
                tuple(interaction_params),
            )
            interaction_rows = cur.fetchall()

    return {
        "object_id": object_id,
        "scene_captions": [
            {"caption": r[0], "timestamp": str(r[1])} for r in scene_rows
        ],
        "interaction_captions": [
            {"action": r[0], "caption": r[1], "created_at": str(r[2])} for r in interaction_rows
        ],
    }


def get_object_image(object_id: str) -> dict:
    """Return the most recent cropped image reference for an object."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {
                    "object_id": object_id,
                    "found": False,
                    "error": "No cropped image available for this object.",
                }

            map_filter = " AND oo.map_id = %s" if active_map_id is not None else ""
            params = [object_id]
            if active_map_id is not None:
                params.append(active_map_id)
            cur.execute(
                f"""
                SELECT id, created_at
                FROM object_observations
                WHERE object_id = %s {map_filter} AND cropped_image IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 1
                """,
                tuple(params),
            )
            row = cur.fetchone()

    if not row:
        return {
            "object_id": object_id,
            "found": False,
            "error": "No cropped image available for this object.",
        }

    obs_id, created_at = row
    return {
        "object_id": object_id,
        "observation_id": obs_id,
        "found": True,
        "last_seen_at": str(created_at) if created_at else None,
    }


def get_object_summary(object_id: str) -> dict:
    """Return a compact summary of the known data for one object id."""
    _ensure_objects_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    o.id,
                    o.class_id,
                    o.name,
                    o.created_at,
                    COUNT(oo.id) AS observation_count,
                    MIN(oo.created_at) AS first_seen_at,
                    MAX(oo.created_at) AS last_seen_at,
                    (
                        SELECT oo2.id
                        FROM object_observations oo2
                        WHERE oo2.object_id = o.id
                        ORDER BY oo2.created_at DESC, oo2.id DESC
                        LIMIT 1
                    ) AS latest_observation_id
                FROM objects o
                LEFT JOIN object_observations oo ON oo.object_id = o.id
                WHERE o.id = %s
                GROUP BY o.id, o.class_id, o.name, o.created_at
                """,
                (object_id,),
            )
            row = cur.fetchone()

    if not row:
        return {"object_id": object_id, "found": False}

    location = get_object_last_location(object_id)
    captions = get_object_captions(object_id)
    interactions = get_object_interactions(object_id)
    image = get_object_image(object_id)

    scene_captions = captions.get("scene_captions") or []
    interaction_captions = captions.get("interaction_captions") or []
    interaction_list = interactions.get("interactions") or []

    summary = {
        "object_id": row[0],
        "found": True,
        "class_id": row[1],
        "class_name": _class_name(row[1]),
        "name": row[2],
        "created_at": str(row[3]) if row[3] else None,
        "observation_count": int(row[4] or 0),
        "first_seen_at": str(row[5]) if row[5] else None,
        "last_seen_at": str(row[6]) if row[6] else None,
        "latest_observation_id": row[7],
        "observation_id": image.get("observation_id") if image.get("found") else row[7],
        "has_image": bool(image.get("found")),
        "interaction_count": len(interaction_list),
        "scene_caption_count": len(scene_captions),
        "interaction_caption_count": len(interaction_captions),
        "latest_scene_caption": scene_captions[0]["caption"] if scene_captions else None,
        "latest_interaction_caption": (
            interaction_captions[0].get("caption") or interaction_captions[0].get("action")
            if interaction_captions
            else None
        ),
        "latest_interaction_participants": (
            interaction_list[0].get("co_participants") if interaction_list else []
        ),
    }

    if location.get("found"):
        summary.update(
            {
                "x": location.get("x"),
                "y": location.get("y"),
                "z": location.get("z"),
                "room": location.get("room"),
                "last_location_seen_at": location.get("last_seen_at"),
                "last_scene_caption": location.get("scene_caption"),
            }
        )

    return summary


def get_object_observations(
    object_id: str,
    limit: int = DEFAULT_OBJECT_OBSERVATION_LIMIT,
    sort_order: str = "desc",
    time_filter: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict:
    """Return recorded detections for one object, optionally filtered by time and sorted by time."""
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    normalized_sort_order = _normalize_sort_order(sort_order)
    order_direction = "ASC" if normalized_sort_order == "asc" else "DESC"
    timestamp_filter_clause, timestamp_filter_params, normalized_time_filter = _build_timestamp_filter(
        "COALESCE(s.timestamp, oo.created_at)",
        time_filter,
    )

    range_clause, range_params, norm_start, norm_end = _build_time_range_filters(
        "COALESCE(s.timestamp, oo.created_at)",
        start_time,
        end_time,
    )
    if range_clause:
        timestamp_filter_clause += range_clause
        timestamp_filter_params.extend(range_params)
    _ensure_rooms_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {
                    "object_id": object_id,
                    "found": False,
                    "limit": safe_limit,
                    "sort_order": normalized_sort_order,
                    "time_filter": normalized_time_filter,
                    "results": [],
                }

            map_filter = " AND oo.map_id = %s" if active_map_id is not None else ""
            count_params = [object_id]
            if active_map_id is not None:
                count_params.append(active_map_id)
            count_params.extend(timestamp_filter_params)
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.object_id = %s{map_filter}
                {timestamp_filter_clause}
                """,
                tuple(count_params),
            )
            total_count = int(cur.fetchone()[0])

            if total_count == 0:
                return {
                    "object_id": object_id,
                    "found": False,
                    "limit": safe_limit,
                    "sort_order": normalized_sort_order,
                    "time_filter": normalized_time_filter,
                    "results": [],
                }

            select_params = [AGENT_QUERY_TIMEZONE, object_id]
            if active_map_id is not None:
                select_params.append(active_map_id)
            select_params.extend(timestamp_filter_params)
            select_params.append(safe_limit)
            cur.execute(
                f"""
                SELECT
                    oo.id,
                    oo.object_id,
                    oo.scene_id,
                    oo.x,
                    oo.y,
                    oo.z,
                    oo.created_at,
                    timezone(%s, oo.created_at) AS local_created_at,
                    oo.yolo_track_id,
                    oo.person_id,
                    oo.position_source,
                    s.caption,
                    s.timestamp
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.object_id = %s{" AND oo.map_id = %s" if active_map_id is not None else ""}
                {timestamp_filter_clause}
                ORDER BY oo.created_at {order_direction}, oo.id {order_direction}
                LIMIT %s
                """,
                tuple(select_params),
            )
            rows = cur.fetchall()

            observations = []
            for row in rows:
                x, y = row[3], row[4]
                room = None
                if x is not None and y is not None:
                    room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)

                observations.append(
                    {
                        "observation_id": row[0],
                        "object_id": row[1],
                        "scene_id": row[2],
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                        "z": float(row[5]) if row[5] is not None else None,
                        "created_at": str(row[6]) if row[6] else None,
                        "local_created_at": str(row[7]) if row[7] else None,
                        "yolo_track_id": row[8],
                        "person_id": row[9],
                        "position_source": row[10],
                        "navigation_usable": row[10] in {"dynosam", "depth"},
                        "scene_caption": row[11],
                        "scene_timestamp": str(row[12]) if row[12] else None,
                        "room": room,
                    }
                )

    return {
        "object_id": object_id,
        "found": True,
        "total_count": total_count,
        "returned_count": len(observations),
        "limit": safe_limit,
        "sort_order": normalized_sort_order,
        "time_filter": normalized_time_filter,
        "results": observations,
    }


def get_person_interaction_summary(
    person_name: str,
    other_person_name: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    room_name: str | None = None,
    limit: int = 10,
) -> dict:
    """Return aggregated interaction statistics for a person, grouped by co-participant.
    
    Use this for:
    - 'Who did X talk to most often?'
    - 'Who talked to X during the workday?'
    - 'Which people interacted with each other?'
    """
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))

    events_result = get_interaction_events(
        person_name=person_name,
        other_person_name=other_person_name,
        start_time=start_time,
        end_time=end_time,
        room_name=room_name,
        limit=MAX_HISTORY_RESULT_LIMIT,
        sort_order="desc",
    )
    events = events_result.get("events") or []

    with _get_conn() as conn:
        with conn.cursor() as cur:
            person_rows = _lookup_named_object_ids(cur, person_name) if person_name else []
            person_ids = {str(row[0]) for row in person_rows}
            other_rows = _lookup_named_object_ids(cur, other_person_name) if other_person_name else []
            other_ids = {str(row[0]) for row in other_rows}

    stats: dict[str, dict] = {}
    for event in events:
        iid = event.get("interaction_id")
        action = event.get("action")
        created_at = event.get("created_at")
        room = event.get("room")

        for participant in event.get("participants") or []:
            pid = str(participant.get("object_id") or "")
            pname = participant.get("name")
            pclass = participant.get("class_name")

            if pid in person_ids or pname == person_name:
                continue
            if other_person_name and (pid not in other_ids and pname != other_person_name):
                continue

            key = pid or pname or "unknown"
            entry = stats.setdefault(key, {
                "object_id": pid or None,
                "name": pname,
                "class_name": pclass,
                "interaction_count": 0,
                "interaction_ids": [],
                "actions": Counter(),
                "latest_interaction_at": None,
                "earliest_interaction_at": None,
                "rooms": set(),
            })
            entry["interaction_count"] += 1
            if iid:
                entry["interaction_ids"].append(iid)
            entry["actions"][action] += 1
            if created_at:
                if entry["latest_interaction_at"] is None or created_at > entry["latest_interaction_at"]:
                    entry["latest_interaction_at"] = created_at
                if entry["earliest_interaction_at"] is None or created_at < entry["earliest_interaction_at"]:
                    entry["earliest_interaction_at"] = created_at
            if room:
                entry["rooms"].add(room)

    ranked = sorted(
        stats.values(),
        key=lambda x: (-x["interaction_count"], str(x["latest_interaction_at"] or "")),
    )

    def _format_participant(item):
        return {
            "object_id": item["object_id"],
            "name": item["name"],
            "class_name": item["class_name"],
            "interaction_count": item["interaction_count"],
            "interaction_ids": item["interaction_ids"][:10],
            "actions": [{"action": a, "count": c} for a, c in item["actions"].most_common()],
            "latest_interaction_at": item["latest_interaction_at"],
            "earliest_interaction_at": item["earliest_interaction_at"],
            "rooms": sorted(item["rooms"]),
        }

    person_only = [p for p in ranked if p["class_name"] == "person"]
    object_only = [p for p in ranked if p["class_name"] != "person"]

    return {
        "person_name": person_name,
        "other_person_name": other_person_name,
        "start_time": start_time,
        "end_time": end_time,
        "room_name": room_name,
        "total_events_analyzed": len(events),
        "co_participant_count": len(ranked),
        "co_participants": [_format_participant(item) for item in ranked[:safe_limit]],
        "person_co_participants": [_format_participant(item) for item in person_only[:safe_limit]],
        "object_co_participants": [_format_participant(item) for item in object_only[:safe_limit]],
    }


def find_person_by_name(person_name: str) -> dict:
    """Find a tracked person by name and return their last known location.

    Use this for:
    - "Where is Mufasa now?"
    - "Find the person Simba"
    - "Locate Nala"
    - "Go to person Scar" (first find them, then use their coordinates with move_to_position)

    Do not use this for historical questions such as "where was Scar first
    observed?" or "where was Scar at 15:45"; resolve the person and call
    get_object_observations/get_object_first_location instead.
    """
    normalized_name = str(person_name or "").strip()
    if not normalized_name:
        return {"found": False, "error": "person_name is required"}

    search_result = search_objects_by_class_id(class_id=0, object_name=normalized_name, limit=5)
    candidates = search_result.get("results") or []
    if not candidates:
        return {"found": False, "person_name": normalized_name, "error": f"No tracked person named '{normalized_name}' found."}

    person = candidates[0]
    object_id = str(person.get("object_id") or "")
    if not object_id:
        return {"found": False, "person_name": normalized_name, "error": f"Person '{normalized_name}' has no valid object_id."}

    loc = get_object_last_location(object_id, require_navigation_usable=False)
    if not loc.get("found"):
        return {
            "found": True,
            "person_name": normalized_name,
            "object_id": object_id,
            "class_name": person.get("class_name"),
            "room": person.get("room"),
            "x": person.get("x"),
            "y": person.get("y"),
            "last_seen_at": person.get("last_seen_at"),
            "note": "Person found but no location observations available.",
        }

    return {
        "found": True,
        "person_name": normalized_name,
        "object_id": object_id,
        "class_name": person.get("class_name"),
        "room": loc.get("room") or person.get("room"),
        "x": loc.get("x"),
        "y": loc.get("y"),
        "z": loc.get("z"),
        "last_seen_at": loc.get("last_seen_at"),
        "scene_caption": loc.get("scene_caption"),
        "position_source": loc.get("position_source"),
        "navigation_usable": loc.get("navigation_usable"),
        "note": "Latest location only. For first/earliest/exact-time history, use get_object_observations or get_object_first_location.",
    }


def get_latest_scene(offset: int = 0) -> dict:
    """Return the most recent recorded scene, or an older one by offset."""
    safe_offset = max(0, int(offset))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"found": False, "error": "No scenes available."}

            map_filter = "WHERE map_id = %s" if active_map_id is not None else ""
            params = [active_map_id] if active_map_id is not None else []
            params.append(safe_offset)
            cur.execute(
                f"""
                SELECT id, x, y, caption, timestamp, scene_image IS NOT NULL
                FROM scenes
                {map_filter}
                ORDER BY timestamp DESC NULLS LAST, id DESC
                LIMIT 1 OFFSET %s
                """,
                tuple(params),
            )
            row = cur.fetchone()

    if not row:
        return {"found": False, "error": "No scenes available."}

    scene_id, x, y, caption, timestamp, has_image = row
    return {
        "found": True,
        "scene_id": scene_id,
        "x": float(x) if x is not None else None,
        "y": float(y) if y is not None else None,
        "caption": caption,
        "timestamp": str(timestamp) if timestamp else None,
        "has_image": bool(has_image),
    }


def count_scenes_for_active_map() -> int:
    """Return total scene count for the currently active map."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return 0
            if active_map_id is not None:
                cur.execute("SELECT COUNT(*) FROM scenes WHERE map_id = %s", (active_map_id,))
            else:
                cur.execute("SELECT COUNT(*) FROM scenes")
            row = cur.fetchone()
    return int(row[0]) if row else 0


def get_latest_observation() -> dict:
    """Return the most recent recorded object observation."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            _active_map_name_value, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"found": False, "error": "No observations available."}

            map_filter = "WHERE oo.map_id = %s" if active_map_id is not None else ""
            params = [active_map_id] if active_map_id is not None else []
            cur.execute(
                f"""
                SELECT
                    oo.id,
                    oo.object_id,
                    oo.class_id,
                    oo.x,
                    oo.y,
                    oo.z,
                    oo.created_at,
                    oo.scene_id,
                    s.caption
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                {map_filter}
                ORDER BY oo.created_at DESC NULLS LAST, oo.id DESC
                LIMIT 1
                """,
                tuple(params),
            )
            row = cur.fetchone()

    if not row:
        return {"found": False, "error": "No observations available."}

    obs_id, object_id, class_id, x, y, z, created_at, scene_id, scene_caption = row
    return {
        "found": True,
        "observation_id": obs_id,
        "object_id": object_id,
        "class_id": class_id,
        "class_name": _class_name(class_id),
        "x": float(x) if x is not None else None,
        "y": float(y) if y is not None else None,
        "z": float(z) if z is not None else None,
        "timestamp": str(created_at) if created_at else None,
        "scene_id": scene_id,
        "scene_caption": scene_caption,
    }


def _ensure_strategy_schema() -> None:
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS robot_visits (
                    id BIGSERIAL PRIMARY KEY,
                    room_name TEXT NOT NULL,
                    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL,
                    arrived_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    departed_at TIMESTAMPTZ,
                    scene_count INT DEFAULT 0,
                    interaction_count INT DEFAULT 0,
                    distance_travelled_m FLOAT DEFAULT 0
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_robot_visits_room ON robot_visits(room_name, map_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_robot_visits_arrived ON robot_visits(arrived_at DESC)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS navigation_decisions (
                    id BIGSERIAL PRIMARY KEY,
                    decision_type TEXT NOT NULL,
                    target_room TEXT,
                    target_x FLOAT,
                    target_y FLOAT,
                    dwell_time_seconds INT,
                    reasoning TEXT,
                    scene_change_prediction TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL
                )
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nav_decisions_type ON navigation_decisions(decision_type, created_at DESC)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_nav_decisions_created ON navigation_decisions(created_at DESC)")
        conn.commit()


def get_room_visit_history(limit: int = 20) -> dict:
    """Return visit history per room: last visit time, total visits, scenes, interactions."""
    _ensure_strategy_schema()
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"rooms": [], "source": "robot_visits", "limit": safe_limit}

            # Prefer explicit robot_visits table
            map_clause = "AND map_id = %s" if active_map_id is not None else ""
            params: list[object] = [safe_limit]
            if active_map_id is not None:
                params.append(active_map_id)
            cur.execute(
                f"""
                SELECT
                    room_name,
                    MAX(arrived_at) AS last_arrived,
                    COUNT(*) AS visit_count,
                    SUM(scene_count) AS total_scenes,
                    SUM(interaction_count) AS total_interactions
                FROM robot_visits
                WHERE 1=1 {map_clause}
                GROUP BY room_name
                ORDER BY MAX(arrived_at) ASC NULLS FIRST
                LIMIT %s
                """,
                tuple(params),
            )
            visit_rows = cur.fetchall()

    if visit_rows:
        rooms = []
        for row in visit_rows:
            rooms.append(
                {
                    "room_name": row[0],
                    "last_visit_at": str(row[1]) if row[1] else None,
                    "visit_count": int(row[2] or 0),
                    "total_scenes": int(row[3] or 0),
                    "total_interactions": int(row[4] or 0),
                }
            )
        return {"rooms": rooms, "source": "robot_visits", "limit": safe_limit}

    # Fallback: derive from scenes via geometric room join
    with _get_conn() as conn:
        with conn.cursor() as cur:
            map_clause = "AND s.map_id = %s" if active_map_id is not None else ""
            params: list[object] = []
            if active_map_id is not None:
                params.append(active_map_id)
            params.append(safe_limit)
            cur.execute(
                f"""
                SELECT
                    COALESCE(r.name, 'unknown') AS room_name,
                    MAX(s.timestamp) AS last_seen,
                    COUNT(*) AS scene_count
                FROM scenes s
                LEFT JOIN rooms r ON r.map_id = s.map_id
                    AND s.x BETWEEN LEAST(r.x1, r.x2, r.x3, r.x4) AND GREATEST(r.x1, r.x2, r.x3, r.x4)
                    AND s.y BETWEEN LEAST(r.y1, r.y2, r.y3, r.y4) AND GREATEST(r.y1, r.y2, r.y3, r.y4)
                WHERE 1=1 {map_clause}
                GROUP BY COALESCE(r.name, 'unknown')
                ORDER BY MAX(s.timestamp) ASC NULLS FIRST
                LIMIT %s
                """,
                tuple(params),
            )
            scene_rows = cur.fetchall()

    rooms = []
    for row in scene_rows:
        rooms.append(
            {
                "room_name": row[0],
                "last_visit_at": str(row[1]) if row[1] else None,
                "visit_count": None,
                "total_scenes": int(row[2] or 0),
                "total_interactions": None,
                "note": "Derived from scene timestamps; visit/interaction counts unavailable.",
            }
        )
    return {"rooms": rooms, "source": "scenes", "limit": safe_limit}


def get_room_event_density(hours: int = 24, limit: int = 20) -> dict:
    """Return interaction density per room over the last N hours."""
    safe_hours = max(1, int(hours))
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"rooms": [], "hours": safe_hours, "limit": safe_limit}

            map_clause = "AND i.map_id = %s" if active_map_id is not None else ""
            params: list[object] = [safe_hours]
            if active_map_id is not None:
                params.append(active_map_id)
            params.append(safe_limit)
            cur.execute(
                f"""
                SELECT
                    COALESCE(r.name, 'unknown') AS room_name,
                    COUNT(i.id) AS interaction_count,
                    COUNT(DISTINCT i.action) AS distinct_actions,
                    array_agg(DISTINCT i.action) FILTER (WHERE i.action IS NOT NULL) AS actions
                FROM interactions i
                LEFT JOIN scenes s ON s.id = i.scene_id
                LEFT JOIN rooms r ON r.map_id = i.map_id
                    AND s.x BETWEEN LEAST(r.x1, r.x2, r.x3, r.x4) AND GREATEST(r.x1, r.x2, r.x3, r.x4)
                    AND s.y BETWEEN LEAST(r.y1, r.y2, r.y3, r.y4) AND GREATEST(r.y1, r.y2, r.y3, r.y4)
                WHERE i.created_at >= NOW() - INTERVAL '%s hours'
                {map_clause}
                GROUP BY COALESCE(r.name, 'unknown')
                ORDER BY COUNT(i.id) DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    rooms = []
    for row in rows:
        actions = row[3] or []
        if isinstance(actions, list):
            action_list = [str(a) for a in actions if a]
        else:
            action_list = [str(a).strip() for a in str(actions).strip('{}').split(',') if a.strip()]
        rooms.append(
            {
                "room_name": row[0],
                "interaction_count": int(row[1] or 0),
                "distinct_actions": int(row[2] or 0),
                "actions": action_list[:10],
            }
        )
    return {"rooms": rooms, "hours": safe_hours, "limit": safe_limit}


def get_stale_rooms(threshold_minutes: int = 30, limit: int = 20) -> dict:
    """Return rooms not visited in the last N minutes, ordered by staleness."""
    safe_threshold = max(1, int(threshold_minutes))
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    history = get_room_visit_history(limit=safe_limit * 2)
    rooms = history.get("rooms") or []
    stale = []
    for room in rooms:
        last_visit = room.get("last_visit_at")
        if not last_visit:
            stale.append(room)
            continue
        try:
            from datetime import datetime as _dt, timezone as _tz
            parsed = _dt.fromisoformat(str(last_visit).replace("Z", "+00:00"))
            age_minutes = (_dt.now(_tz.utc) - parsed).total_seconds() / 60.0
        except Exception:
            stale.append(room)
            continue
        if age_minutes >= safe_threshold:
            stale.append({**room, "minutes_since_visit": round(age_minutes, 1)})
    return {
        "threshold_minutes": safe_threshold,
        "stale_rooms": stale[:safe_limit],
        "count": len(stale),
    }


def get_current_scene_description(minutes: int = 5, offset: int = 0) -> dict:
    """Return the latest scene plus recent objects, people, and interactions."""
    safe_minutes = max(1, int(minutes))
    latest = get_latest_scene(offset=offset)
    if not latest.get("found"):
        return {"found": False, "error": "No scenes available."}

    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"found": True, "scene": latest, "recent_objects": [], "recent_interactions": []}

            map_clause = "AND oo.map_id = %s" if active_map_id is not None else ""
            params: list[object] = [safe_minutes]
            if active_map_id is not None:
                params.append(active_map_id)
            cur.execute(
                f"""
                SELECT
                    o.id,
                    o.class_id,
                    o.name,
                    oo.x,
                    oo.y,
                    oo.created_at
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                WHERE oo.created_at >= NOW() - INTERVAL '%s minutes'
                {map_clause}
                ORDER BY oo.created_at DESC
                LIMIT 20
                """,
                tuple(params),
            )
            obs_rows = cur.fetchall()

            map_clause_i = "AND i.map_id = %s" if active_map_id is not None else ""
            params_i: list[object] = [safe_minutes]
            if active_map_id is not None:
                params_i.append(active_map_id)
            cur.execute(
                f"""
                SELECT i.action, i.caption, i.created_at
                FROM interactions i
                WHERE i.created_at >= NOW() - INTERVAL '%s minutes'
                {map_clause_i}
                ORDER BY i.created_at DESC
                LIMIT 10
                """,
                tuple(params_i),
            )
            interaction_rows = cur.fetchall()

    recent_objects = []
    seen_ids: set[int] = set()
    for row in obs_rows:
        obj_id = int(row[0])
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        recent_objects.append(
            {
                "object_id": obj_id,
                "class_name": _class_name(row[1]),
                "name": row[2],
                "x": float(row[3]) if row[3] is not None else None,
                "y": float(row[4]) if row[4] is not None else None,
                "observed_at": str(row[5]) if row[5] else None,
            }
        )

    recent_interactions = [
        {
            "action": row[0],
            "caption": row[1],
            "created_at": str(row[2]) if row[2] else None,
        }
        for row in interaction_rows
    ]

    return {
        "found": True,
        "scene": latest,
        "recent_objects": recent_objects[:10],
        "recent_interactions": recent_interactions,
        "minutes_window": safe_minutes,
    }


def log_navigation_decision(
    decision_type: str,
    target_room: str | None = None,
    target_x: float | None = None,
    target_y: float | None = None,
    dwell_time_seconds: int | None = None,
    reasoning: str | None = None,
    scene_change_prediction: str | None = None,
) -> dict:
    """Log a navigation decision for later metrics and comparison."""
    _ensure_strategy_schema()
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, _scope = _get_active_map_scope(cur)
            cur.execute(
                """
                INSERT INTO navigation_decisions
                (decision_type, target_room, target_x, target_y, dwell_time_seconds, reasoning, scene_change_prediction, map_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    str(decision_type),
                    target_room,
                    target_x,
                    target_y,
                    dwell_time_seconds,
                    reasoning,
                    scene_change_prediction,
                    active_map_id,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {
        "logged": True,
        "decision_id": int(row[0]) if row else None,
        "decision_type": decision_type,
        "target_room": target_room,
    }


def get_recent_interactions(minutes: int = 10, limit: int = 20) -> dict:
    """Return recent interaction events with room names."""
    safe_minutes = max(1, int(minutes))
    safe_limit = max(1, min(int(limit), MAX_HISTORY_RESULT_LIMIT))
    with _get_conn() as conn:
        with conn.cursor() as cur:
            active_map_name, active_map_id, scope_requested = _get_active_map_scope(cur)
            if scope_requested and active_map_id is None:
                return {"interactions": [], "minutes": safe_minutes, "limit": safe_limit}

            map_clause = "AND i.map_id = %s" if active_map_id is not None else ""
            params: list[object] = [safe_minutes]
            if active_map_id is not None:
                params.append(active_map_id)
            params.append(safe_limit)
            cur.execute(
                f"""
                SELECT
                    i.id,
                    i.action,
                    i.caption,
                    i.created_at,
                    s.x,
                    s.y
                FROM interactions i
                LEFT JOIN scenes s ON s.id = i.scene_id
                WHERE i.created_at >= NOW() - INTERVAL '%s minutes'
                {map_clause}
                ORDER BY i.created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    interactions = []
    for row in rows:
        x, y = row[4], row[5]
        room = None
        if x is not None and y is not None:
            with _get_conn() as conn:
                with conn.cursor() as cur:
                    active_map_name, active_map_id, _ = _get_active_map_scope(cur)
                    room = _resolve_room_name(cur, active_map_name, active_map_id, x, y)
        interactions.append(
            {
                "interaction_id": row[0],
                "action": row[1],
                "caption": row[2],
                "created_at": str(row[3]) if row[3] else None,
                "room": room or "unknown",
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
            }
        )
    return {"interactions": interactions, "minutes": safe_minutes, "limit": safe_limit}


_DISABLED_AGENT_TOOL_NAMES = {
    "get_interaction_map_summary",
    "get_room_exploration_status",
    "get_latest_scene",
    "get_latest_observation",
    "get_object_captions",
    "get_room_visit_history",
    "get_room_event_density",
    "get_stale_rooms",
}


TOOL_SCHEMAS = [
  {
    "type": "function",
    "function": {
      "name": "search_objects_by_class_id",
      "description": "Find tracked objects or people by YOLO class_id and optional name. Returns object_id, class, room, and x/y. Use this first to resolve any entity.",
      "parameters": {
        "type": "object",
        "properties": {
          "class_id": {
            "type": "integer",
            "description": "Optional YOLO class id for the requested object type, for example 0 for person,."
          },
          "object_name": {
            "type": "string",
            "description": "Optional exact tracked name such as 'bottle_001', 'dishwasher_001', or 'Simba'."
          },
          "min_observations": {
            "type": "integer",
            "description": "Optional minimum observation count for returned objects."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of matching objects to return."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_interactions",
      "description": "Return interaction history with people using, picking up, or otherwise interacting with this object.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "UUID object_id (TEXT) from the objects table, e."
          },
          "limit": {
            "type": "integer",
            "description": "Optional maximum number of interactions to return."
          },
          "sort_order": {
            "type": "string",
            "enum": [
              "desc",
              "asc"
            ],
            "description": "Sort by interaction time."
          },
          "action": {
            "type": "string",
            "description": "Optional action filter, e."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_person_interactions",
      "description": "Aggregate interaction information for a specific object, listing the people who used it along with interaction counts.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "UUID object_id (TEXT) from the objects table."
          },
          "action": {
            "type": "string",
            "description": "Optional action filter, e."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_last_location",
      "description": "Quickly get the absolute most recent recorded location of an object. Set require_navigation_usable to true to only get positions backed by depth or DynoSAM.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The object_id (TEXT) from the objects table."
          },
          "require_navigation_usable": {
            "type": "boolean",
            "description": "If true, only return positions with depth or DynoSAM source."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_first_location",
      "description": "Quickly get the absolute first recorded location of an object.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The object_id (TEXT) from the objects table."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "list_objects",
      "description": "List all tracked objects, optionally filtered by class name. Useful for enumerating all persons (class 'person') or items of a specific type.",
      "parameters": {
        "type": "object",
        "properties": {
          "class_name": {
            "type": "string",
            "description": "Optional filter, e.g. 'person', 'handbag'. Omit for all objects."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of objects to return."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_room_exploration_status",
      "description": "List recorded rooms for the current building/map, ordered by the oldest last-seen scene first. Use this both to answer which rooms exist and to decide which places have not been visited for a long time or were never seen.",
      "parameters": {
        "type": "object",
        "properties": {
          "limit": {
            "type": "integer",
            "description": "Maximum number of rooms to return, default 10."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_interaction_map_summary",
      "description": "Summarize how many interactions happened in each room, including rooms with many interactions and rooms with only a few. Use this to reason about interaction hotspots on the map.",
      "parameters": {
        "type": "object",
        "properties": {
          "limit": {
            "type": "integer",
            "description": "Maximum number of rooms to return, default 10."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_room_navigation_target",
      "description": "Resolve a room name into a navigable entry point on the room boundary (or center fallback). Use this only when the user wants to go to a room in general, or when no more specific person, object, or interaction coordinates are available.",
      "parameters": {
        "type": "object",
        "properties": {
          "room_name": {
            "type": "string",
            "description": "Exact room name to navigate to, for example 'Kitchen', 'Open Office', or 'Room."
          }
        },
        "required": [
          "room_name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_objects_in_room",
      "description": "List tracked objects and people whose latest known position is inside a named room. Use this when the user asks what is in a room, which objects are in a room, or who is in a room. Returns object_id, name, class, and position for each entity found.",
      "parameters": {
        "type": "object",
        "properties": {
          "room_name": {
            "type": "string",
            "description": "Exact room name to look inside, for example 'Kitchen', 'Open Office', or 'Room."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of objects to return, default 20."
          }
        },
        "required": [
          "room_name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "move_to_position",
      "description": "Navigation action that accepts the final x/y target chosen by the agent. Use this only after you have already identified the best coordinates from another tool result. Do not use it for lookup or reasoning; use it once to commit to the navigation target.",
      "parameters": {
        "type": "object",
        "properties": {
          "x": {
            "type": "number",
            "description": "Target x coordinate in map/world coordinates taken from a prior tool result."
          },
          "y": {
            "type": "number",
            "description": "Target y coordinate in map/world coordinates taken from a prior tool result."
          },
          "reason": {
            "type": "string",
            "description": "Short explanation of why this position was chosen, for example 'latest."
          },
          "standoff_m": {
            "type": "number",
            "description": "Distance in metres to stop short of the target. Use 1.0 when navigating to an object so the robot stops 1 metre away. Use 0.0 (default) for room centres or arbitrary map coordinates."
          },
          "yaw": {
            "type": "number",
            "description": "Desired final orientation in radians. 0 faces +x, pi/2 faces +y. Only pass this when the prior tool result explicitly provides a yaw value (e.g. room navigation target)."
          }
        },
        "required": [
          "x",
          "y"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_robot_location",
      "description": "Return the robot's current pose and inferred room from the latest map snapshot. Use this for questions about where the robot is right now. Do not use it for object locations, interaction locations, or navigation targets unless the user explicitly asks about the robot itself.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_captions",
      "description": "Deprecated.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The UUID object_id (TEXT) from the objects table."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_image",
      "description": "Fetch the latest cropped image for one resolved object so the UI can show it. Use this for display requests such as 'show me the backpack' or 'show the latest image'.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The resolved object_id from the objects table, typically taken from."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_summary",
      "description": "Get a spatial summary of all the places an object has been.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The UUID object_id (TEXT) from the objects table."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_object_observations",
      "description": "Return the observed location history of one entity. Use for 'where is', 'where was', 'current position', 'latest location', or 'earliest location'. Prefer over get_interaction_events for pure location questions.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "The resolved object_id from the objects table for the object whose position."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of observations to return."
          },
          "sort_order": {
            "type": "string",
            "enum": [
              "desc",
              "asc"
            ],
            "description": "Sort by observation time."
          },
          "time_filter": {
            "type": "string",
            "description": "Optional timestamp filter when the question mentions a specific time or date."
          },
          "start_time": {
            "type": "string",
            "description": "Optional inclusive lower bound for observations."
          },
          "end_time": {
            "type": "string",
            "description": "Optional inclusive upper bound for observations."
          }
        },
        "required": [
          "object_id"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_person_interaction_summary",
      "description": "Aggregate interactions for one person and return co-participants ranked by frequency. Use this for 'who did X talk to most often', 'who interacted with X most', 'which people talked to X during the workday', or 'what relationship exists between X and Y'. It returns counts, interaction IDs, and time ranges per co-participant.",
      "parameters": {
        "type": "object",
        "properties": {
          "person_name": {
            "type": "string",
            "description": "Tracked person name such as 'Simba' or 'Nala'."
          },
          "other_person_name": {
            "type": "string",
            "description": "Optional second person to narrow to a specific pair."
          },
          "start_time": {
            "type": "string",
            "description": "Optional lower time bound."
          },
          "end_time": {
            "type": "string",
            "description": "Optional upper time bound."
          },
          "room_name": {
            "type": "string",
            "description": "Optional room filter, e.g. 'Meeting Room'."
          },
          "limit": {
            "type": "integer",
            "description": "Max co-participants to return."
          }
        },
        "required": [
          "person_name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "find_person_by_name",
      "description": "Find a tracked person by their name and return their latest known location including room and x/y coordinates. Use only for current/live person lookup or navigation. Do not use for first/earliest, historical, or exact-time questions; use search_objects_by_class_id plus get_object_observations/get_object_first_location instead.",
      "parameters": {
        "type": "object",
        "properties": {
          "person_name": {
            "type": "string",
            "description": "The tracked person's name, e.g. 'Simba', 'Nala', 'Kiara'."
          }
        },
        "required": [
          "person_name"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_interaction_events",
      "description": "Return interaction events with timestamps, rooms, participants, and coordinates. Use for who did what, where, or when. Supports action, time, and room filters. If the question names an action such as opening, holding, carrying, picked up, used, or talked, pass it in action. person_name and other_person_name can be person or object names. co_participant_class_name filters to events where a co-participant matches the given class (e.g., 'ball' matches 'sports ball').",
      "parameters": {
        "type": "object",
        "properties": {
          "object_id": {
            "type": "string",
            "description": "Optional resolved object_id when the question is about interactions involving."
          },
          "person_name": {
            "type": "string",
            "description": "Optional exact tracked person name or object name such as 'Simba', 'Nala', or."
          },
          "other_person_name": {
            "type": "string",
            "description": "Optional second tracked person name or object name for pairwise questions such."
          },
          "co_participant_class_name": {
            "type": "string",
            "description": "Optional class name of the desired co-participant. Partial matches work (e.g., 'ball' matches 'sports ball')."
          },
          "start_time": {
            "type": "string",
            "description": "Optional inclusive lower time bound for the interaction search."
          },
          "end_time": {
            "type": "string",
            "description": "Optional inclusive upper time bound for the interaction search."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of interaction events to return."
          },
          "sort_order": {
            "type": "string",
            "enum": [
              "desc",
              "asc"
            ],
            "description": "Sort by interaction time."
          },
          "room_name": {
            "type": "string",
            "description": "Optional room name to filter interactions to a specific room, e."
          },
          "action": {
            "type": "string",
            "description": "Optional action filter, e."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_latest_scene",
      "description": "Fetch the most recent recorded scene from the database. Use this for questions like 'what was the last scene?', 'which scene was your last one?', or 'what did you see most recently?'.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_latest_observation",
      "description": "Fetch the most recent recorded object observation from the database. Use this for questions like 'what is the last observation you had?' or 'what was the latest observation?'.",
      "parameters": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_room_visit_history",
      "description": "Return per-room visit history including last visit time, total visits, scenes observed, and interactions. Use this to find stale rooms or compare room activity levels.",
      "parameters": {
        "type": "object",
        "properties": {
          "limit": {
            "type": "integer",
            "description": "Maximum number of rooms to return."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_room_event_density",
      "description": "Return interaction counts per room over a time window. Use this to identify which rooms historically contain the most activity and interesting events.",
      "parameters": {
        "type": "object",
        "properties": {
          "hours": {
            "type": "integer",
            "description": "Lookback window in hours. Default 24."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of rooms to return."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_stale_rooms",
      "description": "Return rooms that have not been visited recently, ordered by how long ago they were last seen. Use this for least-recently-visited navigation strategies.",
      "parameters": {
        "type": "object",
        "properties": {
          "threshold_minutes": {
            "type": "integer",
            "description": "A room is stale if it was not visited in the last N minutes. Default 30."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of stale rooms to return."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_current_scene_description",
      "description": "Fetch the latest scene caption plus recent objects, people, and interactions observed. Use this to understand what is happening right now before deciding whether to stay or move.",
      "parameters": {
        "type": "object",
        "properties": {
          "minutes": {
            "type": "integer",
            "description": "How many minutes of recent history to include. Default 5."
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "log_navigation_decision",
      "description": "Log a navigation decision into the database for metrics and baseline comparison. Use this after choosing a target room and dwell time.",
      "parameters": {
        "type": "object",
        "properties": {
          "decision_type": {
            "type": "string",
            "description": "Strategy name: agent, fixed_5min, fixed_10min, random, lrv, event_rich."
          },
          "target_room": {
            "type": "string",
            "description": "Chosen destination room name."
          },
          "target_x": {
            "type": "number",
            "description": "Target x coordinate."
          },
          "target_y": {
            "type": "number",
            "description": "Target y coordinate."
          },
          "dwell_time_seconds": {
            "type": "integer",
            "description": "Planned dwell time in seconds."
          },
          "reasoning": {
            "type": "string",
            "description": "Explanation for the decision."
          },
          "scene_change_prediction": {
            "type": "string",
            "description": "Scene change prediction if available: 3min, 10min, 30min, >30min."
          }
        },
        "required": [
          "decision_type"
        ]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "get_recent_interactions",
      "description": "Return recent interaction events with room names. Use this to see what interesting events happened lately.",
      "parameters": {
        "type": "object",
        "properties": {
          "minutes": {
            "type": "integer",
            "description": "How many minutes back to look. Default 10."
          },
          "limit": {
            "type": "integer",
            "description": "Maximum number of interactions to return."
          }
        },
        "required": []
      }
    }
  }
]

TOOL_SCHEMAS = [
    schema
    for schema in TOOL_SCHEMAS
    if schema.get("function", {}).get("name") not in _DISABLED_AGENT_TOOL_NAMES
]


_ALL_TOOL_DISPATCH = {
    "search_objects_by_class_id": search_objects_by_class_id,
    "list_objects": list_objects,
    "get_object_summary": get_object_summary,
    "get_object_observations": get_object_observations,
    "get_object_interactions": get_object_interactions,
    "get_object_person_interactions": get_object_person_interactions,
    "get_object_last_location": get_object_last_location,
    "get_object_first_location": get_object_first_location,
    "get_robot_location": get_robot_location,
    "get_room_navigation_target": get_room_navigation_target,
    "get_objects_in_room": get_objects_in_room,
    "move_to_position": move_to_position,
    "get_object_captions": get_object_captions,
    "get_object_image": get_object_image,
    "get_interaction_events": get_interaction_events,
    "get_latest_scene": get_latest_scene,
    "get_latest_observation": get_latest_observation,
    "get_person_interaction_summary": get_person_interaction_summary,
    "find_person_by_name": find_person_by_name,
    "get_room_visit_history": get_room_visit_history,
    "get_room_event_density": get_room_event_density,
    "get_stale_rooms": get_stale_rooms,
    "get_current_scene_description": get_current_scene_description,
    "log_navigation_decision": log_navigation_decision,
    "get_recent_interactions": get_recent_interactions,
}

TOOL_DISPATCH = {
    name: fn
    for name, fn in _ALL_TOOL_DISPATCH.items()
    if name not in _DISABLED_AGENT_TOOL_NAMES
}
