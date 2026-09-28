"""V4 strategy decision logic for the robot pipeline simulator.

Runs the scene-change-aware v4 strategy against real DB scenes,
using tool results computed from the robot's own observation history
(scoped by map_id).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

import psycopg2

from .prompts import NAVIGATION_SCENE_CHANGE_V4, NAVIGATION_SCENE_CHANGE_V5, NAVIGATION_SCENE_CHANGE_V6
from .tool_adapter import RealToolAdapter
from .vlm_client import get_vlm_client, get_vlm_model

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


def _build_tool_results(map_id: int, room_list: list[str], current_room: str) -> dict[str, Any]:
    """Compute tool results from DB, scoped to the given map_id.

    Uses RealToolAdapter so that the exact same code runs in production
    and in integration tests.
    """
    adapter = RealToolAdapter(map_id=map_id)
    return {
        "get_room_visit_history": adapter.get_room_visit_history(room_list),
        "get_room_change_rates": adapter.get_room_change_rates(room_list),
        "get_time_since_last_change": adapter.get_time_since_last_change(room_list),
        "get_stale_rooms": adapter.get_stale_rooms(room_list),
    }


def _extract_json(text: str) -> dict:
    """Extract JSON object from text, handling markdown fences."""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


def _get_scene_objects(scene_id: int) -> list[dict]:
    """Return object observations for a scene, joined with object names."""
    # Lazy import to avoid circular dependency with tools -> slam_exploration -> pipeline_strategy
    from .tools import YOLO_CLASS_NAMES

    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT oo.object_id, o.name, oo.class_id, oo.x, oo.y, oo.z
                FROM object_observations oo
                LEFT JOIN objects o ON o.id = oo.object_id
                WHERE oo.scene_id = %s
                ORDER BY oo.class_id, oo.object_id
                """,
                (scene_id,),
            )
            return [
                {
                    "object_id": row[0],
                    "name": (
                        row[1]
                        or (YOLO_CLASS_NAMES.get(int(row[2])) if row[2] is not None else None)
                        or f"class_{row[2]}"
                    ),
                    "class_id": row[2],
                    "x": row[3],
                    "y": row[4],
                    "z": row[5],
                }
                for row in cur.fetchall()
            ]


def run_v4_decision(
    scene_id: int,
    room_name: str,
    scene_caption: str,
    map_id: int,
    room_list: list[str],
    step_number: int,
    dwell_minutes: int | None = None,
    min_dwell_minutes: int = 10,
) -> dict[str, Any]:
    """Run the v4 strategy against a real DB scene.

    Args:
        scene_id: The scene row ID.
        room_name: Current room name.
        scene_caption: The VLM-generated scene caption.
        map_id: Current run's map ID (scopes history).
        room_list: Full list of rooms in the run.
        step_number: Current step number (for "just arrived" logic).
        dwell_minutes: Optional override for dwell time in minutes.
        min_dwell_minutes: Minimum dwell time before the agent considers moving.

    Returns:
        Decision dict with keys:
        scene_changed, change, activities_changed, action,
        target_room, reasoning, tool_calls, error
    """
    # Build tool results
    tool_results = _build_tool_results(map_id, room_list, room_name)

    # Build prompt
    dwell_min = dwell_minutes if dwell_minutes is not None else step_number * 5
    prompt = f"""CURRENT SCENE:
Room: {room_name}
Caption: {scene_caption}

You have been in {room_name} for approximately {dwell_min} minutes.
Minimum required dwell time: {min_dwell_minutes} minutes.

TOOL RESULTS:
{json.dumps(tool_results, indent=2, default=str)}

Use the available tools to query your observation history if you need it.
Then decide: stay or move, and output ONLY JSON.
"""

    system_prompt = NAVIGATION_SCENE_CHANGE_V4.replace("{min_dwell}", str(min_dwell_minutes))

    # Call VLM
    client = get_vlm_client()
    model = get_vlm_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=30,
        )
        raw_content = response.choices[0].message.content or "{}"
    except Exception as exc:
        return {
            "scene_changed": False,
            "change": "false",
            "activities_changed": False,
            "action": "stay",
            "target_room": room_name,
            "reasoning": f"VLM call failed: {exc}",
            "tool_calls": [],
            "error": str(exc),
            "raw_response": "",
        }

    # Parse JSON
    parsed = _extract_json(raw_content)
    if not isinstance(parsed, dict):
        parsed = {}

    action = str(parsed.get("action") or "").strip().lower()
    if action not in ("stay", "move"):
        action = "stay"

    target_room = str(parsed.get("target_room") or "").strip()
    if not target_room or target_room.lower() == "none":
        target_room = room_name

    # Normalize target_room to known room list (case-insensitive)
    lower_target = target_room.lower()
    matched = False
    for r in room_list:
        if r.lower() == lower_target:
            target_room = r
            matched = True
            break
    if not matched:
        # VLM returned a room not in the allowed list — fall back to staying
        logger.info(f"[V4_DEFENSE] unknown target '{target_room}' not in {room_list}, falling back to stay in {room_name}")
        target_room = room_name
        action = "stay"

    scene_changed = bool(parsed.get("scene_changed"))
    change_val = str(parsed.get("change") or parsed.get("change_severity") or "false").strip().lower()
    if change_val in ("true", "1", "yes"):
        change_val = "true"
    else:
        change_val = "false"

    activities_changed = bool(parsed.get("activities_changed"))

    return {
        "scene_changed": scene_changed,
        "change": change_val,
        "activities_changed": activities_changed,
        "action": action,
        "target_room": target_room,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "tool_calls": [
            {"tool": name, "result": result}
            for name, result in tool_results.items()
        ],
        "error": None,
        "raw_response": raw_content,
        "prompt": prompt,
    }


def run_v5_decision(
    scene_id: int,
    room_name: str,
    scene_caption: str,
    map_id: int,
    room_list: list[str],
    step_number: int,
    dwell_minutes: int | None = None,
    min_dwell_minutes: int = 10,
) -> dict[str, Any]:
    """Run the v5 strategy against a real DB scene.

    Same as v4 but uses the stricter scene-change definition from NAVIGATION_SCENE_CHANGE_V5.
    """
    tool_results = _build_tool_results(map_id, room_list, room_name)

    dwell_min = dwell_minutes if dwell_minutes is not None else step_number * 5
    prompt = f"""CURRENT SCENE:
Room: {room_name}
Caption: {scene_caption}

You have been in {room_name} for approximately {dwell_min} minutes.

TOOL RESULTS:
{json.dumps(tool_results, indent=2, default=str)}

Use the available tools to query your observation history if you need it.
Then decide: stay or move, and output ONLY JSON.
"""

    system_prompt = NAVIGATION_SCENE_CHANGE_V5

    client = get_vlm_client()
    model = get_vlm_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=30,
        )
        raw_content = response.choices[0].message.content or "{}"
    except Exception as exc:
        return {
            "scene_changed": False,
            "change_severity": "no_change",
            "activities_changed": False,
            "action": "stay",
            "target_room": room_name,
            "reasoning": f"VLM call failed: {exc}",
            "tool_calls": [],
            "error": str(exc),
            "raw_response": "",
        }

    parsed = _extract_json(raw_content)
    if not isinstance(parsed, dict):
        parsed = {}

    action = str(parsed.get("action") or "").strip().lower()
    if action not in ("stay", "move"):
        action = "stay"

    target_room = str(parsed.get("target_room") or "").strip()
    if not target_room or target_room.lower() == "none":
        target_room = room_name

    lower_target = target_room.lower()
    matched = False
    for r in room_list:
        if r.lower() == lower_target:
            target_room = r
            matched = True
            break
    if not matched:
        # VLM returned a room not in the allowed list — fall back to staying
        target_room = room_name
        action = "stay"

    scene_changed = bool(parsed.get("scene_changed"))
    change_severity = str(parsed.get("change_severity") or "no_change").strip().lower()
    if change_severity not in ("no_change", "minor_change", "major_change"):
        change_severity = "no_change"

    activities_changed = bool(parsed.get("activities_changed"))

    return {
        "scene_changed": scene_changed,
        "change_severity": change_severity,
        "activities_changed": activities_changed,
        "action": action,
        "target_room": target_room,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "tool_calls": [
            {"tool": name, "result": result}
            for name, result in tool_results.items()
        ],
        "error": None,
        "raw_response": raw_content,
        "prompt": prompt,
    }


def run_v6_decision(
    scene_id: int,
    room_name: str,
    scene_caption: str,
    map_id: int,
    room_list: list[str],
    step_number: int,
    dwell_minutes: int | None = None,
    min_dwell_minutes: int = 10,
    previous_scene_objects: list[dict] | None = None,
) -> dict[str, Any]:
    """Run the v6 strategy against a real DB scene.

    Extends v5 with structured current and previous object lists
    to help the VLM detect entity changes more accurately.
    """
    tool_results = _build_tool_results(map_id, room_list, room_name)
    current_objects = _get_scene_objects(scene_id)

    def _fmt_object(obj: dict) -> str:
        return f"  - object_id={obj['object_id']}, class={obj['name']}"

    current_obj_lines = "\n".join(_fmt_object(o) for o in current_objects) or "  (none)"
    if previous_scene_objects:
        prev_obj_lines = "\n".join(_fmt_object(o) for o in previous_scene_objects) or "  (none)"
    else:
        prev_obj_lines = "  (no previous observation at this location)"

    dwell_min = dwell_minutes if dwell_minutes is not None else step_number * 5
    prompt = f"""CURRENT SCENE:
Room: {room_name}
Caption: {scene_caption}

You have been in {room_name} for approximately {dwell_min} minutes.

CURRENT OBJECTS IN THIS SCENE:
{current_obj_lines}

PREVIOUS OBJECTS AT THIS ROOM (last visit):
{prev_obj_lines}

TOOL RESULTS:
{json.dumps(tool_results, indent=2, default=str)}

Compare the CURRENT OBJECTS list against the PREVIOUS OBJECTS list using the object_id to identify specific physical instances.
Then decide: stay or move, and output ONLY JSON.
"""

    system_prompt = NAVIGATION_SCENE_CHANGE_V6

    client = get_vlm_client()
    model = get_vlm_model()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=512,
            timeout=30,
        )
        raw_content = response.choices[0].message.content or "{}"
    except Exception as exc:
        return {
            "scene_changed": False,
            "change": "false",
            "activities_changed": False,
            "action": "stay",
            "target_room": room_name,
            "reasoning": f"VLM call failed: {exc}",
            "tool_calls": [],
            "error": str(exc),
            "raw_response": "",
        }

    parsed = _extract_json(raw_content)
    if not isinstance(parsed, dict):
        parsed = {}

    action = str(parsed.get("action") or "").strip().lower()
    if action not in ("stay", "move"):
        action = "stay"

    target_room = str(parsed.get("target_room") or "").strip()
    if not target_room or target_room.lower() == "none":
        target_room = room_name

    lower_target = target_room.lower()
    matched = False
    for r in room_list:
        if r.lower() == lower_target:
            target_room = r
            matched = True
            break
    if not matched:
        target_room = room_name
        action = "stay"

    scene_changed = bool(parsed.get("scene_changed"))
    change_val = str(parsed.get("change") or parsed.get("change_severity") or "false").strip().lower()
    if change_val in ("true", "1", "yes"):
        change_val = "true"
    else:
        change_val = "false"

    activities_changed = bool(parsed.get("activities_changed"))

    return {
        "scene_changed": scene_changed,
        "change": change_val,
        "activities_changed": activities_changed,
        "action": action,
        "target_room": target_room,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "tool_calls": [
            {"tool": name, "result": result}
            for name, result in tool_results.items()
        ],
        "error": None,
        "raw_response": raw_content,
        "prompt": prompt,
    }
