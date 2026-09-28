"""Deterministic navigation intent resolver.

This module is intentionally used only for explicit navigation mode. It keeps
benchmark and robot-command target selection out of the general QA agent, so
ordinary factual questions can keep using the broader conversational flow.
"""

from __future__ import annotations

import re
from typing import Any

from .tools import (
    get_interaction_events,
    get_object_interactions,
    get_object_last_location,
    get_object_person_interactions,
    list_objects,
    move_to_position,
    search_objects_by_class_id,
)


def _normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower().replace("_", " ")))


def _person_names() -> list[str]:
    result = list_objects(class_name="person", limit=50)
    return [str(item.get("name")) for item in result.get("objects") or [] if item.get("name")]


def _title_name(name: str, people: list[str] | None = None) -> str:
    for known in people or _person_names():
        if known.lower() == str(name or "").lower():
            return known
    return str(name or "").strip().title()


def _candidate_object_names(phrase: str) -> list[str]:
    normalized = _normalize(phrase)
    if not normalized:
        return []
    return [normalized.replace(" ", "_")]


def _resolve_object(phrase: str) -> tuple[dict | None, list[dict]]:
    tool_log: list[dict] = []
    for object_name in _candidate_object_names(phrase):
        result = search_objects_by_class_id(object_name=object_name, limit=5)
        tool_log.append(
            {
                "tool": "search_objects_by_class_id",
                "args": {"object_name": object_name, "limit": 5},
                "result": result,
            }
        )
        rows = result.get("results") or []
        if rows:
            return rows[0], tool_log
    return None, tool_log


def _search_object_candidates(phrase: str) -> tuple[list[dict], list[dict]]:
    tool_log: list[dict] = []
    seen: set[str] = set()
    candidates: list[dict] = []
    for object_name in _candidate_object_names(phrase):
        result = search_objects_by_class_id(object_name=object_name, limit=10)
        tool_log.append(
            {
                "tool": "search_objects_by_class_id",
                "args": {"object_name": object_name, "limit": 10},
                "result": result,
            }
        )
        for row in result.get("results") or []:
            key = str(row.get("object_id"))
            if key not in seen:
                seen.add(key)
                candidates.append(row)
    return candidates, tool_log


def _move(x: Any, y: Any, reason: str, *, standoff_m: float = 0.0) -> tuple[str, list[dict]]:
    args = {"x": float(x), "y": float(y), "reason": reason}
    if standoff_m:
        args["standoff_m"] = float(standoff_m)
    result = move_to_position(**args)
    return result.get("message") or "Navigation target selected.", [
        {"tool": "move_to_position", "args": args, "result": result}
    ]


def _participant(event: dict, *, class_name: str | None = None, name: str | None = None, object_id: Any = None) -> dict | None:
    for item in event.get("participants") or []:
        if class_name and item.get("class_name") != class_name:
            continue
        if name and str(item.get("name") or "").lower() != str(name).lower():
            continue
        if object_id is not None and str(item.get("object_id")) != str(object_id):
            continue
        return item
    return None


def _object_location(object_name: str, reason: str) -> tuple[str, list[dict]] | None:
    obj, log = _resolve_object(object_name)
    if not obj:
        return None
    loc = get_object_last_location(str(obj["object_id"]), require_navigation_usable=False)
    log.append(
        {
            "tool": "get_object_last_location",
            "args": {"object_id": str(obj["object_id"]), "require_navigation_usable": False},
            "result": loc,
        }
    )
    if not loc.get("found") or loc.get("x") is None or loc.get("y") is None:
        return None
    answer, move_log = _move(loc["x"], loc["y"], reason, standoff_m=1.0)
    return answer, log + move_log


def _navigate_to_event(event: dict, reason: str, *, target: str = "event", object_id: Any = None) -> tuple[str, list[dict]] | None:
    selected = None
    standoff_m = 0.0
    if target == "person":
        selected = _participant(event, class_name="person")
    elif target == "object":
        selected = _participant(event, object_id=object_id) if object_id is not None else None
        selected = selected or ((event.get("object_participants") or [None])[0])
        standoff_m = 1.0

    x = selected.get("x") if selected else event.get("x")
    y = selected.get("y") if selected else event.get("y")
    if x is None or y is None:
        return None
    return _move(x, y, reason, standoff_m=standoff_m)


def _object_phrase_from_question(question: str) -> str | None:
    q = _normalize(question)
    objects = list_objects(limit=50).get("objects") or []
    candidate_phrases: list[str] = []
    for item in objects:
        for value in (item.get("name"), item.get("class_name")):
            normalized = _normalize(str(value or ""))
            if normalized:
                candidate_phrases.append(normalized)
                candidate_phrases.append(re.sub(r"\s+\d+$", "", normalized).strip())
    for phrase in sorted(set(candidate_phrases), key=len, reverse=True):
        if phrase and phrase in q:
            return phrase
    match = re.search(r"\b([a-z]+_\d+)\b", question.lower())
    return match.group(1) if match else None


def _navigate_action_user(question: str, *, most_frequent: bool) -> tuple[str, list[dict]] | None:
    phrase = _object_phrase_from_question(question)
    if not phrase:
        return None
    obj, log = _resolve_object(phrase)
    if not obj:
        return None
    object_id = str(obj["object_id"])
    if most_frequent:
        people = get_object_person_interactions(object_id, action="using")
        log.append(
            {
                "tool": "get_object_person_interactions",
                "args": {"object_id": object_id, "action": "using"},
                "result": people,
            }
        )
        entries = people.get("person_interactions") or []
        if not entries:
            return None
        selected = entries[0]
        x = selected.get("latest_relevant_x")
        y = selected.get("latest_relevant_y")
        if x is None or y is None:
            return None
        answer, move_log = _move(
            x,
            y,
            f"{selected.get('name')} used {obj.get('name')} most frequently; using latest relevant interaction coordinate.",
        )
        return answer, log + move_log

    interactions = get_object_interactions(object_id, action="using", sort_order="desc", limit=1)
    log.append(
        {
            "tool": "get_object_interactions",
            "args": {"object_id": object_id, "action": "using", "sort_order": "desc", "limit": 1},
            "result": interactions,
        }
    )
    events = interactions.get("interactions") or []
    if not events:
        return None
    event = events[0]
    person = next((p for p in event.get("co_participants") or [] if p.get("class_name") == "person"), None)
    if not person or person.get("x") is None or person.get("y") is None:
        return None
    answer, move_log = _move(
        person["x"],
        person["y"],
        f"{person.get('name')} is the most recent person observed using {obj.get('name')}.",
    )
    return answer, log + move_log


def _navigate_time_object_or_person(question: str) -> tuple[str, list[dict]] | None:
    time_match = re.search(r"\bat\s+(\d{1,2}:\d{2})\b", question.lower())
    if not time_match:
        return None
    time_value = time_match.group(1)

    people = _person_names()
    person_name = next((name for name in people if name.lower() in question.lower()), None)
    phrase = _object_phrase_from_question(question)

    if "person who interacted with" in question.lower() and phrase:
        obj, resolve_log = _resolve_object(phrase)
        if not obj:
            return None
        events = get_interaction_events(object_id=str(obj["object_id"]), start_time=time_value, end_time=time_value, limit=5)
        log = [
            *resolve_log,
            {
                "tool": "get_interaction_events",
                "args": {"object_id": str(obj["object_id"]), "start_time": time_value, "end_time": time_value, "limit": 5},
                "result": events,
            }
        ]
        if events.get("events"):
            moved = _navigate_to_event(events["events"][0], f"Person interacted with {phrase} at {time_value}.", target="person")
            if moved:
                return moved[0], log + moved[1]

    if "object" in question.lower() and "interacting with" in question.lower() and person_name:
        events = get_interaction_events(person_name=person_name, start_time=time_value, end_time=time_value, limit=5)
        log = [
            {
                "tool": "get_interaction_events",
                "args": {"person_name": person_name, "start_time": time_value, "end_time": time_value, "limit": 5},
                "result": events,
            }
        ]
        if events.get("events"):
            event = events["events"][0]
            obj_participant = (event.get("object_participants") or [None])[0]
            moved = _navigate_to_event(
                event,
                f"{person_name} interacted with {obj_participant.get('name') if obj_participant else 'an object'} at {time_value}.",
                target="object",
                object_id=obj_participant.get("object_id") if obj_participant else None,
            )
            if moved:
                return moved[0], log + moved[1]

    if "where" in question.lower() and phrase:
        obj, resolve_log = _resolve_object(phrase)
        if not obj:
            return None
        action = "opening" if "opened" in question.lower() or "opening" in question.lower() else None
        events = get_interaction_events(
            object_id=str(obj["object_id"]),
            action=action,
            start_time=time_value,
            end_time=time_value,
            limit=5,
        )
        log = [
            *resolve_log,
            {
                "tool": "get_interaction_events",
                "args": {"object_id": str(obj["object_id"]), "action": action, "start_time": time_value, "end_time": time_value, "limit": 5},
                "result": events,
            }
        ]
        if events.get("events"):
            event = events["events"][0]
            obj = (event.get("object_participants") or [None])[0]
            moved = _navigate_to_event(
                event,
                f"{obj.get('name')} {action or 'interaction'} at {time_value}.",
                target="object",
                object_id=obj.get("object_id") if obj else None,
            )
            if moved:
                return moved[0], log + moved[1]
    return None


def _navigate_owned_object(question: str) -> tuple[str, list[dict]] | None:
    match = re.search(r"([A-Za-z]+)\s*['\u2019]s\s+([A-Za-z][A-Za-z0-9_]*)", question)
    if not match:
        return None
    owner = _title_name(match.group(1))
    noun = match.group(2)
    candidates, log = _search_object_candidates(noun)
    if not candidates:
        return None

    selected = None
    selected_score = -1
    for candidate in candidates:
        people = get_object_person_interactions(str(candidate["object_id"]))
        log.append(
            {
                "tool": "get_object_person_interactions",
                "args": {"object_id": str(candidate["object_id"])},
                "result": people,
            }
        )
        for item in people.get("person_interactions") or []:
            if str(item.get("name") or "").lower() == owner.lower():
                score = int(item.get("interaction_count") or 0)
                if score > selected_score:
                    selected = candidate
                    selected_score = score
    if selected is None:
        return None
    moved = _object_location(
        str(selected["name"]),
        f"Resolved {selected.get('name')} as {owner}'s {noun}; using latest object location.",
    )
    if not moved:
        return None
    return moved[0], log + moved[1]


def _navigate_shared_or_last_object(question: str) -> tuple[str, list[dict]] | None:
    phrase = _object_phrase_from_question(question)
    if not phrase:
        return None
    if "last known location" not in question.lower() and "that " not in question.lower() and "shared" not in question.lower():
        return None
    return _object_location(phrase, f"Selected latest location for {phrase}.")


def _navigate_person_pair(question: str) -> tuple[str, list[dict]] | None:
    lower = question.lower()
    if "talk" not in lower:
        return None
    names = [name for name in _person_names() if name.lower() in lower]
    if len(names) < 2:
        return None
    events = get_interaction_events(person_name=names[0], other_person_name=names[1], action="talking_to", limit=10)
    log = [
        {
            "tool": "get_interaction_events",
            "args": {"person_name": names[0], "other_person_name": names[1], "action": "talking_to", "limit": 10},
            "result": events,
        }
    ]
    if not events.get("events"):
        return None
    event = events["events"][0]
    moved = _navigate_to_event(event, f"{names[0]} and {names[1]} talked here.", target="event")
    if moved:
        return moved[0], log + moved[1]
    return None


def _navigate_transport(question: str) -> tuple[str, list[dict]] | None:
    lower = question.lower()
    if "brought" not in lower and "bring" not in lower:
        return None
    actor_match = re.search(r"\b([A-Za-z]+)\s+(?:brought|bring|brings)\b", question)
    actor = _title_name(actor_match.group(1), _person_names()) if actor_match else None
    match = re.search(r"['\u2019]s\s+([A-Za-z][A-Za-z0-9_]*)", question)
    phrase = match.group(1) if match else _object_phrase_from_question(question)
    if not phrase or not actor:
        return None
    obj, log = _resolve_object(phrase)
    if not obj:
        return None
    events = get_interaction_events(object_id=str(obj["object_id"]), person_name=actor, action="bring", limit=5)
    log.append(
        {
            "tool": "get_interaction_events",
            "args": {"object_id": str(obj["object_id"]), "person_name": actor, "action": "bring", "limit": 5},
            "result": events,
        }
    )
    if events.get("events"):
        event = events["events"][0]
        moved = _navigate_to_event(
            event,
            f"{actor} brought {obj.get('name')} here.",
            target="object",
            object_id=obj["object_id"],
        )
        if moved:
            return moved[0], log + moved[1]
    return None


def resolve_navigation_command(question: str) -> tuple[str, list[dict]] | None:
    """Resolve explicit navigation tasks without affecting regular QA."""
    lower = question.lower()
    strategies = []
    if "most recently used" in lower:
        strategies.append(lambda q: _navigate_action_user(q, most_frequent=False))
    if "most frequently" in lower:
        strategies.append(lambda q: _navigate_action_user(q, most_frequent=True))
    strategies.extend(
        [
            _navigate_transport,
            _navigate_time_object_or_person,
            _navigate_owned_object,
            _navigate_shared_or_last_object,
            _navigate_person_pair,
        ]
    )
    for strategy in strategies:
        result = strategy(question)
        if result is not None:
            return result
    return None
