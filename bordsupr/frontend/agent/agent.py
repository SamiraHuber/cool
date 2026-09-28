import ast
import os
from collections import Counter
from typing import Any
import inspect
import json
import re
import uuid

from openai import BadRequestError

from .prompts import CHAT_SYSTEM, CHAT_ONLY
from .tools import TOOL_DISPATCH, TOOL_SCHEMAS, YOLO_CLASS_NAMES
from .vlm_client import (
    active_backend_supports_required_tool_choice,
    get_vlm_client,
    get_vlm_extra_body,
    get_vlm_model,
)

_SYSTEM_PROMPT = CHAT_SYSTEM


def _build_tool_usage_guide() -> str:
    lines = ["AVAILABLE TOOLS:"]
    for schema in TOOL_SCHEMAS:
        function = schema.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        lines.append(f"- {name}")
    return "\n".join(lines)


def _build_yolo_class_id_guide() -> str:
    lines = ["YOLO CLASS IDS:"]
    entries = [f"{class_id}={name}" for class_id, name in sorted(YOLO_CLASS_NAMES.items())]
    chunk_size = 8
    for start in range(0, len(entries), chunk_size):
        lines.append("- " + ", ".join(entries[start:start + chunk_size]))
    return "\n".join(lines)


_YOLO_CLASS_ID_GUIDE = _build_yolo_class_id_guide()
_TOOL_USAGE_GUIDE = _build_tool_usage_guide()

_AGENT_REASONING_PROMPT = _SYSTEM_PROMPT + "\n\n" + _TOOL_USAGE_GUIDE + """

WORKFLOW:
- Follow this process for every incoming question, task, or message:
  1. Understand what the user wants to know or do.
  2. Decide whether tool calls are needed.
  3. If tools are needed, choose the best tool and arguments.
  4. Review each tool result and decide whether another tool call is needed, up to the iteration limit.
  5. Synthesize the final answer from the collected evidence.
- Do not rely on brittle keyword matching. Reason from the user's intent.
- Handle ownership, temporal, location, and person-to-person interaction questions by reasoning over the tool outputs instead of matching fixed question templates.
- If no tool is needed, answer directly.
- If tools are needed for factual claims about the robot's observations, locations, scenes, or interactions, use them before answering.
"""

_CHAT_ONLY_SYSTEM_PROMPT = CHAT_ONLY

def _select_tools_with_vlm(question: str, all_schemas: list[dict]) -> set[str] | None:
    """Ask the VLM which tools are relevant for this question. Returns None on failure."""
    try:
        tool_list = []
        for schema in all_schemas:
            func = schema.get("function", {})
            name = func.get("name", "")
            desc = func.get("description", "")
            short_desc = desc.split(".")[0] + "." if desc else ""
            tool_list.append(f"- {name}: {short_desc}")

        router_prompt = (
            "You are a tool router. Given a user question, select which tools from the list below are likely needed.\n"
            'Return ONLY a JSON object in this exact format: {\"tools\": [\"tool_name1\", \"tool_name2\", ...]}\n'
            'If the question is a simple greeting or chat with no tool need, return {\"tools\": []}.\n'
            "Do not include any explanation, only the JSON.\n\n"
            "Available tools:\n" + "\n".join(tool_list) + "\n\n"
            f"User question: {question}\n"
        )

        client = get_vlm_client()
        response = client.chat.completions.create(
            model=get_vlm_model(),
            messages=[{"role": "user", "content": router_prompt}],
            temperature=0.0,
            max_tokens=256,
            extra_body=get_vlm_extra_body(),
        )
        raw = response.choices[0].message.content or ""
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            tools = set(parsed.get("tools", []))
            valid_names = {s.get("function", {}).get("name") for s in all_schemas}
            return tools & valid_names
    except Exception:
        pass
    return None


def _estimate_token_count(messages: list[dict], tools: list[dict] | None) -> int:
    """Rough token estimate using char count / 4."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(part.get("text", ""))
    if tools:
        total += len(json.dumps(tools))
    return total // 4


def _trim_messages_for_budget(
    messages: list[dict],
    tools: list[dict] | None,
    max_model_len: int = 8192,
) -> tuple[list[dict], list[dict] | None]:
    """Progressively trim messages and tools to fit under the context budget."""
    budget = int(max_model_len * 0.85)

    est = _estimate_token_count(messages, tools)
    if est <= budget:
        return messages, tools

    # Step 1: trim history to 2 turns
    messages = _truncate_messages(messages, max_turns=2)
    est = _estimate_token_count(messages, tools)
    if est <= budget:
        return messages, tools

    # Step 2: strip parameter descriptions from tools
    if tools:
        minimal_tools = []
        for schema in tools:
            minimal = json.loads(json.dumps(schema))
            func = minimal.get("function", {})
            params = func.get("parameters", {}).get("properties", {})
            for prop_info in params.values():
                if "description" in prop_info:
                    prop_info["description"] = ""
            minimal_tools.append(minimal)
        est = _estimate_token_count(messages, minimal_tools)
        if est <= budget:
            return messages, minimal_tools
    else:
        minimal_tools = None

    # Step 3: emergency - keep only system + last user message
    system_msg = None
    user_msg = None
    for m in messages:
        if m.get("role") == "system":
            system_msg = m
        elif m.get("role") == "user":
            user_msg = m
    emergency = []
    if system_msg:
        emergency.append(system_msg)
    if user_msg:
        emergency.append(user_msg)
    return emergency, minimal_tools




_UUID_PATTERN = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _empty_usage() -> dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _make_response_metadata(
    *,
    model: str | None = None,
    used_model: bool = False,
    iterations: int = 0,
    system_prompt: str | None = None,
) -> dict:
    return {
        "model": model or get_vlm_model(),
        "used_model": used_model,
        "iterations": iterations,
        "usage": _empty_usage(),
        "trace": [],
        "system_prompt": system_prompt,
    }


def _accumulate_usage(metadata: dict, response) -> None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        metadata["usage"][key] += int(getattr(usage, key, 0) or 0)


def _append_trace_step(
    metadata: dict,
    *,
    iteration: int,
    assistant_content: str,
    planned_tool_calls: list[dict[str, Any]] | None = None,
    tool_results: list[dict[str, Any]] | None = None,
    final_answer: bool = False,
) -> None:
    metadata.setdefault("trace", []).append(
        {
            "iteration": iteration,
            "assistant_content": assistant_content,
            "planned_tool_calls": planned_tool_calls or [],
            "tool_results": tool_results or [],
            "final_answer": final_answer,
        }
    )


def _run_tool(fn_name: str, args: dict) -> dict:
    fn = TOOL_DISPATCH.get(fn_name)
    if fn is None:
        return {"error": f"Unknown tool: {fn_name}"}

    try:
        if not isinstance(args, dict):
            return {"error": f"Tool arguments for {fn_name} must be a JSON object."}

        valid = inspect.signature(fn).parameters
        unexpected = sorted(key for key in args if key not in valid)
        if unexpected:
            return {
                "error": (
                    f"Unsupported argument(s) for {fn_name}: {', '.join(unexpected)}. "
                    f"Allowed arguments: {', '.join(valid)}"
                )
            }

        return fn(**args)
    except Exception as exc:
        return {"error": str(exc)}


def _filter_tool_dispatch(allowed_tool_names: set[str] | None) -> dict:
    if allowed_tool_names is None:
        return TOOL_DISPATCH
    return {name: fn for name, fn in TOOL_DISPATCH.items() if name in allowed_tool_names}


def _filter_tool_schemas(allowed_tool_names: set[str] | None) -> list[dict]:
    if allowed_tool_names is None:
        return TOOL_SCHEMAS
    return [
        schema for schema in TOOL_SCHEMAS
        if schema.get("function", {}).get("name") in allowed_tool_names
    ]


def _coerce_coordinate(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_coordinate_candidates(payload: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            x = _coerce_coordinate(node.get("x"))
            y = _coerce_coordinate(node.get("y"))
            if x is not None and y is not None:
                label = (
                    node.get("name")
                    or node.get("object_id")
                    or node.get("interaction_id")
                    or node.get("observation_id")
                    or node.get("room")
                )
                navigation_usable = node.get("navigation_usable")
                if navigation_usable is not None:
                    navigation_usable = bool(navigation_usable)
                position_source = node.get("position_source")
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "label": label,
                        "navigation_usable": navigation_usable,
                        "position_source": position_source,
                    }
                )
            for child in node.values():
                visit(child)
            return

        if isinstance(node, list):
            for child in node:
                visit(child)

    visit(payload)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[float, float, str]] = set()
    for candidate in candidates:
        key = (
            candidate["x"],
            candidate["y"],
            str(candidate.get("label") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _validate_move_to_position_args(args: dict[str, Any], tool_log: list[dict]) -> str | None:
    x = _coerce_coordinate(args.get("x"))
    y = _coerce_coordinate(args.get("y"))
    if x is None or y is None:
        return "move_to_position requires numeric x and y arguments."

    recent_candidates: list[dict[str, Any]] = []
    for item in reversed(tool_log):
        tool_name = str(item.get("tool") or "")
        if tool_name == "move_to_position":
            continue
        recent_candidates.extend(_extract_coordinate_candidates(item.get("result")))
        if len(recent_candidates) >= 12:
            break

    if not recent_candidates:
        return None

    matching_candidates = [
        candidate for candidate in recent_candidates
        if candidate["x"] == x and candidate["y"] == y
    ]
    if matching_candidates:
        if any(candidate.get("navigation_usable") is not False for candidate in matching_candidates):
            return None
        return (
            "move_to_position cannot use YOLO-only object coordinates for navigation. "
            "Pick coordinates backed by depth or DynoSAM instead."
        )

    # Coordinates don't match any prior tool result, but the user may have
    # explicitly provided them. Allow the call rather than blocking re-planning.
    return None


def _history_to_messages(history: list[dict] | None) -> list[dict]:
    """Convert recent chat history into chat-completions messages."""
    if not history:
        return []

    messages: list[dict] = []
    for entry in history[-8:]:
        role = entry.get("role")
        content = str(entry.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue

        tool_log = entry.get("tool_log")
        if role == "assistant" and tool_log:
            content = (
                f"{content}\n\n"
                "Previous tool results for context:\n"
                f"{_summarize_tool_log(tool_log)}"
            )

        messages.append({"role": role, "content": content})

    return messages


def _summarize_tool_log(tool_log: list[dict] | None) -> str:
    """Return a compact plain-text summary instead of embedding raw JSON blobs."""
    if not tool_log:
        return "No tool results recorded."

    lines: list[str] = []
    for item in tool_log[:6]:
        tool_name = str(item.get("tool") or "unknown_tool")
        args = item.get("args") or {}
        result = item.get("result") or {}

        arg_bits: list[str] = []
        for key in ("room_name", "object_id", "x", "y", "reason", "limit", "start_time", "end_time", "time_filter"):
            if key in args:
                value = str(args[key]).replace("\n", " ").strip()
                if len(value) > 80:
                    value = value[:77] + "..."
                arg_bits.append(f"{key}={value}")

        status_bits: list[str] = []
        for key in ("found", "ok", "executed", "status", "room_name", "x", "y", "error", "message", "note"):
            if key not in result:
                continue
            value = str(result[key]).replace("\n", " ").strip()
            if len(value) > 120:
                value = value[:117] + "..."
            status_bits.append(f"{key}={value}")

        line = f"- {tool_name}"
        if arg_bits:
            line += f"({', '.join(arg_bits)})"
        if status_bits:
            line += f" -> {', '.join(status_bits)}"
        lines.append(line)

    remaining = len(tool_log) - len(lines)
    if remaining > 0:
        lines.append(f"- ... {remaining} more tool call(s)")

    summary = "\n".join(lines)
    if len(summary) > 1800:
        summary = summary[:1797] + "..."
    return summary


def _participant_names(participants: list[dict] | None) -> list[str]:
    names: list[str] = []
    for participant in participants or []:
        name = participant.get("name") or participant.get("class_name") or participant.get("object_id")
        if name is not None:
            names.append(str(name))
    return names


def _sort_timestamp(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _top_action_counts(counter: Counter[str], limit: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "action": action,
            "count": count,
        }
        for action, count in counter.most_common(limit)
    ]


def _aggregate_interaction_participants(events: list[dict], participant_key: str) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = {}

    for event in events:
        created_at = _sort_timestamp(event.get("created_at") or event.get("local_created_at"))
        action = str(event.get("action") or "unknown")
        event_x = event.get("x")
        event_y = event.get("y")
        room = event.get("room")
        interaction_id = event.get("interaction_id")

        for participant in event.get(participant_key) or []:
            entity_key = str(
                participant.get("object_id")
                or participant.get("name")
                or participant.get("class_name")
                or "unknown"
            )
            entry = aggregates.setdefault(
                entity_key,
                {
                    "object_id": participant.get("object_id"),
                    "name": participant.get("name") or participant.get("class_name") or participant.get("object_id"),
                    "class_name": participant.get("class_name"),
                    "role": participant.get("role"),
                    "interaction_count": 0,
                    "latest_created_at": "",
                    "latest_interaction_id": None,
                    "latest_room": None,
                    "latest_x": None,
                    "latest_y": None,
                    "action_counts": Counter(),
                },
            )
            entry["interaction_count"] += 1
            entry["action_counts"][action] += 1

            if created_at >= entry["latest_created_at"]:
                entry["latest_created_at"] = created_at
                entry["latest_interaction_id"] = interaction_id
                entry["latest_room"] = room
                entry["latest_x"] = participant.get("x") if participant.get("x") is not None else event_x
                entry["latest_y"] = participant.get("y") if participant.get("y") is not None else event_y

    ranked = sorted(
        aggregates.values(),
        key=lambda item: (
            -int(item["interaction_count"]),
            _sort_timestamp(item["latest_created_at"]),
            str(item.get("name") or ""),
        ),
        reverse=False,
    )
    ranked.sort(
        key=lambda item: (
            int(item["interaction_count"]),
            _sort_timestamp(item["latest_created_at"]),
            str(item.get("name") or ""),
        ),
        reverse=True,
    )

    return [
        {
            "object_id": item.get("object_id"),
            "name": item.get("name"),
            "class_name": item.get("class_name"),
            "role": item.get("role"),
            "interaction_count": item.get("interaction_count"),
            "latest_created_at": item.get("latest_created_at") or None,
            "latest_interaction_id": item.get("latest_interaction_id"),
            "latest_room": item.get("latest_room"),
            "latest_x": item.get("latest_x"),
            "latest_y": item.get("latest_y"),
            "top_actions": _top_action_counts(item.get("action_counts") or Counter()),
        }
        for item in ranked[:5]
    ]


def _summarize_participants(participants: list[dict] | None) -> list[dict]:
    return [
        {
            "object_id": item.get("object_id"),
            "name": item.get("name") or item.get("class_name") or item.get("object_id"),
            "role": item.get("role"),
            "x": item.get("x"),
            "y": item.get("y"),
        }
        for item in (participants or [])[:4]
    ]


def _summarize_search_results(results: list[dict]) -> list[dict]:
    return [
        {
            "object_id": item.get("object_id"),
            "name": item.get("name"),
            "class_name": item.get("class_name"),
            "room": item.get("room"),
            "last_seen_at": item.get("last_seen_at"),
            "observation_count": item.get("observation_count"),
            "observation_id": item.get("observation_id"),
            "x": item.get("x"),
            "y": item.get("y"),
        }
        for item in results[:3]
    ]


def _summarize_interaction_events(events: list[dict]) -> list[dict]:
    summary: list[dict] = []
    for event in events[:5]:
        summary.append(
            {
                "interaction_id": event.get("interaction_id"),
                "action": event.get("action"),
                "caption": event.get("caption"),
                "created_at": event.get("created_at"),
                "local_created_at": event.get("local_created_at"),
                "room": event.get("room"),
                "x": event.get("x"),
                "y": event.get("y"),
                "people": _summarize_participants(event.get("person_participants")),
                "objects": _summarize_participants(event.get("object_participants")),
            }
        )
    return summary


def _summarize_object_interactions(interactions: list[dict], limit: int = 5) -> list[dict]:
    summary: list[dict] = []
    for interaction in interactions[:limit]:
        summary.append(
            {
                "interaction_id": interaction.get("interaction_id"),
                "action": interaction.get("action"),
                "caption": interaction.get("caption"),
                "created_at": interaction.get("created_at"),
                "local_created_at": interaction.get("local_created_at"),
                "room": interaction.get("room"),
                "co_participants": _participant_names(interaction.get("co_participants")),
            }
        )
    return summary


def _summarize_interaction_breakdown(items: list[dict] | None) -> list[dict]:
    return [
        {
            "action": item.get("action"),
            "count": item.get("count"),
        }
        for item in (items or [])[:5]
    ]


def _summarize_top_participants(items: list[dict] | None) -> list[dict]:
    return [
        {
            "name": item.get("name") or item.get("class_name") or item.get("object_id"),
            "object_id": item.get("object_id"),
            "count": item.get("count"),
        }
        for item in (items or [])[:5]
    ]


def _summarize_unique_co_participants(interactions: list[dict], limit: int = 10) -> list[dict]:
    counts: Counter[str] = Counter()
    labels: dict[str, dict[str, Any]] = {}
    for interaction in interactions:
        for participant in interaction.get("co_participants") or []:
            key = str(
                participant.get("object_id")
                or participant.get("name")
                or participant.get("class_name")
                or "unknown"
            )
            counts[key] += 1
            labels.setdefault(
                key,
                {
                    "name": participant.get("name") or participant.get("class_name") or participant.get("object_id"),
                    "object_id": participant.get("object_id"),
                },
            )
    return [
        {
            **labels[key],
            "count": count,
        }
        for key, count in counts.most_common(limit)
    ]


def _summarize_object_person_interactions(items: list[dict] | None) -> list[dict]:
    return [
        {
            "name": item.get("name") or item.get("object_id"),
            "object_id": item.get("object_id"),
            "interaction_count": item.get("interaction_count"),
            "actions": _summarize_interaction_breakdown(item.get("actions")),
            "interaction_ids": (item.get("interaction_ids") or [])[:5],
        }
        for item in (items or [])[:5]
    ]


def _summarize_observations(results: list[dict]) -> list[dict]:
    return [
        {
            "observation_id": item.get("observation_id"),
            "scene_id": item.get("scene_id"),
            "created_at": item.get("created_at"),
            "room": item.get("room"),
            "x": item.get("x"),
            "y": item.get("y"),
            "z": item.get("z"),
        }
        for item in results[:3]
    ]


def _summarize_objects(results: list[dict]) -> list[dict]:
    return [
        {
            "object_id": item.get("object_id"),
            "name": item.get("name"),
            "class_name": item.get("class_name"),
            "last_seen_at": item.get("last_seen_at"),
            "observation_count": item.get("observation_count"),
        }
        for item in results[:3]
    ]


def _summarize_tool_result_for_model(tool_name: str, args: dict[str, Any], result: Any) -> str:
    """Return a sparse tool summary for the next model turn instead of raw JSON.
    
    If the raw result is small enough, pass it through verbatim so the model has
    full details and does not feel the need to re-query."""
    if isinstance(result, dict):
        raw_content = json.dumps({"tool": tool_name, "result": result}, default=str)
        if len(raw_content) <= 3000:
            return raw_content
    if not isinstance(result, dict):
        content = json.dumps({"tool": tool_name, "result": result}, default=str)
        return content if len(content) <= 1800 else content[:1797] + "..."

    summary: dict[str, Any] = {"tool": tool_name}
    for key in (
        "found",
        "error",
        "hint",
        "message",
        "query",
        "class_id",
        "class_name",
        "object_name",
        "object_id",
        "person_name",
        "other_person_name",
        "room",
        "limit",
        "start_time",
        "end_time",
        "time_filter",
        "action_filter",
    ):
        if key in result:
            summary[key] = result.get(key)

    if tool_name == "search_objects_by_class_id":
        matches = result.get("results") or []
        summary["match_count"] = len(matches)
        summary["top_matches"] = _summarize_search_results(matches)
    elif tool_name == "get_interaction_events":
        events = result.get("events") or []
        summary["event_count"] = len(events)
        summary["recent_events"] = _summarize_interaction_events(events)
        summary["top_people"] = _aggregate_interaction_participants(events, "person_participants")
        summary["top_objects"] = _aggregate_interaction_participants(events, "object_participants")
        # If exact-time matches exist, add a hard stop directive
        if any(e.get("matched_time_filter") for e in events):
            summary["note"] = "EXACT TIME MATCH — these events are the complete answer. STOP and synthesize immediately. Do not verify with other tools."
    elif tool_name == "get_object_observations":
        observations = result.get("results") or []
        summary["total_count"] = result.get("total_count")
        summary["returned_count"] = result.get("returned_count", len(observations))
        summary["recent_observations"] = _summarize_observations(observations)
    elif tool_name == "get_object_interactions":
        interactions = result.get("interactions") or []
        summary["interaction_count"] = result.get("interaction_count")
        summary["returned_count"] = result.get("returned_count", len(interactions))
        summary["action_breakdown"] = _summarize_interaction_breakdown(result.get("action_breakdown"))
        summary["top_co_participants"] = _summarize_top_participants(result.get("top_co_participants"))
        pba = result.get("people_by_action") or {}
        if pba:
            summary["people_by_action"] = {
                action: [
                    {"name": p.get("name") or p.get("class_name"), "count": p.get("count")}
                    for p in people[:3]
                ]
                for action, people in pba.items()
            }
            summary["note"] = "The answer is in people_by_action above. STOP and answer immediately. Do not verify with other tools."
        summary["recent_interactions"] = _summarize_object_interactions(interactions)
    elif tool_name == "get_object_person_interactions":
        people = result.get("person_interactions") or []
        summary["person_count"] = result.get("person_count")
        summary["interaction_count"] = result.get("interaction_count")
        summary["people"] = _summarize_object_person_interactions(people)
        if len(people) >= 3:
            summary["note"] = (
                "3+ people interacted with this object. Decide ownership by BOTH frequency AND action type: "
                "it is PUBLIC/SHARED only if their interaction_count values are similar (no clearly dominant person) "
                "AND they share similar ownership-relevant actions (e.g. 'using','holding','carrying'). "
                "If one person has a clearly higher count or is the only one performing the meaningful ownership actions "
                "(others only 'next_to'/'near'), name that person as the primary owner instead of calling it shared."
            )
    elif tool_name == "get_person_interaction_summary":
        co_participants = result.get("co_participants") or []
        person_co_participants = result.get("person_co_participants") or []
        summary["total_events_analyzed"] = result.get("total_events_analyzed")
        summary["co_participant_count"] = result.get("co_participant_count")
        summary["top_co_participants"] = [
            {
                "name": item.get("name") or item.get("class_name") or item.get("object_id"),
                "object_id": item.get("object_id"),
                "interaction_count": item.get("interaction_count"),
                "actions": _summarize_interaction_breakdown(item.get("actions")),
            }
            for item in co_participants[:3]
        ]
        if person_co_participants:
            summary["person_co_participants"] = [
                {
                    "name": item.get("name") or item.get("object_id"),
                    "object_id": item.get("object_id"),
                    "interaction_count": item.get("interaction_count"),
                }
                for item in person_co_participants[:5]
            ]
    elif tool_name in ("get_object_last_location", "get_object_first_location", "get_robot_location"):
        for key in ("found", "x", "y", "z", "room", "last_seen_at", "first_seen_at", "name"):
            if key in result:
                summary[key] = result.get(key)
    elif tool_name == "get_room_navigation_target":
        for key in ("found", "room_name", "x", "y"):
            if key in result:
                summary[key] = result.get(key)
    elif tool_name == "move_to_position":
        for key in ("ok", "x", "y", "reason", "status", "message"):
            if key in result:
                summary[key] = result.get(key)
    elif tool_name == "get_object_summary":
        for key in ("found", "class_name", "name", "observation_count", "interaction_count", "x", "y", "room", "last_seen_at"):
            if key in result:
                summary[key] = result.get(key)
    else:
        for key in ("x", "y", "z", "last_seen_at", "first_seen_at", "timestamp", "scene_id", "name"):
            if key in result:
                summary[key] = result.get(key)
        for list_key in ("results", "events", "objects", "interactions", "person_interactions", "co_participants"):
            value = result.get(list_key)
            if isinstance(value, list):
                summary[f"{list_key}_count"] = len(value)
                if list_key in ("results", "person_interactions", "co_participants"):
                    summary["sample"] = value[:2]

    summary["requested_args"] = args
    content = json.dumps(summary, default=str)
    if len(content) <= 1800:
        return content

    summary.pop("requested_args", None)
    content = json.dumps(summary, default=str)
    if len(content) <= 1800:
        return content

    fallback = {"tool": tool_name, "summary": f"Sparse summary only. See counts and top items.", **{
        key: value for key, value in summary.items()
        if key in {"tool", "found", "error", "query", "object_id", "person_name", "other_person_name", "room", "limit", "start_time", "end_time", "time_filter", "action_filter",
               "match_count", "event_count", "interaction_count", "object_count", "total_count", "returned_count", "person_count", "co_participant_count"}
    }}
    content = json.dumps(fallback, default=str)
    return content if len(content) <= 1800 else content[:1797] + "..."


def _extract_textual_tool_calls(content: str | None, allowed_tool_names: set[str] | None = None) -> list[dict]:
    """Best-effort fallback for models that print function calls instead of emitting tool_calls."""
    if not isinstance(content, str) or not content:
        return []

    allowed_dispatch = _filter_tool_dispatch(allowed_tool_names)
    tool_names = "|".join(re.escape(name) for name in allowed_dispatch)
    if not tool_names:
        return []
    pattern = re.compile(rf"(?P<call>(?P<name>{tool_names})\s*\((?P<args>.*?)\))", re.DOTALL)
    parsed_calls: list[dict] = []

    for match in pattern.finditer(content):
        fn_name = match.group("name")
        raw_args = match.group("args").strip()
        try:
            expr = ast.parse(f"{fn_name}({raw_args})", mode="eval").body
        except SyntaxError:
            continue

        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name):
            continue
        if expr.func.id != fn_name:
            continue

        kwargs: dict = {}
        param_names = list(inspect.signature(allowed_dispatch[fn_name]).parameters)
        valid = set(param_names)
        try:
            if expr.args:
                if len(expr.args) > len(param_names):
                    raise ValueError("Unsupported positional arguments")
                for index, arg in enumerate(expr.args):
                    kwargs[param_names[index]] = ast.literal_eval(arg)
            for kw in expr.keywords:
                if kw.arg is None or kw.arg not in valid:
                    raise ValueError("Unsupported argument")
                kwargs[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            continue

        if kwargs:
            parsed_calls.append(
                {
                    "id": f"fallback_{uuid.uuid4().hex}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": json.dumps(kwargs),
                    },
                }
            )

    return parsed_calls


def _truncate_messages(messages: list[dict], max_turns: int = 4) -> list[dict]:
    """Keep system prompt + user question + last N assistant/tool pairs to stay under context limits."""
    # Find system and user messages (typically first and last of the prefix)
    system_msg = None
    user_msg = None
    prefix_indices = set()
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "system":
            system_msg = m
            prefix_indices.add(i)
        elif role == "user" and i not in prefix_indices:
            # Keep the last user message as the question
            user_msg = m
    # Rebuild: keep prefix, then last N assistant/tool pairs
    body = []
    pair: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role in ("system", "user"):
            continue
        if role == "assistant":
            if pair:
                body.append(pair)
            pair = [m]
        elif role == "tool" and pair:
            pair.append(m)
    if pair:
        body.append(pair)
    kept_pairs = body[-max_turns:] if len(body) > max_turns else body
    result: list[dict] = []
    if system_msg:
        result.append(system_msg)
    if user_msg:
        result.append(user_msg)
    for pair in kept_pairs:
        result.extend(pair)
    return result


def run_custom_agent(
    question: str,
    system_prompt: str,
    history: list[dict] | None = None,
    max_iterations: int = 10,
    allow_tools: bool = True,
    require_tool_use: bool = False,
    force_first_tool_use: bool = True,
    allowed_tool_names: set[str] | None = None,
    return_metadata: bool = False,
) -> tuple[str, list[dict]] | tuple[str, list[dict], dict]:
    """Run the shared VLM agent loop with a custom system prompt."""
    metadata = _make_response_metadata(system_prompt=system_prompt)
    client = get_vlm_client()
    tool_schemas = _filter_tool_schemas(allowed_tool_names) if allow_tools else None
    allowed_dispatch = _filter_tool_dispatch(allowed_tool_names) if allow_tools else {}
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        *_history_to_messages(history),
        {"role": "user", "content": question},
    ]
    tool_log: list[dict] = []

    max_model_len = int(os.getenv("VLM_MAX_MODEL_LEN", "8192"))
    trimmed_for_emergency = False

    for iteration_index in range(1, max_iterations + 1):
        messages = _truncate_messages(messages, max_turns=4)
        tool_choice = None
        if allow_tools:
            if require_tool_use and force_first_tool_use and not tool_log and active_backend_supports_required_tool_choice():
                tool_choice = "required"
            else:
                tool_choice = "auto"

        call_messages, call_tools = messages, tool_schemas
        if not trimmed_for_emergency:
            call_messages, call_tools = _trim_messages_for_budget(messages, tool_schemas, max_model_len)

        try:
            response = client.chat.completions.create(
                model=get_vlm_model(),
                messages=call_messages,
                tools=call_tools,
                tool_choice=tool_choice,
                extra_body=get_vlm_extra_body(),
            )
        except BadRequestError as bre:
            err_msg = str(bre).lower()
            if "maximum context length" in err_msg or "too long" in err_msg or "context length" in err_msg:
                if not trimmed_for_emergency:
                    call_messages, call_tools = _trim_messages_for_budget(messages, tool_schemas, max_model_len)
                    trimmed_for_emergency = True
                    response = client.chat.completions.create(
                        model=get_vlm_model(),
                        messages=call_messages,
                        tools=call_tools,
                        tool_choice=tool_choice,
                        extra_body=get_vlm_extra_body(),
                    )
                else:
                    raise
            else:
                raise
        metadata["used_model"] = True
        metadata["iterations"] += 1
        metadata["model"] = getattr(response, "model", None) or metadata["model"]
        _accumulate_usage(metadata, response)
        msg = response.choices[0].message
        tool_calls = list(msg.tool_calls or []) if allow_tools else []

        if allow_tools and not tool_calls:
            tool_calls = _extract_textual_tool_calls(msg.content, allowed_tool_names=allowed_tool_names)

        if not tool_calls:
            if require_tool_use and not tool_log and not active_backend_supports_required_tool_choice():
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({
                    "role": "user",
                    "content": "Please select a tool to handle the current issue.",
                })
                continue
            answer = msg.content or "I could not find an answer."
            _append_trace_step(
                metadata,
                iteration=iteration_index,
                assistant_content=msg.content or "",
                final_answer=True,
            )
            if return_metadata:
                return answer, tool_log, metadata
            return answer, tool_log

        planned_tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            fn = tc.function if hasattr(tc, "function") else tc["function"]
            fn_name = fn.name if hasattr(fn, "name") else fn["name"]
            try:
                raw_arguments = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
                args = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            planned_tool_calls.append({"tool": fn_name, "args": args})

        trace_step = {
            "iteration": iteration_index,
            "assistant_content": msg.content or "",
            "planned_tool_calls": planned_tool_calls,
            "tool_results": [],
            "final_answer": False,
        }
        metadata.setdefault("trace", []).append(trace_step)

        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)

        for tc in tool_calls:
            fn = tc.function if hasattr(tc, "function") else tc["function"]
            tc_id = tc.id if hasattr(tc, "id") else tc["id"]
            fn_name = fn.name if hasattr(fn, "name") else fn["name"]
            try:
                raw_arguments = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
                args = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}

            def _semantic_args(tool_name: str, a: dict) -> dict:
                """Return the 'semantic' arguments that define a unique query, ignoring pagination/sort."""
                if tool_name in {"get_interaction_events", "get_object_interactions", "get_object_observations"}:
                    return {k: v for k, v in a.items() if k not in {"limit", "sort_order"}}
                return a

            if tool_log and len(tool_log) >= 2:
                last_two = tool_log[-2:]
                repeated = any(
                    prev["tool"] == fn_name and prev["args"] == args
                    for prev in last_two
                )
                semantic_repeated = any(
                    prev["tool"] == fn_name and _semantic_args(fn_name, prev["args"]) == _semantic_args(fn_name, args)
                    for prev in last_two
                )
                if repeated or semantic_repeated:
                    last_result = last_two[-1].get("result") or {}
                    # Allow retry if the previous result was an error
                    if isinstance(last_result, dict) and last_result.get("error"):
                        pass  # fall through to normal execution
                    else:
                        result = {
                            "error": (
                                f"You already called {fn_name} with these exact arguments (only limit/sort_order changed). "
                                f"STOP and synthesize your answer from the previous result. "
                                f"Previous result: {json.dumps(last_result, default=str)[:400]}. "
                                f"Do not repeat the same call; answer immediately."
                            )
                        }
                        tool_log.append({"tool": fn_name, "args": args, "result": result})
                        trace_step["tool_results"].append({"tool": fn_name, "args": args, "result": result})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": json.dumps(result),
                        })
                        continue

            if fn_name not in allowed_dispatch:
                result = {"error": f"Tool not allowed in this context: {fn_name}"}
            elif fn_name == "move_to_position":
                move_validation_error = _validate_move_to_position_args(args, tool_log)
                result = {"error": move_validation_error} if move_validation_error else _run_tool(fn_name, args)
            else:
                result = _run_tool(fn_name, args)

            # Phase 3: error recovery — add hints for common failure patterns
            if isinstance(result, dict) and result.get("error"):
                err_text = str(result["error"]).lower()
                if "bigint" in err_text or "invalid input syntax" in err_text:
                    result["hint"] = (
                        "You passed a name string where a numeric object_id is required. "
                        "Call search_objects_by_class_id with the object_name FIRST to get the object_id, "
                        "then retry with that object_id."
                    )
                elif "not found" in err_text or "no results" in err_text:
                    result["hint"] = (
                        "No results were found. Try broadening your search: remove filters, "
                        "use a different name, or call list_objects to see available objects."
                    )
                elif "required" in err_text and "missing" in err_text:
                    result["hint"] = (
                        "A required argument is missing. Check the tool schema and provide all required parameters."
                    )

            tool_log.append({"tool": fn_name, "args": args, "result": result})
            trace_step["tool_results"].append({"tool": fn_name, "args": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": _summarize_tool_result_for_model(fn_name, args, result),
                }
            )

    if tool_log:
        last_tool = tool_log[-1]
        last_result = last_tool.get("result") or {}
        if last_tool.get("tool") == "move_to_position" and last_result.get("ok"):
            answer = str(last_result.get("message") or "Navigation target selected.")
            if return_metadata:
                return answer, tool_log, metadata
            return answer, tool_log

    # The loop exhausted max_iterations without the model producing a final answer.
    # Instead of giving up, make one final no-tools synthesis call so the model answers
    # from the facts it already gathered in tool_log.
    answer = _synthesize_final_answer_from_tool_log(question, tool_log, metadata, max_model_len)
    if return_metadata:
        return answer, tool_log, metadata
    return answer, tool_log


def _synthesize_final_answer_from_tool_log(
    question: str,
    tool_log: list[dict],
    metadata: dict,
    max_model_len: int,
) -> str:
    """Best-effort final answer when the reasoning loop hits the iteration cap.

    Makes one last LLM call (no tools) over a compact rendering of the gathered tool
    results. Falls back to a deterministic summary if the call fails or there is nothing
    to synthesize.
    """
    fallback = "I reached the maximum reasoning steps without a conclusive answer."
    if not tool_log:
        return fallback

    # Compact rendering of what was gathered (cap size to stay within budget).
    lines = []
    for entry in tool_log:
        tool = entry.get("tool")
        result = entry.get("result")
        try:
            rendered = json.dumps(result, default=str)
        except Exception:
            rendered = str(result)
        lines.append(f"- {tool}: {rendered[:600]}")
    gathered = "\n".join(lines)
    # Hard cap on the evidence blob.
    if len(gathered) > 6000:
        gathered = gathered[:6000] + "\n... (truncated)"

    synthesis_messages = [
        {
            "role": "system",
            "content": (
                "You are answering a question using data already retrieved from a robot's "
                "object/interaction database. Answer the user's question directly and concisely "
                "using ONLY the evidence below. If the evidence identifies a specific object or "
                "person, name it (include the object_id if present). If the evidence is clearly "
                "insufficient, say so briefly and state what is known. Do NOT ask for more tool calls."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nEvidence gathered:\n{gathered}\n\nAnswer the question now.",
        },
    ]

    try:
        client = get_vlm_client()
        response = client.chat.completions.create(
            model=get_vlm_model(),
            messages=synthesis_messages,
            extra_body=get_vlm_extra_body(),
        )
        metadata["used_model"] = True
        _accumulate_usage(metadata, response)
        content = (response.choices[0].message.content or "").strip()
        _append_trace_step(
            metadata,
            iteration=metadata.get("iterations", 0) + 1,
            assistant_content=content,
            final_answer=True,
        )
        if content:
            return content
    except Exception:
        pass

    # Deterministic fallback: surface a short summary rather than the bare "max steps" line.
    return (
        "I gathered some information but reached the maximum reasoning steps before "
        "forming a complete answer. Here is what I found:\n" + gathered[:1500]
    )


def _direct_tool_answer(answer: str, tool_log: list[dict], return_metadata: bool):
    if return_metadata:
        metadata = _make_response_metadata(system_prompt=_AGENT_REASONING_PROMPT)
        return answer, tool_log, metadata
    return answer, tool_log


def _record_tool(tool_log: list[dict], tool: str, args: dict[str, Any]) -> Any:
    result = _run_tool(tool, args)
    tool_log.append({"tool": tool, "args": args, "result": result})
    return result


def _try_possessive_navigation(question: str) -> tuple[str, list[dict]] | None:
    """Handle 'go to X's Y' directly in code, bypassing LLM reasoning."""
    # Match patterns like: "go to names's ball", "names's ball", "navigate to names's ball"
    match = re.match(
        r"\s*(?:go\s+(?:to\s+)?|navigate\s+(?:to\s+)?)?([A-Za-z_][A-Za-z0-9_]*)\s*['\u2019]?s\s+([A-Za-z][A-Za-z0-9_]*(?:\s+[A-Za-z][A-Za-z0-9_]*)*)",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None

    x_name = match.group(1).strip()
    y_name = match.group(2).strip().lower()
    tool_log: list[dict] = []

    # Step 1: Find X (no class_id)
    x_result = _record_tool(tool_log, "search_objects_by_class_id", {"object_name": x_name})
    x_results = x_result.get("results", [])
    if not x_results:
        return None
    x = x_results[0]
    x_id = str(x["object_id"])

    # Step 2: Resolve Y to exact YOLO class name (with space-normalized fallback)
    y_normalized = y_name.replace(" ", "").replace("_", "")
    y_exact_name = None
    for cid, cname in sorted(YOLO_CLASS_NAMES.items()):
        cname_lower = cname.lower()
        cname_normalized = cname_lower.replace(" ", "").replace("_", "")
        if y_name == cname_lower or y_name in cname_lower or cname_lower in y_name or y_normalized == cname_normalized:
            y_exact_name = cname
            break

    # Step 3: Find interactions between X and Y-class objects
    interactions = _record_tool(
        tool_log,
        "get_interaction_events",
        {
            "object_id": x_id,
            "co_participant_class_name": y_exact_name or y_name,
            "limit": 20,
        },
    )
    events = interactions.get("events", [])

    # Step 4: Extract the most recent Y-class participant
    target_y_id = None
    for event in events:
        for participant in event.get("object_participants", []):
            p_class = str(participant.get("class_name") or "").lower()
            p_class_normalized = p_class.replace(" ", "").replace("_", "")
            if y_exact_name and p_class == y_exact_name.lower():
                target_y_id = str(participant["object_id"])
                break
            elif not y_exact_name and (y_name in p_class or p_class in y_name):
                target_y_id = str(participant["object_id"])
                break
            elif y_normalized and y_normalized == p_class_normalized:
                target_y_id = str(participant["object_id"])
                break
        if target_y_id:
            break

    if target_y_id:
        # Prefer depth-backed position
        loc = _record_tool(
            tool_log,
            "get_object_last_location",
            {"object_id": target_y_id, "require_navigation_usable": True},
        )
        if not loc.get("found"):
            loc = _record_tool(
                tool_log,
                "get_object_last_location",
                {"object_id": target_y_id, "require_navigation_usable": False},
            )
        if loc.get("found") and loc.get("x") is not None and loc.get("y") is not None:
            _record_tool(
                tool_log,
                "move_to_position",
                {
                    "x": loc["x"],
                    "y": loc["y"],
                    "reason": f"Possessive navigation: {x_name}'s {y_name}",
                },
            )
            room = loc.get("room") or "unknown"
            source_note = " (depth-backed)" if loc.get("navigation_usable") else ""
            answer = (
                f"Navigating to {x_name}'s {y_name} (object {target_y_id}) at "
                f"({loc['x']:.2f}, {loc['y']:.2f}) in {room}{source_note}."
            )
            return answer, tool_log

    # Fallback: search for Y independently and navigate to most recent depth-backed position
    y_search_args: dict[str, Any] = {"object_name": y_name} if y_exact_name is None else {"class_id": next((cid for cid, cname in YOLO_CLASS_NAMES.items() if cname == y_exact_name), None)}
    if y_search_args.get("class_id") is not None:
        y_search = _record_tool(tool_log, "search_objects_by_class_id", y_search_args)
        y_results = y_search.get("results", [])
        if y_results:
            y = y_results[0]
            y_id = str(y["object_id"])
            loc = _record_tool(
                tool_log,
                "get_object_last_location",
                {"object_id": y_id, "require_navigation_usable": True},
            )
            if not loc.get("found"):
                loc = _record_tool(
                    tool_log,
                    "get_object_last_location",
                    {"object_id": y_id, "require_navigation_usable": False},
                )
            if loc.get("found") and loc.get("x") is not None and loc.get("y") is not None:
                _record_tool(
                    tool_log,
                    "move_to_position",
                    {
                        "x": loc["x"],
                        "y": loc["y"],
                        "reason": f"Possessive fallback: {x_name}'s {y_name}",
                    },
                )
                room = loc.get("room") or "unknown"
                source_note = " (depth-backed)" if loc.get("navigation_usable") else ""
                answer = (
                    f"Navigating to the most recently seen {y_name} at "
                    f"({loc['x']:.2f}, {loc['y']:.2f}) in {room}{source_note}."
                )
                return answer, tool_log

    return None


def _try_person_navigation(question: str) -> tuple[str, list[dict]] | None:
    """Handle 'go to [person_name]' / 'find [person_name]' directly in code, bypassing LLM reasoning."""
    # Match patterns like: "go to mufasa", "navigate to mufasa", "find mufasa", "where is mufasa"
    match = re.match(
        r"\s*(?:go\s+(?:to\s+)?|navigate\s+(?:to\s+)?|find\s+|locate\s+|where\s+is\s+)([A-Za-z_][A-Za-z0-9_]*)\s*\?*\s*$",
        question,
        re.IGNORECASE,
    )
    if not match:
        return None

    person_name = match.group(1).strip()
    if not person_name:
        return None

    # Avoid matching common room words
    room_keywords = {"room", "kitchen", "office", "lobby", "hallway", "corridor", "entrance", "exit", "toilet", "bathroom", "meeting", "conference", "lab", "storage", "warehouse", "garage", "elevator", "stairs", "building", "floor", "area", "zone", "space", "desk", "table", "chair", "couch", "sofa", "window", "door", "wall"}
    if person_name.lower() in room_keywords:
        return None

    tool_log: list[dict] = []
    result = _record_tool(tool_log, "find_person_by_name", {"person_name": person_name})

    if not result.get("found"):
        # Try search_objects_by_class_id as fallback
        search_result = _record_tool(
            tool_log, "search_objects_by_class_id", {"class_id": 0, "object_name": person_name}
        )
        candidates = search_result.get("results", [])
        if not candidates:
            return None
        person = candidates[0]
        loc = _record_tool(
            tool_log,
            "get_object_last_location",
            {"object_id": str(person["object_id"]), "require_navigation_usable": False},
        )
        if not loc.get("found"):
            return None
        result = {
            "found": True,
            "person_name": person_name,
            "object_id": str(person["object_id"]),
            "room": loc.get("room") or person.get("room"),
            "x": loc.get("x"),
            "y": loc.get("y"),
            "z": loc.get("z"),
            "last_seen_at": loc.get("last_seen_at"),
            "scene_caption": loc.get("scene_caption"),
            "position_source": loc.get("position_source"),
            "navigation_usable": loc.get("navigation_usable"),
        }

    room = result.get("room") or "unknown"
    x = result.get("x")
    y = result.get("y")
    nav_usable = result.get("navigation_usable")
    source_note = " (depth-backed)" if nav_usable else ""

    # Check if the original question implies navigation
    nav_match = re.match(
        r"\s*(?:go\s+(?:to\s+)?|navigate\s+(?:to\s+)?)" + re.escape(person_name) + r"\s*\?*\s*$",
        question,
        re.IGNORECASE,
    )
    if nav_match and x is not None and y is not None:
        _record_tool(
            tool_log,
            "move_to_position",
            {
                "x": x,
                "y": y,
                "reason": f"Navigate to person {person_name}",
            },
        )
        answer = (
            f"Navigating to **{person_name}** at "
            f"({x:.2f}, {y:.2f}) in **{room}**{source_note}."
        )
    else:
        answer = (
            f"**{person_name}** was last seen in **{room}** at "
            f"({x:.2f}, {y:.2f}){source_note}."
        )
    return answer, tool_log


def _maybe_answer_with_local_tools(
    question: str,
    allowed_tool_names: set[str] | None,
    return_metadata: bool,
):
    tool_log: list[dict] = []

    if allowed_tool_names == {"get_robot_location"}:
        result = _record_tool(tool_log, "get_robot_location", {})
        answer = f"The robot is in {result.get('room') or 'an unknown room'}"
        if result.get("map_name"):
            answer += f" on map {result.get('map_name')}"
        answer += f" at x={result.get('x')}, y={result.get('y')}."
        return _direct_tool_answer(answer, tool_log, return_metadata)

    # Try code-level possessive navigation before involving the LLM
    possessive_result = _try_possessive_navigation(question)
    if possessive_result is not None:
        answer, tool_log = possessive_result
        return _direct_tool_answer(answer, tool_log, return_metadata)

    # Try code-level person navigation before involving the LLM
    person_nav_result = _try_person_navigation(question)
    if person_nav_result is not None:
        answer, tool_log = person_nav_result
        return _direct_tool_answer(answer, tool_log, return_metadata)

    return None


def run_agent(
    question: str,
    history: list[dict] | None = None,
    max_iterations: int = 12,
    allowed_tool_names: set[str] | None = None,
    return_metadata: bool = False,
) -> tuple[str, list[dict]] | tuple[str, list[dict], dict]:
    """Run the reasoning-first tool-use agent loop. Returns (answer, tool_call_log)."""
    direct_answer = _maybe_answer_with_local_tools(question, allowed_tool_names, return_metadata)
    if direct_answer is not None:
        return direct_answer

    # If no tool filter was provided, use the VLM router to keep the tool set
    # small enough for the 4B model to reason effectively, but always keep the
    # foundational discovery and history tools available.
    if allowed_tool_names is None:
        routed = _select_tools_with_vlm(question, TOOL_SCHEMAS)
        if routed:
            # Foundational tools that almost every query needs access to.
            # The router may omit these if it misjudges the question.
            essential_tools = {
                "search_objects_by_class_id",
                "get_object_observations",
                "get_interaction_events",
                "get_object_person_interactions",
                "get_object_interactions",
                "get_person_interaction_summary",
                "find_person_by_name",
                "list_objects",
                "get_object_last_location",
                "get_object_first_location",
                "get_room_navigation_target",
                "move_to_position",
                "get_robot_location",
            }
            allowed_tool_names = routed | essential_tools

    return run_custom_agent(
        question=question,
        system_prompt=_AGENT_REASONING_PROMPT,
        history=history,
        max_iterations=max_iterations,
        allow_tools=True,
        require_tool_use=False,
        force_first_tool_use=False,
        allowed_tool_names=allowed_tool_names,
        return_metadata=return_metadata,
    )


def run_model_chat(
    question: str,
    history: list[dict] | None = None,
    return_metadata: bool = False,
) -> tuple[str, list[dict]] | tuple[str, list[dict], dict]:
    """Run a plain model-only chat without tools or database access."""
    return run_custom_agent(
        question=question,
        system_prompt=_CHAT_ONLY_SYSTEM_PROMPT,
        history=history,
        allow_tools=False,
        require_tool_use=False,
        force_first_tool_use=False,
        return_metadata=return_metadata,
    )
