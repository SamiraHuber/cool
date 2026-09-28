"""VLM-based scene dynamics evaluator.

Predicts how likely the current scene is to change in the near future,
based on the latest scene caption, observed objects/people, and recent interactions.
"""

import json
import os
from typing import Any

from . import tools
from .prompts import SCENE_EVALUATION
from .vlm_client import get_vlm_client, get_vlm_model

_SCENE_EVALUATION_SYSTEM_PROMPT = SCENE_EVALUATION


def _extract_json_block(text: str) -> str:
    """Best-effort extraction of a JSON object from model output."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if text.startswith("```"):
        # Strip markdown code fences
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def evaluate_scene_from_data(
    caption: str | None = None,
    recent_objects: list[dict] | None = None,
    recent_interactions: list[dict] | None = None,
) -> dict[str, Any]:
    """Evaluate scene dynamics from provided data (no DB query).

    Returns:
        dict with keys: prediction, confidence, reasoning, activity_type, scene_summary
    """
    recent_objects = recent_objects or []
    recent_interactions = recent_interactions or []

    object_summary = ", ".join(
        f"{obj.get('class_name') or 'object'}" + (f" ({obj.get('name')})" if obj.get("name") else "")
        for obj in recent_objects[:8]
    ) or "none"

    interaction_summary = "; ".join(
        f"{inter.get('action') or 'interaction'}: {inter.get('caption') or 'no caption'}"
        for inter in recent_interactions[:5]
    ) or "none"

    prompt = f"""Latest scene caption: {caption or 'N/A'}
Recent objects/people: {object_summary}
Recent interactions: {interaction_summary}

Classify the scene dynamics."""

    client = get_vlm_client()
    response = client.chat.completions.create(
        model=get_vlm_model(),
        messages=[
            {"role": "system", "content": _SCENE_EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    raw_json = _extract_json_block(content)

    default_result = {
        "prediction": ">30min",
        "confidence": 0.5,
        "reasoning": "Failed to parse VLM response; defaulting to static.",
        "activity_type": "unknown",
        "scene_summary": {
            "caption": caption,
            "recent_objects": recent_objects,
            "recent_interactions": recent_interactions,
        },
    }

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError:
        return {**default_result, "raw_response": content}

    if not isinstance(parsed, dict):
        return {**default_result, "raw_response": content}

    prediction = str(parsed.get("prediction") or ">30min").strip()
    if prediction not in {"3min", "10min", "30min", ">30min"}:
        prediction = ">30min"

    confidence = float(parsed.get("confidence") or 0.5)
    confidence = max(0.0, min(1.0, confidence))

    return {
        "prediction": prediction,
        "confidence": confidence,
        "reasoning": str(parsed.get("reasoning") or "").strip(),
        "activity_type": str(parsed.get("activity_type") or "unknown").strip(),
        "scene_summary": {
            "caption": caption,
            "recent_objects": recent_objects,
            "recent_interactions": recent_interactions,
        },
    }


def evaluate_current_scene(minutes: int = 5, offset: int = 0) -> dict[str, Any]:
    """Query the latest scene and ask the VLM to classify its change dynamics.

    Args:
        minutes: Time window for recent objects/interactions.
        offset: Skip N newest scenes (0 = latest, 1 = second latest, etc.).

    Returns:
        dict with keys: prediction, confidence, reasoning, activity_type, scene_summary
    """
    scene_data = tools.get_current_scene_description(minutes=minutes, offset=offset)
    if not scene_data.get("found"):
        return {
            "prediction": ">30min",
            "confidence": 0.5,
            "reasoning": "No scene data available; assuming static environment.",
            "activity_type": "unknown",
            "scene_summary": {},
        }

    scene = scene_data.get("scene") or {}
    result = evaluate_scene_from_data(
        caption=scene.get("caption"),
        recent_objects=scene_data.get("recent_objects"),
        recent_interactions=scene_data.get("recent_interactions"),
    )
    # Enrich summary with scene identity for UI timeline
    summary = result.get("scene_summary") or {}
    summary["scene_id"] = scene.get("scene_id")
    summary["timestamp"] = scene.get("timestamp")
    summary["has_image"] = scene.get("has_image", False)
    result["scene_summary"] = summary
    return result
