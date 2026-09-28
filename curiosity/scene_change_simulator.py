#!/usr/bin/env python3
"""Scene-change-aware navigation simulator.

Replays 5-minute synthetic scene logs against navigation strategies.
Computes:
  • Scene-change detection accuracy (TP/TN/FP/FN)
  • Navigation-to-change quality
  • Exploration coverage
  • Event recall (backward-compatible)

Usage:
    from scene_change_simulator import run_scene_change_simulation
    result = run_scene_change_simulation("agent_scene_change")
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCENE_CHANGE_ROOM_ORDER = [
    "kitchen",
    "storage",
    "office 1",
    "office 2",
    "office 3",
    "meeting room",
]

SCENE_CHANGE_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "curiosity" / "datasets" / "default"

# ---------------------------------------------------------------------------
# Tool schemas for v3 agent (OpenAI function-calling format)
# ---------------------------------------------------------------------------

SCENE_CHANGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_room_change_rates",
            "description": "Returns the change rate (changes per visit) for each room based ONLY on what YOU have personally detected. A room you have never visited will show 'unknown', not zero. Higher rate means the room has been a hotspot in YOUR experience.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time_since_last_change",
            "description": "Returns how many minutes have passed since YOU last detected a scene change in each room. This is based only on your own observations — if you were not present when a change happened, you do not know about it.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stale_rooms",
            "description": "Returns rooms you have not visited recently, ordered by how long ago you last visited them.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_room_visit_history",
            "description": "Returns your visit counts and total changes YOU detected per room. Never-visited rooms show 'unknown'.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_predicted_change_probability",
            "description": "Returns the predicted probability that a change will happen next in each room, based ONLY on your own detection history (hazard model: how often the room changed vs. how long since the last change you saw). Rooms you know nothing about carry an 'optimism prior' so they are worth exploring.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _normalize_room(room: str) -> str:
    value = room.strip().lower()
    aliases = {
        "office1": "office 1",
        "office2": "office 2",
        "office3": "office 3",
        "meetingroom": "meeting room",
        "meeting": "meeting room",
        "storageroom": "storage",
        "storage room": "storage",
    }
    return aliases.get(value, value)


@dataclass(frozen=True)
class SceneLogEntry:
    timestamp: datetime
    room: str
    scene: str
    people_present: list[str]
    objects_present: list[str]
    change_type: str  # "no_change", "minor_change", "major_change"
    people_added: list[str]
    people_removed: list[str]
    objects_added: list[str]
    objects_removed: list[str]
    activities_changed: bool = False
    activity_change_severity: str = "none"
    activities_added: list[str] = field(default_factory=list)
    activities_removed: list[str] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)


class SceneChangeTimeLog:
    """Ground-truth scene-change log with fast lookups."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or SCENE_CHANGE_DATA_DIR
        self.entries: list[SceneLogEntry] = []
        self.scene_index: dict[tuple[datetime, str], SceneLogEntry] = {}
        self.change_index: dict[tuple[datetime, str], str] = {}
        self.interaction_index: dict[tuple[datetime, str], list[dict]] = {}
        self.total_changes = 0  # major + minor changes
        self.total_major_changes = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        # Room universe for this log; recomputed in _load(). Default keeps the
        # classic 6-room order so unloaded/empty logs behave as before.
        self.rooms: list[str] = list(SCENE_CHANGE_ROOM_ORDER)
        # Per-dataset walking distances (minutes), from the optional
        # distances.json sidecar (scripts/dump_dataset_distances.py). None
        # means "no sidecar" -> distance() falls back to the legacy 6-room
        # DISTANCES table.
        self._distance_default: int | None = None
        self._distance_exceptions: dict[frozenset, int] = {}
        self._load_distances()
        self._load()

    def _load_distances(self) -> None:
        path = self.data_dir / "distances.json"
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self._distance_default = int(payload.get("default", 2))
            for a, b, d in payload.get("exceptions", []):
                pair = frozenset((_normalize_room(a), _normalize_room(b)))
                self._distance_exceptions[pair] = int(d)
        except Exception:
            self._distance_default = None
            self._distance_exceptions = {}

    def distance(self, room_a: str, room_b: str) -> int:
        """Walking minutes between rooms for THIS log's world.

        Uses the dataset's distances.json sidecar when present (v2 worlds,
        incl. 24-room); otherwise falls back to the legacy 6-room table.
        """
        a, b = _normalize_room(room_a), _normalize_room(room_b)
        if a == b:
            return 0
        if self._distance_default is not None:
            return self._distance_exceptions.get(frozenset((a, b)), self._distance_default)
        return travel_time(a, b)

    def _load(self) -> None:
        combined_path = self.data_dir / "scene_changes_all.json"
        if not combined_path.exists():
            raise FileNotFoundError(f"Scene change log not found: {combined_path}")

        with combined_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        # Load interactions if available
        interactions_path = self.data_dir / "synthetic_interactions_all.json"
        interactions_by_key: dict[tuple[datetime, str], list[dict]] = {}
        if interactions_path.exists():
            with interactions_path.open("r", encoding="utf-8") as f:
                inter_raw = json.load(f)
            for item in inter_raw:
                try:
                    ts = datetime.strptime(f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M")
                    room = _normalize_room(item["room"])
                    interactions_by_key.setdefault((ts, room), []).append(item)
                except Exception:
                    continue

        for item in raw:
            try:
                ts = datetime.strptime(f"{item['date']} {item['time']}", "%Y-%m-%d %H:%M")
                room = _normalize_room(item["room"])
                change_type = item.get("change_type", "no_change")
            except Exception:
                continue

            interactions = interactions_by_key.get((ts, room))
            if interactions is None and not interactions_path.exists():
                # v2 datasets (generate_scene_change_dataset_v2.py) ship no
                # synthetic_interactions_all.json sidecar; activities live in
                # the inline person_activities map (person -> free-text phrase).
                # Map to the v1 interaction shape so prompts render them.
                pa = item.get("person_activities") or {}
                interactions = [
                    {"subject": person, "action": phrase, "target": "",
                     "verb_category": "unknown"}
                    for person, phrase in pa.items()
                ]
            interactions = interactions or []

            entry = SceneLogEntry(
                timestamp=ts,
                room=room,
                scene=item.get("scene", ""),
                people_present=item.get("people_present", []),
                objects_present=item.get("objects_present", []),
                change_type=change_type,
                people_added=item.get("people_added", []),
                people_removed=item.get("people_removed", []),
                objects_added=item.get("objects_added", []),
                objects_removed=item.get("objects_removed", []),
                activities_changed=item.get("activities_changed", False),
                activity_change_severity=item.get("activity_change_severity", "none"),
                activities_added=item.get("activities_added", []),
                activities_removed=item.get("activities_removed", []),
                interactions=interactions,
            )
            self.entries.append(entry)
            self.scene_index[(ts, room)] = entry
            self.change_index[(ts, room)] = change_type
            self.interaction_index[(ts, room)] = entry.interactions
            if change_type in ("minor_change", "major_change"):
                self.total_changes += 1
            if change_type == "major_change":
                self.total_major_changes += 1

        if self.entries:
            # Classic 6-room order first (6-room datasets are then
            # byte-identical to the pre-12-room behavior), extra rooms
            # (e.g. 12-room scaled datasets) appended sorted.
            found_rooms = {e.room for e in self.entries}
            self.rooms = ([r for r in SCENE_CHANGE_ROOM_ORDER if r in found_rooms]
                          + sorted(found_rooms - set(SCENE_CHANGE_ROOM_ORDER)))
            self.start_time = min(e.timestamp for e in self.entries)
            self.end_time = max(e.timestamp for e in self.entries)
            self.unique_timestamps = sorted(set(e.timestamp for e in self.entries))
            # timestamp -> step index map (§4.2-CORE): kills O(n) list.index /
            # linear scans in the hazard model.
            self.ts_index = {ts: i for i, ts in enumerate(self.unique_timestamps)}
        else:
            self.unique_timestamps = []
            self.ts_index = {}

    def scene_at(self, timestamp: datetime, room: str) -> SceneLogEntry | None:
        return self.scene_index.get((timestamp, _normalize_room(room)))

    def change_type_at(self, timestamp: datetime, room: str) -> str:
        return self.change_index.get((timestamp, _normalize_room(room)), "no_change")

    def interactions_at(self, timestamp: datetime, room: str) -> list[dict]:
        return self.interaction_index.get((timestamp, _normalize_room(room)), [])

    def is_changed(self, timestamp: datetime, room: str) -> bool:
        return self.change_type_at(timestamp, room) in ("minor_change", "major_change")

    def changes_in_window(self, timestamp: datetime, window_steps: int = 1) -> dict[str, int]:
        """Count changes per room in the next N steps (looking forward)."""
        counts = {r: 0 for r in self.rooms}
        for i in range(1, window_steps + 1):
            t = timestamp + timedelta(minutes=5 * i)
            for room in self.rooms:
                if self.is_changed(t, room):
                    counts[room] += 1
        return counts

    @property
    def theoretical_max_recall(self) -> float:
        """Max achievable: unique change-steps / total changes."""
        if self.total_changes == 0:
            return 0.0
        steps_with_changes = set()
        for e in self.entries:
            if e.change_type in ("minor_change", "major_change"):
                steps_with_changes.add(e.timestamp)
        return len(steps_with_changes) / self.total_changes


# ---------------------------------------------------------------------------
# Simulation result
# ---------------------------------------------------------------------------


@dataclass
class SceneChangeVisit:
    visit_number: int
    room: str
    start: datetime
    end: datetime
    changes_observed: int = 0
    dwell_seconds: int = 0


@dataclass
class SceneChangeMinuteStep:
    timestamp: datetime
    time_str: str
    room: str
    scene_text: str
    gt_change_type: str
    vlm_detected_change: bool | None = None
    vlm_change: str | None = None
    vlm_detected_activity_change: bool | None = None
    action: str = "stay"
    target_room: str | None = None
    decision: dict[str, Any] | None = None
    fresh_decision: bool = False
    people_present: list[str] = field(default_factory=list)
    objects_present: list[str] = field(default_factory=list)
    interactions: list[dict] = field(default_factory=list)
    gt_activity_changed: bool = False
    prompt: str | None = None
    raw_response: str | None = None
    system_prompt: str | None = None
    tool_calls: list[dict] | None = None
    error: str | None = None
    # Set when the simulator overrode the agent's decision (forced dwell move,
    # unseen-room override). Overridden steps are excluded from nav-precision.
    decision_override: str | None = None


@dataclass
class SceneChangeSimulationResult:
    strategy: str
    total_changes: int
    total_major_changes: int
    observed_changes: int = 0
    total_visits: int = 0
    visits_with_changes: int = 0
    change_recall: float = 0.0
    change_precision: float = 0.0
    change_f1: float = 0.0
    hit_rate: float = 0.0
    elapsed_seconds: float = 0.0
    # Scene-change detection confusion
    scene_change_tp: int = 0
    scene_change_tn: int = 0
    scene_change_fp: int = 0
    scene_change_fn: int = 0
    scene_change_accuracy: float = 0.0
    no_change_accuracy: float = 0.0
    change_accuracy: float = 0.0
    # Per-severity confusion
    minor_tp: int = 0
    minor_fp: int = 0
    minor_fn: int = 0
    major_tp: int = 0
    major_fp: int = 0
    major_fn: int = 0
    minor_recall: float = 0.0
    minor_precision: float = 0.0
    major_recall: float = 0.0
    major_precision: float = 0.0
    # Activity-change detection confusion
    activity_change_tp: int = 0
    activity_change_tn: int = 0
    activity_change_fp: int = 0
    activity_change_fn: int = 0
    activity_change_accuracy: float = 0.0
    # Navigation quality
    navigation_to_changed_room: int = 0
    total_moves: int = 0
    navigation_precision: float = 0.0
    # Post-hoc navigation-quality metrics (computed from minute_trace +
    # scene log in _compute_gt_metrics, uniformly for all strategies)
    move_value_k3: float = 0.0
    move_value_k6: float = 0.0
    wasted_dwell_rate: float = 0.0
    coverage_halftime: int | None = None
    # Map-completeness metrics (active-mapping vs COOL comparison):
    # coverage_milestones = trace-step index at which visit coverage first
    # reaches 25/50/75/100% of the room universe (None if never);
    # avg_mean_staleness_minutes = time-averaged mean per-room staleness
    # (map freshness; unvisited rooms age from log start);
    # visits_per_room_gini = Gini of per-room step counts (0 = perfectly even
    # sweep, higher = targeted revisits).
    coverage_milestones: dict[str, Any] = field(default_factory=dict)
    avg_mean_staleness_minutes: float = 0.0
    visits_per_room_gini: float = 0.0
    # Memory-quality metrics (active-mapping comparison): what the policy's
    # trajectory is good for beyond change-catching. people_observation_rate =
    # share of trace steps with ≥1 person in the observed scene;
    # interaction_cells_* count (timestamp, room) cells with ≥1 interaction:
    # total across the log vs witnessed by the robot (trace steps have unique
    # timestamps, so witnessed cells never double-count).
    people_observed_steps: int = 0
    people_observation_rate: float = 0.0
    interaction_cells_witnessed: int = 0
    interaction_cells_total: int = 0
    interaction_witness_rate: float = 0.0
    # Exploration
    rooms_visited: set[str] = field(default_factory=set)
    exploration_coverage: float = 0.0
    # Legacy / VLM
    vlm_calls_made: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    system_prompt: str = ""
    visits: list[SceneChangeVisit] = field(default_factory=list)
    minute_trace: list[SceneChangeMinuteStep] = field(default_factory=list)
    # Ground-truth cross-room metrics
    cross_room_misses: int = 0
    cross_room_miss_rate: float = 0.0
    avg_detection_latency_minutes: float = 0.0
    median_detection_latency_minutes: float = 0.0
    max_detection_latency_minutes: float = 0.0
    event_timesteps_caught: int = 0
    total_event_timesteps: int = 0
    event_timestep_rate: float = 0.0
    changes_never_detected: int = 0
    latency_histogram: dict[str, int] = field(default_factory=dict)
    # Timestep-based detection latency (new)
    detection_latency_histogram: dict[str, int] = field(default_factory=dict)
    avg_detection_latency_steps: float = 0.0
    median_detection_latency_steps: float = 0.0
    max_detection_latency_steps: int = 0
    changes_detected_immediately: int = 0
    per_room_latency: dict[str, Any] = field(default_factory=dict)
    cumulative_detection_curve: dict[str, Any] = field(default_factory=dict)
    catchable_changes: int = 0
    detected_among_catchable: float = 0.0
    absent_changes: int = 0
    blind_changes: int = 0
    conditional_latency_histogram: dict[str, int] = field(default_factory=dict)
    visit_opportunity_histogram: dict[str, int] = field(default_factory=dict)
    # Paper-ready upper-bound / efficiency metrics
    path_max_changes: int = 0
    theoretical_max_changes: int = 0
    detection_efficiency: float = 0.0
    navigation_efficiency: float = 0.0
    normalized_recall: float = 0.0
    # Optimal-path metrics (DP: max events with min moves)
    optimal_path_changes: int = 0
    optimal_path_moves: int = 0
    optimal_path_efficiency: float = 0.0
    # Robot-memory-relative metrics (fair comparison: change since robot's last visit)
    robot_memory_tp: int = 0
    robot_memory_tn: int = 0
    robot_memory_fp: int = 0
    robot_memory_fn: int = 0
    robot_memory_accuracy: float = 0.0
    robot_memory_precision: float = 0.0
    robot_memory_recall: float = 0.0
    robot_memory_f1: float = 0.0
    # Presence- vs detection-based recall (change_recall stays presence-based
    # for backward comparability with older runs)
    presence_recall: float = 0.0
    detected_recall: float = 0.0
    # What the latency/detection metrics are grounded in for this strategy
    detection_basis: str = ""
    # Agent-relative state-diff metrics (_compute_state_diff_metrics):
    # a change exists only if the room state differs from the agent's last
    # observation; reverted-before-visit changes are sd_expired_changes.
    sd_visible_instances: int = 0
    sd_detected: int = 0
    sd_visible_recall: float = 0.0
    sd_blind: int = 0
    sd_unvisited: int = 0
    sd_expired_changes: int = 0
    sd_false_positives: int = 0
    sd_fp_rate_per_step: float = 0.0
    sd_avg_latency_steps: float = 0.0
    sd_median_latency_steps: float = 0.0
    sd_max_latency_steps: int = 0
    sd_latency_histogram: dict[str, int] = field(default_factory=dict)
    sd_cumulative_detection_curve: dict[str, Any] = field(default_factory=dict)
    sd_per_room: dict[str, Any] = field(default_factory=dict)
    # VLM error steps (excluded from detection metrics)
    error_steps: int = 0
    error_rate: float = 0.0
    # Moves forced by the simulator (dwell/unseen overrides), excluded from nav-precision
    overridden_moves: int = 0
    # Rank-divergence tracking (NavSplit strategies): how often the VLM
    # navigator's decision differed from plain argmax of its own ROOM RANKING
    # table. rank_divergence_log has one entry per navigator call.
    rank_greedy_calls: int = 0
    rank_greedy_divergences: int = 0
    rank_greedy_override_moves: int = 0
    rank_divergence_rate: float = 0.0
    rank_divergence_log: list[dict] = field(default_factory=list)
    # Top-K agreement (NavSplit): distribution of where the VLM's chosen
    # target landed in the ROOM RANKING table it saw.
    # rank_target_histogram keys: "stay", "rank1".."rank4", "rank5plus",
    # "off_table" (moved to a room not in the table), "no_table" (empty table).
    # rank_topk_rates: share of tabled calls whose target rank <= K
    # (top1/top3/top5), plus "stay_rate" and "off_table_rate".
    rank_target_histogram: dict[str, int] = field(default_factory=dict)
    rank_topk_rates: dict[str, float] = field(default_factory=dict)
    # Reproduction snapshot (§0.3.3): strategy, resolved flags, dataset path,
    # simulator file SHA256. Populated by run_scene_change_simulation.
    config_snapshot: dict[str, Any] = field(default_factory=dict)


def _compute_rank_topk(rank_track: list[dict]) -> tuple[dict[str, int], dict[str, float]]:
    """Aggregate per-call rank tracking into a target-rank histogram and
    top-K agreement rates.

    Histogram buckets (one per navigator call):
      stay      — navigator chose stay
      rank1..4  — moved to the room at that position in the ROOM RANKING table
      rank5plus — moved to a room ranked 5 or lower
      off_table — moved to a room not present in the table
      no_table  — the table was empty (excluded from top-K rates)
    Rates are over "tabled" calls (table non-empty): topK = share with
    target_rank <= K; stay/off_table get their own rates on the same base.
    """
    hist = {"stay": 0, "rank1": 0, "rank2": 0, "rank3": 0, "rank4": 0,
            "rank5plus": 0, "off_table": 0, "no_table": 0}
    for e in rank_track:
        if e.get("greedy_target") is None:
            hist["no_table"] += 1
            continue
        if e.get("vlm_action") == "stay":
            hist["stay"] += 1
            continue
        r = e.get("vlm_target_rank")
        if r is None:
            hist["off_table"] += 1
        elif r >= 5:
            hist["rank5plus"] += 1
        else:
            hist[f"rank{r}"] += 1
    tabled = len(rank_track) - hist["no_table"]
    rates: dict[str, float] = {}
    if tabled:
        rates = {
            "top1": hist["rank1"] / tabled,
            "top3": (hist["rank1"] + hist["rank2"] + hist["rank3"]) / tabled,
            "top5": (tabled - hist["stay"] - hist["off_table"]
                     - sum(1 for e in rank_track
                           if e.get("vlm_target_rank") is not None
                           and e["vlm_target_rank"] > 5)) / tabled,
            "stay": hist["stay"] / tabled,
            "off_table": hist["off_table"] / tabled,
        }
    return hist, rates


def _compute_robot_memory_metrics(sim) -> dict[str, Any]:
    """Compute scene-change confusion where GT = change since robot's last visit to each room.

    This is fairer to the robot because it compares the current scene to what the robot
    actually remembers, rather than to the previous 5-minute timestep.
    """
    if not sim.minute_trace:
        return {
            "robot_memory_tp": 0, "robot_memory_tn": 0,
            "robot_memory_fp": 0, "robot_memory_fn": 0,
            "robot_memory_accuracy": 0.0, "robot_memory_precision": 0.0,
            "robot_memory_recall": 0.0, "robot_memory_f1": 0.0,
        }

    tp = tn = fp = fn = 0

    # Track last observation per room from the trace
    last_obs_by_room: dict[str, SceneChangeMinuteStep] = {}

    for step in sim.minute_trace:
        room = step.room
        prev_step = last_obs_by_room.get(room)

        # Compute change since robot's last visit
        if prev_step is None:
            # First visit to this room: no baseline → no_change by definition
            robot_memory_gt = "no_change"
        else:
            cur_people = set(step.people_present or [])
            prev_people = set(prev_step.people_present or [])
            cur_objects = set(step.objects_present or [])
            prev_objects = set(prev_step.objects_present or [])

            people_diffs = len(cur_people - prev_people) + len(prev_people - cur_people)
            objects_diffs = len(cur_objects - prev_objects) + len(prev_objects - cur_objects)
            total_diffs = people_diffs + objects_diffs

            if total_diffs == 0:
                robot_memory_gt = "no_change"
            elif total_diffs == 1:
                robot_memory_gt = "minor_change"
            else:
                robot_memory_gt = "major_change"

        # Compare to VLM detection (error steps carry no detection signal)
        if step.error is None:
            vlm_detected = step.vlm_detected_change in (True, "true", "True", 1, "1")
            is_changed = robot_memory_gt in ("minor_change", "major_change")

            if vlm_detected and is_changed:
                tp += 1
            elif not vlm_detected and not is_changed:
                tn += 1
            elif vlm_detected and not is_changed:
                fp += 1
            else:
                fn += 1

        # Update last observation for this room
        last_obs_by_room[room] = step

    total = tp + tn + fp + fn
    return {
        "robot_memory_tp": tp,
        "robot_memory_tn": tn,
        "robot_memory_fp": fp,
        "robot_memory_fn": fn,
        "robot_memory_accuracy": (tp + tn) / total if total > 0 else 0.0,
        "robot_memory_precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "robot_memory_recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "robot_memory_f1": (2 * (tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))) if (tp + fp) > 0 and (tp + fn) > 0 else 0.0,
    }


def _sd_entry_state(entry: SceneLogEntry | None) -> tuple[frozenset, frozenset]:
    """GT room state (people, objects) for state-diff comparison."""
    if entry is None:
        return (frozenset(), frozenset())
    return (frozenset(entry.people_present or []), frozenset(entry.objects_present or []))


def _sd_step_state(step: SceneChangeMinuteStep) -> tuple[frozenset, frozenset]:
    return (frozenset(step.people_present or []), frozenset(step.objects_present or []))


def _compute_state_diff_metrics(sim) -> dict[str, Any]:
    """Agent-relative (state-diff) detection metrics.

    A change only exists FOR THE AGENT if the room's GT state (people +
    objects) differs from the GT state at the agent's last observation of
    that room (the exact question posed to the VLM). Consequences:

    - visible instance: state diverged from last-observed; the first
      non-error step in the room after divergence is the single detection
      opportunity. detected iff the VLM reported a change there (non-VLM
      strategies: implicit detection, same convention as _is_detected).
      latency = 5-min steps from divergence start to that opportunity, so
      "detected 3 steps late" still counts.
    - expired change: state diverged then reverted to last-observed before
      any visit -> nothing detectable at visit time. EXCLUDED from recall,
      reported separately (also a dataset-quality statistic).
    - unvisited: still diverged at end of log, never visited afterwards.
    - false positive: step where state == last-observed but VLM said changed.
    - multiple GT change events between two observations collapse into ONE
      visible instance (the VLM sees "different", not "changed k times").

    Memory absorbs the scene on every visit (even a blind one), consistent
    with _compute_robot_memory_metrics.
    """
    zero = {
        "sd_visible_instances": 0, "sd_detected": 0, "sd_visible_recall": 0.0,
        "sd_blind": 0, "sd_unvisited": 0, "sd_expired_changes": 0,
        "sd_false_positives": 0, "sd_fp_rate_per_step": 0.0,
        "sd_avg_latency_steps": 0.0, "sd_median_latency_steps": 0.0,
        "sd_max_latency_steps": 0, "sd_latency_histogram": {},
        "sd_cumulative_detection_curve": {}, "sd_per_room": {},
    }
    if not sim.minute_trace or not sim.log:
        return zero

    import statistics as _stats

    all_rooms = sim.log.rooms
    timeline = sim.log.unique_timestamps
    detected_true = (True, "true", "True", 1, "1")

    # GT state per (timestamp, room)
    states: dict[tuple[datetime, str], tuple[frozenset, frozenset]] = {}
    for ts in timeline:
        for room in all_rooms:
            states[(ts, room)] = _sd_entry_state(sim.log.scene_at(ts, room))

    latencies: list[int] = []
    hist: dict[str, int] = {}
    visible = detected = blind = unvisited = expired = fp = 0
    baseline_steps = 0  # steps with a previous observation (FP denominator)
    per_room: dict[str, dict[str, Any]] = {
        r: {"visible": 0, "detected": 0, "blind": 0, "unvisited": 0,
            "expired": 0, "latencies": []} for r in all_rooms}

    for room in all_rooms:
        steps = [s for s in sim.minute_trace if s.room == room and s.error is None]
        if not steps:
            continue
        pr = per_room[room]
        last_obs = _sd_step_state(steps[0])
        for prev, cur in zip(steps, steps[1:]):
            baseline_steps += 1
            # Track divergence/reversion cycles strictly between observations
            open_start: datetime | None = None
            for ts in timeline:
                if ts <= prev.timestamp or ts >= cur.timestamp:
                    continue
                st = states[(ts, room)]
                if st != last_obs and open_start is None:
                    open_start = ts
                elif st == last_obs and open_start is not None:
                    expired += 1
                    pr["expired"] += 1
                    open_start = None
            cur_state = _sd_step_state(cur)
            vlm = cur.vlm_detected_change
            # Non-VLM strategies: implicit detection on visiting a changed room
            is_detected = (vlm in detected_true) if vlm is not None else True
            if cur_state != last_obs:
                visible += 1
                pr["visible"] += 1
                div = open_start if open_start is not None else cur.timestamp
                lat = int(round((cur.timestamp - div).total_seconds() / 300))
                if is_detected:
                    detected += 1
                    pr["detected"] += 1
                    latencies.append(lat)
                    pr["latencies"].append(lat)
                    key = "T=0" if lat == 0 else f"T+{lat}"
                    hist[key] = hist.get(key, 0) + 1
                else:
                    blind += 1
                    pr["blind"] += 1
            else:
                if open_start is not None:
                    # Reverted exactly at this observation -> invisible
                    expired += 1
                    pr["expired"] += 1
                if vlm in detected_true:
                    fp += 1
            last_obs = cur_state
        # Tail after the last observation of this room
        open_start = None
        for ts in timeline:
            if ts <= steps[-1].timestamp:
                continue
            st = states[(ts, room)]
            if st != last_obs and open_start is None:
                open_start = ts
            elif st == last_obs and open_start is not None:
                expired += 1
                pr["expired"] += 1
                open_start = None
        if open_start is not None:
            unvisited += 1
            pr["unvisited"] += 1

    # Cumulative detection curve: fraction of visible instances detected
    # within k steps of divergence (headline latency plot).
    thresholds = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 144, 192, 288, 576]
    curve: dict[str, float] = {}
    for k in thresholds:
        curve[f"T<={k}"] = round(sum(1 for l in latencies if l <= k) / visible, 4) if visible else 0.0
    curve["detected_total"] = round(detected / visible, 4) if visible else 0.0

    per_room_out: dict[str, Any] = {}
    for r, pr in per_room.items():
        per_room_out[r] = {
            "visible": pr["visible"], "detected": pr["detected"],
            "blind": pr["blind"], "unvisited": pr["unvisited"],
            "expired": pr["expired"],
            "median_latency_steps": (_stats.median(pr["latencies"]) if pr["latencies"] else None),
        }

    return {
        "sd_visible_instances": visible,
        "sd_detected": detected,
        "sd_visible_recall": round(detected / visible, 4) if visible else 0.0,
        "sd_blind": blind,
        "sd_unvisited": unvisited,
        "sd_expired_changes": expired,
        "sd_false_positives": fp,
        "sd_fp_rate_per_step": round(fp / baseline_steps, 4) if baseline_steps else 0.0,
        "sd_avg_latency_steps": round(_stats.mean(latencies), 2) if latencies else 0.0,
        "sd_median_latency_steps": _stats.median(latencies) if latencies else 0.0,
        "sd_max_latency_steps": max(latencies) if latencies else 0,
        "sd_latency_histogram": hist,
        "sd_cumulative_detection_curve": curve,
        "sd_per_room": per_room_out,
    }


def _compute_optimal_path_metrics(log: SceneChangeTimeLog) -> dict[str, Any]:
    """Find the path that catches the maximum events with the minimum moves.

    Dynamic programming over timesteps:
      dp[t][r] = (max_events, min_moves) achievable at timestep t in room r.
    Transition: from any prev_room at t-1, either stay (0 moves) or move (1 move).
    Objective: maximize events caught; break ties by minimizing moves.

    Initial placement in any room at t=0 costs 0 moves (matches simulator).
    """
    timestamps = log.unique_timestamps
    rooms = log.rooms
    n_steps = len(timestamps)
    n_rooms = len(rooms)
    if n_steps == 0 or n_rooms == 0:
        return {"optimal_path_changes": 0, "optimal_path_moves": 0, "optimal_path_efficiency": 0.0}

    # Build change mask: change_at[t][r] = 1 if room r has a change at timestep t
    change_at: list[list[int]] = []
    for ts in timestamps:
        row = []
        for room in rooms:
            entry = log.scene_at(ts, room)
            row.append(1 if entry is not None and entry.change_type in ("minor_change", "major_change") else 0)
        change_at.append(row)

    # DP tables: best_events[t][r], best_moves[t][r]
    INF_MOVES = 10**9
    # Initialize t=0: can start in any room with 0 moves
    prev_events = [change_at[0][r] for r in range(n_rooms)]
    prev_moves = [0] * n_rooms

    for t in range(1, n_steps):
        cur_events = [0] * n_rooms
        cur_moves = [INF_MOVES] * n_rooms
        for r in range(n_rooms):
            gain = change_at[t][r]
            best_e = -1
            best_m = INF_MOVES
            for prev_r in range(n_rooms):
                cand_e = prev_events[prev_r] + gain
                cand_m = prev_moves[prev_r] + (0 if prev_r == r else 1)
                if cand_e > best_e or (cand_e == best_e and cand_m < best_m):
                    best_e = cand_e
                    best_m = cand_m
            cur_events[r] = best_e
            cur_moves[r] = best_m
        prev_events = cur_events
        prev_moves = cur_moves

    # Pick best final state
    best_events = max(prev_events)
    best_moves = min(m for e, m in zip(prev_events, prev_moves) if e == best_events)

    return {
        "optimal_path_changes": best_events,
        "optimal_path_moves": best_moves,
        "optimal_path_efficiency": 0.0,
    }


def _compute_gt_metrics(sim) -> dict[str, Any]:
    """Compute cross-room miss and detection-latency metrics from ground truth.

    Cross-room miss: robot is in an unchanged room while changes happen elsewhere.
    Detection latency: timesteps from a change occurring to the robot's first
    *detection* of that change (vlm_detected_change=True for VLM agents,
    or implicit detection via visiting a changed room for non-VLM agents).
    """
    if not sim.minute_trace or not sim.log:
        return {
            "cross_room_misses": 0,
            "cross_room_miss_rate": 0.0,
            "avg_detection_latency_minutes": 0.0,
            "median_detection_latency_minutes": 0.0,
            "max_detection_latency_minutes": 0.0,
            "changes_never_detected": 0,
            "latency_histogram": {},
            "path_max_changes": 0,
            "theoretical_max_changes": 0,
            "detection_efficiency": 0.0,
            "navigation_efficiency": 0.0,
            "normalized_recall": 0.0,
            "optimal_path_changes": 0,
            "optimal_path_moves": 0,
            "optimal_path_efficiency": 0.0,
            "detection_latency_histogram": {},
            "avg_detection_latency_steps": 0.0,
            "median_detection_latency_steps": 0.0,
            "max_detection_latency_steps": 0,
            "changes_detected_immediately": 0,
            "per_room_latency": {},
            "cumulative_detection_curve": {},
            "catchable_changes": 0,
            "detected_among_catchable": 0.0,
            "absent_changes": 0,
            "blind_changes": 0,
            "conditional_latency_histogram": {},
            "visit_opportunity_histogram": {},
            "presence_recall": 0.0,
            "detected_recall": 0.0,
            "detection_basis": "none",
            "move_value_k3": 0.0,
            "move_value_k6": 0.0,
            "wasted_dwell_rate": 0.0,
            "coverage_halftime": None,
            "coverage_milestones": {},
            "avg_mean_staleness_minutes": 0.0,
            "visits_per_room_gini": 0.0,
            "people_observed_steps": 0,
            "people_observation_rate": 0.0,
            "interaction_cells_witnessed": 0,
            "interaction_cells_total": 0,
            "interaction_witness_rate": 0.0,
        }

    all_rooms = sim.log.rooms
    total_steps = len(sim.minute_trace)

    # --- Cross-room misses ---
    cross_room_misses = 0
    for step in sim.minute_trace:
        robot_room = step.room
        robot_entry = sim.log.scene_at(step.timestamp, robot_room)
        robot_changed = robot_entry is not None and robot_entry.change_type in ("minor_change", "major_change")
        if robot_changed:
            continue
        other_changed = False
        for room in all_rooms:
            if room == robot_room:
                continue
            entry = sim.log.scene_at(step.timestamp, room)
            if entry is not None and entry.change_type in ("minor_change", "major_change"):
                other_changed = True
                break
        if other_changed:
            cross_room_misses += 1

    cross_room_miss_rate = cross_room_misses / total_steps if total_steps > 0 else 0.0

    # --- Event timesteps (max 1 catchable per timestamp) ---
    event_timesteps: set[datetime] = set()
    caught_event_timesteps: set[datetime] = set()
    for step in sim.minute_trace:
        any_change = False
        robot_room_changed = False
        for room in all_rooms:
            entry = sim.log.scene_at(step.timestamp, room)
            if entry is not None and entry.change_type in ("minor_change", "major_change"):
                any_change = True
                if room == step.room:
                    robot_room_changed = True
        if any_change:
            event_timesteps.add(step.timestamp)
        if robot_room_changed:
            caught_event_timesteps.add(step.timestamp)

    total_event_timesteps = len(event_timesteps)
    event_timesteps_caught = len(caught_event_timesteps)
    event_timestep_rate = event_timesteps_caught / total_event_timesteps if total_event_timesteps > 0 else 0.0

    # --- Detection latency (timestep-based + visit-opportunity-based) ---
    def _is_detected(step: SceneChangeMinuteStep) -> bool:
        """A step counts as a detection if the VLM said so, or (for non-VLM
        strategies) if the room actually had a GT change at that moment."""
        if step.vlm_detected_change is not None:
            return step.vlm_detected_change
        # Non-VLM agents: detection is implicit on visiting a changed room
        return step.gt_change_type in ("minor_change", "major_change")

    # Build robot step lookup per room: list of (timestamp, step)
    # Error steps (VLM/parse/timeout failures) are excluded from detection
    # metrics — they carry no detection signal either way.
    # setdefault: trace rooms outside the log's room universe get their own
    # bucket instead of crashing (robustness for partial-room fixtures/logs).
    robot_steps_by_room: dict[str, list[tuple[datetime, SceneChangeMinuteStep]]] = {r: [] for r in all_rooms}
    for step in sim.minute_trace:
        if step.error is not None:
            continue
        robot_steps_by_room.setdefault(step.room, []).append((step.timestamp, step))

    # Collect all GT change events (use unique_timestamps to avoid duplicates)
    gt_changes: list[tuple[datetime, str, str]] = []
    for timestamp in sim.log.unique_timestamps:
        for room in all_rooms:
            entry = sim.log.scene_at(timestamp, room)
            if entry is not None and entry.change_type in ("minor_change", "major_change"):
                gt_changes.append((timestamp, room, entry.change_type))

    # --- Post-hoc navigation-quality metrics (N2/N3/N5) ---
    # N2 move_value_kK: a move is a trace step whose room differs from the
    # previous step's room (the first trace step is the initial placement, not
    # a move); moves forced by the simulator (decision_override set on the
    # departing or arriving step) are excluded. A counted move is "valuable"
    # if its destination room has a GT change (minor/major) at any timestamp
    # from the arrival step up to and including K steps later. Generalizes
    # the arrival-instant navigation_precision.
    # N3 wasted_dwell_rate: fraction of ALL trace steps that re-observe an
    # unchanged room — a step is "wasted" when it is not the first step of
    # the current visit (same room as the previous step) and its
    # gt_change_type is "no_change".
    # N5 coverage_halftime: 0-based trace step index at which the set of
    # distinct rooms visited so far first covers the log's room universe
    # (SCENE_CHANGE_ROOM_ORDER for classic 6-room datasets, plus any extra
    # rooms for scaled datasets); None if coverage never completes.
    # Map-completeness companions (active-mapping comparison):
    # coverage_milestones p25/p50/p75/p100 = same index at partial coverage;
    # avg_mean_staleness_minutes = time-averaged mean per-room staleness
    # (unvisited rooms age from log start) — the "map freshness" an active
    # mapper optimizes; visits_per_room_gini = Gini of per-room trace-step
    # counts (0 = even sweep, higher = targeted revisits).
    move_value_hits = {3: 0, 6: 0}
    counted_moves = 0
    wasted_steps = 0
    rooms_seen: set[str] = set()
    coverage_halftime: int | None = None
    full_room_set = set(all_rooms)
    n_rooms = len(full_room_set)
    coverage_milestones: dict[str, Any] = {}
    milestone_fracs = ((0.25, "p25"), (0.5, "p50"), (0.75, "p75"), (1.0, "p100"))
    last_seen: dict[str, datetime] = {}
    staleness_sum_min = 0.0
    staleness_steps = 0
    steps_per_room: dict[str, int] = {}
    log_start_ts = sim.log.unique_timestamps[0] if sim.log.unique_timestamps else None
    for i, step in enumerate(sim.minute_trace):
        rooms_seen.add(step.room)
        if coverage_halftime is None and rooms_seen >= full_room_set:
            coverage_halftime = i
        coverage_frac = len(rooms_seen) / n_rooms if n_rooms else 0.0
        for frac, key in milestone_fracs:
            if key not in coverage_milestones and coverage_frac >= frac:
                coverage_milestones[key] = i
        last_seen[step.room] = step.timestamp
        steps_per_room[step.room] = steps_per_room.get(step.room, 0) + 1
        if log_start_ts is not None and n_rooms:
            staleness_sum_min += sum(
                (step.timestamp - last_seen.get(r, log_start_ts)).total_seconds() / 60.0
                for r in full_room_set
            ) / n_rooms
            staleness_steps += 1
        if i == 0:
            continue  # initial placement, not a move
        prev_step = sim.minute_trace[i - 1]
        if step.room == prev_step.room:
            if step.gt_change_type == "no_change":
                wasted_steps += 1
            continue
        if prev_step.room is None:
            continue
        if prev_step.decision_override or step.decision_override:
            continue  # simulator-forced move, not the agent's choice
        counted_moves += 1
        for k in move_value_hits:
            horizon_ts = step.timestamp + timedelta(minutes=5 * k)
            if any(room == step.room and step.timestamp <= change_ts <= horizon_ts
                   for change_ts, room, _ctype in gt_changes):
                move_value_hits[k] += 1
    move_value_k3 = move_value_hits[3] / counted_moves if counted_moves > 0 else 0.0
    move_value_k6 = move_value_hits[6] / counted_moves if counted_moves > 0 else 0.0
    wasted_dwell_rate = wasted_steps / total_steps if total_steps > 0 else 0.0
    for _frac, _key in milestone_fracs:
        coverage_milestones.setdefault(_key, None)
    avg_mean_staleness_minutes = staleness_sum_min / staleness_steps if staleness_steps else 0.0
    room_step_counts = sorted(steps_per_room.get(r, 0) for r in full_room_set)
    counts_total = sum(room_step_counts)
    if counts_total > 0 and n_rooms > 0:
        visits_per_room_gini = (
            (2 * sum((idx + 1) * c for idx, c in enumerate(room_step_counts)))
            / (n_rooms * counts_total) - (n_rooms + 1) / n_rooms
        )
    else:
        visits_per_room_gini = 0.0

    # Memory-quality: people/interactions witnessed along the trajectory.
    # Read from the log (not step fields) so non-VLM strategies that leave
    # step.people_present/interactions empty are measured uniformly.
    people_observed_steps = 0
    interaction_cells_witnessed = 0
    for step in sim.minute_trace:
        entry = sim.log.scene_at(step.timestamp, step.room)
        if entry is None:
            continue
        if entry.people_present:
            people_observed_steps += 1
        if entry.interactions:
            interaction_cells_witnessed += 1
    interaction_cells_total = sum(1 for v in sim.log.interaction_index.values() if v)
    people_observation_rate = people_observed_steps / total_steps if total_steps else 0.0
    interaction_witness_rate = (
        interaction_cells_witnessed / interaction_cells_total if interaction_cells_total else 0.0
    )

    # Compute latencies, catchability, absent vs blind, visit-opportunities
    latencies: list[int] = []
    per_room_latencies: dict[str, list[int]] = {r: [] for r in all_rooms}
    per_room_never: dict[str, int] = {r: 0 for r in all_rooms}
    changes_never_detected = 0
    ts_histogram: dict[str, int] = {}
    catchable_changes = 0
    absent_changes = 0
    blind_changes = 0
    visit_opportunities: list[int] = []  # 0 = detected on 1st visit, 1 = 2nd visit, ...
    per_room_visit_opp: dict[str, list[int]] = {r: [] for r in all_rooms}
    per_room_absent: dict[str, int] = {r: 0 for r in all_rooms}
    per_room_blind: dict[str, int] = {r: 0 for r in all_rooms}
    per_room_catchable: dict[str, int] = {r: 0 for r in all_rooms}

    for change_ts, room, _ctype in gt_changes:
        room_steps = robot_steps_by_room.get(room, [])
        # Collect all visits at or after the change time
        visits_after = [(ts, step) for ts, step in room_steps if ts >= change_ts]
        was_visited = len(visits_after) > 0

        detected = False
        detection_latency = None
        detection_visit_idx = None

        for visit_idx, (visit_ts, step) in enumerate(visits_after):
            if _is_detected(step):
                detection_latency = int((visit_ts - change_ts).total_seconds() / 300)
                detection_visit_idx = visit_idx  # 0-based visit count after change
                detected = True
                break

        if detected:
            latencies.append(detection_latency)
            per_room_latencies[room].append(detection_latency)
            visit_opportunities.append(detection_visit_idx)
            per_room_visit_opp[room].append(detection_visit_idx)
            bin_key = "T=0" if detection_latency == 0 else f"T+{detection_latency}"
            ts_histogram[bin_key] = ts_histogram.get(bin_key, 0) + 1
            catchable_changes += 1
            per_room_catchable[room] += 1
        elif was_visited:
            # Visited but never detected -> BLIND
            blind_changes += 1
            per_room_blind[room] += 1
            catchable_changes += 1
            per_room_catchable[room] += 1
            changes_never_detected += 1
            ts_histogram["blind"] = ts_histogram.get("blind", 0) + 1
        else:
            # Never visited after change -> ABSENT
            absent_changes += 1
            per_room_absent[room] += 1
            changes_never_detected += 1
            per_room_never[room] += 1
            ts_histogram["absent"] = ts_histogram.get("absent", 0) + 1

    # Conditional latency histogram (detected only)
    conditional_histogram: dict[str, int] = {}
    for latency in latencies:
        bin_key = "T=0" if latency == 0 else f"T+{latency}"
        conditional_histogram[bin_key] = conditional_histogram.get(bin_key, 0) + 1

    # Visit-opportunity histogram
    visit_opp_histogram: dict[str, int] = {}
    for opp in visit_opportunities:
        bin_key = "V=0" if opp == 0 else f"V+{opp}"
        visit_opp_histogram[bin_key] = visit_opp_histogram.get(bin_key, 0) + 1
    if blind_changes > 0:
        visit_opp_histogram["blind"] = blind_changes
    if absent_changes > 0:
        visit_opp_histogram["absent"] = absent_changes

    # Per-room latency breakdown
    per_room_latency: dict[str, Any] = {}
    for room in all_rooms:
        room_hist: dict[str, int] = {}
        room_latencies = per_room_latencies[room]
        for latency in room_latencies:
            bin_key = "T=0" if latency == 0 else f"T+{latency}"
            room_hist[bin_key] = room_hist.get(bin_key, 0) + 1
        if per_room_absent[room] > 0:
            room_hist["absent"] = per_room_absent[room]
        if per_room_blind[room] > 0:
            room_hist["blind"] = per_room_blind[room]

        room_total = len(room_latencies) + per_room_absent[room] + per_room_blind[room]
        room_catchable = per_room_catchable[room]
        per_room_latency[room] = {
            "histogram": room_hist,
            "avg": round(sum(room_latencies) / len(room_latencies), 2) if room_latencies else None,
            "median": int(sorted(room_latencies)[len(room_latencies) // 2]) if room_latencies else None,
            "max": max(room_latencies) if room_latencies else None,
            "total_changes": room_total,
            "catchable": room_catchable,
            "detected": len(room_latencies),
            "absent": per_room_absent[room],
            "blind": per_room_blind[room],
            "detection_rate": round(len(room_latencies) / room_catchable, 4) if room_catchable > 0 else 0.0,
        }

    # Cumulative detection curve: % detected within N steps (of ALL changes)
    cumulative: dict[str, Any] = {}
    if latencies:
        max_latency = max(latencies)
        total_detected = len(latencies)
        total_changes = len(gt_changes)
        sorted_latencies = sorted(latencies)
        for n in range(0, max_latency + 1):
            count = sum(1 for l in sorted_latencies if l <= n)
            cumulative[f"≤T+{n}"] = {
                "count": count,
                "pct_of_detected": round(count / total_detected * 100, 1),
                "pct_of_all": round(count / total_changes * 100, 1),
            }

    avg_latency_steps = sum(latencies) / len(latencies) if latencies else 0.0
    median_latency_steps = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
    max_latency_steps = max(latencies) if latencies else 0
    detected_among_catchable = round(len(latencies) / catchable_changes, 4) if catchable_changes > 0 else 0.0

    # Keep minute-based metrics for backward compatibility
    avg_latency_min = avg_latency_steps * 5
    median_latency_min = median_latency_steps * 5
    max_latency_min = max_latency_steps * 5

    # Old minute-based histogram (for backward compat)
    min_histogram = {"0min": 0, "1-5min": 0, "6-15min": 0, "16-30min": 0, "31-60min": 0, "61+min": 0}
    for lat in latencies:
        lat_min = lat * 5
        if lat_min == 0:
            min_histogram["0min"] += 1
        elif lat_min <= 5:
            min_histogram["1-5min"] += 1
        elif lat_min <= 15:
            min_histogram["6-15min"] += 1
        elif lat_min <= 30:
            min_histogram["16-30min"] += 1
        elif lat_min <= 60:
            min_histogram["31-60min"] += 1
        else:
            min_histogram["61+min"] += 1

    # Paper-ready efficiency metrics
    path_max_changes = 0
    for step in sim.minute_trace:
        entry = sim.log.scene_at(step.timestamp, step.room)
        if entry is not None and entry.change_type in ("minor_change", "major_change"):
            path_max_changes += 1

    theoretical_max_changes = len(
        set(e.timestamp for e in sim.log.entries if e.change_type in ("minor_change", "major_change"))
    )
    detection_efficiency = (
        sim.observed_changes / path_max_changes if path_max_changes > 0 else 0.0
    )
    navigation_efficiency = (
        path_max_changes / theoretical_max_changes if theoretical_max_changes > 0 else 0.0
    )
    normalized_recall = (
        (sim.observed_changes / sim.log.total_changes) / (theoretical_max_changes / sim.log.total_changes)
        if theoretical_max_changes > 0 and sim.log.total_changes > 0
        else 0.0
    )

    # Optimal path: maximum events with minimum moves
    optimal = _compute_optimal_path_metrics(sim.log)
    optimal_path_changes = optimal["optimal_path_changes"]
    optimal_path_moves = optimal["optimal_path_moves"]
    optimal_path_efficiency = (
        sim.observed_changes / optimal_path_changes if optimal_path_changes > 0 else 0.0
    )

    # Paired recall metrics: presence (physically present when the GT change
    # happened — the old change_recall semantics) vs detected (VLM actually
    # flagged the change while present). For non-VLM strategies there is no
    # detector, so detected_recall stays 0 and detection_basis says the
    # latency metrics rest on implicit presence instead.
    presence_recall = sim.observed_changes / sim.log.total_changes if sim.log.total_changes else 0.0
    detected_changes = getattr(sim, "detected_changes", 0)
    detected_recall = detected_changes / sim.log.total_changes if sim.log.total_changes else 0.0
    detection_basis = (
        "vlm_detected_change"
        if any(step.vlm_detected_change is not None for step in sim.minute_trace)
        else "implicit_presence"
    )

    return {
        "cross_room_misses": cross_room_misses,
        "cross_room_miss_rate": round(cross_room_miss_rate, 4),
        "total_event_timesteps": total_event_timesteps,
        "event_timesteps_caught": event_timesteps_caught,
        "event_timestep_rate": round(event_timestep_rate, 4),
        "avg_detection_latency_minutes": round(avg_latency_min, 1),
        "median_detection_latency_minutes": round(median_latency_min, 1),
        "max_detection_latency_minutes": round(max_latency_min, 1),
        "changes_never_detected": changes_never_detected,
        "latency_histogram": min_histogram,
        "path_max_changes": path_max_changes,
        "theoretical_max_changes": theoretical_max_changes,
        "detection_efficiency": round(detection_efficiency, 4),
        "navigation_efficiency": round(navigation_efficiency, 4),
        "normalized_recall": round(normalized_recall, 4),
        "optimal_path_changes": optimal_path_changes,
        "optimal_path_moves": optimal_path_moves,
        "optimal_path_efficiency": round(optimal_path_efficiency, 4),
        "detection_latency_histogram": ts_histogram,
        "avg_detection_latency_steps": round(avg_latency_steps, 2),
        "median_detection_latency_steps": round(median_latency_steps, 2),
        "max_detection_latency_steps": max_latency_steps,
        "changes_detected_immediately": ts_histogram.get("T=0", 0),
        "per_room_latency": per_room_latency,
        "cumulative_detection_curve": cumulative,
        "catchable_changes": catchable_changes,
        "detected_among_catchable": detected_among_catchable,
        "absent_changes": absent_changes,
        "blind_changes": blind_changes,
        "conditional_latency_histogram": conditional_histogram,
        "visit_opportunity_histogram": visit_opp_histogram,
        "presence_recall": round(presence_recall, 4),
        "detected_recall": round(detected_recall, 4),
        "detection_basis": detection_basis,
        "move_value_k3": round(move_value_k3, 4),
        "move_value_k6": round(move_value_k6, 4),
        "wasted_dwell_rate": round(wasted_dwell_rate, 4),
        "coverage_halftime": coverage_halftime,
        "coverage_milestones": coverage_milestones,
        "avg_mean_staleness_minutes": round(avg_mean_staleness_minutes, 1),
        "visits_per_room_gini": round(visits_per_room_gini, 4),
        "people_observed_steps": people_observed_steps,
        "people_observation_rate": round(people_observation_rate, 4),
        "interaction_cells_witnessed": interaction_cells_witnessed,
        "interaction_cells_total": interaction_cells_total,
        "interaction_witness_rate": round(interaction_witness_rate, 4),
    }


# ---------------------------------------------------------------------------
# Distance matrix (walking minutes between rooms)
# ---------------------------------------------------------------------------

_distances = {
    ("kitchen", "meeting room"): 2,
    ("kitchen", "office 1"): 2,
    ("kitchen", "office 2"): 3,
    ("kitchen", "office 3"): 4,
    ("kitchen", "storage"): 3,
    ("meeting room", "office 1"): 3,
    ("meeting room", "office 2"): 2,
    ("meeting room", "office 3"): 3,
    ("meeting room", "storage"): 2,
    ("office 1", "office 2"): 1,
    ("office 1", "office 3"): 2,
    ("office 1", "storage"): 4,
    ("office 2", "office 3"): 1,
    ("office 2", "storage"): 3,
    ("office 3", "storage"): 2,
}

DISTANCES: dict[str, dict[str, int]] = defaultdict(dict)
for r in SCENE_CHANGE_ROOM_ORDER:
    DISTANCES[r][r] = 0
for (r1, r2), d in _distances.items():
    DISTANCES[r1][r2] = d
    DISTANCES[r2][r1] = d


def travel_time(room_a: str, room_b: str) -> int:
    a, b = _normalize_room(room_a), _normalize_room(room_b)
    if a == b:
        return 0
    return DISTANCES.get(a, {}).get(b, 2)


def _policy_travel_time(log: Any, room_a: str, room_b: str) -> int:
    """Dataset-aware walking minutes for nav policies and prompts.

    Uses the log's own distances (SceneChangeTimeLog.distance) when the
    dataset ships a distances.json sidecar; falls back to the legacy 6-room
    DISTANCES table otherwise.
    """
    dist = getattr(log, "distance", None)
    if callable(dist):
        return dist(room_a, room_b)
    return travel_time(room_a, room_b)


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------


class _SteppableFixedRotation:
    def __init__(self, log: SceneChangeTimeLog, dwell_steps: int):
        self.log = log
        self.dwell_steps = dwell_steps
        self.current_room: str | None = None
        self.room_idx = 0
        self.steps_in_room = 0
        self.visits: list[SceneChangeVisit] = []
        self.minute_trace: list[SceneChangeMinuteStep] = []
        self.visit_number = 0
        self.observed_changes = 0
        self.hits = 0
        self.rooms_visited: set[str] = set()
        self.total_moves = 0
        self.navigation_to_changed_room = 0

    def step(self, current_time: datetime) -> SceneChangeMinuteStep:
        if self.current_room is None:
            self.current_room = self.log.rooms[0]
            self.visit_number = 1
            self.visits.append(SceneChangeVisit(
                visit_number=self.visit_number,
                room=self.current_room,
                start=current_time,
                end=current_time,
                dwell_seconds=0,
            ))

        # Check if we should move
        if self.steps_in_room >= self.dwell_steps:
            # Finalize current visit
            self.visits[-1].end = current_time
            self.visits[-1].dwell_seconds = int((self.visits[-1].end - self.visits[-1].start).total_seconds())

            # Move to next room
            self.room_idx = (self.room_idx + 1) % len(self.log.rooms)
            next_room = self.log.rooms[self.room_idx]
            # Check if target room has a change (navigation quality)
            entry = self.log.scene_at(current_time, next_room)
            if entry and entry.change_type in ("minor_change", "major_change"):
                self.navigation_to_changed_room += 1
            self.total_moves += 1
            self.current_room = next_room
            self.steps_in_room = 0
            self.visit_number += 1
            self.visits.append(SceneChangeVisit(
                visit_number=self.visit_number,
                room=self.current_room,
                start=current_time,
                end=current_time,
                dwell_seconds=0,
            ))

        self.rooms_visited.add(self.current_room)

        entry = self.log.scene_at(current_time, self.current_room)
        gt_change = entry.change_type if entry else "no_change"
        is_changed = gt_change in ("minor_change", "major_change")

        if is_changed:
            self.observed_changes += 1
            self.visits[-1].changes_observed += 1

        step = SceneChangeMinuteStep(
            timestamp=current_time,
            time_str=current_time.strftime("%H:%M"),
            room=self.current_room,
            scene_text=entry.scene if entry else "No data.",
            gt_change_type=gt_change,
            people_present=entry.people_present if entry else [],
            objects_present=entry.objects_present if entry else [],
        )
        self.minute_trace.append(step)
        self.steps_in_room += 1
        return step

    def finalize(self) -> SceneChangeSimulationResult:
        if self.visits:
            self.visits[-1].end = self.log.end_time
            self.visits[-1].dwell_seconds = int((self.visits[-1].end - self.visits[-1].start).total_seconds())
            if self.visits[-1].changes_observed > 0:
                self.hits += 1

        nav_precision = self.navigation_to_changed_room / self.total_moves if self.total_moves > 0 else 0.0
        gt_metrics = _compute_gt_metrics(self)
        robot_memory_metrics = _compute_robot_memory_metrics(self)
        state_diff_metrics = _compute_state_diff_metrics(self)
        return SceneChangeSimulationResult(
            strategy=f"fixed_{self.dwell_steps * 5}min",
            total_changes=self.log.total_changes,
            total_major_changes=self.log.total_major_changes,
            observed_changes=self.observed_changes,
            total_visits=self.visit_number,
            visits_with_changes=self.hits,
            change_recall=self.observed_changes / self.log.total_changes if self.log.total_changes else 0.0,
            hit_rate=self.hits / self.visit_number if self.visit_number else 0.0,
            total_moves=self.total_moves,
            navigation_to_changed_room=self.navigation_to_changed_room,
            navigation_precision=nav_precision,
            rooms_visited=set(self.rooms_visited),
            exploration_coverage=len(self.rooms_visited) / len(self.log.rooms),
            visits=self.visits,
            minute_trace=self.minute_trace,
            **gt_metrics,
            **robot_memory_metrics,
            **state_diff_metrics,
        )


class _SteppablePerfectOracle:
    """Perfect oracle: always pre-position in a room that will have a GT change.

    This is equivalent to the greedy one-step lookahead strategy and achieves
    the global theoretical upper bound on observed changes (one per change
    timestamp, since the robot can only occupy one room at a time).
    """

    def __init__(self, log: SceneChangeTimeLog, seed: int | None = None):
        self.log = log
        self.rng = random.Random(seed) if seed is not None else random
        self.current_room: str | None = None
        self.visits: list[SceneChangeVisit] = []
        self.minute_trace: list[SceneChangeMinuteStep] = []
        self.visit_number = 0
        self.observed_changes = 0
        self.hits = 0
        self.rooms_visited: set[str] = set()
        self.total_moves = 0
        self.navigation_to_changed_room = 0
        self.steps_in_room = 0
        self.dwell_steps = 1
        self.pending_move_room: str | None = None

    def step(self, current_time: datetime) -> SceneChangeMinuteStep:
        # Apply pending move from previous step (baseline pattern)
        if self.pending_move_room is not None:
            self.current_room = self.pending_move_room
            self.pending_move_room = None
            self.steps_in_room = 0
            self.visit_number += 1
            self.visits.append(SceneChangeVisit(
                visit_number=self.visit_number,
                room=self.current_room,
                start=current_time,
                end=current_time,
                dwell_seconds=0,
            ))

        if self.current_room is None:
            self.current_room = self.log.rooms[0]
            self.visit_number = 1
            self.visits.append(SceneChangeVisit(
                visit_number=self.visit_number,
                room=self.current_room,
                start=current_time,
                end=current_time,
                dwell_seconds=0,
            ))

        entry = self.log.scene_at(current_time, self.current_room)
        gt_change = entry.change_type if entry else "no_change"
        is_changed = gt_change in ("minor_change", "major_change")

        if is_changed:
            self.observed_changes += 1
            self.visits[-1].changes_observed += 1

        # Look ahead 1 step and decide next room (move applied at NEXT step)
        if self.steps_in_room >= self.dwell_steps - 1:
            next_changes = self.log.changes_in_window(current_time, window_steps=1)
            next_room = max(next_changes, key=next_changes.get)
            if next_changes[next_room] == 0:
                next_room = self.rng.choice(self.log.rooms)

            if next_room != self.current_room:
                re = self.log.scene_at(current_time, next_room)
                if re and re.change_type in ("minor_change", "major_change"):
                    self.navigation_to_changed_room += 1
                self.total_moves += 1
                self.pending_move_room = next_room

                self.visits[-1].end = current_time
                self.visits[-1].dwell_seconds = int(
                    (self.visits[-1].end - self.visits[-1].start).total_seconds()
                )
                if self.visits[-1].changes_observed > 0:
                    self.hits += 1

        self.rooms_visited.add(self.current_room)

        step = SceneChangeMinuteStep(
            timestamp=current_time,
            time_str=current_time.strftime("%H:%M"),
            room=self.current_room,
            scene_text=entry.scene if entry else "No data.",
            gt_change_type=gt_change,
            people_present=entry.people_present if entry else [],
            objects_present=entry.objects_present if entry else [],
        )
        self.minute_trace.append(step)
        self.steps_in_room += 1
        return step

    def finalize(self) -> SceneChangeSimulationResult:
        if self.visits:
            self.visits[-1].end = self.log.end_time
            self.visits[-1].dwell_seconds = int(
                (self.visits[-1].end - self.visits[-1].start).total_seconds()
            )
            if self.visits[-1].changes_observed > 0:
                self.hits += 1

        nav_precision = (
            self.navigation_to_changed_room / self.total_moves if self.total_moves > 0 else 0.0
        )
        gt_metrics = _compute_gt_metrics(self)
        robot_memory_metrics = _compute_robot_memory_metrics(self)
        state_diff_metrics = _compute_state_diff_metrics(self)
        return SceneChangeSimulationResult(
            strategy="perfect_oracle",
            total_changes=self.log.total_changes,
            total_major_changes=self.log.total_major_changes,
            observed_changes=self.observed_changes,
            total_visits=self.visit_number,
            visits_with_changes=self.hits,
            change_recall=self.observed_changes / self.log.total_changes if self.log.total_changes else 0.0,
            hit_rate=self.hits / self.visit_number if self.visit_number else 0.0,
            total_moves=self.total_moves,
            navigation_to_changed_room=self.navigation_to_changed_room,
            navigation_precision=nav_precision,
            rooms_visited=set(self.rooms_visited),
            exploration_coverage=len(self.rooms_visited) / len(self.log.rooms),
            visits=self.visits,
            minute_trace=self.minute_trace,
            **gt_metrics,
            **robot_memory_metrics,
            **state_diff_metrics,
        )


# ---------------------------------------------------------------------------
# VLM-based scene-change agent
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VLM client helper
# ---------------------------------------------------------------------------

class _NoThinkingCompletions:
    """Proxy for client.chat.completions that disables model "thinking".

    Thinking models (e.g. Qwen3.5-9B) otherwise burn the whole max_tokens
    budget on hidden reasoning and never emit the JSON verdict
    (finish_reason="length"), which the simulator counts as error steps
    (observed: error_rate=1.0 across an entire 9B grid). Enabled via
    VLM_DISABLE_THINKING=1; off by default so existing behavior is unchanged.
    """

    def __init__(self, inner: Any):
        self._inner = inner

    def create(self, **kwargs: Any) -> Any:
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        kwargs["extra_body"] = extra_body
        return self._inner.create(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _NoThinkingClient:
    """OpenAI client proxy exposing chat.completions with thinking disabled."""

    def __init__(self, client: Any):
        self._client = client

        class _Chat:
            completions = _NoThinkingCompletions(client.chat.completions)

        self.chat = _Chat()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def _get_vlm_client() -> tuple[Any, str]:
    """Return a VLM client pointing to the local Qwen3-VL server.

    Bypasses the frontend's global VLM selection to ensure the simulator
    always uses Qwen3-VL as specified.

    Performs a lightweight warmup chat completion so that model-loading
    delay is paid up-front rather than mid-simulation.
    """
    from openai import OpenAI

    # Env override first (e.g. VLM_API_URL=http://host.docker.internal:8002/v1
    # when the VLM runs on the host or is reached via an SSH tunnel), then
    # container-internal hostname, then localhost fallbacks.
    import os
    candidate_urls = []
    for env_var in ("VLM_API_URL", "VLM_BASE_URL", "OPENAI_BASE_URL"):
        env_url = os.environ.get(env_var)
        if env_url and env_url not in candidate_urls:
            candidate_urls.append(env_url)
    candidate_urls += [
        "http://host.docker.internal:8002/v1",
        "http://host.docker.internal:8000/v1",
        "http://42b9e761e7e5:8000/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://localhost:8002/v1",
        "http://127.0.0.1:8002/v1",
    ]
    # Model name override (e.g. VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct when
    # running against a different-size server); default keeps the 4B.
    model_name = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
    disable_thinking = os.environ.get("VLM_DISABLE_THINKING", "").lower() in ("1", "true", "yes")
    last_exc: Exception | None = None
    for url in candidate_urls:
        client = OpenAI(base_url=url, api_key="not-needed", timeout=60.0)
        if disable_thinking:
            client = _NoThinkingClient(client)
        try:
            client.models.list()
        except Exception as exc:
            last_exc = exc
            continue

        # URL is reachable — do a tiny warmup so model-loading happens now
        # rather than on the first real step.
        try:
            client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=1,
                timeout=120,
            )
            return client, model_name
        except Exception as exc:
            last_exc = exc
            # If the server is reachable but the model isn't ready, don't
            # keep trying other URLs — the same issue will happen there.
            break

    raise RuntimeError(
        f"Could not connect to local Qwen3-VL server (or model is still loading). "
        f"Tried: {', '.join(candidate_urls)}. Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Perfect simulation tool back-end
# ---------------------------------------------------------------------------

class _PerfectSimBackend:
    """Simulation back-end that answers tool queries from the robot's memory.

    This is the simulation equivalent of RealToolAdapter: both implement the
    same logical queries, but this one reads from in-memory counters while
    RealToolAdapter reads from PostgreSQL.
    """

    def __init__(
        self,
        room_visit_counts: Counter,
        room_change_counts: Counter,
        room_perceived_change_counts: Counter,
        room_last_visit_time: dict[str, datetime],
        room_last_change_time: dict[str, datetime],
        room_perceived_last_change_time: dict[str, datetime],
        strategy_type: str,
        current_time: datetime | None = None,
        log: SceneChangeTimeLog | None = None,
        hazard_fix: bool = False,
        signal_source: str = "perceived",
    ):
        # Mutable references — updates in the simulator are visible here
        self.room_visit_counts = room_visit_counts
        self.room_change_counts = room_change_counts
        self.room_perceived_change_counts = room_perceived_change_counts
        self.room_last_visit_time = room_last_visit_time
        self.room_last_change_time = room_last_change_time
        self.room_perceived_last_change_time = room_perceived_last_change_time
        self.strategy_type = strategy_type
        self.current_time = current_time
        self.log = log
        # False (default) = legacy future-horizon hazard denominator (old runs
        # comparable). True = §4.2-CORE: denominators use only elapsed steps
        # up to current_time, so no future GT leaks into the signal.
        self.hazard_fix = hazard_fix
        # §4.1: which counters feed the agent-facing tools. "perceived"
        # (default) = the VLM's own past verdicts — what production derives
        # from navigation_decisions. "gt" = changes-while-present ground
        # truth (oracle_tools upper-bound variant).
        if signal_source not in ("perceived", "gt"):
            raise ValueError(f"Invalid signal_source: {signal_source!r}. Expected 'perceived' or 'gt'.")
        self.signal_source = signal_source

    def _signal_counts(self) -> Counter:
        """Change counters matching signal_source (§4.1)."""
        return self.room_perceived_change_counts if self.signal_source == "perceived" else self.room_change_counts

    def _signal_last_times(self) -> dict[str, datetime]:
        """Last-change timestamps matching signal_source (§4.1)."""
        return self.room_perceived_last_change_time if self.signal_source == "perceived" else self.room_last_change_time

    # --- Raw helpers (same shapes as before, for _build_context_block) ---

    def _tool_get_room_event_density(self) -> dict[str, int]:
        counts = self._signal_counts()
        return {room: counts.get(room, 0) for room in self.log.rooms}

    def _tool_get_stale_rooms(self, current_time: datetime) -> list[dict]:
        stale = []
        for room in self.log.rooms:
            last = self.room_last_visit_time.get(room)
            if last is None:
                stale.append({"room": room, "minutes_since_visit": 999999, "never_visited": True})
            else:
                ago = int((current_time - last).total_seconds() / 60)
                if ago >= 15:
                    stale.append({"room": room, "minutes_since_visit": ago, "never_visited": False})
        return sorted(stale, key=lambda x: x["minutes_since_visit"], reverse=True)

    def _tool_get_room_visit_history(self) -> list[dict]:
        counts = self._signal_counts()
        return [
            {"room": room, "visits": self.room_visit_counts.get(room, 0), "changes_observed": counts.get(room, 0)}
            for room in self.log.rooms
        ]

    def _tool_get_room_change_rates(self) -> list[dict]:
        counts = self._signal_counts()
        result = []
        for room in self.log.rooms:
            visits = self.room_visit_counts.get(room, 0)
            changes = counts.get(room, 0)
            if visits == 0:
                result.append({"room": room, "visits": 0, "changes": 0, "rate": None, "status": "never visited"})
            elif self.signal_source == "perceived":
                # Laplace smoothing (§4.1): dampens 1/1 = 1.0 spikes from the
                # noisy perceived signal. GT keeps the raw legacy rate so
                # oracle_tools rows stay comparable to pre-flip runs.
                rate = (changes + 1) / (visits + 2)
                result.append({"room": room, "visits": visits, "changes": changes, "rate": round(rate, 2)})
            else:
                rate = changes / visits
                result.append({"room": room, "visits": visits, "changes": changes, "rate": round(rate, 2)})
        return sorted(result, key=lambda x: (x["rate"] is None, -(x["rate"] or 0)))

    def _tool_get_predicted_change_probability(self, current_time: datetime) -> list[dict]:
        """Predict P(change in next step) for each room using a simple hazard model.

        For each room:
        - base_rate = historical_changes / horizon_steps
        - avg_interval = horizon_steps / historical_changes (steps between changes)
        - time_since = steps since last observed change
        - predicted_prob = base_rate * max(1, time_since / avg_interval)
        - clamp to [0, 0.99]

        horizon_steps: legacy behavior (hazard_fix=False) uses the FULL dataset
        horizon including the future; hazard_fix=True (§4.2-CORE) uses only the
        elapsed steps up to current_time, so the signal never reads ahead.
        """
        total_steps = len(self.log.unique_timestamps) if self.log else 0
        if total_steps == 0:
            return []

        if self.hazard_fix:
            # Elapsed steps up to (and including) current_time — no future data.
            current_idx = self.log.ts_index.get(current_time)
            if current_idx is None:
                # current_time not in the log: first timestamp >= current_time.
                current_idx = next((i for i, ts in enumerate(self.log.unique_timestamps) if ts >= current_time), total_steps - 1)
            horizon = current_idx + 1
        else:
            current_idx = None
            horizon = total_steps

        # Counters matching signal_source (§4.1): perceived = the VLM's own
        # past verdicts (production-shaped); gt = changes-while-present.
        counts = self._signal_counts()
        last_times = self._signal_last_times()

        # §4.2 optimism prior (fixed branch only): without it, a room with
        # zero observed changes scores 0.0 forever — ranking the unexplored
        # below any stale explored room, which makes the expected-change
        # signal structurally anti-curiosity. Prior = half the global
        # per-room change rate over elapsed steps.
        n_rooms = len(self.log.rooms)
        base_rate_global = sum(counts.values()) / max(horizon * n_rooms, 1)
        optimism_prior = 0.5 * base_rate_global

        result = []
        for room in self.log.rooms:
            changes = counts.get(room, 0)
            if changes == 0:
                if self.hazard_fix and optimism_prior > 0.0:
                    result.append({
                        "room": room,
                        "probability": round(min(0.99, optimism_prior), 3),
                        "base_rate": round(base_rate_global, 4),
                        "time_since_steps": None,
                        "avg_interval_steps": None,
                        "status": "optimism prior (unexplored)",
                    })
                else:
                    result.append({
                        "room": room,
                        "probability": 0.0,
                        "base_rate": 0.0,
                        "time_since_steps": None,
                        "avg_interval_steps": None,
                        "status": "no changes observed yet",
                    })
                continue

            base_rate = changes / horizon
            avg_interval = horizon / changes

            last = last_times.get(room)
            if last is None:
                time_since_steps = horizon  # never observed
            elif self.hazard_fix:
                last_idx = self.log.ts_index.get(last)
                if last_idx is None:
                    time_since_steps = horizon
                else:
                    time_since_steps = current_idx - last_idx
            else:
                # Find index of last change timestamp in unique_timestamps
                try:
                    last_idx = self.log.unique_timestamps.index(last)
                    legacy_current_idx = 0
                    for i, ts in enumerate(self.log.unique_timestamps):
                        if ts >= current_time:
                            legacy_current_idx = i
                            break
                    else:
                        legacy_current_idx = total_steps - 1
                    time_since_steps = legacy_current_idx - last_idx
                except (ValueError, AttributeError):
                    time_since_steps = total_steps

            # Hazard model: probability scales with staleness. §4.2 (fixed
            # branch): raw multiplier — fresh rooms score BELOW base_rate,
            # overdue rooms above. The legacy max(1, …) floor made every
            # fresher-than-average room indistinguishable.
            if self.hazard_fix:
                prob = base_rate * (time_since_steps / avg_interval)
            else:
                prob = base_rate * max(1.0, time_since_steps / avg_interval)
            prob = min(0.99, prob)

            result.append({
                "room": room,
                "probability": round(prob, 3),
                "base_rate": round(base_rate, 4),
                "time_since_steps": time_since_steps,
                "avg_interval_steps": round(avg_interval, 1),
                "changes_observed": changes,
            })

        return sorted(result, key=lambda x: x["probability"], reverse=True)

    def _tool_get_time_since_last_change(self, current_time: datetime) -> list[dict]:
        counts = self._signal_counts()
        last_times = self._signal_last_times()
        result = []
        for room in self.log.rooms:
            last = last_times.get(room)
            changes = counts.get(room, 0)
            if last is None:
                result.append({"room": room, "minutes_ago": None, "total_changes": changes, "status": "never observed a change"})
            else:
                ago = int((current_time - last).total_seconds() / 60)
                result.append({"room": room, "minutes_ago": ago, "total_changes": changes})
        return sorted(result, key=lambda x: (x["minutes_ago"] is None, -(x["minutes_ago"] or 0)))

    # --- Normalised methods (same return shapes as RealToolAdapter) ---

    def get_room_visit_history(self, room_list: list[str]) -> dict[str, Any]:
        raw = self._tool_get_room_visit_history()
        lookup = {r["room"]: r for r in raw}
        return {
            "rooms": [
                {
                    "room": room,
                    "visits": lookup.get(room, {}).get("visits", 0),
                    "changes": lookup.get(room, {}).get("changes_observed", 0),
                    "last_arrived": None,  # simulator does not track ISO timestamps
                }
                for room in room_list
            ]
        }

    def get_room_change_rates(self, room_list: list[str]) -> dict[str, Any]:
        raw = self._tool_get_room_change_rates()
        lookup = {r["room"]: r for r in raw}
        return {
            "rooms": [
                {
                    "room": room,
                    "visits": lookup.get(room, {}).get("visits", 0),
                    "changes": lookup.get(room, {}).get("changes", 0),
                    "rate": lookup.get(room, {}).get("rate") if room in lookup else None,
                    "status": "never visited" if room not in lookup else "visited",
                }
                for room in room_list
            ]
        }

    def get_time_since_last_change(self, room_list: list[str]) -> dict[str, Any]:
        if self.current_time is None:
            return {"rooms": []}
        raw = self._tool_get_time_since_last_change(self.current_time)
        lookup = {r["room"]: r for r in raw}
        return {
            "rooms": [
                {
                    "room": room,
                    "minutes_ago": lookup.get(room, {}).get("minutes_ago"),
                    "last_change": None,
                    "status": lookup.get(room, {}).get("status", "never observed a change"),
                }
                for room in room_list
            ]
        }

    def get_stale_rooms(self, room_list: list[str], threshold_minutes: int = 15) -> dict[str, Any]:
        if self.current_time is None:
            return {"stale_rooms": []}
        raw = self._tool_get_stale_rooms(self.current_time)
        filtered = [r for r in raw if r.get("minutes_since_visit", 0) >= threshold_minutes]
        return {
            "stale_rooms": [
                {
                    "room": r["room"],
                    "minutes_since_visit": r.get("minutes_since_visit"),
                    "never_visited": r.get("never_visited", False),
                }
                for r in filtered  # §5.2: no [:5] cap — with 6 rooms one could vanish from the prompt
            ]
        }

    def get_predicted_change_probability(self, room_list: list[str]) -> dict[str, Any]:
        if self.current_time is None:
            return {"rooms": []}
        raw = self._tool_get_predicted_change_probability(self.current_time)
        lookup = {r["room"]: r for r in raw}
        return {
            "rooms": [
                {
                    "room": room,
                    "probability": lookup.get(room, {}).get("probability", 0.0),
                    "base_rate": lookup.get(room, {}).get("base_rate", 0.0),
                    "time_since_steps": lookup.get(room, {}).get("time_since_steps"),
                    "avg_interval_steps": lookup.get(room, {}).get("avg_interval_steps"),
                    "status": lookup.get(room, {}).get("status", "unknown"),
                }
                for room in room_list
            ]
        }


# ---------------------------------------------------------------------------
# Base VLM agent with tool-based context
# ---------------------------------------------------------------------------

class _SteppableSceneChangeVLMBase:
    """Base class with robot memory, tool context, and metrics."""

    def __init__(
        self,
        log: SceneChangeTimeLog,
        strategy_type: str = "agent_scene_change",
        system_prompt: str | None = None,
        seed: int | None = None,
        no_tools: bool = False,
        dwell_config: dict | None = None,
        detector_type: str | None = None,
        initial_pick: str = "vlm",
        hazard_fix: bool = False,
        signal_source: str = "perceived",
        mask_tools: frozenset[str] = frozenset(),
        anti_camping: bool = False,
        move_budget: int = 0,
        navigator_prompt: str | None = None,
        score_weights: dict | None = None,
        staleness_deadband: int = 6,
    ):
        self.log = log
        self.strategy_type = strategy_type
        self.prompt = system_prompt or "You are a robot navigation strategist. Output only JSON."
        self.rng = random.Random(seed) if seed is not None else random
        self.seed = seed
        self.no_tools = no_tools
        # Explicit dwell-enforcement config. None = derive from strategy_type
        # substring matching (legacy behaviour, keeps old runs comparable).
        # Keys: enabled (bool), min_steps, max_steps, unseen_override (bool).
        self.dwell_config = dwell_config
        # Detector identity for _call_vlm dispatch. None = use strategy_type
        # (backward compat). Baseline wrappers set this to the detector they
        # actually run (e.g. "agent_scene_change_v8_short") so the detector's
        # input does not depend on the wrapper name.
        self.detector_type = detector_type
        # How the initial room is chosen (§3.3): "vlm" = legacy full VLM call
        # whose detection fields are discarded; "random" = seeded RNG pick;
        # "fixed" = first room in self.log.rooms. "random"/"fixed"
        # skip the discarded VLM call entirely.
        self.initial_pick = initial_pick
        # Hazard-model denominator fix (§4.2-CORE). False = legacy
        # future-horizon behavior; greedy_hazard strategies force True.
        self.hazard_fix = hazard_fix
        # §4.1: which counters the agent-facing tools serve. "perceived"
        # (default) = the VLM's own past verdicts, matching production;
        # "gt" = oracle_tools upper-bound variant. Default flip recorded
        # per §0.3 rule 2 — post-flip runs are not comparable to pre-flip
        # GT-fed rows in saved reports.
        if signal_source not in ("perceived", "gt"):
            raise ValueError(f"Invalid signal_source: {signal_source!r}. Expected 'perceived' or 'gt'.")
        self.signal_source = signal_source
        # §15 curiosity-signal ablation: masked tools are unavailable to
        # the agent (error result, no signal in the prompt).
        self.mask_tools = frozenset(mask_tools)
        # Code-enforced anti-camping (v9 variants; both default off = legacy
        # behavior). anti_camping: force a move when the VLM keeps voting
        # stay in an unchanged room. move_budget (0 = off): force a move
        # after N consecutive detection-free steps without moving.
        self.anti_camping = anti_camping
        self.move_budget = int(move_budget)
        # v10_split: navigator system prompt (TURN 2) and weights for the
        # code-computed room scores (_compute_room_scores). Defaults match
        # the Tier-2 plan: score = 1.0·rate + 1.0·hazard + 0.5·staleness.
        self.navigator_prompt = navigator_prompt
        self.score_weights = score_weights or {"w_hot": 1.0, "w_exp": 1.0, "w_stale": 0.5}
        # Tier-3 ping-pong fix: a visited room contributes 0 staleness for
        # this many steps (5 min each) after the last visit; never-visited
        # rooms are exempt. 0 = legacy minutes-normalized scoring. Only
        # consumed by _compute_room_scores (v10_split family).
        self.staleness_deadband = int(staleness_deadband)

        # Robot's own memory (only what it has observed)
        self.room_last_observed: dict[str, dict] = {}
        # Rolling per-room observation window (oldest → newest). Maintained
        # alongside room_last_observed; only injected into prompts when a
        # strategy sets obs_history_size > 0 (v13_split family), so legacy
        # strategies are unaffected. Entries: time_str, people, objects,
        # activities, scene_text, vlm_changed (None on error steps).
        self.room_obs_history: dict[str, list[dict]] = defaultdict(list)
        # Exact datetimes of the last observation per room (HH:MM strings are
        # ambiguous across days; v14 injects the computed gap in minutes).
        self.room_last_observed_dt: dict[str, datetime] = {}
        self.room_visit_counts: Counter = Counter()
        self.room_last_visit_time: dict[str, datetime] = {}

        # Perceived person-object interaction history (v20r3 ownership term).
        # Counted ONLY from observations the robot actually made: an
        # observation counts as an interaction observation when the activity
        # list rendered for that step is non-empty. Never reads
        # log.interaction_index (that is the GT metric) — mirrors the
        # perceived/oracle separation used for room_perceived_change_*.
        self.room_interaction_obs_counts: Counter = Counter()
        self.room_observation_counts: Counter = Counter()

        # GT-based history (what actually happened while robot was present)
        self.room_change_counts: Counter = Counter()
        self.room_last_change_time: dict[str, datetime] = {}
        # Per-room change-type distribution over every step present
        # (no_change / minor_change / major_change) — GT counterpart of
        # room_perceived_change_type_counts below.
        self.room_change_type_counts: dict[str, Counter] = defaultdict(Counter)

        # Perceived history (what the VLM detected — only used by v3 tools)
        self.room_perceived_change_counts: Counter = Counter()
        self.room_perceived_last_change_time: dict[str, datetime] = {}
        # Perceived per-room change-type distribution over every step present
        # (from the VLM's own verdicts, never GT) — used by the entropy policy.
        self.room_perceived_change_type_counts: dict[str, Counter] = defaultdict(Counter)

        # State
        self.current_room: str | None = None
        self.current_visit_room: str | None = None
        self.current_visit_start: datetime | None = None
        self.current_visit_changes = 0
        self.steps_in_room = 0
        self.pending_move_room: str | None = None
        # Anti-camping counters: votes reset on a truthy vlm_detected_change,
        # increment on falsy non-error verdicts (error steps leave both
        # unchanged); steps_since_last_move resets in _apply_move.
        self.consecutive_no_change_votes = 0
        self.steps_since_last_move = 0

        self.visits: list[SceneChangeVisit] = []
        self.minute_trace: list[SceneChangeMinuteStep] = []
        self.visit_number = 0
        self.observed_changes = 0
        self.detected_changes = 0  # GT changes while present AND VLM-detected
        self.error_steps = 0  # VLM/parse/timeout failures (excluded from detection metrics)
        self.hits = 0
        self.rooms_visited: set[str] = set()
        self.total_moves = 0
        self.overridden_moves = 0  # simulator-forced moves (excluded from nav-precision)
        self.pending_move_overridden = False

        # Metrics — scene change
        self.scene_change_tp = 0
        self.scene_change_tn = 0
        self.scene_change_fp = 0
        self.scene_change_fn = 0

        # Per-severity metrics
        self.minor_tp = 0
        self.minor_fp = 0
        self.minor_fn = 0
        self.major_tp = 0
        self.major_fp = 0
        self.major_fn = 0

        # Metrics — activity change
        self.activity_change_tp = 0
        self.activity_change_tn = 0
        self.activity_change_fp = 0
        self.activity_change_fn = 0

        self.navigation_to_changed_room = 0

        # VLM stats
        self.stat_vlm_calls = 0
        self.stat_prompt_tokens = 0
        self.stat_completion_tokens = 0
        self.last_prompt: str | None = None
        self.last_raw_response: str | None = None
        self.last_tool_calls: list[dict] | None = None
        self.last_target_room: str | None = None

    # --- Perfect simulation back-end (unified tool interface) ---

    def _make_sim_backend(self, current_time: datetime | None = None) -> "_PerfectSimBackend":
        """Factory for the simulation tool back-end."""
        return _PerfectSimBackend(
            room_visit_counts=self.room_visit_counts,
            room_change_counts=self.room_change_counts,
            room_perceived_change_counts=self.room_perceived_change_counts,
            room_last_visit_time=self.room_last_visit_time,
            room_last_change_time=self.room_last_change_time,
            room_perceived_last_change_time=self.room_perceived_last_change_time,
            strategy_type=self.strategy_type,
            current_time=current_time,
            log=self.log,
            hazard_fix=self.hazard_fix,
            signal_source=self.signal_source,
        )

    # --- Delegated tool-like helpers (kept for _build_context_block compatibility) ---

    def _tool_get_room_event_density(self) -> dict[str, int]:
        return self._make_sim_backend()._tool_get_room_event_density()

    def _tool_get_stale_rooms(self, current_time: datetime) -> list[dict]:
        return self._make_sim_backend()._tool_get_stale_rooms(current_time)

    def _tool_get_room_visit_history(self) -> list[dict]:
        return self._make_sim_backend()._tool_get_room_visit_history()

    def _tool_get_room_change_rates(self) -> list[dict]:
        return self._make_sim_backend()._tool_get_room_change_rates()

    def _tool_get_predicted_change_probability(self, current_time: datetime) -> list[dict]:
        return self._make_sim_backend(current_time=current_time)._tool_get_predicted_change_probability(current_time)

    def _tool_get_time_since_last_change(self, current_time: datetime) -> list[dict]:
        return self._make_sim_backend()._tool_get_time_since_last_change(current_time)

    def _execute_tool(self, name: str, current_time: datetime) -> dict:
        """Execute a tool by name and return serializable results.

        Uses the unified format so that output matches RealToolAdapter.
        Masked tools (§15 ablation) return an error instead — the signal
        never reaches the prompt.
        """
        if name in self.mask_tools:
            return {"error": f"tool unavailable: {name}"}
        backend = self._make_sim_backend(current_time=current_time)
        room_list = list(self.log.rooms)
        if name == "get_room_change_rates":
            return backend.get_room_change_rates(room_list)
        elif name == "get_time_since_last_change":
            return backend.get_time_since_last_change(room_list)
        elif name == "get_stale_rooms":
            return backend.get_stale_rooms(room_list)
        elif name == "get_room_visit_history":
            return backend.get_room_visit_history(room_list)
        elif name == "get_predicted_change_probability":
            return backend.get_predicted_change_probability(room_list)
        else:
            return {"error": f"Unknown tool: {name}"}

    def _perceived_interaction_rates(self) -> dict[str, float]:
        """Laplace-smoothed P(person-object interaction | observation) per room.

        Perceived signal only: built from room_observation_counts /
        room_interaction_obs_counts, which are updated in
        _record_room_observation from the activity list the robot actually
        saw. Rooms never observed fall back to the Laplace prior 0.5, the
        same convention _compute_room_scores uses for an unknown change rate,
        so an unobserved room is never scored as uninteresting.
        """
        rates: dict[str, float] = {}
        for room in self.log.rooms:
            obs = self.room_observation_counts.get(room, 0)
            hits = self.room_interaction_obs_counts.get(room, 0)
            rates[room] = 0.5 if obs == 0 else (hits + 0.5) / (obs + 1.0)
        return rates

    def _compute_room_scores(self, current_time: datetime) -> list[dict]:
        """Code-computed room ranking for the v10_split navigator (Tier 2).

        score = w_hot·laplace_rate + w_exp·hazard + w_stale·staleness_norm
        - laplace_rate: changes-per-visit from _tool_get_room_change_rates
          (already Laplace-smoothed under signal_source="perceived";
          never-visited rooms count as the Laplace prior 0.5).
        - hazard: P(change next step) from _tool_get_predicted_change_probability
          (§4.2-fixed branch when hazard_fix=True).
        - staleness_norm: EXCESS steps since last visit over the
          staleness_deadband, normalized by the max excess over rooms
          (never-visited rooms are exempt and keep 1.0 — zeroing them would
          kill exploration). The deadband (Tier 3) kills the A→B→A
          ping-pong: a just-left room contributes 0 staleness for
          staleness_deadband steps instead of topping the ranking again
          within 1-2 navigator calls. staleness_deadband=0 reproduces the
          legacy minutes-normalized scoring.
        Returns a ranked list of {"room", "score", "reason", ...} dicts where
        reason names the strongest contributor. Deterministic per seed:
        ties broken by a (seed, room)-derived value, not by consuming rng.
        """
        w_hot = float(self.score_weights.get("w_hot", 1.0))
        w_exp = float(self.score_weights.get("w_exp", 1.0))
        w_stale = float(self.score_weights.get("w_stale", 0.5))
        # v20r3 ownership term; 0.0 for every legacy strategy, so the
        # three-term score is bit-identical to before.
        w_int = float(self.score_weights.get("w_int", 0.0))
        interactions = self._perceived_interaction_rates() if w_int > 0 else {}
        deadband = float(self.staleness_deadband)  # steps (5 min each)

        rates = {
            e["room"]: (0.5 if e.get("rate") is None else float(e["rate"]))
            for e in self._tool_get_room_change_rates()
        }
        hazards = {
            e["room"]: float(e.get("probability") or 0.0)
            for e in self._tool_get_predicted_change_probability(current_time)
        }
        stale_minutes: dict[str, float | None] = {}
        stale_excess_steps: dict[str, float | None] = {}
        for room in self.log.rooms:
            last = self.room_last_visit_time.get(room)
            if last is None:
                stale_minutes[room] = None
                stale_excess_steps[room] = None
            else:
                stale_minutes[room] = max(0.0, (current_time - last).total_seconds() / 60.0)
                steps_since = (current_time - last).total_seconds() / 300.0
                stale_excess_steps[room] = max(0.0, steps_since - deadband)
        visited_excess = [x for x in stale_excess_steps.values() if x is not None]
        max_excess = max(visited_excess) if visited_excess else 0.0

        scored = []
        for room in self.log.rooms:
            rate = rates.get(room, 0.5)
            hazard = hazards.get(room, 0.0)
            minutes = stale_minutes[room]
            excess = stale_excess_steps[room]
            stale_norm = 1.0 if excess is None else (excess / max_excess if max_excess > 0 else 0.0)
            interaction = interactions.get(room, 0.5) if w_int > 0 else 0.0
            t_hot = w_hot * rate
            t_exp = w_exp * hazard
            t_stale = w_stale * stale_norm
            t_int = w_int * interaction
            score = t_hot + t_exp + t_stale + t_int
            # Reason names the strongest contributor; a deadbanded staleness
            # (t_stale == 0) must not surface as "last seen"/"overdue".
            if minutes is None:
                reason = "never visited"
            elif t_int > 0 and t_int >= t_hot and t_int >= t_exp and t_int >= t_stale:
                reason = f"interactions: {interaction:.2f}/obs"
            elif t_hot >= t_exp and t_hot >= t_stale and t_hot > 0:
                reason = f"hot: {rate:.2f}/visit"
            elif t_exp >= t_stale and t_exp > 0:
                reason = f"overdue {minutes:.0f} min" if t_stale > 0 else f"predicted {hazard:.2f}"
            elif t_stale > 0:
                reason = f"last seen {minutes:.0f} min ago"
            else:
                reason = "no strong signal"
            scored.append({
                "room": room,
                "score": round(score, 3),
                "reason": reason,
                "rate": round(rate, 3),
                "hazard": round(hazard, 3),
                "stale_norm": round(stale_norm, 3),
                "interaction": round(interaction, 3),
                "_tie": random.Random(f"{self.seed}:{room}").random(),
            })
        scored.sort(key=lambda e: (-e["score"], e["_tie"]))
        for e in scored:
            del e["_tie"]
        return scored

    def _format_room_ranking(self, scores: list[dict]) -> str:
        """Compact ROOM RANKING block for the v10_split navigator prompt."""
        lines = ["ROOM RANKING (highest priority first):"]
        for i, e in enumerate(scores, start=1):
            lines.append(f"{i}. {e['room']:<12} score {e['score']:.2f}  ({e['reason']})")
        return "\n".join(lines)

    def _build_context_block_short(self, current_time: datetime) -> str:
        """Compact context block: top-3 hotspots + stale rooms only."""
        density = self._tool_get_room_event_density()
        stale = self._tool_get_stale_rooms(current_time)
        rates = self._tool_get_room_change_rates()

        # Top 3 by changes observed
        top_hotspots = sorted(density.items(), key=lambda x: x[1], reverse=True)[:3]
        hotspot_lines = [f"  {room}: {count} changes" for room, count in top_hotspots]

        # Stale rooms (never visited or >30 min)
        stale_lines = [
            f"  {s['room']}: never visited" if s['never_visited'] else f"  {s['room']}: {s['minutes_since_visit']} min ago"
            for s in stale if s.get('never_visited') or s.get('minutes_since_visit', 0) > 30
        ]

        return f"""Hotspots:
{chr(10).join(hotspot_lines)}
Stale:{chr(10).join(stale_lines) if stale_lines else '  none'}
"""

    def _build_context_block_v11(self, current_time: datetime) -> str:
        """Compromise context block: top-3 hotspots with last-visit time + stale."""
        density = self._tool_get_room_event_density()
        stale = self._tool_get_stale_rooms(current_time)

        # Top 3 by total changes observed, with last visit time
        top_hotspots = sorted(density.items(), key=lambda x: x[1], reverse=True)[:3]
        hotspot_lines = []
        for room, count in top_hotspots:
            last = self.room_last_visit_time.get(room)
            if last is not None:
                ago = int((current_time - last).total_seconds() / 60)
                hotspot_lines.append(f"  {room}: {count} changes (last visit {ago} min ago)")
            else:
                hotspot_lines.append(f"  {room}: {count} changes (never visited)")

        # Stale rooms (never visited or >30 min)
        stale_lines = [
            f"  {s['room']}: never visited" if s["never_visited"] else f"  {s['room']}: {s['minutes_since_visit']} min ago"
            for s in stale if s.get("never_visited") or s.get("minutes_since_visit", 0) > 30
        ]

        return f"""Hotspots:
{chr(10).join(hotspot_lines)}

Stale:
{chr(10).join(stale_lines) if stale_lines else '  none'}
"""

    def _build_context_block_predictive(self, current_time: datetime) -> str:
        """Predictive context block: likelihood scores + stale rooms (avoids 'change' word to prevent detection bleed)."""
        predictions = self._tool_get_predicted_change_probability(current_time)
        stale = self._tool_get_stale_rooms(current_time)

        # Prediction lines sorted by probability — use "activity likelihood" not "P(change)"
        pred_lines = []
        for p in predictions:
            room = p["room"]
            prob = p["probability"]
            status = p.get("status", "")
            if status == "no changes observed yet":
                pred_lines.append(f"  {room}: unknown pattern")
            elif status == "optimism prior (unexplored)":
                pred_lines.append(f"  {room}: unknown pattern (unexplored — small exploration prior)")
            else:
                score = int(prob * 100)
                overdue = "  OVERDUE" if p.get("time_since_steps") and p.get("avg_interval_steps") and p["time_since_steps"] > p["avg_interval_steps"] else ""
                pred_lines.append(f"  {room}: score {score}/100{overdue} ({p['changes_observed']} events, every ~{p['avg_interval_steps']:.0f} steps)")

        # Stale rooms (never visited or >30 min)
        stale_lines = [
            f"  {s['room']}: never visited" if s['never_visited'] else f"  {s['room']}: {s['minutes_since_visit']} min ago"
            for s in stale if s.get('never_visited') or s.get('minutes_since_visit', 0) > 30
        ]

        return f"""Activity likelihood (next step):
{chr(10).join(pred_lines)}

Stale:{chr(10).join(stale_lines) if stale_lines else '  none'}
"""

    def _record_room_observation(self, room: str, current_time: datetime, entry: Any,
                                 activity_list: list[str], vlm_changed: bool | None) -> None:
        """Update room_last_observed (unchanged legacy content) and append to
        the rolling room_obs_history window. The history is only injected into
        prompts by strategies with obs_history_size > 0 (v13_split family).
        vlm_changed records the detector's verdict for this observation
        (None on error steps) so the model can stay self-consistent."""
        obs = {
            "time_str": current_time.strftime("%H:%M"),
            "people": entry.people_present,
            "objects": entry.objects_present,
            "activities": activity_list,
            "scene_text": entry.scene,
            "vlm_changed": vlm_changed,
            "dt": current_time,
        }
        self.room_last_observed[room] = {k: v for k, v in obs.items() if k not in ("vlm_changed", "dt")}
        self.room_last_observed_dt[room] = current_time
        self.room_obs_history[room].append(obs)
        # Perceived interaction bookkeeping (v20r3). activity_list is derived
        # from what this observation showed, so this is memory, not oracle.
        self.room_observation_counts[room] += 1
        if activity_list:
            self.room_interaction_obs_counts[room] += 1

    def _build_context_block(self, current_time: datetime) -> str:
        """Build a context block from tool-like queries."""
        density = self._tool_get_room_event_density()
        stale = self._tool_get_stale_rooms(current_time)
        history = self._tool_get_room_visit_history()
        rates = self._tool_get_room_change_rates()
        time_since = self._tool_get_time_since_last_change(current_time)

        density_lines = [f"  {room}: {count} changes observed" for room, count in density.items()]
        stale_lines = [
            f"  {s['room']}: never visited" if s['never_visited'] else f"  {s['room']}: {s['minutes_since_visit']} min ago"
            for s in stale[:6]
        ]
        history_lines = [
            f"  {h['room']}: {h['visits']} visits, {h['changes_observed']} changes"
            for h in history
        ]
        rate_lines = [
            f"  {r['room']}: {r['rate']} changes/visit ({r['changes']} changes / {r['visits']} visits)"
            if r['rate'] is not None
            else f"  {r['room']}: unknown rate (never visited)"
            for r in rates
        ]
        time_lines = [
            f"  {t['room']}: {t['minutes_ago']} min since last change ({t['total_changes']} total changes)"
            if t['minutes_ago'] is not None
            else f"  {t['room']}: no change observed yet ({t['total_changes']} total changes)"
            for t in time_since
        ]

        return f"""ROOM EVENT DENSITY (from YOUR observations):
{chr(10).join(density_lines)}

ROOM CHANGE RATES (changes per visit — HIGHER is better):
{chr(10).join(rate_lines)}

TIME SINCE LAST CHANGE (rooms at top are most overdue):
{chr(10).join(time_lines)}

STALE ROOMS (not seen recently):
{chr(10).join(stale_lines) if stale_lines else '  None — all rooms visited recently'}

VISIT HISTORY:
{chr(10).join(history_lines)}
"""

    def _build_omniscient_context_block(self, current_time: datetime) -> str:
        """Build a context block using ground-truth data from the log (omniscient)."""
        # Compute actual change counts per room up to current_time
        actual_changes: Counter = Counter()
        last_change_time: dict[str, datetime] = {}
        for entry in self.log.entries:
            if entry.timestamp > current_time:
                break
            if entry.change_type in ("minor_change", "major_change"):
                actual_changes[entry.room] += 1
                last_change_time[entry.room] = entry.timestamp

        density_lines = [f"  {room}: {actual_changes.get(room, 0)} actual changes" for room in self.log.rooms]

        rate_lines = []
        for room in self.log.rooms:
            visits = self.room_visit_counts.get(room, 0)
            changes = actual_changes.get(room, 0)
            if visits == 0:
                rate_lines.append(f"  {room}: unknown rate (never visited)")
            else:
                rate_lines.append(f"  {room}: {changes / visits:.2f} changes/visit ({changes} changes / {visits} visits)")

        time_lines = []
        for room in self.log.rooms:
            last = last_change_time.get(room)
            changes = actual_changes.get(room, 0)
            if last is None:
                time_lines.append(f"  {room}: no actual change yet ({changes} total)")
            else:
                ago = int((current_time - last).total_seconds() / 60)
                time_lines.append(f"  {room}: {ago} min since last actual change ({changes} total)")

        stale_lines = []
        for room in self.log.rooms:
            last = self.room_last_visit_time.get(room)
            if last is None:
                stale_lines.append(f"  {room}: never visited")
            else:
                ago = int((current_time - last).total_seconds() / 60)
                stale_lines.append(f"  {room}: {ago} min ago")

        return f"""ROOM EVENT DENSITY (GROUND TRUTH for ALL rooms):
{chr(10).join(density_lines)}

ROOM CHANGE RATES (GROUND TRUTH):
{chr(10).join(rate_lines)}

TIME SINCE LAST ACTUAL CHANGE:
{chr(10).join(time_lines)}

STALE ROOMS (your visit history):
{chr(10).join(stale_lines)}

VISIT HISTORY:
{chr(10).join(f"  {room}: {self.room_visit_counts.get(room, 0)} visits" for room in self.log.rooms)}
"""

    def _format_current_scene_full(self, current_room: str, entry: SceneLogEntry) -> tuple[str, str]:
        """Full current-scene block (v8/is_v3 format): scene + people +
        objects + activities. Shared by _build_prompt and the v9_split
        navigator's CURRENT SCENE block so the formatting cannot diverge."""
        current_scene = f"Current room: {current_room}\nScene: {entry.scene}\nPeople: {entry.people_present}\nObjects: {entry.objects_present}"
        if entry.interactions:
            act_lines = [f"  - {i['subject']} {i['action']} {i['target']} ({i.get('verb_category', 'unknown')})" for i in entry.interactions]
            current_activities = "\nCurrent activities:\n" + "\n".join(act_lines)
        else:
            current_activities = "\nCurrent activities: none"
        return current_scene, current_activities

    def _build_prompt(self, current_time: datetime, current_room: str | None) -> str:
        """Build the user prompt for the VLM."""
        # v9 forcing variants run the identical v8 prompt path; they differ
        # from v8 only by code-enforced moves in _make_step.
        stype = self.strategy_type
        if "v9_antipcamp" in stype or "v9_budget" in stype:
            stype = "agent_scene_change_v8"
        # Split detector turns (v9_split*/v10_split): identical v8 observation
        # block, detection-only question/JSON contract at the tail (see is_v3
        # branch below). detector_tail: None = fused v8 tail (default),
        # "strict" = v9_split, "soft" = v9_split_soft (adds
        # interesting/observation keys), "v10" = v10_split (adds
        # entity_diffs/confidence keys).
        # NB: "v9_split_soft" must be matched BEFORE "v9_split" (substring),
        # and "v10_split" BEFORE the plain "v10" flag.
        detector_tail = None
        if "v11r" in stype or "v20r" in stype:
            # v11r/v20r ("redesigned" pair from PROMPTS_V11_VS_V20.md) TURN 1:
            # v8 observation block + RECENT OBSERVATIONS window; the JSON
            # contract lives in the SYSTEM prompt for these two, so the tail
            # is only the closing question. Also matches the cross variants
            # (v11r_det_v20r_nav / v20r_det_v11r_nav) — the DETECTOR half
            # (the part right after "agent_scene_change_") decides the tail.
            # Matched FIRST.
            detector_tail = "v11r" if "agent_scene_change_v11r" in self.strategy_type else "v20r"
            stype = "agent_scene_change_v8"
        elif "v22_split" in stype:
            # v22 TURN 1: v15 blocks (observation + gap + history + single
            # braces) but the contract DROPS departure_destination — the
            # navigator reasons about the destination itself. Matched FIRST.
            stype = "agent_scene_change_v8"
            detector_tail = "v22"
        elif "v21_split" in stype:
            # v21 TURN 1: identical to v15 (same contract) plus a cross-room
            # LAST OBSERVATIONS OF OTHER ROOMS block (cross_room_obs flag).
            stype = "agent_scene_change_v8"
            detector_tail = "v21"
        elif any(f"v{n}_split" in stype for n in (16, 17, 18, 19, 20)):
            # v16-v20 TURN 1: identical to v15 (observation block + gap +
            # history window + single-brace contract). Prompt-only changes;
            # the JSON contract and parsing are the v15 one. Matched FIRST.
            stype = "agent_scene_change_v8"
            detector_tail = "v15"
        elif "v15_split" in stype:
            # v15_split TURN 1: identical to v14 (observation block + gap +
            # history window + single-brace v13 contract). Matched FIRST.
            stype = "agent_scene_change_v8"
            detector_tail = "v15"
        elif "v14_split" in stype:
            # v14_split TURN 1: v8 observation block + RECENT OBSERVATIONS
            # window + injected gap-in-minutes, v13 detection contract with
            # single-brace JSON (v13 tails rendered literal {{ }}). FIRST.
            stype = "agent_scene_change_v8"
            detector_tail = "v14"
        elif "v13_split" in stype:
            # v13_split TURN 1: v8 observation block + RECENT OBSERVATIONS
            # window (obs_history_size > 0), detection-only tail with the v12
            # contract plus structured movement keys (leaving /
            # departure_destination / people_trend). Matched FIRST.
            stype = "agent_scene_change_v8"
            detector_tail = "v13"
        elif "v12_split" in stype:
            # v12_split TURN 1: v8 observation block, detection-only tail with
            # the combined v10 (entity_diffs/confidence) + soft
            # (interesting/observation) contract. Matched FIRST — "v12_split"
            # shares no substring with the other split flags, but keep the
            # explicit ordering convention of the branches below.
            stype = "agent_scene_change_v8"
            detector_tail = "v12"
        elif "v10_split" in stype:
            # v10_split TURN 1: identical v8 observation block, detection-only
            # tail with confidence/entity_diffs contract. Matched BEFORE the
            # plain "v10" flag below so the legacy v10 path is not taken.
            stype = "agent_scene_change_v8"
            detector_tail = "v10"
        elif "v9_split_soft" in stype or "v11_split" in stype:
            # v11_split uses the same soft detection-only tail (interesting /
            # observation keys) as v9_split_soft. Matched BEFORE the plain
            # "v11" flag so the legacy fused v11 path is not taken.
            stype = "agent_scene_change_v8"
            detector_tail = "soft"
        elif "v9_split" in stype:
            stype = "agent_scene_change_v8"
            detector_tail = "strict"
        # Strategy flags (always defined so they are available after the if/else)
        is_v2 = "v2" in stype or "history" in stype
        is_v3 = "v3" in stype or "v4" in stype or "v5" in stype or "v6" in stype or "v7" in stype or "v8" in stype
        is_v5 = "v5" in stype and "no_tools" not in stype
        is_v6 = "v6" in stype and "no_tools" not in stype
        is_v7 = "v7" in stype and "no_tools" not in stype
        is_v8 = "v8" in stype and "no_tools" not in stype and "short" not in stype and "quick" not in stype and "clean" not in stype
        is_v8_short = "v8_short" in stype
        is_v8_quick = "v8_quick" in stype
        is_v8_clean = "v8_clean" in stype
        is_v8_active = "v8_active" in stype
        is_v8_smart = "v8_smart" in stype
        is_v9 = "v9" in stype
        is_v10 = "v10" in stype
        is_v11 = "v11" in stype
        is_omniscient = "omniscient" in stype
        is_text_only = "text_only" in stype
        is_reactive_tools = "reactive_tools" in stype
        is_free_json = "free_json" in stype
        is_visual = "visual" in stype and "no_tools" not in stype
        is_visual_no_tools = "visual" in stype and "no_tools" in stype

        entry = None
        last = None
        if current_room is None:
            current_scene = "You are just starting. No current room."
            last_observed = "No previous observations."
            current_activities = ""
            last_activities = ""
        else:
            entry = self.log.scene_at(current_time, current_room)
            if entry:
                if is_v8_short or is_v8_quick or is_v8_clean or is_v8_active or is_v8_smart or is_v9 or is_v10 or is_v11:
                    current_scene = f"Current room: {current_room}\nScene: {entry.scene}"
                    if entry.interactions:
                        act_summary = ", ".join(f"{i['subject']} {i['action']} {i['target']}" for i in entry.interactions)
                        current_activities = f"\nActivities: {act_summary}"
                    else:
                        current_activities = "\nActivities: none"
                else:
                    current_scene, current_activities = self._format_current_scene_full(current_room, entry)
            else:
                current_scene = f"Current room: {current_room}\nNo scene data available."
                current_activities = ""

            last = self.room_last_observed.get(current_room)
            if last:
                if is_v8_short or is_v8_quick or is_v8_clean or is_v8_active or is_v8_smart or is_v9 or is_v10 or is_v11:
                    last_observed = f"\nLast ({last['time_str']}): {last.get('scene_text', 'No description.')}"
                    last_activities = ""  # removed for short/quick/clean variants
                else:
                    gap_note = ""
                    if detector_tail in ("v14", "v15", "v21", "v22", "v20r"):
                        # v14: inject the exact gap — the model cannot compute
                        # cross-day gaps from HH:MM strings (arithmetic errors
                        # caused FNs and truncated JSON in the v13 capture).
                        last_dt = self.room_last_observed_dt.get(current_room)
                        if last_dt is not None:
                            gap_min = max(0, int((current_time - last_dt).total_seconds() / 60))
                            gap_note = f", {gap_min} min ago"
                    last_observed = f"\nLast time you were in {current_room} ({last['time_str']}{gap_note}):\nPeople: {last['people']}\nObjects: {last['objects']}"
                    if last.get('activities'):
                        last_activities = f"\nLast activities: {', '.join(last['activities'])}"
                    else:
                        last_activities = "\nLast activities: none"
            else:
                last_observed = f"\nYou have never observed {current_room} before."
                last_activities = ""

        if is_v10:
            context_str = self._build_context_block_predictive(current_time)
        elif is_v8_short or is_v8_quick or is_v8_clean or is_v8_active or is_v8_smart or is_v9:
            context_str = self._build_context_block_short(current_time)
        elif is_v11:
            context_str = self._build_context_block_v11(current_time)
        else:
            context_str = self._build_context_block(current_time)

        # Dwell time string (used by tool-based agents)
        dwell_min = 0
        if self.current_visit_start is not None:
            dwell_min = int((current_time - self.current_visit_start).total_seconds() / 60)
        dwell_str = f"\nYou have been in {current_room} for approximately {dwell_min} minutes." if current_room and dwell_min > 0 else ""

        # v13_split: rolling per-room observation window for trend reasoning
        # and self-consistency. Empty for all legacy strategies
        # (obs_history_size defaults to 0; only the NavSplit v13 family sets
        # it). obs_history_include_caption additionally appends each entry's
        # scene caption to the line.
        history_str = ""
        obs_hist_size = getattr(self, "obs_history_size", 0)
        if obs_hist_size > 0 and current_room:
            hist = self.room_obs_history.get(current_room) or []
            # Exclude the latest entry — it is already shown as the LAST
            # OBSERVATION block above.
            window = hist[:-1][-obs_hist_size:]
            if window:
                include_caption = getattr(self, "obs_history_include_caption", False)
                lines = []
                for h in window:
                    people = h.get("people") or []
                    objects = h.get("objects") or []
                    acts = h.get("activities") or []
                    verdict = h.get("vlm_changed")
                    verdict_str = "changed" if verdict else ("no_change" if verdict is False else "error")
                    # v14: previous-day entries carry their date — bare HH:MM
                    # is ambiguous across days (cross-day arithmetic failures
                    # in the v13 capture).
                    hdt = h.get("dt")
                    if detector_tail in ("v14", "v15", "v21", "v22", "v11r", "v20r") and isinstance(hdt, datetime):
                        t_disp = hdt.strftime("%H:%M") if hdt.date() == current_time.date() else hdt.strftime("%m-%d %H:%M")
                    else:
                        t_disp = h.get("time_str", "?")
                    line = (f"- {t_disp}: {len(people)} people ({', '.join(people) or 'none'}), "
                            f"{len(objects)} objects — activities: {', '.join(acts) or 'none'} "
                            f"— your verdict: {verdict_str}")
                    if include_caption:
                        line += f' — scene: "{h.get("scene_text", "")}"'
                    lines.append(line)
                history_str = "\nRECENT OBSERVATIONS of this room (oldest → newest):\n" + "\n".join(lines) + "\n"

        # v21_split: cross-room context — the detector also sees its last
        # observation of every OTHER room (gap injected in minutes, same as
        # the current-room block). Empty unless the cross_room_obs flag is set.
        cross_room_str = ""
        if getattr(self, "cross_room_obs", False) and current_room:
            cr_lines = []
            for room in self.log.rooms:
                if room == current_room:
                    continue
                obs = self.room_last_observed.get(room)
                if not obs:
                    continue
                gap_note = ""
                last_dt = self.room_last_observed_dt.get(room)
                if last_dt is not None:
                    gap_min = max(0, int((current_time - last_dt).total_seconds() / 60))
                    gap_note = f", {gap_min} min ago"
                people = obs.get("people") or []
                objects = obs.get("objects") or []
                cr_lines.append(f"- {room} ({obs['time_str']}{gap_note}): "
                                f"{len(people)} people ({', '.join(people) or 'none'}), "
                                f"{len(objects)} objects")
            if cr_lines:
                cross_room_str = ("\nLAST OBSERVATIONS OF OTHER ROOMS (context only — "
                                  "not part of this room's diff):\n" + "\n".join(cr_lines) + "\n")

        # Entity diff for v5-v8 (compact for short/quick variants)
        entity_diff_str = ""
        if (is_v5 or is_v6 or is_v7 or is_v8 or is_v8_short or is_v8_quick or is_v8_clean or is_v8_active or is_v8_smart) and current_room and entry and last:
            cur_people = set(entry.people_present or [])
            last_people = set(last.get("people") or [])
            cur_objects = set(entry.objects_present or [])
            last_objects = set(last.get("objects") or [])
            people_added = sorted(cur_people - last_people)
            people_removed = sorted(last_people - cur_people)
            objects_added = sorted(cur_objects - last_objects)
            objects_removed = sorted(last_objects - cur_objects)
            total_diffs = len(people_added) + len(people_removed) + len(objects_added) + len(objects_removed)
            if is_v8_short or is_v8_quick or is_v8_clean or is_v8_active or is_v8_smart or is_v9 or is_v10 or is_v11:
                # Compact single-line diff
                parts = []
                if people_added:
                    parts.append(f"+{len(people_added)} people ({', '.join(people_added)})")
                if people_removed:
                    parts.append(f"-{len(people_removed)} people ({', '.join(people_removed)})")
                if objects_added:
                    parts.append(f"+{len(objects_added)} objects ({', '.join(objects_added)})")
                if objects_removed:
                    parts.append(f"-{len(objects_removed)} objects ({', '.join(objects_removed)})")
                diff_line = "; ".join(parts) if parts else "none"
                entity_diff_str = f"\nDiffs: {diff_line} → total: {total_diffs}\n"
            else:
                entity_diff_str = f"""
ENTITY DIFFERENCE ANALYSIS:
- Current people: {sorted(cur_people)}
- Last people:    {sorted(last_people)}
- People added:   {people_added if people_added else '(none)'}
- People removed: {people_removed if people_removed else '(none)'}
- Current objects: {sorted(cur_objects)}
- Last objects:    {sorted(last_objects)}
- Objects added:   {objects_added if objects_added else '(none)'}
- Objects removed: {objects_removed if objects_removed else '(none)'}
- TOTAL ENTITY DIFFERENCES: {total_diffs}
"""

        if is_text_only:
            # Text-only: raw scene description only
            scene_text = entry.scene if entry else "No scene data available."
            last_text = ""
            if last:
                last_text = f"\nLast time you were in {current_room} ({last['time_str']}): {last.get('scene_text', 'No description.')}"
            prompt = f"""Current room: {current_room}
Scene: {scene_text}
{last_text}
{dwell_str}

Current time: {current_time.strftime('%H:%M')}

Has the scene changed compared to your last observation? Should you STAY or MOVE?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        elif is_omniscient:
            omniscient_ctx = self._build_omniscient_context_block(current_time)
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{last_activities}
{dwell_str}

{omniscient_ctx}

Current time: {current_time.strftime('%H:%M')}

ACTIVE verbs indicate ongoing action: taking, holding, carrying, preparing, cutting, drinking, talking to, picking up, putting, placing onto.
PASSIVE verbs indicate static state: sitting, standing, stands next to, uses (when stationary).

Decide:
1. Has the scene changed (people/objects different)?
2. Have activities changed significantly?
3. Should you STAY or MOVE?
4. If moving, which room is most likely to have changed?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        elif is_reactive_tools:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{last_activities}
{dwell_str}

Current time: {current_time.strftime('%H:%M')}

You have access to these tools:
- get_room_change_rates
- get_time_since_last_change
- get_stale_rooms
- get_room_visit_history

Decide which tools you need (if any), then output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name", "tool_calls": []}}
If you don't need any tools, set tool_calls to [].
"""
        elif is_free_json:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{last_activities}

{context_str}

Current time: {current_time.strftime('%H:%M')}

ACTIVE verbs indicate ongoing action: taking, holding, carrying, preparing, cutting, drinking, talking to, picking up, putting, placing onto.
PASSIVE verbs indicate static state: sitting, standing, stands next to, uses (when stationary).

Decide:
1. Has the scene changed (people/objects different)?
2. Have activities changed significantly?
3. Should you STAY or MOVE?
4. If moving, which room is most likely to have changed?

Explain your reasoning freely in plain text. At the VERY END of your response, output this JSON:
{{"reasoning": "brief summary", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        elif is_v8_active:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{dwell_str}
{entity_diff_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Decide: scene changed? severity? stay or move? target room?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay"/"move", "target_room": "Room Name"}}
"""
        elif is_v8_smart:
            # Staleness warning for V8_SMART
            stale_warning = ""
            if last and current_room:
                try:
                    last_dt = datetime.strptime(last['time_str'], "%H:%M")
                    current_dt = current_time
                    # Handle same-day comparison
                    gap_min = int((current_dt - last_dt.replace(year=current_dt.year, month=current_dt.month, day=current_dt.day)).total_seconds() / 60)
                    if gap_min > 15:
                        stale_warning = f"\n⚠️ STALE MEMORY: Your last observation was {gap_min} min ago. Differences may be normal turnover, not a scene change. Only declare changed if you see active verbs indicating VERY RECENT activity.\n"
                except Exception:
                    pass
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{stale_warning}{dwell_str}
{entity_diff_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Decide: scene changed? severity? stay or move? target room?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay"/"move", "target_room": "Room Name"}}
"""
        elif is_v11:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{dwell_str}
{entity_diff_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Decide: scene changed? severity? stay or move? target room?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay"/"move", "target_room": "Room Name"}}
"""
        elif is_v8_clean:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{dwell_str}
{entity_diff_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Decide: scene changed? severity? stay or move? target room?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay"/"move", "target_room": "Room Name"}}
"""
        elif is_v3:
            if detector_tail in ("v11r", "v20r"):
                # v11r/v20r: the JSON contract lives in the SYSTEM prompt
                # (user's redesigned pair); the tail is only the closing
                # question from their template.
                if detector_tail == "v11r":
                    tail = "Has the scene changed, and is it interesting? Fill the JSON."
                else:
                    tail = "Is the current scene interesting? Fill the JSON."
            elif detector_tail in ("v14", "v15", "v21"):
                # v14/v15/v21 TURN 1: same contract as v13 but with single
                # braces — the v13 tails below render literal {{ }} because
                # they are plain strings inserted into an f-string.
                tail = """Decide: scene changed? severity? activities changed? confidence? interesting? people movement?

At the end, output this JSON:
{"reasoning": "max 80 words", "entity_diffs": 0, "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "confidence": 0.0-1.0, "interesting": true/false, "leaving": true/false, "departure_destination": "", "people_trend": "arriving"/"leaving"/"stable"/"mixed", "observation": "..."}"""
            elif detector_tail == "v22":
                # v22 TURN 1: same as v15 but WITHOUT departure_destination —
                # the navigator infers the destination itself from the
                # detector's observation note and the ranked table.
                tail = """Decide: scene changed? severity? activities changed? confidence? interesting? people movement?

At the end, output this JSON:
{"reasoning": "max 80 words", "entity_diffs": 0, "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "confidence": 0.0-1.0, "interesting": true/false, "leaving": true/false, "people_trend": "arriving"/"leaving"/"stable"/"mixed", "observation": "..."}"""
            elif detector_tail == "v13":
                # v13_split TURN 1: detection only — v12 contract (entity_diffs,
                # confidence, interesting) plus structured movement keys.
                tail = """Decide: scene changed? severity? activities changed? confidence? interesting? people movement?

At the end, output this JSON:
{{"reasoning": "...", "entity_diffs": 0, "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "confidence": 0.0-1.0, "interesting": true/false, "leaving": true/false, "departure_destination": "", "people_trend": "arriving"/"leaving"/"stable"/"mixed", "observation": "..."}}"""
            elif detector_tail == "v12":
                # v12_split TURN 1: detection only — v10 confidence/entity_diffs
                # contract plus soft interesting/observation keys.
                tail = """Decide: scene changed? severity? activities changed? confidence? interesting?

At the end, output this JSON:
{{"reasoning": "...", "entity_diffs": 0, "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "confidence": 0.0-1.0, "interesting": true/false, "observation": "..."}}"""
            elif detector_tail == "v10":
                # v10_split TURN 1: detection only — confidence-calibrated,
                # no navigation question, no action/target_room keys.
                tail = """Decide: scene changed? severity? activities changed? confidence?

At the end, output this JSON:
{{"reasoning": "...", "entity_diffs": 0, "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "confidence": 0.0-1.0}}"""
            elif detector_tail == "strict":
                # v9_split TURN 1: detection only — no navigation question,
                # no action/target_room keys in the contract.
                tail = """Decide: scene changed? severity? activities changed?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false}}"""
            elif detector_tail == "soft":
                # v9_split_soft TURN 1: detection + interestingness/observation.
                tail = """Decide: scene changed? severity? activities changed? interesting?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "interesting": true/false, "observation": "..."}}"""
            else:
                tail = """Decide: scene changed? severity? stay or move? target room?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay"/"move", "target_room": "Room Name"}}"""
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{dwell_str}
{history_str}{cross_room_str}{entity_diff_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

{tail}
"""
        elif is_visual:
            # Visual comparison: format previous and current observations as two image descriptions
            if current_room is None:
                image_a = "You are just starting. No previous image."
                image_b = "No current image."
            else:
                if last:
                    image_a = f"""Room: {current_room} (observed at {last['time_str']})
Scene: {last.get('scene_text', 'No description.')}
People visible: {last.get('people', [])}
Objects visible: {last.get('objects', [])}
Activities: {', '.join(last.get('activities', [])) or 'none'}"""
                else:
                    image_a = f"You have never observed {current_room} before."
                if entry:
                    image_b = f"""Room: {current_room} (current observation)
Scene: {entry.scene}
People visible: {entry.people_present}
Objects visible: {entry.objects_present}
Activities: {', '.join([i['subject'] + ' ' + i['action'] + ' ' + i['target'] for i in entry.interactions]) or 'none'}"""
                else:
                    image_b = f"No current scene data for {current_room}."
            prompt = f"""Image A (Last observation of this room):
{image_a}

Image B (Current observation of this room):
{image_b}

{dwell_str}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Compare Image A and Image B visually. Look for differences in people, objects, activities, and layout.
Ignore camera angle and lighting differences.

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        elif is_visual_no_tools:
            # Visual comparison without tools and without structured lists — prose only
            if current_room is None:
                image_a = "You are just starting. No previous observation."
                image_b = "No current observation."
            else:
                if last:
                    image_a = f"""Room: {current_room} (observed at {last['time_str']})
Scene: {last.get('scene_text', 'No description.')}"""
                else:
                    image_a = f"You have never observed {current_room} before."
                if entry:
                    image_b = f"""Room: {current_room} (current observation)
Scene: {entry.scene}"""
                else:
                    image_b = f"No current scene data for {current_room}."
            prompt = f"""Image A (Last observation of this room):
{image_a}

Image B (Current observation of this room):
{image_b}

{dwell_str}

Current time: {current_time.strftime('%H:%M')}

Compare Image A and Image B. Look for differences in people, objects, activities, and layout.
Ignore minor wording differences that do not change the meaning.

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        elif is_v2:
            prompt = f"""{current_scene}
{current_activities}
{last_observed}
{last_activities}

{context_str}

Current time: {current_time.strftime('%H:%M')}

ACTIVE verbs indicate ongoing action: taking, holding, carrying, preparing, cutting, drinking, talking to, picking up, putting, placing onto.
PASSIVE verbs indicate static state: sitting, standing, stands next to, uses (when stationary).

Decide:
1. Has the scene changed (people/objects different)?
2. Have activities changed significantly?
3. Should you STAY or MOVE?
4. If moving, which room is most likely to have changed?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
        else:
            prompt = f"""{current_scene}
{last_observed}

{context_str}

Current time: {current_time.strftime('%H:%M')}

Based on the current scene and your memory of this room, decide:
1. Has the scene changed compared to your last observation of this room?
2. Should you STAY to observe more, or MOVE to another room?
3. If moving, which room is most likely to have changed since you last saw it?

At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "action": "stay" or "move", "target_room": "Room Name"}}
"""
        return prompt

    def _update_memory_and_metrics(self, current_time: datetime, entry: SceneLogEntry | None,
                                    vlm_detected: bool, vlm_activity_detected: bool,
                                    gt_change: str, gt_activity_changed: bool,
                                    interactions: list[dict], vlm_severity: str = "no_change") -> None:
        """Update robot memory and confusion matrices."""
        is_changed = gt_change in ("minor_change", "major_change")
        # First-ever observation of this room: the robot has no baseline to
        # compare against, so scoring a detection here is meaningless.
        first_observation = self.current_room not in self.room_last_observed

        if not first_observation:
            # Scene-change confusion (binary)
            if vlm_detected and is_changed:
                self.scene_change_tp += 1
            elif not vlm_detected and not is_changed:
                self.scene_change_tn += 1
            elif vlm_detected and not is_changed:
                self.scene_change_fp += 1
            else:
                self.scene_change_fn += 1

            # Per-severity confusion, routed by GT severity. FPs have no GT
            # severity and stay in minor_fp (major_fp remains 0 by design).
            if gt_change == "minor_change":
                if vlm_detected:
                    self.minor_tp += 1
                else:
                    self.minor_fn += 1
            elif gt_change == "major_change":
                if vlm_detected:
                    self.major_tp += 1
                else:
                    self.major_fn += 1
            else:  # no_change
                if vlm_detected:
                    self.minor_fp += 1

            # Activity-change confusion
            if vlm_activity_detected and gt_activity_changed:
                self.activity_change_tp += 1
            elif not vlm_activity_detected and not gt_activity_changed:
                self.activity_change_tn += 1
            elif vlm_activity_detected and not gt_activity_changed:
                self.activity_change_fp += 1
            else:
                self.activity_change_fn += 1

        # Memory (including scene_text for text-only / no_tools agents)
        if entry:
            activity_list = [f"{i['subject']} {i['action']} {i['target']}" for i in interactions]
            self._record_room_observation(self.current_room, current_time, entry, activity_list, bool(vlm_detected))

        # Perceived history (v3 tools use this — reflects what the VLM actually detected)
        if vlm_detected:
            self.room_perceived_change_counts[self.current_room] += 1
            self.room_perceived_last_change_time[self.current_room] = current_time

        # Perceived change-type distribution over every step present (VLM
        # verdicts only — the deployable counterpart of the GT
        # room_change_type_counts below). Detected changes without a
        # minor/major severity ("true"/"yes") count as minor, mirroring the
        # "FPs stay in minor_fp" convention above.
        if not vlm_detected:
            perceived_type = "no_change"
        elif vlm_severity in ("major", "major_change"):
            perceived_type = "major_change"
        else:
            perceived_type = "minor_change"
        self.room_perceived_change_type_counts[self.current_room][perceived_type] += 1

        # Visit stats (metrics use GT)
        if is_changed:
            self.observed_changes += 1
            self.current_visit_changes += 1
            self.room_change_counts[self.current_room] += 1
            self.room_last_change_time[self.current_room] = current_time
            if vlm_detected:
                self.detected_changes += 1

        # Per-room change-type distribution (all steps present, incl. no_change).
        if gt_change:
            self.room_change_type_counts[self.current_room][gt_change] += 1

        # room_visit_counts is incremented per actual visit in _apply_move;
        # here we only refresh the last-observed time for staleness tracking.
        self.room_last_visit_time[self.current_room] = current_time

    def _apply_move(self, current_time: datetime, target_room: str, overridden: bool = False) -> None:
        """Initiate a move to target_room (no travel time).

        ``overridden`` marks simulator-forced moves (dwell/unseen overrides);
        they are counted separately and excluded from nav-precision metrics.
        """
        if self.current_room is not None and target_room == self.current_room:
            # Not a real move — no move accounting, no phantom visit.
            return
        if self.current_room is not None:
            if self.current_visit_start is not None and self.current_visit_room is not None:
                # Visit ends at current_time (start of new timestep)
                self.visits[-1].end = current_time
                self.visits[-1].dwell_seconds = int((self.visits[-1].end - self.visits[-1].start).total_seconds())
                if self.current_visit_changes > 0:
                    self.hits += 1

            if overridden:
                self.overridden_moves += 1
            else:
                entry = self.log.scene_at(current_time, target_room)
                if entry and entry.change_type in ("minor_change", "major_change"):
                    self.navigation_to_changed_room += 1
                self.total_moves += 1

        self.current_room = target_room
        self.current_visit_room = target_room
        self.current_visit_start = current_time
        self.current_visit_changes = 0
        self.steps_in_room = 0
        self.steps_since_last_move = 0
        self.visit_number += 1
        # One visit-count per actual visit (not per 5-min step), so
        # get_room_change_rates is truly changes-per-visit.
        self.room_visit_counts[target_room] += 1
        self.visits.append(SceneChangeVisit(
            visit_number=self.visit_number,
            room=self.current_room,
            start=current_time,
            end=current_time,
            dwell_seconds=0,
        ))

    def _make_step(self, current_time: datetime, parsed: dict[str, Any],
                   gt_change: str, gt_activity_changed: bool,
                   entry: SceneLogEntry | None, interactions: list[dict],
                   tool_calls: list[dict] | None = None) -> SceneChangeMinuteStep:
        """Create a SceneChangeMinuteStep from decision and GT."""
        vlm_detected = parsed.get("scene_changed") in (True, "true", "True", 1, "1")
        vlm_severity = str(parsed.get("change") or parsed.get("change_severity") or "false").lower()
        vlm_activity_detected = parsed.get("activities_changed") in (True, "true", "True", 1, "1")
        action = str(parsed.get("action", "stay")).lower()
        target_room = str(parsed.get("target_room", "")).strip()

        self._update_memory_and_metrics(
            current_time, entry, vlm_detected, vlm_activity_detected,
            gt_change, gt_activity_changed, interactions, vlm_severity,
        )

        # Anti-camping counters (v9 forcing). Error steps never reach
        # _make_step, so they leave both counters unchanged.
        if vlm_detected:
            self.consecutive_no_change_votes = 0
        else:
            self.consecutive_no_change_votes += 1
        self.steps_since_last_move += 1

        def _staleness(room: str) -> float:
            last = self.room_last_visit_time.get(room)
            return float("inf") if last is None else (current_time - last).total_seconds()

        # Tool-based dwell enforcement (applied to older agents only):
        # - Minimum: stay at least 2 steps (10 min) unless scene changed
        # - Maximum: move after 4 steps (20 min) if no change detected
        # v6 and newer agents have free navigation — no forced dwell or exploration override.
        # An explicit dwell_config overrides the legacy strategy-name matching.
        decision_override: str | None = None
        if self.dwell_config is not None:
            has_dwell_enforcement = bool(self.dwell_config.get("enabled", True))
            min_dwell_steps = int(self.dwell_config.get("min_steps", 2))
            max_dwell_steps = int(self.dwell_config.get("max_steps", 4))
            unseen_room_override = bool(self.dwell_config.get("unseen_override", True))
        else:
            has_dwell_enforcement = any(x in self.strategy_type for x in ["v3", "v4", "v5", "omniscient", "reactive_tools"])
            min_dwell_steps = 2
            max_dwell_steps = 4
            unseen_room_override = True
        if has_dwell_enforcement:
            if action == "move" and self.steps_in_room < min_dwell_steps and not vlm_detected:
                action = "stay"
                target_room = self.current_room
                decision_override = "forced_min_dwell"
            elif action == "stay" and self.steps_in_room >= max_dwell_steps and not vlm_detected:
                action = "move"
                decision_override = "forced_max_dwell"
                # Actually move: pick the most-stale room ≠ current room so a
                # real pending_move_room is set (previously the robot stayed
                # while the trace logged a move).
                candidates = [r for r in self.log.rooms if r != self.current_room]
                best = max(_staleness(r) for r in candidates)
                target_room = self.rng.choice([r for r in candidates if _staleness(r) == best])

        # Code-enforced anti-camping (v9 variants; default off = legacy).
        # Fires only when the VLM chose stay and no dwell override already
        # rewrote the decision. The forced move targets the argmax-staleness
        # room ≠ current, exactly like forced_max_dwell.
        if decision_override is None and action == "stay":
            if self.anti_camping and self.steps_in_room >= 4 and self.consecutive_no_change_votes >= 3:
                decision_override = "anti_camping"
            elif (self.move_budget > 0 and self.steps_since_last_move >= self.move_budget
                  and self.consecutive_no_change_votes >= self.move_budget):
                decision_override = "move_budget"
            if decision_override is not None:
                action = "move"
                candidates = [r for r in self.log.rooms if r != self.current_room]
                best = max(_staleness(r) for r in candidates)
                target_room = self.rng.choice([r for r in candidates if _staleness(r) == best])

        if action == "move" and target_room and target_room != self.current_room:
            for r in self.log.rooms:
                if r.lower() == target_room.lower():
                    target_room = r
                    break
            else:
                # Unknown room from the VLM: fall back to a random room ≠ current
                target_room = self.rng.choice([r for r in self.log.rooms if r != self.current_room])

            # Exploration override for older tool-based agents only
            if has_dwell_enforcement and unseen_room_override:
                unseen = [r for r in self.log.rooms if r not in self.room_last_visit_time]
                if unseen and target_room not in unseen:
                    target_room = self.rng.choice(unseen)
                    decision_override = "unseen_room_override"

            self.pending_move_room = target_room
            self.pending_move_overridden = decision_override is not None
            self.last_target_room = target_room

        return SceneChangeMinuteStep(
            timestamp=current_time,
            time_str=current_time.strftime("%H:%M"),
            room=self.current_room,
            scene_text=entry.scene if entry else "No data.",
            gt_change_type=gt_change,
            gt_activity_changed=gt_activity_changed,
            vlm_detected_change=vlm_detected,
            vlm_change=vlm_severity,
            vlm_detected_activity_change=vlm_activity_detected,
            action=action,
            target_room=target_room if action == "move" else self.current_room,
            decision=parsed,
            fresh_decision=True,
            people_present=entry.people_present if entry else [],
            objects_present=entry.objects_present if entry else [],
            interactions=interactions,
            prompt=self.last_prompt,
            raw_response=self.last_raw_response,
            system_prompt=self.prompt,
            tool_calls=tool_calls,
            decision_override=decision_override,
        )

    def finalize(self) -> SceneChangeSimulationResult:
        if self.visits:
            self.visits[-1].end = self.log.end_time
            self.visits[-1].dwell_seconds = int((self.visits[-1].end - self.visits[-1].start).total_seconds())
            if self.current_visit_changes > 0:
                self.hits += 1

        total = self.scene_change_tp + self.scene_change_tn + self.scene_change_fp + self.scene_change_fn
        accuracy = (self.scene_change_tp + self.scene_change_tn) / total if total > 0 else 0.0
        precision = self.scene_change_tp / (self.scene_change_tp + self.scene_change_fp) if (self.scene_change_tp + self.scene_change_fp) > 0 else 0.0
        recall = self.scene_change_tp / (self.scene_change_tp + self.scene_change_fn) if (self.scene_change_tp + self.scene_change_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        nav_precision = self.navigation_to_changed_room / self.total_moves if self.total_moves > 0 else 0.0

        # No-change / change accuracy (split)
        no_change_total = self.scene_change_tn + self.scene_change_fp
        no_change_accuracy = self.scene_change_tn / no_change_total if no_change_total > 0 else 0.0
        change_total = self.scene_change_tp + self.scene_change_fn
        change_accuracy = self.scene_change_tp / change_total if change_total > 0 else 0.0

        # Per-severity derived metrics
        minor_recall = self.minor_tp / (self.minor_tp + self.minor_fn) if (self.minor_tp + self.minor_fn) > 0 else 0.0
        minor_precision = self.minor_tp / (self.minor_tp + self.minor_fp) if (self.minor_tp + self.minor_fp) > 0 else 0.0
        major_recall = self.major_tp / (self.major_tp + self.major_fn) if (self.major_tp + self.major_fn) > 0 else 0.0
        major_precision = self.major_tp / (self.major_tp + self.major_fp) if (self.major_tp + self.major_fp) > 0 else 0.0

        act_total = self.activity_change_tp + self.activity_change_tn + self.activity_change_fp + self.activity_change_fn
        act_accuracy = (self.activity_change_tp + self.activity_change_tn) / act_total if act_total > 0 else 0.0
        gt_metrics = _compute_gt_metrics(self)
        robot_memory_metrics = _compute_robot_memory_metrics(self)
        state_diff_metrics = _compute_state_diff_metrics(self)

        return SceneChangeSimulationResult(
            strategy=self.strategy_type,
            total_changes=self.log.total_changes,
            total_major_changes=self.log.total_major_changes,
            observed_changes=self.observed_changes,
            total_visits=self.visit_number,
            visits_with_changes=self.hits,
            change_recall=self.observed_changes / self.log.total_changes if self.log.total_changes else 0.0,
            change_precision=precision,
            change_f1=f1,
            hit_rate=self.hits / self.visit_number if self.visit_number else 0.0,
            scene_change_tp=self.scene_change_tp,
            scene_change_tn=self.scene_change_tn,
            scene_change_fp=self.scene_change_fp,
            scene_change_fn=self.scene_change_fn,
            scene_change_accuracy=accuracy,
            no_change_accuracy=no_change_accuracy,
            change_accuracy=change_accuracy,
            minor_tp=self.minor_tp,
            minor_fp=self.minor_fp,
            minor_fn=self.minor_fn,
            major_tp=self.major_tp,
            major_fp=self.major_fp,
            major_fn=self.major_fn,
            minor_recall=minor_recall,
            minor_precision=minor_precision,
            major_recall=major_recall,
            major_precision=major_precision,
            activity_change_tp=self.activity_change_tp,
            activity_change_tn=self.activity_change_tn,
            activity_change_fp=self.activity_change_fp,
            activity_change_fn=self.activity_change_fn,
            activity_change_accuracy=act_accuracy,
            navigation_to_changed_room=self.navigation_to_changed_room,
            total_moves=self.total_moves,
            navigation_precision=nav_precision,
            overridden_moves=self.overridden_moves,
            error_steps=self.error_steps,
            error_rate=self.error_steps / len(self.minute_trace) if self.minute_trace else 0.0,
            rooms_visited=set(self.rooms_visited),
            exploration_coverage=len(self.rooms_visited) / len(self.log.rooms),
            vlm_calls_made=self.stat_vlm_calls,
            total_prompt_tokens=self.stat_prompt_tokens,
            total_completion_tokens=self.stat_completion_tokens,
            system_prompt=self.prompt,
            visits=self.visits,
            minute_trace=self.minute_trace,
            rank_greedy_calls=len(getattr(self, "rank_track", [])),
            rank_greedy_divergences=getattr(self, "stat_rank_divergences", 0),
            rank_greedy_override_moves=getattr(self, "stat_rank_override_moves", 0),
            rank_divergence_rate=(getattr(self, "stat_rank_divergences", 0) / len(getattr(self, "rank_track", []))
                                  if getattr(self, "rank_track", []) else 0.0),
            rank_divergence_log=list(getattr(self, "rank_track", [])),
            rank_target_histogram=_compute_rank_topk(getattr(self, "rank_track", []))[0],
            rank_topk_rates=_compute_rank_topk(getattr(self, "rank_track", []))[1],
            **gt_metrics,
            **robot_memory_metrics,
            **state_diff_metrics,
        )


class _SteppableSceneChangeVLM(_SteppableSceneChangeVLMBase):
    """VLM agent that uses the REAL VLM backend (Qwen3-VL or Kimi)."""

    def __init__(self, log: SceneChangeTimeLog, strategy_type: str = "agent_scene_change",
                 system_prompt: str | None = None, seed: int | None = None, no_tools: bool = False,
                 dwell_config: dict | None = None, detector_type: str | None = None,
                 anti_camping: bool = False, move_budget: int = 0,
                 navigator_prompt: str | None = None, score_weights: dict | None = None,
                 nav_walk_times: bool = True, nav_scene_context: bool = False,
                 staleness_deadband: int = 6):
        super().__init__(log, strategy_type, system_prompt, seed, no_tools=no_tools, dwell_config=dwell_config,
                         detector_type=detector_type, anti_camping=anti_camping, move_budget=move_budget,
                         navigator_prompt=navigator_prompt, score_weights=score_weights,
                         staleness_deadband=staleness_deadband)
        # v10_split ablations (consumed by _call_vlm_split; irrelevant for all
        # other strategies). nav_walk_times=False omits the WALK TIMES table
        # from the navigator prompt (ping-pong ablation); nav_scene_context=True
        # adds a per-room LAST OBSERVED SCENES block from perceived memory.
        self.nav_walk_times = nav_walk_times
        self.nav_scene_context = nav_scene_context
        self._vlm_client_info = _get_vlm_client()

    @staticmethod
    def _extract_json(raw_content: str) -> dict[str, Any]:
        """Robustly extract and parse a JSON object from VLM output."""
        content = raw_content.strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
        if content.endswith("```"):
            content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            content = match.group(0)

        content = _SteppableSceneChangeVLM._fix_truncated_json(content)
        return json.loads(content)

    @staticmethod
    def _format_tool_results_compact(tool_results: dict[str, Any]) -> str:
        """Compact single-line table format for tool results (v8_clean)."""
        lines = []
        # Change rates: room | visits | changes | rate
        rates = tool_results.get("get_room_change_rates", {}).get("rooms", [])
        if rates:
            lines.append("RATES (room: visits/changes/rate):")
            for r in rates:
                rate_str = f"{r['rate']:.2f}" if r['rate'] is not None else "n/a"
                lines.append(f"  {r['room']}: {r['visits']}v / {r['changes']}c / {rate_str}")
        # Time since last change
        times = tool_results.get("get_time_since_last_change", {}).get("rooms", [])
        if times:
            lines.append("STALE (room: min since last change):")
            for t in times:
                ago = f"{t['minutes_ago']}m" if t['minutes_ago'] is not None else "never"
                lines.append(f"  {t['room']}: {ago}")
        # Stale rooms
        stale = tool_results.get("get_stale_rooms", {}).get("stale_rooms", [])
        if stale:
            stale_list = [f"{s['room']}({s['minutes_since_visit']}m)" for s in stale if not s.get('never_visited')]
            never = [s['room'] for s in stale if s.get('never_visited')]
            parts = []
            if stale_list:
                parts.append("stale: " + ", ".join(stale_list))
            if never:
                parts.append("never: " + ", ".join(never))
            if parts:
                lines.append("VISIT AGE: " + "; ".join(parts))
        # Visit history (only rooms with visits)
        history = tool_results.get("get_room_visit_history", {}).get("rooms", [])
        if history:
            visited = [f"{h['room']}: {h['visits']}v/{h['changes']}c" for h in history if h.get('visits', 0) > 0]
            if visited:
                lines.append("VISITS: " + "; ".join(visited))
        # Predicted change probability (hazard model, §4.2)
        hazard = tool_results.get("get_predicted_change_probability", {}).get("rooms", [])
        if hazard:
            lines.append("PREDICTED P(change next): " + ", ".join(
                f"{h['room']}: {h.get('probability', 0.0)}" for h in hazard))
        return "\n".join(lines)

    def _call_vlm_with_tools(self, current_time: datetime, current_room: str | None,
                              client, model: str | None) -> dict[str, Any]:
        """Single-call tool flow: pre-execute all tools, embed results in prompt, then decide."""
        prompt = self._build_prompt(current_time, current_room)

        # Pre-execute all tools and format results
        tool_results = {
            "get_room_change_rates": self._execute_tool("get_room_change_rates", current_time),
            "get_time_since_last_change": self._execute_tool("get_time_since_last_change", current_time),
            "get_stale_rooms": self._execute_tool("get_stale_rooms", current_time),
            "get_room_visit_history": self._execute_tool("get_room_visit_history", current_time),
            "get_predicted_change_probability": self._execute_tool("get_predicted_change_probability", current_time),
        }

        is_v8_clean = "v8_clean" in self.strategy_type
        if is_v8_clean:
            tool_result_text = self._format_tool_results_compact(tool_results)
        else:
            tool_result_text = "\n".join(
                f"--- {name} ---\n{json.dumps(result, indent=2, default=str)}"
                for name, result in tool_results.items()
            )

        full_prompt = f"""{prompt}

TOOL RESULTS (your observation history):
{tool_result_text}
"""
        self.last_prompt = full_prompt
        self.last_tool_calls = [
            {"tool": name, "arguments": "{}", "result": result}
            for name, result in tool_results.items()
        ]

        self.stat_vlm_calls += 1
        response = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": self.prompt or "You are a robot navigation strategist. Output only JSON."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
            seed=self.seed,
        )
        raw_content = response.choices[0].message.content or "{}"
        self.last_raw_response = raw_content
        self.stat_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.stat_completion_tokens += response.usage.completion_tokens if response.usage else 0

        return self._extract_json(raw_content)

    def _call_vlm_plain(self, current_time: datetime, current_room: str | None,
                         client, model: str | None) -> dict[str, Any]:
        """Plain prompt-based VLM call. No retries — errors propagate to the UI."""
        self.stat_vlm_calls += 1
        prompt = self._build_prompt(current_time, current_room)
        self.last_prompt = prompt

        response = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": self.prompt or "You are a robot navigation strategist. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
            seed=self.seed,
        )
        raw_content = response.choices[0].message.content or "{}"
        self.last_raw_response = raw_content
        self.stat_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.stat_completion_tokens += response.usage.completion_tokens if response.usage else 0

        return self._extract_json(raw_content)

    def _call_vlm_reactive_tools(self, current_time: datetime, current_room: str | None,
                                  client, model: str | None) -> dict[str, Any]:
        """Two-turn tool flow: VLM requests tools, we execute them, VLM decides."""
        prompt = self._build_prompt(current_time, current_room)
        self.last_prompt = prompt

        self.stat_vlm_calls += 1
        response = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": self.prompt or "You are a robot navigation strategist. Output only JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
            seed=self.seed,
        )
        raw_content = response.choices[0].message.content or "{}"
        self.last_raw_response = raw_content
        self.stat_prompt_tokens += response.usage.prompt_tokens if response.usage else 0
        self.stat_completion_tokens += response.usage.completion_tokens if response.usage else 0

        parsed = self._extract_json(raw_content)
        tool_calls = parsed.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        if tool_calls:
            executed = []
            for tc in tool_calls:
                tool_name = tc.get("tool", "") if isinstance(tc, dict) else str(tc)
                result = self._execute_tool(tool_name, current_time)
                executed.append({"tool": tool_name, "arguments": tc.get("args", {}) if isinstance(tc, dict) else {}, "result": result})
            self.last_tool_calls = executed

            tool_result_text = "\n".join(
                f"--- {t['tool']} ---\n{json.dumps(t['result'], indent=2, default=str)}"
                for t in executed
            )

            second_prompt = f"""{prompt}

TOOL RESULTS (you requested these):
{tool_result_text}

Now make your final decision. At the end, output this JSON:
{{"reasoning": "...", "scene_changed": true/false, "change": "true"/"false", "activities_changed": true/false, "action": "stay" or "move", "target_room": "Room Name"}}
"""
            self.stat_vlm_calls += 1
            response2 = client.chat.completions.create(
                model=model or "Qwen/Qwen3-VL-4B-Instruct",
                messages=[
                    {"role": "system", "content": self.prompt or "You are a robot navigation strategist. Output only JSON."},
                    {"role": "user", "content": second_prompt},
                ],
                temperature=0,
                max_tokens=8192,
                timeout=60,
                seed=self.seed,
            )
            raw_content2 = response2.choices[0].message.content or "{}"
            self.last_raw_response = raw_content2
            self.stat_prompt_tokens += response2.usage.prompt_tokens if response2.usage else 0
            self.stat_completion_tokens += response2.usage.completion_tokens if response2.usage else 0
            return self._extract_json(raw_content2)

        self.last_tool_calls = []
        return parsed

    def _call_vlm(self, current_time: datetime, current_room: str | None) -> dict[str, Any]:
        """Call the real VLM backend.

        Dispatch keys on detector_type (the detector actually in use), not the
        wrapper strategy name, so top-level baselines get the same detector
        input as their ablation counterparts. v3-v8 detectors use pre-executed
        tools; v10_split uses the split detector→navigator flow; reactive_tools
        uses 2-turn; others use plain prompting.
        """
        client, model = self._vlm_client_info
        detector = self.detector_type or self.strategy_type

        if self.no_tools or "no_tools" in self.strategy_type:
            return self._call_vlm_plain(current_time, current_room, client, model)
        elif "v10_split" in detector:
            return self._call_vlm_split(current_time, current_room)
        elif any(x in detector for x in ["v3", "v4", "v5", "v6", "v7", "v8"]):
            return self._call_vlm_with_tools(current_time, current_room, client, model)
        elif "reactive_tools" in detector:
            return self._call_vlm_reactive_tools(current_time, current_room, client, model)
        else:
            return self._call_vlm_plain(current_time, current_room, client, model)

    def _call_vlm_split(self, current_time: datetime, current_room: str | None) -> dict[str, Any]:
        """v10_split two-turn flow (Tier 2): detector → code state machine →
        navigator only on certain no-change.

        TURN 1 (detector): same observation context the v8 detector gets
        (_build_prompt v8 path with a detection-only tail), V10_DETECTOR
        system prompt, V10 schema with confidence. TURN 2 (navigator) runs
        ONLY when the detector is certain there is no change:
        - scene_changed true            → stay, NO navigator call (token
          saving, and the navigator cannot conflict with the detector);
        - uncertain (0.3 <= confidence <= 0.5) → stay, no navigator call;
        - certain no-change             → navigator with the code-computed
          ROOM RANKING + walk times; action/target come from its JSON.
        vlm_detected_change comes ONLY from the detector — the navigator has
        no say in detection. vlm_calls/token counters count both calls.
        """
        client, model = self._vlm_client_info

        # ------------------------------------------------------------------
        # TURN 1 — Detector
        # ------------------------------------------------------------------
        base_prompt = self._build_prompt(current_time, current_room)
        detector_system = self.prompt or "You are a scene-change detector. Output only JSON."
        self.last_prompt = base_prompt

        self.stat_vlm_calls += 1
        response1 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": detector_system},
                {"role": "user", "content": base_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
            seed=self.seed,
        )
        raw_detector = response1.choices[0].message.content or "{}"
        self._last_detector_raw = raw_detector
        self.stat_prompt_tokens += response1.usage.prompt_tokens if response1.usage else 0
        self.stat_completion_tokens += response1.usage.completion_tokens if response1.usage else 0

        detector_parsed = self._extract_json(raw_detector)

        # Normalise detector fields (V10 schema)
        scene_changed = detector_parsed.get("scene_changed") in (True, "true", "True", 1, "1")
        change_val = str(detector_parsed.get("change") or "false").lower()
        if change_val not in ("true", "false"):
            change_val = "false"
        activities_changed = detector_parsed.get("activities_changed") in (True, "true", "True", 1, "1")
        try:
            confidence = float(detector_parsed.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5  # unparseable confidence → treat as uncertain
        detector_reasoning = str(detector_parsed.get("reasoning", ""))

        # ------------------------------------------------------------------
        # State machine in code: detection or uncertainty → stay, no
        # navigator call this step.
        # ------------------------------------------------------------------
        if scene_changed or 0.3 <= confidence <= 0.5:
            why = ("detection — stay and watch" if scene_changed
                   else f"uncertain (confidence {confidence:.2f}) — gather more evidence")
            self.last_tool_calls = []
            self.last_raw_response = raw_detector
            return {
                "reasoning": f"DETECTOR: {detector_reasoning}\nNAVIGATOR: skipped ({why})",
                "scene_changed": scene_changed,
                "change": change_val,
                "activities_changed": activities_changed,
                "action": "stay",
                "target_room": current_room or "",
            }

        # ------------------------------------------------------------------
        # TURN 2 — Navigator (certain no-change): code-computed ROOM RANKING
        # + walk times; no raw tool JSON dumps.
        # ------------------------------------------------------------------
        scored = self._compute_room_scores(current_time)
        # Same rule as the NavSplit variants (general fix): the current room
        # is not a candidate destination — never passed to the navigator.
        # Full scores are still recorded in last_tool_calls for analysis.
        ranking_entries = [e for e in scored if e["room"] != current_room]
        ranking_block = self._format_room_ranking(ranking_entries)
        # nav_scene_context ablation: per-room last-observed scenes from the
        # agent's OWN perceived memory (room_last_observed) — never GT of
        # unvisited rooms. Scenes truncated to ~120 chars (~≤500 tokens total).
        scene_block = ""
        if self.nav_scene_context:
            scene_lines = []
            for room in self.log.rooms:
                last_obs = self.room_last_observed.get(room)
                if last_obs is None:
                    scene_lines.append(f"  {room} (never observed)")
                else:
                    last_visit = self.room_last_visit_time.get(room)
                    ago = int((current_time - last_visit).total_seconds() / 60) if last_visit else 0
                    text = (last_obs.get("scene_text") or "No description.")[:120]
                    scene_lines.append(f'  {room} ({ago} min ago): "{text}"')
            scene_block = "\n\nLAST OBSERVED SCENES (your own memory):\n" + "\n".join(scene_lines)
        # nav_walk_times ablation (ping-pong suspect): omit the WALK TIMES
        # table entirely — the NOTIME system prompt drops the tie-break rule.
        walk_block = ""
        if self.nav_walk_times:
            walk_lines = [
                f"  {room}: {self.log.distance(current_room, room)} min"
                for room in self.log.rooms if room != current_room
            ]
            walk_block = f"\n\nWALK TIMES from {current_room} (minutes):\n" + "\n".join(walk_lines)
        nav_prompt = f"""CURRENT ROOM: {current_room}
CURRENT TIME: {current_time.strftime('%H:%M')}
DETECTOR: scene stable (scene_changed=false, confidence {confidence:.2f}, activities_changed={str(activities_changed).lower()})

{ranking_block}{scene_block}{walk_block}
"""
        navigator_system = self.navigator_prompt or "You are a robot navigation strategist. Output only JSON."
        self.last_prompt = nav_prompt
        self.last_tool_calls = [{
            "tool": "computed_room_scores",
            "arguments": json.dumps(self.score_weights),
            "result": scored,
        }]

        self.stat_vlm_calls += 1
        response2 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": navigator_system},
                {"role": "user", "content": nav_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
            seed=self.seed,
        )
        raw_navigator = response2.choices[0].message.content or "{}"
        self._last_navigator_raw = raw_navigator
        self.stat_prompt_tokens += response2.usage.prompt_tokens if response2.usage else 0
        self.stat_completion_tokens += response2.usage.completion_tokens if response2.usage else 0

        navigator_parsed = self._extract_json(raw_navigator)

        # Normalise navigator fields
        action = str(navigator_parsed.get("action", "stay")).lower().strip()
        if action not in ("stay", "move"):
            action = "stay"
        target_room = str(navigator_parsed.get("target_room", current_room or "")).strip()
        nav_reasoning = str(navigator_parsed.get("reason") or navigator_parsed.get("reasoning") or "")

        # Combine into the shape expected by _make_step
        merged = {
            "reasoning": f"DETECTOR: {detector_reasoning}\nNAVIGATOR: {nav_reasoning}",
            "scene_changed": scene_changed,
            "change": change_val,
            "activities_changed": activities_changed,
            "action": action,
            "target_room": target_room,
        }
        self.last_raw_response = (
            f"=== DETECTOR RAW ===\n{raw_detector}\n"
            f"=== NAVIGATOR RAW ===\n{raw_navigator}"
        )
        return merged

    @staticmethod
    def _fix_truncated_json(text: str) -> str:
        """Attempt to repair common truncation patterns in VLM JSON output."""
        # Close unterminated string values
        lines = text.split('\n')
        fixed_lines = []
        for line in lines:
            stripped = line.rstrip()
            # If line ends mid-string (odd number of unescaped quotes), close it
            quote_count = stripped.count('"') - stripped.count('\\"')
            if quote_count % 2 == 1:
                # Find last colon to determine if we're inside a string value
                if ':' in stripped:
                    after_colon = stripped.split(':', 1)[1].strip()
                    if after_colon.startswith('"') and not after_colon.endswith('"'):
                        stripped = stripped + '"'
            fixed_lines.append(stripped)
        text = '\n'.join(fixed_lines)

        # Ensure JSON object is closed
        open_braces = text.count('{') - text.count('}')
        if open_braces > 0:
            text = text + ('}' * open_braces)

        # Remove trailing commas before closing braces
        text = re.sub(r',(\s*[}\]])', r'\1', text)

        return text

    def step(self, current_time: datetime) -> SceneChangeMinuteStep:
        if self.pending_move_room is not None:
            self._apply_move(current_time, self.pending_move_room, overridden=self.pending_move_overridden)
            self.pending_move_room = None
            self.pending_move_overridden = False

        if self.current_room is None:
            # Initial-room pick (§3.3): "vlm" = legacy full VLM call (whose
            # detection fields are discarded); "random"/"fixed" skip it.
            if self.initial_pick == "fixed":
                target = self.log.rooms[0]
            elif self.initial_pick == "random":
                target = self.rng.choice(self.log.rooms)
            else:
                try:
                    try:
                        initial_decision = self._call_vlm(current_time, None)
                    except Exception:
                        # Same one-retry-on-failure policy as the main step
                        # path (§2.3); on double failure fall back to a random
                        # room instead of aborting the run.
                        initial_decision = self._call_vlm(current_time, None)
                except Exception:
                    initial_decision = {}
                target = str(initial_decision.get("target_room") or "").strip()
                if not target or target.lower() not in {r.lower() for r in self.log.rooms}:
                    target = self.rng.choice(self.log.rooms)
                for r in self.log.rooms:
                    if r.lower() == target.lower():
                        target = r
                        break
            self._apply_move(current_time, target)

        self.rooms_visited.add(self.current_room)

        entry = self.log.scene_at(current_time, self.current_room)
        gt_change = entry.change_type if entry else "no_change"
        gt_activity_changed = entry.activities_changed if entry else False
        interactions = entry.interactions if entry else []

        try:
            try:
                parsed = self._call_vlm(current_time, self.current_room)
            except Exception:
                # Retry once — a transient parse/timeout failure should not
                # silently become a stay-step counted as a non-detection.
                parsed = self._call_vlm(current_time, self.current_room)
            step = self._make_step(current_time, parsed, gt_change, gt_activity_changed, entry, interactions, tool_calls=self.last_tool_calls)
        except Exception as exc:
            # Record the error in the step but keep the simulation running.
            # Error steps carry NO detection signal (vlm_detected_change=None)
            # and are excluded from detection/confusion metrics.
            error_msg = f"{type(exc).__name__}: {exc}"
            self.error_steps += 1
            step = SceneChangeMinuteStep(
                timestamp=current_time,
                time_str=current_time.strftime("%H:%M"),
                room=self.current_room,
                scene_text=entry.scene if entry else "No data.",
                gt_change_type=gt_change,
                gt_activity_changed=gt_activity_changed,
                vlm_detected_change=None,
                vlm_change="no_change",
                vlm_detected_activity_change=None,
                action="stay",
                target_room=self.current_room,
                decision={"reasoning": error_msg, "scene_changed": False, "change": "false", "action": "stay", "target_room": self.current_room or ""},
                fresh_decision=True,
                people_present=entry.people_present if entry else [],
                objects_present=entry.objects_present if entry else [],
                interactions=interactions,
                prompt=self.last_prompt,
                raw_response=self.last_raw_response or error_msg,
                system_prompt=self.prompt,
                tool_calls=self.last_tool_calls,
                error=error_msg,
            )
            # The robot still physically observed the room: refresh memory so
            # the next diff is not computed against an older scene, and update
            # last-visit time (but NOT room_visit_counts — that is per-visit,
            # handled in _apply_move).
            if entry:
                activity_list = [f"{i['subject']} {i['action']} {i['target']}" for i in interactions]
                self._record_room_observation(self.current_room, current_time, entry, activity_list, None)
            self.room_last_visit_time[self.current_room] = current_time
        self.minute_trace.append(step)
        self.steps_in_room += 1
        self.last_tool_calls = None
        return step


class _SteppableSceneChangeVLMSplit(_SteppableSceneChangeVLM):
    """Two-step VLM agent: detector decides IF scene changed, navigator decides WHERE to go."""

    def __init__(self, log: SceneChangeTimeLog, strategy_type: str = "agent_scene_change_split",
                 detector_prompt: str | None = None, navigator_prompt: str | None = None,
                 seed: int | None = None, no_tools: bool = False, dwell_config: dict | None = None):
        # Initialise as a normal VLM agent; we will override _call_vlm
        super().__init__(log, strategy_type=strategy_type, system_prompt=detector_prompt, seed=seed, no_tools=no_tools, dwell_config=dwell_config)
        self.detector_prompt = detector_prompt
        self.navigator_prompt = navigator_prompt
        self._last_detector_raw = ""
        self._last_navigator_raw = ""

    def _call_vlm(self, current_time: datetime, current_room: str | None) -> dict[str, Any]:
        """Two-turn flow: detector → navigator."""
        client, model = self._vlm_client_info

        # ------------------------------------------------------------------
        # TURN 1 — Detector (plain prompt, no tools)
        # ------------------------------------------------------------------
        base_prompt = self._build_prompt(current_time, current_room)
        detector_system = self.detector_prompt or "You are a scene-change detector. Output only JSON."

        self.stat_vlm_calls += 1
        response1 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": detector_system},
                {"role": "user", "content": base_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
        )
        raw_detector = response1.choices[0].message.content or "{}"
        self._last_detector_raw = raw_detector
        self.stat_prompt_tokens += response1.usage.prompt_tokens if response1.usage else 0
        self.stat_completion_tokens += response1.usage.completion_tokens if response1.usage else 0

        detector_parsed = self._extract_json(raw_detector)

        # Normalise detector fields
        scene_changed = bool(detector_parsed.get("scene_changed", False))
        change_val = detector_parsed.get("change") or detector_parsed.get("change", "false")
        if change_val not in ("true", "false"):
            change_val = "false"
        activities_changed = bool(detector_parsed.get("activities_changed", False))
        detector_reasoning = str(detector_parsed.get("reasoning", ""))

        # ------------------------------------------------------------------
        # TURN 2 — Navigator (prompt + detector result + tool results)
        # ------------------------------------------------------------------
        detector_block = json.dumps({
            "scene_changed": scene_changed,
            "change": change_val,
            "activities_changed": activities_changed,
            "reasoning": detector_reasoning,
        }, indent=2)

        # Pre-execute tools for the navigator (same as v4/v5/v6)
        tool_results = {
            "get_room_change_rates": self._execute_tool("get_room_change_rates", current_time),
            "get_time_since_last_change": self._execute_tool("get_time_since_last_change", current_time),
            "get_stale_rooms": self._execute_tool("get_stale_rooms", current_time),
            "get_room_visit_history": self._execute_tool("get_room_visit_history", current_time),
            "get_predicted_change_probability": self._execute_tool("get_predicted_change_probability", current_time),
        }
        tool_result_text = "\n".join(
            f"--- {name} ---\n{json.dumps(result, indent=2, default=str)}"
            for name, result in tool_results.items()
        )

        nav_prompt = f"""{base_prompt}

DETECTOR ASSESSMENT:
{detector_block}

TOOL RESULTS (your observation history):
{tool_result_text}
"""
        navigator_system = self.navigator_prompt or "You are a robot navigation strategist. Output only JSON."
        self.last_prompt = nav_prompt
        self.last_tool_calls = [
            {"tool": name, "arguments": "{}", "result": result}
            for name, result in tool_results.items()
        ]

        self.stat_vlm_calls += 1
        response2 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": navigator_system},
                {"role": "user", "content": nav_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
        )
        raw_navigator = response2.choices[0].message.content or "{}"
        self._last_navigator_raw = raw_navigator
        self.stat_prompt_tokens += response2.usage.prompt_tokens if response2.usage else 0
        self.stat_completion_tokens += response2.usage.completion_tokens if response2.usage else 0

        navigator_parsed = self._extract_json(raw_navigator)

        # Normalise navigator fields
        action = str(navigator_parsed.get("action", "stay")).lower().strip()
        if action not in ("stay", "move"):
            action = "stay"
        target_room = str(navigator_parsed.get("target_room", current_room or "")).strip()
        nav_reasoning = str(navigator_parsed.get("reasoning", ""))

        # Combine into the shape expected by _make_step
        merged = {
            "reasoning": f"DETECTOR: {detector_reasoning}\nNAVIGATOR: {nav_reasoning}",
            "scene_changed": scene_changed,
            "change": change_val,
            "activities_changed": activities_changed,
            "action": action,
            "target_room": target_room,
        }
        self.last_raw_response = (
            f"=== DETECTOR RAW ===\n{raw_detector}\n"
            f"=== NAVIGATOR RAW ===\n{raw_navigator}"
        )
        return merged


class _SteppableSceneChangeVLMNavSplit(_SteppableSceneChangeVLM):
    """Tier-2 v9_split (REBUTTAL_NAVIGATION_IMPROVEMENT_PLAN.md §Tier 2).

    Two-turn flow like _SteppableSceneChangeVLMSplit, but:
    - the detector sees a detection-only prompt (no navigation question);
    - the navigator is SKIPPED on detection steps (detector verdict =
      stay-and-watch, one VLM call that step);
    - the navigator receives a Python-computed ranked score table
      (w_hot·laplace_rate + w_exp·hazard + w_stale·staleness) instead of
      raw tool JSON dumps.
    """

    def __init__(self, log: SceneChangeTimeLog, strategy_type: str = "agent_scene_change_v9_split",
                 detector_prompt: str | None = None, navigator_prompt: str | None = None,
                 seed: int | None = None, no_tools: bool = False, dwell_config: dict | None = None,
                 detector_type: str | None = None, anti_camping: bool = False, move_budget: int = 0,
                 nav_score_weights: tuple[float, ...] = (1.0, 1.0, 0.5, 0.0),
                 skip_on_detection: bool = True,
                 skip_mode: str | None = None,
                 exclude_current_room_from_ranking: bool = True,
                 obs_history_size: int = 0,
                 obs_history_include_caption: bool = False,
                 fta_unblock: bool = False,
                 detector_max_tokens: int = 8192,
                 fta_mode: str | None = None,
                 cross_room_obs: bool = False,
                 argmax_nav: bool = False,
                 skip_nav_policy: Any = None):
        super().__init__(log, strategy_type=strategy_type, system_prompt=detector_prompt,
                         seed=seed, no_tools=no_tools, dwell_config=dwell_config,
                         detector_type=detector_type, anti_camping=anti_camping,
                         move_budget=move_budget)
        self.detector_prompt = detector_prompt
        self.navigator_prompt = navigator_prompt
        # (w_hot, w_exp, w_stale[, w_int]). A 3-tuple is accepted and padded
        # with w_int=0.0 so every pre-v20r3 call site stays valid.
        self.nav_score_weights = tuple(nav_score_weights)
        # Navigator skip mode:
        # - "detection" (strict v9_split): skip the navigator on scene-change
        #   steps (stay-and-watch).
        # - "never" (v9_split_soft): always call the navigator — it may move
        #   despite a change or follow the action.
        # - "interesting" (v11_split, synthesis): skip whenever the soft
        #   detector judges the scene INTERESTING (covers detections AND
        #   interesting activity); the navigator IS called on changed-but-
        #   boring scenes so the robot can move on or follow people. Falls
        #   back to scene_changed when the detector emits no `interesting`.
        if skip_mode is None:
            skip_mode = "detection" if skip_on_detection else "never"
        if skip_mode not in ("detection", "never", "interesting"):
            raise ValueError(f"Invalid skip_mode: {skip_mode!r}")
        self.skip_mode = skip_mode
        self.skip_on_detection = skip_mode == "detection"
        # Default (all split variants): the current room is NOT passed in the
        # ROOM RANKING table — a top-ranked current room invites self-target
        # "moves" that never execute (the stuck-agent failure mode). The full
        # scored list is still recorded in last_tool_calls for analysis. Pass
        # False only to reproduce the pre-fix behavior.
        self.exclude_current_room_from_ranking = exclude_current_room_from_ranking
        # v13_split: per-room observation window injected into the detector
        # prompt (0 = off, legacy behavior). obs_history_include_caption
        # appends each history entry's scene caption to the window lines.
        self.obs_history_size = int(obs_history_size)
        self.obs_history_include_caption = bool(obs_history_include_caption)
        # v14_split: FTA (follow-the-action) unblock. In v13 the interesting
        # checklist made navigator rule 3 unreachable: leaving=true always
        # implied interesting=true, so the navigator was skipped on every
        # departure (0/20 opportunities in the prompt capture). When enabled,
        # the navigator is NOT skipped when leaving=true with a valid
        # departure_destination, and destinations are validated against the
        # room list ("door"/"outside"/"unknown" -> "").
        self.fta_unblock = bool(fta_unblock)
        # v15_split: fta_mode="v15" tightens the v14 unblock:
        # - departure_destination == current room is normalized to "" (6/21
        #   v14 FTA opportunities were this contradiction);
        # - the navigator is only unblocked when people_trend == "leaving"
        #   (winding-down room) — v14 followed departures out of still-active
        #   rooms (moves 105 -> 149, recall and precision both dropped).
        # None = legacy v14 behavior.
        if fta_mode not in (None, "v15", "v22"):
            raise ValueError(f"Invalid fta_mode: {fta_mode!r}. Expected None, 'v15' or 'v22'.")
        self.fta_mode = fta_mode
        # v21_split: also show the detector its last observation of every
        # OTHER room (cross-room context — did people who left here arrive
        # elsewhere, what is normal turnover building-wide).
        self.cross_room_obs = bool(cross_room_obs)
        # *_argmax_split ablations: keep the detector + skip logic (dwell),
        # but replace the VLM navigator with plain argmax of the ROOM RANKING
        # table — separates "detector+skip" value from "navigator" value.
        self.argmax_nav = bool(argmax_nav)
        # *_skip_<policy>_split ablations: like argmax_nav (detector + skip
        # logic own stay/move, no navigator VLM call), but the destination is
        # chosen by a heuristic nav policy (same _nav_policy_* functions the
        # *_pure baselines use) instead of argmax of the ranking table.
        # Answers: is semantic dwell valuable on top of ANY where-policy?
        self.skip_nav_policy = skip_nav_policy
        # v14_split: detector turn token budget (v13's 512 truncated the JSON
        # contract mid-response on verbose reasoning).
        self.detector_max_tokens = int(detector_max_tokens)
        self._last_detector_raw = ""
        self._last_navigator_raw = ""
        # Rank-divergence tracking: on every navigator call, record what the
        # pure argmax of the ROOM RANKING table would have done vs. what the
        # VLM navigator actually decided — quantifies how often the LLM
        # overrides (or just follows) its own ranking suggestion.
        self.rank_track: list[dict] = []
        self.stat_rank_divergences = 0
        self.stat_rank_override_moves = 0

    def _nav_weights4(self) -> tuple[float, float, float, float]:
        """nav_score_weights padded to (w_hot, w_exp, w_stale, w_int).

        Pre-v20r3 call sites pass a 3-tuple; w_int then defaults to 0.0 and
        the score is bit-identical to the three-term version.
        """
        w = tuple(float(x) for x in self.nav_score_weights)
        return (w + (0.0, 0.0, 0.0, 0.0))[:4]

    def _build_current_scene_block(self, current_time: datetime, current_room: str | None) -> str:
        """Live observation of the current room for the navigator, using the
        same data source and formatting as _build_prompt's current_scene /
        current_activities blocks (v8 full format)."""
        if current_room is None:
            return "You are just starting. No current room."
        entry = self.log.scene_at(current_time, current_room)
        if not entry:
            return f"Current room: {current_room}\nNo scene data available."
        current_scene, current_activities = self._format_current_scene_full(current_room, entry)
        return current_scene + current_activities

    def _compute_room_scores(self, current_time: datetime) -> list[dict]:
        """Ranked room table:
        score = w_hot·rate + w_exp·hazard + w_stale·staleness + w_int·interaction.

        w_int is 0.0 for every strategy before v20r3, which leaves the
        three-term score bit-identical. The interaction term is the perceived
        person-object interaction rate per observation (see
        _perceived_interaction_rates) — the ownership-relevant signal the
        first three terms do not carry.

        Deterministic per seed: ties broken with the seeded rng (shuffle
        before sort). Masked tools (error dicts) contribute 0.0.
        """
        w_hot, w_exp, w_stale, w_int = self._nav_weights4()
        interactions = self._perceived_interaction_rates() if w_int > 0 else {}

        rates_raw = self._execute_tool("get_room_change_rates", current_time)
        rates: dict[str, float] = {}
        if isinstance(rates_raw, list):
            # rate=None (never visited) counts as the Laplace prior 0.5
            rates = {
                e["room"]: (0.5 if e.get("rate") is None else float(e["rate"]))
                for e in rates_raw
            }

        hazard_raw = self._execute_tool("get_predicted_change_probability", current_time)
        hazards: dict[str, float] = {}
        if isinstance(hazard_raw, list):
            hazards = {
                e["room"]: float(e.get("probability") or 0.0)
                for e in hazard_raw
            }

        log_start = self.log.unique_timestamps[0] if self.log.unique_timestamps else current_time
        stale_minutes: dict[str, float] = {}
        for room in self.log.rooms:
            last = self.room_last_visit_time.get(room)
            since = last if last is not None else log_start
            stale_minutes[room] = max(0.0, (current_time - since).total_seconds() / 60.0)
        max_minutes = max(stale_minutes.values()) if stale_minutes else 0.0

        scored = []
        for room in self.log.rooms:
            rate = rates.get(room, 0.0)
            hazard = hazards.get(room, 0.0)
            minutes = stale_minutes[room]
            stale_norm = minutes / max(max_minutes, 1)
            interaction = interactions.get(room, 0.5) if w_int > 0 else 0.0
            t_hot = w_hot * rate
            t_exp = w_exp * hazard
            t_stale = w_stale * stale_norm
            t_int = w_int * interaction
            score = t_hot + t_exp + t_stale + t_int
            if room not in self.room_last_visit_time:
                reason = "never visited"
            else:
                # Dominant reason among ACTIVE terms only: a zero-weighted
                # (ablated) term must never be named in the "why" note —
                # otherwise the no-mention ablation prompts still leak it.
                candidates = []
                if w_stale > 0:
                    candidates.append((t_stale, f"overdue {minutes:.0f} min"))
                if w_hot > 0:
                    candidates.append((t_hot, f"hot: {rate:.2f}/visit"))
                if w_exp > 0:
                    candidates.append((t_exp, f"predicted {hazard:.2f}"))
                if w_int > 0:
                    candidates.append((t_int, f"interactions {interaction:.2f}/obs"))
                reason = max(candidates)[1] if candidates else "tie"
            scored.append({
                "room": room,
                "score": round(score, 3),
                "reason": reason,
                "rate": round(rate, 3),
                "hazard": round(hazard, 3),
                "stale_norm": round(stale_norm, 3),
                "interaction": round(interaction, 3),
                # Seeded tie-break: derived from (seed, room) instead of
                # consuming self.rng, so repeated calls at the same sim state
                # give an identical ordering.
                "_tie": random.Random(f"{self.seed}:{room}").random(),
            })
        scored.sort(key=lambda e: (-e["score"], e["_tie"]))
        for e in scored:
            del e["_tie"]
        return scored

    def _call_vlm(self, current_time: datetime, current_room: str | None) -> dict[str, Any]:
        """Two-turn flow: detector → navigator (navigator skipped on detection)."""
        client, model = self._vlm_client_info

        # ------------------------------------------------------------------
        # TURN 1 — Detector (identical call shape to _SteppableSceneChangeVLMSplit)
        # ------------------------------------------------------------------
        base_prompt = self._build_prompt(current_time, current_room)
        detector_system = self.detector_prompt or "You are a scene-change detector. Output only JSON."

        self.stat_vlm_calls += 1
        response1 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": detector_system},
                {"role": "user", "content": base_prompt},
            ],
            temperature=0,
            max_tokens=self.detector_max_tokens,
            timeout=60,
        )
        raw_detector = response1.choices[0].message.content or "{}"
        self._last_detector_raw = raw_detector
        self.stat_prompt_tokens += response1.usage.prompt_tokens if response1.usage else 0
        self.stat_completion_tokens += response1.usage.completion_tokens if response1.usage else 0

        detector_parsed = self._extract_json(raw_detector)

        # Normalise detector fields
        scene_changed = bool(detector_parsed.get("scene_changed", False))
        change_val = detector_parsed.get("change") or detector_parsed.get("change", "false")
        if change_val not in ("true", "false"):
            change_val = "false"
        activities_changed = bool(detector_parsed.get("activities_changed", False))
        detector_reasoning = str(detector_parsed.get("reasoning", ""))
        # Soft-variant fields (strict detector never emits them → None/"").
        interesting_raw = detector_parsed.get("interesting")
        interesting = None if interesting_raw is None else bool(interesting_raw)
        observation = str(detector_parsed.get("observation") or "")
        # v12 detector field (v10-style confidence contract; absent for
        # strict/soft detectors → None and omitted downstream).
        confidence_raw = detector_parsed.get("confidence")
        confidence: float | None = None
        if confidence_raw is not None:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = None
        # v13 detector fields (structured movement; absent for strict/soft/
        # v12 detectors → None/"" and omitted downstream).
        leaving_raw = detector_parsed.get("leaving")
        leaving = None if leaving_raw is None else bool(leaving_raw)
        departure_destination = str(detector_parsed.get("departure_destination") or "").strip()
        if self.fta_unblock and departure_destination:
            # v14: destinations must be real room names — the model also emits
            # "door"/"outside"/"unknown" (4/20 in the v13 capture). Validate
            # case-insensitively against the room list, else drop to "".
            canon = {r.lower(): r for r in self.log.rooms}
            departure_destination = canon.get(departure_destination.lower(), "")
        if self.fta_mode == "v15" and departure_destination and current_room \
                and departure_destination == current_room:
            # v15: someone leaving this room cannot be heading to this room
            # (6/21 v14 FTA opportunities were exactly this contradiction).
            departure_destination = ""
        people_trend_raw = detector_parsed.get("people_trend")
        people_trend = str(people_trend_raw).strip().lower() if people_trend_raw is not None else ""
        if people_trend and people_trend not in ("arriving", "leaving", "stable", "mixed"):
            people_trend = ""

        # Marker appended to the merged reasoning when the soft detector
        # reported interestingness/dynamics, e.g. [interesting=False; "..."].
        det_marker = ""
        if interesting is not None or observation or confidence is not None or leaving is not None:
            parts = []
            if interesting is not None:
                parts.append(f"interesting={interesting}")
            if confidence is not None:
                parts.append(f"conf={confidence:.2f}")
            if leaving is not None:
                parts.append(f"leaving={leaving}")
            if departure_destination:
                parts.append(f"dest={departure_destination}")
            if people_trend:
                parts.append(f"trend={people_trend}")
            if observation:
                parts.append(f'"{observation}"')
            det_marker = " [" + "; ".join(parts) + "]"

        # ------------------------------------------------------------------
        # SKIP RULE — stay and watch, no navigator call this step.
        # detection: skip on scene change. never: never skip. interesting:
        # skip when the soft detector says the scene is worth watching
        # (detections AND interesting activity), falling back to scene_changed
        # when no `interesting` field was emitted.
        # ------------------------------------------------------------------
        if self.skip_mode == "interesting":
            skip_nav = bool(interesting) if interesting is not None else scene_changed
            skip_why = ("detection — stay and watch" if scene_changed
                        else "interesting activity — stay and watch")
            if self.fta_unblock and skip_nav and leaving and (departure_destination or self.fta_mode == "v22"):
                # v14 FTA unblock: a departure with a known destination must
                # reach the navigator (rule 3) even though the scene is
                # interesting — in v13 this combination was unreachable.
                # v15 narrows it: only when the room is winding down
                # (people_trend == "leaving"), not departures out of
                # still-active rooms.
                # v22: the detector emits NO destination — unblock on
                # leaving=true alone (same winding-down trend gate as v15)
                # and let the navigator infer the destination itself.
                if self.fta_mode not in ("v15", "v22") or people_trend == "leaving":
                    skip_nav, skip_why = False, ""
        elif self.skip_mode == "never":
            skip_nav, skip_why = False, ""
        else:
            skip_nav, skip_why = scene_changed, "detection — stay and watch"
        if skip_nav:
            self.last_prompt = base_prompt
            self.last_tool_calls = []
            self.last_raw_response = raw_detector
            return {
                "reasoning": f"DETECTOR: {detector_reasoning}{det_marker}\nNAVIGATOR: skipped ({skip_why})",
                "scene_changed": scene_changed,
                "change": change_val,
                "activities_changed": activities_changed,
                "action": "stay",
                "target_room": current_room or "",
            }

        # ------------------------------------------------------------------
        # TURN 2 — Navigator (computed score table, no raw tool dumps)
        # ------------------------------------------------------------------
        scored = self._compute_room_scores(current_time)
        w_hot, w_exp, w_stale, w_int = self._nav_weights4()
        table_entries = [
            e for e in scored
            if not (self.exclude_current_room_from_ranking and current_room and e["room"] == current_room)
        ]

        if self.argmax_nav or self.skip_nav_policy is not None:
            # *_argmax_split / *_skip_<policy>_split ablations: navigation is
            # code, not a navigator VLM call — the detector + skip logic
            # (dwell) above is untouched. argmax = top of the ROOM RANKING
            # table; skip_nav_policy = a heuristic _nav_policy_* function.
            # This isolates the navigator's marginal value / tests whether
            # semantic dwell helps any where-policy.
            greedy_target = table_entries[0]["room"] if table_entries else (current_room or "")
            if self.skip_nav_policy is not None:
                _pol_action, _pol_target = self.skip_nav_policy(
                    current_room, current_time, self.steps_in_room, self.log, self)
                nav_target = (_pol_target or "").strip() or (current_room or "")
                nav_reasoning = f"heuristic policy navigation ({_pol_action}: {nav_target})"
            else:
                nav_target = greedy_target
                nav_reasoning = f"argmax of ROOM RANKING ({greedy_target})"
            nav_action = "stay" if (current_room and nav_target == current_room) else "move"
            nav_rank = next((i for i, e in enumerate(table_entries, start=1)
                             if e["room"] == nav_target), None)
            self.rank_track.append({
                "time": current_time.strftime("%Y-%m-%d %H:%M"),
                "current_room": current_room,
                "greedy_target": greedy_target,
                "vlm_action": nav_action,
                "vlm_target": nav_target,
                "vlm_target_rank": nav_rank,
                "diverged": nav_target != greedy_target,
                "reason": nav_reasoning,
            })
            return {
                "reasoning": f"DETECTOR: {detector_reasoning}{det_marker}\nNAVIGATOR: {nav_reasoning}",
                "scene_changed": scene_changed,
                "change": change_val,
                "activities_changed": activities_changed,
                "action": nav_action,
                "target_room": nav_target,
                "greedy_target": greedy_target,
                "rank_diverged": nav_target != greedy_target,
            }
        table_lines = [
            f"{i}. {entry['room']:<12} score {entry['score']:.2f}  ({entry['reason']})"
            # Top-10: 6-room worlds show all-but-current (5 rows) exactly as
            # before; 12-room worlds no longer hide half the map.
            for i, entry in enumerate(table_entries[:10], start=1)
        ]
        if self.skip_on_detection:
            detector_block = f"DETECTOR: scene stable (scene_changed=false, activities_changed={str(activities_changed).lower()})"
        else:
            # Soft variant: always pass the full detector assessment — the
            # observation may note someone heading elsewhere (follow-the-action).
            assessment = {
                "scene_changed": scene_changed,
                "activities_changed": activities_changed,
                "interesting": interesting,
                "observation": observation,
            }
            if confidence is not None:
                # v12 navigator rule 4 keys off detector uncertainty.
                assessment["confidence"] = confidence
            if leaving is not None:
                # v13 navigator rule 3 (follow-the-action) keys off the
                # structured movement fields instead of parsing observation.
                assessment["leaving"] = leaving
                if self.fta_mode != "v22":
                    # v22: the detector emits no destination — the navigator
                    # infers it from the observation note; omit the key
                    # entirely so it cannot anchor on an empty field.
                    assessment["departure_destination"] = departure_destination
            if people_trend:
                assessment["people_trend"] = people_trend
            detector_block = "DETECTOR ASSESSMENT:\n" + json.dumps(assessment)
        scene_block = self._build_current_scene_block(current_time, current_room)
        nav_prompt = f"""CURRENT SCENE ({current_room}, {current_time.strftime('%H:%M')}):
{scene_block}

ROOM RANKING (highest priority first):
{chr(10).join(table_lines)}

CURRENT ROOM: {current_room}
CURRENT TIME: {current_time.strftime('%H:%M')}
{detector_block}"""

        navigator_system = self.navigator_prompt or "You are a robot navigation strategist. Output only JSON."
        self.last_prompt = nav_prompt
        self.last_tool_calls = [{
            "tool": "computed_room_scores",
            "arguments": json.dumps({"w_hot": w_hot, "w_exp": w_exp, "w_stale": w_stale}),
            "result": scored,
        }]

        self.stat_vlm_calls += 1
        response2 = client.chat.completions.create(
            model=model or "Qwen/Qwen3-VL-4B-Instruct",
            messages=[
                {"role": "system", "content": navigator_system},
                {"role": "user", "content": nav_prompt},
            ],
            temperature=0,
            max_tokens=8192,
            timeout=60,
        )
        raw_navigator = response2.choices[0].message.content or "{}"
        self._last_navigator_raw = raw_navigator
        self.stat_prompt_tokens += response2.usage.prompt_tokens if response2.usage else 0
        self.stat_completion_tokens += response2.usage.completion_tokens if response2.usage else 0

        navigator_parsed = self._extract_json(raw_navigator)

        # Normalise navigator fields
        action = str(navigator_parsed.get("action", "stay")).lower().strip()
        if action not in ("stay", "move"):
            action = "stay"
        target_room = str(navigator_parsed.get("target_room", current_room or "")).strip()
        nav_reasoning = str(navigator_parsed.get("reasoning", ""))

        # Rank-divergence tracking: greedy = plain argmax of the ranking
        # table the navigator just saw (current room already excluded).
        # vlm_target_rank = position of the chosen target in the table
        # (None if off-table/stay) — enables "how deep does the LLM reach?"
        # analysis beyond the binary diverged flag.
        greedy_target = table_entries[0]["room"] if table_entries else None
        target_rank = next((i for i, e in enumerate(table_entries, start=1)
                            if e["room"] == target_room), None)
        diverged = greedy_target is not None and (action == "stay" or target_room != greedy_target)
        self.rank_track.append({
            "time": current_time.strftime("%Y-%m-%d %H:%M"),
            "current_room": current_room,
            "greedy_target": greedy_target,
            "vlm_action": action,
            "vlm_target": target_room,
            "vlm_target_rank": target_rank,
            "diverged": diverged,
            "reason": nav_reasoning[:300],
        })
        if diverged:
            self.stat_rank_divergences += 1
            if action == "move":
                self.stat_rank_override_moves += 1

        # Combine into the shape expected by _make_step
        merged = {
            "reasoning": f"DETECTOR: {detector_reasoning}{det_marker}\nNAVIGATOR: {nav_reasoning}",
            "scene_changed": scene_changed,
            "change": change_val,
            "activities_changed": activities_changed,
            "action": action,
            "target_room": target_room,
            "greedy_target": greedy_target,
            "rank_diverged": diverged,
        }
        self.last_raw_response = (
            f"=== DETECTOR RAW ===\n{raw_detector}\n"
            f"=== NAVIGATOR RAW ===\n{raw_navigator}"
        )
        return merged


# ---------------------------------------------------------------------------
# Navigation policy factories for hybrid ablation strategies
# ---------------------------------------------------------------------------

def _nav_policy_round_robin(rooms: list[str] | None = None) -> Any:
    """Return a navigation policy that rotates through rooms (dwell=1 step)."""
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    state: dict[str, Any] = {"room_idx": None, "steps": 0, "dwell": 1}

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if state["room_idx"] is None and current_room is not None:
            try:
                state["room_idx"] = rooms.index(current_room)
            except ValueError:
                state["room_idx"] = 0
        if current_room is None:
            state["room_idx"] = 0
            return "move", rooms[0]
        state["steps"] = steps_in_room
        if state["steps"] >= state["dwell"]:
            state["room_idx"] = (state["room_idx"] + 1) % len(rooms)
            state["steps"] = 0
            return "move", rooms[state["room_idx"]]
        return "stay", current_room

    return policy


def _nav_policy_greedy(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Return a navigation policy that moves to the room with the most GT changes in the next step."""
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed) if seed is not None else random

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        next_changes = log.changes_in_window(current_time, window_steps=1)
        next_room = max(next_changes, key=next_changes.get)
        if next_changes[next_room] == 0:
            next_room = rng.choice(rooms)
        if next_room != current_room:
            return "move", next_room
        return "stay", current_room

    return policy


def _nav_policy_frequency(seed: int | None = None, source: str = "perceived", rooms: list[str] | None = None) -> Any:
    """Return a navigation policy that moves proportionally to change frequency.

    source="perceived" (default, post-§3.2): weights ∝ the VLM's own detected
    change counts + 1 (Laplace smoothing) — deployable, FP/FN-sensitive.
    source="gt": legacy behavior — weights ∝ GT changes-while-present + 1
    (selectable as the named variant `frequency_gt`).
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)
    counts_attr = "room_change_counts" if source == "gt" else "room_perceived_change_counts"

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        counts = getattr(detector, counts_attr)
        # Laplace +1: uniform weights at cold start (no all-zero weights).
        weights = [counts.get(r, 0) + 1 for r in rooms]
        next_room = rng.choices(rooms, weights=weights, k=1)[0]
        if next_room != current_room:
            return "move", next_room
        return "stay", current_room

    return policy


def _nav_policy_greedy_hazard(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Deployable hazard-greedy: argmax predicted change probability per room.

    Reads only the robot's own observed history through the hazard tool —
    never future GT. The factory wires hazard_fix=True for these strategies
    (§4.2-CORE), so the hazard denominators use elapsed steps only. Cold
    start (all hazards 0 — nothing observed anywhere yet): fall back to
    argmax staleness. Ties break randomly (seeded).
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        hazards = detector._tool_get_predicted_change_probability(current_time)
        best_prob = max((h["probability"] for h in hazards), default=0.0)
        if best_prob > 0.0:
            best_rooms = [h["room"] for h in hazards if h["probability"] >= best_prob - 1e-9]
            next_room = rng.choice(best_rooms)
        else:
            # Cold start: no changes observed anywhere yet -> argmax staleness
            # (never-visited rooms sort first with minutes_since_visit=999999).
            stale = detector._tool_get_stale_rooms(current_time)
            candidates = [s for s in stale if s["room"] != current_room]
            if candidates:
                top_minutes = candidates[0]["minutes_since_visit"]
                tied = [s["room"] for s in candidates if s["minutes_since_visit"] == top_minutes]
                next_room = rng.choice(tied)
            else:
                next_room = rng.choice([r for r in rooms if r != current_room])
        if next_room != current_room:
            return "move", next_room
        return "stay", current_room

    return policy


def _nav_policy_rank_greedy(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Pure argmax of the code-computed ROOM RANKING the VLM navigator sees.

    Uses the exact same scoring (_compute_room_scores: w_hot·laplace_rate +
    w_exp·hazard + w_stale·staleness_norm) and always moves to the top-ranked
    room != current. This is the ablation that answers: "does the VLM
    navigator add anything over just following its own ranking table?"
    Deterministic per seed (tie-breaking lives inside _compute_room_scores).
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        scored = detector._compute_room_scores(current_time)
        for entry in scored:
            if entry["room"] != current_room:
                return "move", entry["room"]
        return "stay", current_room

    return policy


def _nav_policy_entropy(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Information-gain baseline: move to the room maximizing entropy x staleness.

    Each room is scored by the Shannon entropy of its PERCEIVED change-type
    distribution (the VLM's own no_change / minor_change / major_change
    verdicts while present — never ground truth) — how unpredictable its
    dynamics are — weighted by staleness. Both factors are bounded: entropy
    is normalized by ln(3) and staleness by the currently stalest room, so
    neither term can dominate the other. Rooms with no observations get a
    uniform prior (maximum entropy and maximum staleness), so unexplored and
    long-unvisited rooms are preferred. Ties break randomly.

    (Historical note: this policy previously read the GT change-type counts —
    oracle information — and multiplied by unbounded staleness minutes, which
    degenerated into staleness sweeping with an oracle tie-breaker.)
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        log_start = log.unique_timestamps[0] if log.unique_timestamps else current_time
        staleness: dict[str, float] = {}
        for room in rooms:
            last_visit = detector.room_last_visit_time.get(room)
            reference = last_visit if last_visit is not None else log_start
            staleness[room] = (current_time - reference).total_seconds() / 60.0
        max_staleness = max(staleness.values()) if staleness else 0.0
        scores: dict[str, float] = {}
        for room in rooms:
            type_counts = detector.room_perceived_change_type_counts.get(room)
            if type_counts:
                total = sum(type_counts.values())
                entropy = -sum(
                    (c / total) * math.log(c / total) for c in type_counts.values()
                )
            else:
                # No observations yet: uniform prior -> maximum entropy.
                entropy = math.log(3)
            entropy_norm = entropy / math.log(3)
            staleness_norm = staleness[room] / max_staleness if max_staleness > 0 else 0.0
            # +0.05 so the just-updated current room keeps a small score
            # instead of exactly zero.
            scores[room] = entropy_norm * (staleness_norm + 0.05)
        best_score = max(scores.values())
        best_rooms = [r for r, s in scores.items() if s >= best_score - 1e-9]
        next_room = rng.choice(best_rooms)
        if next_room != current_room:
            return "move", next_room
        return "stay", current_room

    return policy


def _nav_policy_perfect_oracle(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Return a navigation policy equivalent to greedy (1-step GT lookahead).

    NOTE: In the current dataset the existing perfect_oracle strategy uses the
    same one-step lookahead as greedy. This policy mirrors that behaviour.
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    return _nav_policy_greedy(seed=seed, rooms=rooms)


def _digamma(x: float) -> float:
    """Digamma approximation ψ(x) ≈ ln(x) − 1/(2x) (error < 0.01 for x ≥ 1).

    scipy is not a dependency; our Beta parameters are always ≥ 1, so the
    first-order asymptotic expansion is accurate enough for a ranking score.
    """
    return math.log(x) - 1.0 / (2.0 * x)


def _beta_differential_entropy(a: float, b: float) -> float:
    """Differential entropy of Beta(a, b): ln B − (a−1)ψ(a) − (b−1)ψ(b) + (a+b−2)ψ(a+b)."""
    ln_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    return ln_b - (a - 1) * _digamma(a) - (b - 1) * _digamma(b) + (a + b - 2) * _digamma(a + b)


def _nav_policy_beta_entropy(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Beta-posterior entropy baseline (§19, post-§4.1 perceived counts).

    Per room, treat visits as Bernoulli trials over "did I detect a change":
    posterior Beta(perceived_changes + 1, max(visits − changes, 0) + 1).
    Score = differential entropy of that posterior — a principled
    exploration bonus: unvisited rooms are Beta(1, 1) (uniform, maximum
    entropy) and win on merit; heavily-visited rooms concentrate and sink.
    Like _nav_policy_entropy, this reads only the VLM's own perceived
    verdicts. Argmax, seeded tie-break.
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        scores: dict[str, float] = {}
        for room in rooms:
            changes = detector.room_perceived_change_counts.get(room, 0)
            visits = detector.room_visit_counts.get(room, 0)
            a = changes + 1.0
            b = max(visits - changes, 0) + 1.0
            scores[room] = _beta_differential_entropy(a, b)
        best_score = max(scores.values())
        best_rooms = [r for r, s in scores.items() if s >= best_score - 1e-9]
        next_room = rng.choice(best_rooms)
        if next_room != current_room:
            return "move", next_room
        return "stay", current_room

    return policy


def _nav_policy_frontier(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Frontier-based exploration baseline (Yamauchi 1997), room-graph variant.

    The recognizable active-mapping loop with NO change-rate/ownership
    reasoning: maintain a visit map; each step go to the highest-uncertainty
    region — never-visited rooms first (nearest by walk time, the classic
    nearest-frontier rule), else the stalest room (map freshness decays with
    time). Ties break by shorter walk time, then seeded random. The pure
    variant re-plans every step and never dwells for detection: mapping
    coverage, not change-catching, is its objective. Walk times come from the
    dataset's own distance map when available (24-room worlds); on a
    uniform-cost room graph this approximates least-recently-visited
    sweeping — close to round-robin once the map is complete, which is the
    expected result and part of the rebuttal argument.
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        last_visits = detector.room_last_visit_time
        # 1) Unexplored regions: nearest frontier among never-visited rooms.
        unvisited = [r for r in rooms if r not in last_visits and r != current_room]
        if unvisited:
            best_d = min(_policy_travel_time(log, current_room, r) for r in unvisited)
            nearest = [r for r in unvisited if _policy_travel_time(log, current_room, r) == best_d]
            return "move", rng.choice(nearest)
        # 2) Map complete: revisit the stalest region (uncertainty regrowth).
        log_start = log.unique_timestamps[0] if log.unique_timestamps else current_time

        def _staleness(room: str) -> float:
            ref = last_visits.get(room, log_start)
            return (current_time - ref).total_seconds()

        best_s = max(_staleness(r) for r in rooms)
        candidates = [r for r in rooms if _staleness(r) >= best_s - 1e-9 and r != current_room]
        if not candidates:
            # The current room itself is the stalest — staying refreshes the map.
            return "stay", current_room
        best_d = min(_policy_travel_time(log, current_room, r) for r in candidates)
        nearest = [r for r in candidates if _policy_travel_time(log, current_room, r) == best_d]
        return "move", rng.choice(nearest)

    return policy


def _nav_policy_staleness(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Staleness-only baseline: argmax time since last visit.

    The 'COOL minus ownership/interaction signals' bridge row: of COOL's three
    navigation signals (hotspots, expected change, staleness) only staleness
    remains. Never-visited rooms are maximally stale; ties break randomly
    (seeded). Reads only the robot's own visit memory.
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        last_visits = detector.room_last_visit_time

        def _staleness(room: str) -> float:
            ref = last_visits.get(room)
            if ref is None:
                return float("inf")
            return (current_time - ref).total_seconds()

        best_s = max(_staleness(r) for r in rooms)
        candidates = [r for r in rooms if _staleness(r) >= best_s - 1e-9 and r != current_room]
        if not candidates:
            return "stay", current_room
        return "move", rng.choice(candidates)

    return policy


def _nav_policy_beta_entropy_cost(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Cost-discounted semantic-IG baseline: information gain per unit travel.

    Same Beta-posterior beliefs as _nav_policy_beta_entropy, but scored
    exp(H) / (1 + walk minutes) — the standard information-gain-over-path-cost
    formulation, where candidate targets are OTHER rooms: as in frontier
    exploration, the current position is never a candidate (it is already
    mapped). Without that exclusion the d=0 self-loop wins and the policy
    camps. exp(H) (the posterior's effective support size) is a monotone
    non-negative transform of the differential entropy H, which keeps the
    ratio well-defined: raw H is ≤ 0 and dividing a negative score by cost
    would invert the preference for distant rooms. Seeded tie-break.
    """
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        scores: dict[str, float] = {}
        for room in rooms:
            if room == current_room:
                continue  # current position is already mapped — never a target
            changes = detector.room_perceived_change_counts.get(room, 0)
            visits = detector.room_visit_counts.get(room, 0)
            a = changes + 1.0
            b = max(visits - changes, 0) + 1.0
            scores[room] = math.exp(_beta_differential_entropy(a, b)) / (1.0 + _policy_travel_time(log, current_room, room))
        if not scores:
            return "stay", current_room
        best_score = max(scores.values())
        best_rooms = [r for r, s in scores.items() if s >= best_score - 1e-9]
        return "move", rng.choice(best_rooms)

    return policy


def _nav_policy_random(seed: int | None = None, rooms: list[str] | None = None) -> Any:
    """Return a navigation policy that picks a random room at each step."""
    rooms = rooms if rooms is not None else SCENE_CHANGE_ROOM_ORDER
    rng = random.Random(seed)

    def policy(
        current_room: str | None,
        current_time: datetime,
        steps_in_room: int,
        log: SceneChangeTimeLog,
        detector: Any,
    ) -> tuple[str, str]:
        if current_room is None:
            return "move", rooms[0]
        candidates = [r for r in rooms if r != current_room]
        next_room = rng.choice(candidates)
        return "move", next_room

    return policy


def _get_detector_prompt(detector: str, no_tools: bool = False) -> str:
    """Return the system prompt for a detector strategy name.

    When no_tools=True, prefers the _NO_TOOLS variant of the prompt if available.
    """
    prompt_map = {
        "agent_scene_change_v5": "NAVIGATION_SCENE_CHANGE_V5",
        "agent_scene_change_split": "NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR",
        "agent_scene_change_v5_no_tools": "NAVIGATION_SCENE_CHANGE_V5",
        "agent_scene_change_v8": "NAVIGATION_SCENE_CHANGE_V8",
        "agent_scene_change_v8_short": "NAVIGATION_SCENE_CHANGE_V8_SHORT",
        "agent_scene_change_v8_quick": "NAVIGATION_SCENE_CHANGE_V8_QUICK",
        "agent_scene_change_v8_clean": "NAVIGATION_SCENE_CHANGE_V8_CLEAN",
        "agent_scene_change_v8_active": "NAVIGATION_SCENE_CHANGE_V8_ACTIVE",
        "agent_scene_change_v8_smart": "NAVIGATION_SCENE_CHANGE_V8_SMART",
        "agent_scene_change_v9": "NAVIGATION_SCENE_CHANGE_V9",
        "agent_scene_change_v10": "NAVIGATION_SCENE_CHANGE_V10",
        "agent_scene_change_v8_no_tools": "NAVIGATION_SCENE_CHANGE_V8_NO_TOOLS",
        "agent_scene_change_v8_short_no_tools": "NAVIGATION_SCENE_CHANGE_V8_SHORT_NO_TOOLS",
    }
    prompt_name = prompt_map.get(detector)
    # If no_tools is requested and a _NO_TOOLS variant exists, prefer it
    if no_tools and not detector.endswith("_no_tools"):
        no_tools_detector = detector + "_no_tools"
        no_tools_prompt_name = prompt_map.get(no_tools_detector)
        if no_tools_prompt_name:
            prompt_name = no_tools_prompt_name
    if not prompt_name:
        return "You are a scene-change-aware robot navigation strategist. Output only JSON."
    try:
        module = __import__("bordsupr.frontend.agent.prompts", fromlist=[prompt_name])
        return getattr(module, prompt_name)
    except (ImportError, AttributeError):
        try:
            module = __import__("agent.prompts", fromlist=[prompt_name])
            return getattr(module, prompt_name)
        except (ImportError, AttributeError):
            return "You are a scene-change-aware robot navigation strategist. Output only JSON."


def _get_nav_policy(navigator: str, seed: int | None = None, rooms: list[str] | None = None):
    """Return a navigation policy factory for a navigator name."""
    if navigator == "round_robin":
        return _nav_policy_round_robin(rooms=rooms)
    elif navigator in ("greedy", "greedy_oracle_1step"):
        # "greedy" kept as a deprecated alias for "greedy_oracle_1step"
        # (1-step GT lookahead — an oracle ceiling, not a competitor).
        return _nav_policy_greedy(seed=seed, rooms=rooms)
    elif navigator == "greedy_hazard":
        return _nav_policy_greedy_hazard(seed=seed, rooms=rooms)
    elif navigator == "frequency":
        return _nav_policy_frequency(seed=seed, source="perceived", rooms=rooms)
    elif navigator == "frequency_gt":
        # Named legacy variant: GT changes-while-present counts (pre-§3.2).
        return _nav_policy_frequency(seed=seed, source="gt", rooms=rooms)
    elif navigator == "entropy":
        return _nav_policy_entropy(seed=seed, rooms=rooms)
    elif navigator == "beta_entropy":
        return _nav_policy_beta_entropy(seed=seed, rooms=rooms)
    elif navigator == "perfect_oracle":
        return _nav_policy_perfect_oracle(seed=seed, rooms=rooms)
    elif navigator == "random":
        return _nav_policy_random(seed=seed, rooms=rooms)
    elif navigator == "native":
        return None
    else:
        raise ValueError(f"Unknown navigation policy: {navigator}")


class _SteppableSceneChangeVLMHybridNav(_SteppableSceneChangeVLM):
    """VLM agent where detection is done by the VLM but navigation is overridden by a baseline policy."""

    def __init__(
        self,
        log: SceneChangeTimeLog,
        strategy_type: str = "agent_scene_change_hybrid",
        system_prompt: str | None = None,
        seed: int | None = None,
        nav_policy: Any = None,
        random_target: bool = False,
        no_tools: bool = False,
        dwell_config: dict | None = None,
        policy_controls_action: bool = False,
        detector_type: str | None = None,
    ):
        super().__init__(log, strategy_type=strategy_type, system_prompt=system_prompt, seed=seed, no_tools=no_tools, dwell_config=dwell_config, detector_type=detector_type)
        # nav_policy=None means "use VLM's native navigation" (explicit override).
        # To get round-robin, pass the policy function or omit the arg.
        self.nav_policy = nav_policy
        self.random_target = random_target
        # False (default): VLM decides WHEN to move, policy picks WHERE.
        # True ("pure" baseline): the policy decides both stay/move and the
        # destination; the VLM still runs every step but only for detection.
        self.policy_controls_action = policy_controls_action

    def _make_step(
        self,
        current_time: datetime,
        parsed: dict[str, Any],
        gt_change: str,
        gt_activity_changed: bool,
        entry: SceneLogEntry | None,
        interactions: list[dict],
        tool_calls: list[dict] | None = None,
    ) -> SceneChangeMinuteStep:
        """Use VLM for detection, but override navigation with the policy."""
        vlm_detected = parsed.get("scene_changed") in (True, "true", "True", 1, "1")
        vlm_severity = str(parsed.get("change") or parsed.get("change_severity") or "false").lower()
        vlm_activity_detected = parsed.get("activities_changed") in (True, "true", "True", 1, "1")

        self._update_memory_and_metrics(
            current_time,
            entry,
            vlm_detected,
            vlm_activity_detected,
            gt_change,
            gt_activity_changed,
            interactions,
            vlm_severity,
        )

        # Use VLM for stay/move decision, but let nav_policy override destination.
        # Call nav_policy every step so its internal state advances correctly.
        # With policy_controls_action the policy also decides stay/move; the
        # VLM's action/target outputs are then ignored for motion (detection
        # above is unaffected).
        action = str(parsed.get("action", "stay")).lower()
        target_room = str(parsed.get("target_room", "")).strip()

        if self.nav_policy is not None:
            policy_action, policy_target = self.nav_policy(
                current_room=self.current_room,
                current_time=current_time,
                steps_in_room=self.steps_in_room,
                log=self.log,
                detector=self,
            )
            if self.policy_controls_action:
                action = str(policy_action).lower()
                target_room = policy_target if action == "move" else (self.current_room or "")
            elif action == "move":
                target_room = policy_target

        # If random_target is enabled, keep VLM action but pick random destination
        if self.random_target and action == "move":
            candidates = [r for r in self.log.rooms if r != self.current_room]
            if candidates:
                target_room = self.rng.choice(candidates)

        if action == "move" and target_room and target_room != self.current_room:
            for r in self.log.rooms:
                if r.lower() == target_room.lower():
                    target_room = r
                    break
            else:
                # Unknown room from the VLM: fall back to a random room ≠ current
                target_room = self.rng.choice([r for r in self.log.rooms if r != self.current_room])
            self.pending_move_room = target_room
            self.last_target_room = target_room

        return SceneChangeMinuteStep(
            timestamp=current_time,
            time_str=current_time.strftime("%H:%M"),
            room=self.current_room,
            scene_text=entry.scene if entry else "No data.",
            gt_change_type=gt_change,
            gt_activity_changed=gt_activity_changed,
            vlm_detected_change=vlm_detected,
            vlm_change=vlm_severity,
            vlm_detected_activity_change=vlm_activity_detected,
            action=action,
            target_room=target_room if action == "move" else self.current_room,
            decision=parsed,
            fresh_decision=True,
            people_present=entry.people_present if entry else [],
            objects_present=entry.objects_present if entry else [],
            interactions=interactions,
            prompt=self.last_prompt,
            raw_response=self.last_raw_response,
            system_prompt=self.prompt,
            tool_calls=tool_calls,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _simulator_sha256() -> str:
    """SHA256 of this simulator file, computed at runtime (§0.3.3)."""
    try:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        return ""


_prompts_stub_warning_issued = False


def _warn_if_prompts_unavailable() -> None:
    """Warn (once per process) when the frontend prompt package is missing.

    Every strategy branch below falls back to a one-line STUB system prompt
    when bordsupr.frontend.agent.prompts is not importable. A stub silently
    degrades fused strategies (rules lost) and completely disables the split
    strategies (the navigator's JSON contract lives in the system prompt, so
    its output never parses and the agent never moves). This burned a full
    grid: run_grid_sweep.py historically lacked the repo root on sys.path.
    """
    global _prompts_stub_warning_issued
    if _prompts_stub_warning_issued:
        return
    try:
        import bordsupr.frontend.agent.prompts  # noqa: F401
        return
    except ImportError:
        pass
    try:
        import agent.prompts  # noqa: F401
        return
    except ImportError:
        pass
    _prompts_stub_warning_issued = True
    warnings.warn(
        "bordsupr.frontend.agent.prompts is NOT importable — all VLM "
        "strategies will run with STUB system prompts (navigation rules and "
        "JSON contracts lost; split strategies will not move). Put the repo "
        "root on sys.path or install the bordsupr package.",
        RuntimeWarning, stacklevel=3)


def run_scene_change_simulation(
    strategy: str,
    seed: int | None = 42,
    progress_callback: Any = None,
    data_dir: Path | str | None = None,
    no_tools: bool = False,
    dwell_config: dict | None = None,
    initial_pick: str = "vlm",
    hazard_fix: bool = False,
    signal_source: str = "perceived",
    mask_tools: tuple[str, ...] | frozenset[str] | None = None,
    anti_camping: bool = False,
    move_budget: int = 0,
    score_weights: dict | None = None,
) -> SceneChangeSimulationResult:
    """Run a single strategy against the scene-change log."""
    _warn_if_prompts_unavailable()
    if data_dir is not None:
        data_dir = Path(data_dir)
    log = SceneChangeTimeLog(data_dir=data_dir)
    strategy = str(strategy).strip().lower()
    if initial_pick not in ("vlm", "random", "fixed"):
        raise ValueError(f"Invalid initial_pick: {initial_pick!r}. Expected 'vlm', 'random' or 'fixed'.")
    if signal_source not in ("perceived", "gt"):
        raise ValueError(f"Invalid signal_source: {signal_source!r}. Expected 'perceived' or 'gt'.")
    # §4.1: "+oracle_tools" suffix = named GT-tools upper-bound variant
    # (pre-flip behavior). Stripped before the strategy ladder; applied in
    # the passthrough block below.
    oracle_tools = strategy.endswith("+oracle_tools")
    if oracle_tools:
        strategy = strategy[: -len("+oracle_tools")]
    strategy_label = strategy + ("+oracle_tools" if oracle_tools else "")

    if strategy == "fixed_10min":
        sim = _SteppableFixedRotation(log, dwell_steps=2)  # 2 * 5min = 10min
    elif strategy == "fixed_20min":
        sim = _SteppableFixedRotation(log, dwell_steps=4)
    elif strategy in ("round_robin", "round_robin_vlm_gated"):
        if strategy == "round_robin":
            warnings.warn(
                "'round_robin' is deprecated; use 'round_robin_vlm_gated' "
                "(identical behavior: VLM decides when, policy where) or "
                "'round_robin_pure' (policy decides both).",
                DeprecationWarning,
                stacklevel=2,
            )
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_round_robin(rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "round_robin_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="round_robin_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_round_robin(rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "random":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="random", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_random(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "random_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="random_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_random(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy in ("frequency", "frequency_gt"):
        # §3.2 default flip: "frequency" now weights ∝ PERCEIVED change
        # counts + 1 (Laplace); "frequency_gt" is the named legacy variant
        # (GT changes-while-present). Recorded per §0.3 rule 2.
        source = "gt" if strategy == "frequency_gt" else "perceived"
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_frequency(seed=seed, source=source, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy in ("frequency_pure", "frequency_gt_pure"):
        source = "gt" if strategy == "frequency_gt_pure" else "perceived"
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_frequency(seed=seed, source=source, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "entropy":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="entropy", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_entropy(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "entropy_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="entropy_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_entropy(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "beta_entropy":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="beta_entropy", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_beta_entropy(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "beta_entropy_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="beta_entropy_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_beta_entropy(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy in ("greedy_oracle_1step", "greedy"):
        if strategy == "greedy":
            warnings.warn(
                "'greedy' is deprecated; use 'greedy_oracle_1step' (identical "
                "behavior: 1-step GT lookahead — an oracle ceiling, not a "
                "deployable competitor) or 'greedy_hazard' (deployable).",
                DeprecationWarning,
                stacklevel=2,
            )
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_greedy(rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy in ("greedy_oracle_1step_pure", "greedy_pure"):
        if strategy == "greedy_pure":
            warnings.warn(
                "'greedy_pure' is deprecated; use 'greedy_oracle_1step_pure' "
                "(identical behavior).",
                DeprecationWarning,
                stacklevel=2,
            )
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_greedy(rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "greedy_hazard":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="greedy_hazard", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_greedy_hazard(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "greedy_hazard_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="greedy_hazard_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_greedy_hazard(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "rank_greedy_pure":
        # Ablation: VLM detection (v8_short) + pure argmax navigation over the
        # exact ROOM RANKING score the VLM navigators see (_compute_room_scores:
        # w_hot·rate + w_exp·hazard + w_stale·staleness). Quantifies what the
        # VLM navigator adds over simply following its own ranking table.
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="rank_greedy_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_rank_greedy(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "active_mapping":
        # Frontier-based exploration (Yamauchi 1997), VLM-gated like round_robin:
        # VLM decides WHEN to move, frontier policy picks WHERE.
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="active_mapping", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_frontier(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "active_mapping_pure":
        # Textbook variant: the frontier policy owns stay/move (re-plans every
        # step, never dwells); the VLM runs only as the detector.
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="active_mapping_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_frontier(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "staleness_only":
        # 'COOL minus ownership signals' bridge: staleness is the only signal.
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="staleness_only", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_staleness(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "staleness_only_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="staleness_only_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_staleness(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "beta_entropy_cost":
        # Cost-discounted semantic IG: exp(H)/(1 + walk minutes).
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="beta_entropy_cost", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_beta_entropy_cost(seed=seed, rooms=log.rooms), no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "beta_entropy_cost_pure":
        prompt = _get_detector_prompt("agent_scene_change_v8_short", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type="beta_entropy_cost_pure", system_prompt=prompt, seed=seed, nav_policy=_nav_policy_beta_entropy_cost(seed=seed, rooms=log.rooms), policy_controls_action=True, no_tools=no_tools, detector_type="agent_scene_change_v8_short")
    elif strategy == "perfect_oracle":
        warnings.warn(
            "'perfect_oracle' is deprecated as a name; it is the same 1-step "
            "GT lookahead oracle as 'greedy_oracle_1step' (kept as an alias, "
            "trajectory unchanged).",
            DeprecationWarning,
            stacklevel=2,
        )
        sim = _SteppablePerfectOracle(log, seed=seed)
    elif strategy == "agent_scene_change_v8":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8", system_prompt=NAVIGATION_SCENE_CHANGE_V8, seed=seed, no_tools=no_tools)
    elif strategy in ("agent_scene_change_v9_antipcamp", "agent_scene_change_v9_budget"):
        # v9 forcing variants: IDENTICAL v8 detector/prompt/tools (v8 system
        # prompt, detector_type="agent_scene_change_v8" for the _call_vlm
        # dispatch, "v9_antipcamp"/"v9_budget" remapped to the v8 path in
        # _build_prompt). The only difference is code-enforced movement:
        # anti_camping = no-change-vote hysteresis; move_budget = hard cap
        # on consecutive detection-free steps without moving.
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(
            log, strategy_type=strategy, system_prompt=NAVIGATION_SCENE_CHANGE_V8,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            anti_camping=strategy == "agent_scene_change_v9_antipcamp",
            move_budget=15 if strategy == "agent_scene_change_v9_budget" else 0,
        )
    elif strategy == "agent_scene_change_v10_split":
        # Tier-2 prompt-split (§Tier 2): detection-only V10_DETECTOR turn
        # (confidence-calibrated) with a code state machine — stay on
        # change/uncertainty; the navigator is called ONLY on certain
        # no-change and consumes a code-computed ROOM RANKING + walk times.
        # detector_type="agent_scene_change_v10_split" routes _call_vlm to
        # _call_vlm_split and _build_prompt to the v8 observation path with
        # a v10 detection-only tail (never the legacy v10 branch).
        # signal_source stays "perceived"; hazard_fix forced True in the
        # passthrough below. anti_camping/move_budget stay off (orthogonal).
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR,
            )
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
                    NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR,
                )
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V10_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(
            log, strategy_type="agent_scene_change_v10_split",
            system_prompt=NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
            seed=seed, no_tools=no_tools,
            detector_type="agent_scene_change_v10_split",
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR,
        )
    elif strategy in ("agent_scene_change_v10_split_notime", "agent_scene_change_v10_split_scene"):
        # v10_split ablations (Tier 3 ping-pong investigation). Identical
        # detector/state-machine wiring as v10_split (detector_type keeps the
        # "v10_split" substring so dispatch and the _build_prompt stype
        # mapping both catch it); only the navigator input changes:
        # - notime: nav_walk_times=False — WALK TIMES table omitted, and the
        #   NOTIME system prompt (no tie-break-to-closer-room rule) is used
        #   so the model is not told to use information it no longer gets.
        # - scene: nav_scene_context=True — adds a per-room LAST OBSERVED
        #   SCENES block from the agent's perceived memory; walk times kept.
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_NOTIME,
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_SCENE,
            )
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
                    NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_NOTIME,
                    NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_SCENE,
                )
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V10_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_NOTIME = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_SCENE = "You are a robot navigation strategist. Output only JSON."
        notime = strategy == "agent_scene_change_v10_split_notime"
        sim = _SteppableSceneChangeVLM(
            log, strategy_type=strategy,
            system_prompt=NAVIGATION_SCENE_CHANGE_V10_DETECTOR,
            seed=seed, no_tools=no_tools,
            detector_type="agent_scene_change_v10_split",
            navigator_prompt=(NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_NOTIME if notime
                              else NAVIGATION_SCENE_CHANGE_V10_NAVIGATOR_SCENE),
            nav_walk_times=not notime,
            nav_scene_context=not notime,
        )
    elif strategy == "agent_scene_change_v9_split":
        # Tier-2 v9_split (§Tier 2): v8 detector turn with a detection-only
        # prompt tail + navigator fed a Python-computed score table. The
        # navigator is skipped on detection steps (stay-and-watch).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
        )
    elif strategy == "agent_scene_change_v9_split_soft":
        # Soft v9_split: detector also judges interestingness and notes where
        # people are heading; the navigator is ALWAYS called (skip_on_detection
        # = False) and may move despite a scene change or follow the action.
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_on_detection=False,
        )
    elif strategy == "agent_scene_change_v11_split":
        # v11_split (synthesis of strict + soft): the SOFT detector (judges
        # interestingness, notes where people head) + the SOFT navigator
        # (move-on/follow-the-action), but skip_mode="interesting": the
        # navigator is SKIPPED whenever the scene is worth watching — cheaper
        # than soft on interesting scenes, smarter than strict on boring ones
        # (changed-but-boring still reaches the navigator so it can move on).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_SOFT_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
        )
    elif strategy == "agent_scene_change_v12_split":
        # v12_split (synthesis of v10 + soft): the V12 detector (v10 counting
        # procedure + confidence contract + stale-memory/noise rules, plus
        # soft interesting/observation) + the V12 navigator (v10 ranked table
        # + follow-the-action + action-cost awareness). skip_mode=
        # "interesting": navigator skipped when the scene is worth watching;
        # changed-but-boring reaches the navigator so it can move on, and
        # detector confidence is passed through the DETECTOR ASSESSMENT block
        # (navigator rule 4: stay one more observation when uncertain).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V12_DETECTOR, NAVIGATION_SCENE_CHANGE_V12_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V12_DETECTOR, NAVIGATION_SCENE_CHANGE_V12_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V12_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V12_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V12_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V12_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
        )
    elif strategy in ("agent_scene_change_v13_split", "agent_scene_change_v13_split_caption"):
        # v13_split: v12 detector + RECENT OBSERVATIONS window (last 4
        # perceived observations of the current room incl. past verdicts) and
        # structured movement output (leaving / departure_destination /
        # people_trend); v13 navigator consumes departure_destination for
        # follow-the-action (rule 3). skip_mode="interesting" as in v11/v12.
        # The _caption variant additionally includes each history entry's
        # scene caption in the window (obs_history_include_caption=True).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V13_DETECTOR, NAVIGATION_SCENE_CHANGE_V13_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V13_DETECTOR, NAVIGATION_SCENE_CHANGE_V13_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V13_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V13_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V13_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V13_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            obs_history_include_caption=strategy.endswith("_caption"),
        )
    elif strategy == "agent_scene_change_v14_split":
        # v14_split: v13 + prompt-capture fixes: FTA unblock (navigator called
        # on leaving=true with a VALID destination — validated against the
        # room list), gap-in-minutes injected into the prompt, bounded
        # stale-memory rule + confidence floor (detector prompt), single-brace
        # JSON contract, detector max_tokens=8192 (512 truncated the JSON).
        # No caption variant — the v13 caption ablation was a regression.
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V14_DETECTOR, NAVIGATION_SCENE_CHANGE_V14_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V14_DETECTOR, NAVIGATION_SCENE_CHANGE_V14_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V14_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V14_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V14_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V14_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            fta_unblock=True,
            detector_max_tokens=8192,
        )
    elif strategy == "agent_scene_change_v15_split":
        # v15_split: v14 + fixes from the v14 grid/capture analysis:
        # - departure_destination == current room normalized to "" (fta_mode=
        #   "v15"; 6/21 v14 FTA opportunities were this contradiction);
        # - FTA unblock only when people_trend == "leaving" (winding-down
        #   room) — v14 followed departures out of still-active rooms
        #   (moves 105 -> 149, recall/precision dropped);
        # - navigator rule 3 refuses bottom-half destinations (V15 navigator).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V15_DETECTOR, NAVIGATION_SCENE_CHANGE_V15_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V15_DETECTOR, NAVIGATION_SCENE_CHANGE_V15_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V15_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V15_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V15_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V15_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            fta_unblock=True,
            fta_mode="v15",
            detector_max_tokens=8192,
        )
    elif strategy in ("agent_scene_change_v16_split", "agent_scene_change_v17_split"):
        # v16_split: v15 harness + stale-memory DECISION TABLE in the detector
        #   prompt (fixes the v15 boundary FNs: active verbs never required at
        #   short gaps; large-diff branch covers ANY 2+ total diffs).
        # v17_split: same harness, qualitative thresholds instead of the
        #   hardcoded 15-minute cutoff and exact diff counts — meant to
        #   transfer to real-world scenes with irregular revisit intervals.
        # Both reuse the v15 JSON contract, navigator, and fta_mode="v15".
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V16_DETECTOR, NAVIGATION_SCENE_CHANGE_V16_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V17_DETECTOR, NAVIGATION_SCENE_CHANGE_V17_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V16_DETECTOR, NAVIGATION_SCENE_CHANGE_V16_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V17_DETECTOR, NAVIGATION_SCENE_CHANGE_V17_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V16_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V16_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V17_DETECTOR = NAVIGATION_SCENE_CHANGE_V16_DETECTOR
                NAVIGATION_SCENE_CHANGE_V17_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V16_NAVIGATOR
        det = NAVIGATION_SCENE_CHANGE_V16_DETECTOR if strategy.endswith("v16_split") else NAVIGATION_SCENE_CHANGE_V17_DETECTOR
        nav = NAVIGATION_SCENE_CHANGE_V16_NAVIGATOR if strategy.endswith("v16_split") else NAVIGATION_SCENE_CHANGE_V17_NAVIGATOR
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=nav,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            fta_unblock=True,
            fta_mode="v15",
            detector_max_tokens=8192,
        )
    elif strategy in ("agent_scene_change_v18_split", "agent_scene_change_v19_split", "agent_scene_change_v20_split"):
        # v18_split: v17 + CONSISTENCY CHECK pinning scene_changed to the
        #   model's own entity_diffs value (v16 capture failure class: model
        #   counted 5/7 diffs, then "re-estimated" 0-1 when applying the rule).
        # v19_split: v18 + the check mirrored into the reasoning instruction
        #   (model must state its entity_diffs count and which check line
        #   fired before judging).
        # v20_split: v19 with the entity_diffs = 1 line softened — judged by
        #   MATCH CONFIDENCE instead of gap alone (v19's only remaining FNs
        #   were confident 1-diff cases suppressed by gap + no active verbs).
        # All reuse the v15 JSON contract, navigator, and fta_mode="v15".
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V18_DETECTOR, NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V19_DETECTOR, NAVIGATION_SCENE_CHANGE_V19_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V20_DETECTOR, NAVIGATION_SCENE_CHANGE_V20_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V18_DETECTOR, NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V19_DETECTOR, NAVIGATION_SCENE_CHANGE_V19_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V20_DETECTOR, NAVIGATION_SCENE_CHANGE_V20_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V18_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V19_DETECTOR = NAVIGATION_SCENE_CHANGE_V18_DETECTOR
                NAVIGATION_SCENE_CHANGE_V19_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR
                NAVIGATION_SCENE_CHANGE_V20_DETECTOR = NAVIGATION_SCENE_CHANGE_V18_DETECTOR
                NAVIGATION_SCENE_CHANGE_V20_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR
        _dets = {"v18_split": NAVIGATION_SCENE_CHANGE_V18_DETECTOR,
                 "v19_split": NAVIGATION_SCENE_CHANGE_V19_DETECTOR,
                 "v20_split": NAVIGATION_SCENE_CHANGE_V20_DETECTOR}
        _navs = {"v18_split": NAVIGATION_SCENE_CHANGE_V18_NAVIGATOR,
                 "v19_split": NAVIGATION_SCENE_CHANGE_V19_NAVIGATOR,
                 "v20_split": NAVIGATION_SCENE_CHANGE_V20_NAVIGATOR}
        _tail = strategy.rsplit("_", 2)[-2] + "_split"  # "v18_split" / "v19_split" / "v20_split"
        det = _dets[_tail]
        nav = _navs[_tail]
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=nav,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            fta_unblock=True,
            fta_mode="v15",
            detector_max_tokens=8192,
        )
    elif strategy in ("agent_scene_change_v21_split", "agent_scene_change_v22_split"):
        # v21_split: v20 + cross-room context — the detector also sees its
        #   last observation of every OTHER room (cross_room_obs=True). Same
        #   JSON contract / navigator / fta_mode="v15" as v20.
        # v22_split: division of labor experiment — the detector does NOT
        #   output departure_destination; the navigator infers the
        #   destination itself from the detector's observation note + the
        #   ranked table + walk times (fta_mode="v22": unblock on
        #   leaving=true, same winding-down trend gate as v15).
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V21_DETECTOR, NAVIGATION_SCENE_CHANGE_V21_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V22_DETECTOR, NAVIGATION_SCENE_CHANGE_V22_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V21_DETECTOR, NAVIGATION_SCENE_CHANGE_V21_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V22_DETECTOR, NAVIGATION_SCENE_CHANGE_V22_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V21_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V21_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V22_DETECTOR = NAVIGATION_SCENE_CHANGE_V21_DETECTOR
                NAVIGATION_SCENE_CHANGE_V22_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V21_NAVIGATOR
        if strategy.endswith("v21_split"):
            det, nav, mode, cross = (NAVIGATION_SCENE_CHANGE_V21_DETECTOR,
                                     NAVIGATION_SCENE_CHANGE_V21_NAVIGATOR, "v15", True)
        else:
            det, nav, mode, cross = (NAVIGATION_SCENE_CHANGE_V22_DETECTOR,
                                     NAVIGATION_SCENE_CHANGE_V22_NAVIGATOR, "v22", False)
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=nav,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            fta_unblock=True,
            fta_mode=mode,
            cross_room_obs=cross,
            detector_max_tokens=8192,
        )
    elif strategy in ("agent_scene_change_v11r_split", "agent_scene_change_v20r_split"):
        # "Redesigned" pair from PROMPTS_V11_VS_V20.md (user's re-write):
        # v11r: lean v11 style (free judgment, no structured movement) BUT
        #   with the last-4 history window incl. scene captions; JSON
        #   contract in the system prompt; no FTA unblock.
        # v20r: trimmed v20 — keeps confidence calibration, structured
        #   movement (leaving/destination/trend) and the interesting
        #   checklist; drops the noise rules / stale-memory rule /
        #   consistency check. Same harness as v20 (fta_mode="v15").
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V20R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR = NAVIGATION_SCENE_CHANGE_V11R_DETECTOR
                NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR
        if strategy.endswith("v11r_split"):
            sim = _SteppableSceneChangeVLMNavSplit(
                log, strategy_type=strategy,
                detector_prompt=NAVIGATION_SCENE_CHANGE_V11R_DETECTOR,
                navigator_prompt=NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR,
                seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
                skip_mode="interesting",
                obs_history_size=4,
                obs_history_include_caption=True,
                detector_max_tokens=8192,
            )
        else:
            sim = _SteppableSceneChangeVLMNavSplit(
                log, strategy_type=strategy,
                detector_prompt=NAVIGATION_SCENE_CHANGE_V20R_DETECTOR,
                navigator_prompt=NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR,
                seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
                skip_mode="interesting",
                obs_history_size=4,
                fta_unblock=True,
                fta_mode="v15",
                detector_max_tokens=8192,
            )
    elif strategy in ("agent_scene_change_v11r_det_v20r_nav_split", "agent_scene_change_v20r_det_v11r_nav_split"):
        # Cross combinations of the redesigned pair:
        # v11r_det_v20r_nav: lean free-judgment detector (no structured
        #   movement -> FTA unreachable) + strict-rule v20r navigator.
        # v20r_det_v11r_nav: structured-movement detector (fta_mode="v15")
        #   + prose-rule v11r navigator (rule 3 reasons over observation).
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V20R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR = NAVIGATION_SCENE_CHANGE_V11R_DETECTOR
                NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR
        if strategy.startswith("agent_scene_change_v11r_det"):
            det, nav = NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_NAVIGATOR
            kw = dict(obs_history_include_caption=True)
        else:
            det, nav = NAVIGATION_SCENE_CHANGE_V20R_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R_NAVIGATOR
            kw = dict(fta_unblock=True, fta_mode="v15")
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=nav,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            **kw,
        )
    elif strategy in ("agent_scene_change_v11r_argmax_split", "agent_scene_change_v20r_argmax_split"):
        # Dwell-isolation ablation: v11r / v20r DETECTOR + skip logic exactly
        # as in the full strategies, but navigation is plain argmax of the
        # ROOM RANKING table (no navigator VLM call). Compare against
        # v11r_split / v20r_split to isolate the navigator's marginal value,
        # and against rank_greedy_pure (which also replaces the detector with
        # v8_short and has no skip) to isolate detector+skip value.
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_DETECTOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V11R_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R_DETECTOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V11R_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR = NAVIGATION_SCENE_CHANGE_V11R_DETECTOR
        if strategy.startswith("agent_scene_change_v11r"):
            det = NAVIGATION_SCENE_CHANGE_V11R_DETECTOR
            kw = dict(obs_history_include_caption=True)
        else:
            det = NAVIGATION_SCENE_CHANGE_V20R_DETECTOR
            kw = dict(fta_unblock=False)  # no navigator to unblock — argmax never follows
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=None,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            argmax_nav=True,
            **kw,
        )
    elif strategy.startswith("agent_scene_change_v20r_skip_") and strategy.endswith("_split"):
        # Skip-x-policy ablation: v20r DETECTOR + skip logic own stay/move
        # (dwell while interesting), a HEURISTIC nav policy owns the
        # destination (no navigator VLM call). Extends the argmax ablation:
        # tests whether semantic dwell lifts ANY where-policy, not just
        # argmax of the ranking table. Compare each
        # agent_scene_change_v20r_skip_<P>_split against <P>_pure (same
        # policy, no dwell) and agent_scene_change_v20r_argmax_split.
        _SKIP_POLICIES = {
            "round_robin": lambda: _nav_policy_round_robin(rooms=log.rooms),
            "random": lambda: _nav_policy_random(seed=seed, rooms=log.rooms),
            "frequency": lambda: _nav_policy_frequency(seed=seed, source="perceived", rooms=log.rooms),
            "entropy": lambda: _nav_policy_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy": lambda: _nav_policy_beta_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy_cost": lambda: _nav_policy_beta_entropy_cost(seed=seed, rooms=log.rooms),
            "greedy_hazard": lambda: _nav_policy_greedy_hazard(seed=seed, rooms=log.rooms),
            "active_mapping": lambda: _nav_policy_frontier(seed=seed, rooms=log.rooms),
            "staleness_only": lambda: _nav_policy_staleness(seed=seed, rooms=log.rooms),
        }
        _pol_name = strategy[len("agent_scene_change_v20r_skip_"):-len("_split")]
        if _pol_name not in _SKIP_POLICIES:
            raise ValueError(f"Unknown skip-policy in strategy {strategy!r}; expected one of {sorted(_SKIP_POLICIES)}")
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V20R_DETECTOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V20R_DETECTOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V20R_DETECTOR = "You are a scene-change detector. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V20R_DETECTOR,
            navigator_prompt=None,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            skip_nav_policy=_SKIP_POLICIES[_pol_name](),
            fta_unblock=False,  # no navigator to unblock — policy never follows
        )
        sim._skip_policy_name = _pol_name
    elif strategy.startswith("agent_scene_change_v20r2_skip_") and strategy.endswith("_split"):
        # Skip-x-policy ablation with the V20R2 DETECTOR: same detector and
        # dwell logic as agent_scene_change_v20r2_split, but the destination
        # is owned by a HEURISTIC nav policy (no navigator VLM call). This is
        # the controlled "same detector everywhere" comparison against the
        # heuristic rows, which detect with the v8_short prompt.
        _SKIP_POLICIES = {
            "round_robin": lambda: _nav_policy_round_robin(rooms=log.rooms),
            "random": lambda: _nav_policy_random(seed=seed, rooms=log.rooms),
            "frequency": lambda: _nav_policy_frequency(seed=seed, source="perceived", rooms=log.rooms),
            "entropy": lambda: _nav_policy_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy": lambda: _nav_policy_beta_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy_cost": lambda: _nav_policy_beta_entropy_cost(seed=seed, rooms=log.rooms),
            "greedy_hazard": lambda: _nav_policy_greedy_hazard(seed=seed, rooms=log.rooms),
            "active_mapping": lambda: _nav_policy_frontier(seed=seed, rooms=log.rooms),
            "staleness_only": lambda: _nav_policy_staleness(seed=seed, rooms=log.rooms),
        }
        _pol_name = strategy[len("agent_scene_change_v20r2_skip_"):-len("_split")]
        if _pol_name not in _SKIP_POLICIES:
            raise ValueError(f"Unknown skip-policy in strategy {strategy!r}; expected one of {sorted(_SKIP_POLICIES)}")
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR = "You are a scene-change detector. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
            navigator_prompt=None,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            skip_nav_policy=_SKIP_POLICIES[_pol_name](),
            fta_unblock=False,  # no navigator to unblock — policy never follows
        )
        sim._skip_policy_name = _pol_name
    elif strategy in ("agent_scene_change_v11r2_split", "agent_scene_change_v20r2_split",
                      "agent_scene_change_v11r2_argmax_split", "agent_scene_change_v20r2_argmax_split"):
        # R2 pair (user's rewrite, prompts.py scene_change_v11r2/v20r2_*).
        # Same harness split as the r1 pair: v11r2 = lean, caption-history,
        # no FTA unblock; v20r2 = fta_mode="v15" trend-gated unblock.
        # *_argmax_split keeps the detector + skip logic and replaces the
        # navigator with plain argmax of the ROOM RANKING table.
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V11R2_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R2_NAVIGATOR,
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V11R2_DETECTOR, NAVIGATION_SCENE_CHANGE_V11R2_NAVIGATOR,
                    NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR, NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V11R2_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V11R2_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR = NAVIGATION_SCENE_CHANGE_V11R2_DETECTOR
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR = NAVIGATION_SCENE_CHANGE_V11R2_NAVIGATOR
        is_v11 = strategy.startswith("agent_scene_change_v11r2")
        det = NAVIGATION_SCENE_CHANGE_V11R2_DETECTOR if is_v11 else NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR
        nav = NAVIGATION_SCENE_CHANGE_V11R2_NAVIGATOR if is_v11 else NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR
        argmax = strategy.endswith("_argmax_split")
        kw = dict(obs_history_include_caption=True) if is_v11 else (
            dict(fta_unblock=False) if argmax else dict(fta_unblock=True, fta_mode="v15"))
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=det,
            navigator_prompt=None if argmax else nav,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            argmax_nav=argmax,
            **kw,
        )
    elif strategy in ("agent_scene_change_v20r3_split", "agent_scene_change_v20r3_argmax_split"):
        # v20r3 = v20r2 + a fourth score term: the perceived person-object
        # INTERACTION rate per room (see _perceived_interaction_rates). The
        # first three terms all measure scene change; none of them measures
        # where ownership evidence is produced, which is what the memory is
        # built from. Everything else — detector, dwell, history window,
        # FTA — is byte-identical to agent_scene_change_v20r2_split, so
        # v20r2_split is the paired control and the interaction term is the
        # single ablated variable.
        # *_argmax_split keeps detector + dwell and replaces the navigator
        # with argmax of the same 4-term table: separates "the term helps"
        # from "the VLM uses the term".
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
                NAVIGATION_SCENE_CHANGE_V20R3_NAVIGATOR)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
                    NAVIGATION_SCENE_CHANGE_V20R3_NAVIGATOR)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R3_NAVIGATOR = "You are a robot navigation strategist. Output only JSON."
        _argmax = strategy.endswith("_argmax_split")
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
            navigator_prompt=None if _argmax else NAVIGATION_SCENE_CHANGE_V20R3_NAVIGATOR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            argmax_nav=_argmax,
            nav_score_weights=(1.0, 1.0, 0.5, 1.0),
            **(dict(fta_unblock=False) if _argmax else dict(fta_unblock=True, fta_mode="v15")),
        )
    elif strategy.startswith("agent_scene_change_v20r3_skip_") and strategy.endswith("_split"):
        # Same-dwell control family for v20r3, mirroring
        # agent_scene_change_v20r2_skip_<P>_split: v20r3's detector + dwell
        # own stay/move, a heuristic owns the destination. Re-runs the
        # when/where ablation on the new table so the navigator's marginal
        # value can be measured against the same 11 destination policies as
        # before.
        _SKIP_POLICIES_R3 = {
            "round_robin": lambda: _nav_policy_round_robin(rooms=log.rooms),
            "random": lambda: _nav_policy_random(seed=seed, rooms=log.rooms),
            "frequency": lambda: _nav_policy_frequency(seed=seed, source="perceived", rooms=log.rooms),
            "entropy": lambda: _nav_policy_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy": lambda: _nav_policy_beta_entropy(seed=seed, rooms=log.rooms),
            "beta_entropy_cost": lambda: _nav_policy_beta_entropy_cost(seed=seed, rooms=log.rooms),
            "greedy_hazard": lambda: _nav_policy_greedy_hazard(seed=seed, rooms=log.rooms),
            "active_mapping": lambda: _nav_policy_frontier(seed=seed, rooms=log.rooms),
            "staleness_only": lambda: _nav_policy_staleness(seed=seed, rooms=log.rooms),
        }
        _pol = strategy[len("agent_scene_change_v20r3_skip_"):-len("_split")]
        if _pol not in _SKIP_POLICIES_R3:
            raise ValueError(f"Unknown skip-policy in strategy {strategy!r}; expected one of {sorted(_SKIP_POLICIES_R3)}")
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR = "You are a scene-change detector. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
            navigator_prompt=None,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            fta_unblock=False,
            nav_score_weights=(1.0, 1.0, 0.5, 1.0),
            skip_nav_policy=_SKIP_POLICIES_R3[_pol](),
        )
    elif strategy in ("agent_scene_change_v20r2_split_nohot",
                      "agent_scene_change_v20r2_split_noexp",
                      "agent_scene_change_v20r2_split_nostale",
                      "agent_scene_change_v20r2_split_noscore"):
        # No-mention term ablation of v20r2_split: the ablated objective is
        # BOTH zero-weighted in the score AND absent from the navigator
        # prompt and the table "why" note (the note names only active
        # terms; _compute_room_scores enforces this). Stronger ablation
        # than --score-weights alone, which still mentions the term.
        # _noscore removes ALL three terms (table all zeros, why-note
        # sentence dropped from the prompt entirely).
        try:
            from bordsupr.frontend.agent.prompts import (
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT,
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOEXP,
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSTALE,
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSCORE)
        except ImportError:
            try:
                from agent.prompts import (
                    NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
                    NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT,
                    NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOEXP,
                    NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSTALE,
                    NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSCORE)
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT = "You are a robot navigation strategist. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOEXP = NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSTALE = NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT
                NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSCORE = NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT
        _abl = strategy.rsplit("_", 1)[-1]  # nohot / noexp / nostale / noscore
        _nav_prompt = {"nohot": NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOHOT,
                       "noexp": NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOEXP,
                       "nostale": NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSTALE,
                       "noscore": NAVIGATION_SCENE_CHANGE_V20R2_NAVIGATOR_NOSCORE}[_abl]
        _weights = {"nohot": (0.0, 1.0, 0.5),
                    "noexp": (1.0, 0.0, 0.5),
                    "nostale": (1.0, 1.0, 0.0),
                    "noscore": (0.0, 0.0, 0.0)}[_abl]
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_V20R2_DETECTOR,
            navigator_prompt=_nav_prompt,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            skip_mode="interesting",
            obs_history_size=4,
            detector_max_tokens=8192,
            fta_unblock=True, fta_mode="v15",
            nav_score_weights=_weights,
        )
    elif strategy == "agent_scene_change_v9_split_nocur":
        # v9_split_nocur: strict split, but the current room is NOT passed in
        # the ROOM RANKING table — a top-ranked current room invites
        # self-target "moves" that never execute (stuck-agent failure mode).
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR_NOCUR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR_NOCUR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR_NOCUR = "You are a robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMNavSplit(
            log, strategy_type=strategy,
            detector_prompt=NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_V9_SPLIT_NAVIGATOR_NOCUR,
            seed=seed, no_tools=no_tools, detector_type="agent_scene_change_v8",
            exclude_current_room_from_ranking=True,
        )
    elif strategy == "agent_scene_change_v8_short":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SHORT
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SHORT
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_SHORT = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_short", system_prompt=NAVIGATION_SCENE_CHANGE_V8_SHORT, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_quick":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_QUICK
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_QUICK
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_QUICK = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_quick", system_prompt=NAVIGATION_SCENE_CHANGE_V8_QUICK, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_clean":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_CLEAN
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_CLEAN
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_CLEAN = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_clean", system_prompt=NAVIGATION_SCENE_CHANGE_V8_CLEAN, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_active":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_ACTIVE
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_ACTIVE
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_ACTIVE = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_active", system_prompt=NAVIGATION_SCENE_CHANGE_V8_ACTIVE, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_smart":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SMART
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SMART
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_SMART = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_smart", system_prompt=NAVIGATION_SCENE_CHANGE_V8_SMART, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v9":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V9
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V9
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V9 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v9", system_prompt=NAVIGATION_SCENE_CHANGE_V9, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v10":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V10
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V10
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V10 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v10", system_prompt=NAVIGATION_SCENE_CHANGE_V10, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v11":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V11
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V11
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V11 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v11", system_prompt=NAVIGATION_SCENE_CHANGE_V11, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_no_tools":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_NO_TOOLS
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_NO_TOOLS
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_NO_TOOLS = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_no_tools", system_prompt=NAVIGATION_SCENE_CHANGE_V8_NO_TOOLS, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_v8_short_no_tools":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SHORT_NO_TOOLS
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8_SHORT_NO_TOOLS
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8_SHORT_NO_TOOLS = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLM(log, strategy_type="agent_scene_change_v8_short_no_tools", system_prompt=NAVIGATION_SCENE_CHANGE_V8_SHORT_NO_TOOLS, seed=seed, no_tools=no_tools)
    elif strategy == "agent_scene_change_split":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_SPLIT_NAVIGATOR
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR, NAVIGATION_SCENE_CHANGE_SPLIT_NAVIGATOR
            except ImportError:
                NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR = "You are a scene-change detector. Output only JSON."
                NAVIGATION_SCENE_CHANGE_SPLIT_NAVIGATOR = "You are a navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMSplit(
            log,
            strategy_type="agent_scene_change_split",
            detector_prompt=NAVIGATION_SCENE_CHANGE_SPLIT_DETECTOR,
            navigator_prompt=NAVIGATION_SCENE_CHANGE_SPLIT_NAVIGATOR,
            seed=seed,
        )
    # --- Dynamic ablation: detector + navigator ---
    elif strategy.startswith("ablation:"):
        parts = strategy.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid ablation strategy: {strategy}. Expected ablation:<detector>:<navigator>")
        _, detector, navigator = parts
        # "_pure" suffix on the navigator: the policy controls stay/move AND
        # destination (bare names keep the VLM-gated flavor).
        policy_controls_action = False
        if navigator.endswith("_pure"):
            navigator = navigator[: -len("_pure")]
            if navigator == "native":
                raise ValueError(f"Invalid ablation strategy: {strategy}. 'native' has no nav policy and cannot be combined with '_pure'")
            policy_controls_action = True
        prompt = _get_detector_prompt(detector, no_tools=no_tools)
        nav_policy = _get_nav_policy(navigator, seed=seed, rooms=log.rooms)
        sim = _SteppableSceneChangeVLMHybridNav(
            log,
            strategy_type=strategy,
            system_prompt=prompt,
            seed=seed,
            nav_policy=nav_policy,
            no_tools=no_tools,
            policy_controls_action=policy_controls_action,
            detector_type=detector,
        )
    # --- VLM with random target (uses VLM stay/move decision, random destination) ---
    elif strategy == "agent_scene_change_v8_random_target":
        try:
            from bordsupr.frontend.agent.prompts import NAVIGATION_SCENE_CHANGE_V8
        except ImportError:
            try:
                from agent.prompts import NAVIGATION_SCENE_CHANGE_V8
            except ImportError:
                NAVIGATION_SCENE_CHANGE_V8 = "You are a scene-change-aware robot navigation strategist. Output only JSON."
        sim = _SteppableSceneChangeVLMHybridNav(
            log,
            strategy_type=strategy,
            system_prompt=NAVIGATION_SCENE_CHANGE_V8,
            seed=seed,
            nav_policy=None,
            random_target=True,
            no_tools=no_tools,
        )
    # --- Legacy hardcoded ablations (backward compat for saved files) ---
    elif strategy == "agent_scene_change_v5_nav_round_robin":
        prompt = _get_detector_prompt("agent_scene_change_v5", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_round_robin(rooms=log.rooms), no_tools=no_tools)
    elif strategy == "agent_scene_change_v5_nav_greedy":
        prompt = _get_detector_prompt("agent_scene_change_v5", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_greedy(rooms=log.rooms), no_tools=no_tools)
    elif strategy == "agent_scene_change_v5_nav_frequency":
        prompt = _get_detector_prompt("agent_scene_change_v5", no_tools=no_tools)
        # Legacy hardcoded ablation: keep the pre-§3.2 GT-count behavior so
        # saved files stay comparable.
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_frequency(seed=seed, source="gt", rooms=log.rooms), no_tools=no_tools)
    elif strategy == "agent_scene_change_v5_nav_perfect_oracle":
        prompt = _get_detector_prompt("agent_scene_change_v5", no_tools=no_tools)
        sim = _SteppableSceneChangeVLMHybridNav(log, strategy_type=strategy, system_prompt=prompt, seed=seed, nav_policy=_nav_policy_perfect_oracle(rooms=log.rooms), no_tools=no_tools)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Passthrough knobs (§0.3 contract flags) applied uniformly to all
    # VLM-backed steppables. The attributes are consumed at step time, so
    # setting them here is equivalent to threading them through every
    # factory branch.
    if isinstance(sim, _SteppableSceneChangeVLMBase):
        if dwell_config is not None:
            sim.dwell_config = dwell_config
        sim.initial_pick = initial_pick
        # greedy_hazard internally always uses the §4.2-CORE fixed hazard
        # computation; hazard_fix=True opts any other strategy in as well.
        # v10_split forces it too — its room scores read the hazard signal.
        sim.hazard_fix = (bool(hazard_fix) or "v10_split" in strategy
                          or strategy in ("greedy_hazard", "greedy_hazard_pure", "rank_greedy_pure"))
        # §4.1: oracle_tools suffix forces GT counters; otherwise the
        # caller's signal_source (default "perceived") applies. The label
        # also flows into strategy_type so exported results stay
        # distinguishable from their perceived counterparts (the suffix
        # contains no magic substring, so prompt/dwell flags are unaffected).
        sim.signal_source = "gt" if oracle_tools else signal_source
        if oracle_tools:
            sim.strategy_type = strategy_label
        # §15 curiosity-signal ablation: masked tools unavailable to the agent.
        if mask_tools:
            sim.mask_tools = frozenset(mask_tools)
        # v9 forcing knobs: opt any VLM strategy into code-enforced moves.
        if anti_camping:
            sim.anti_camping = True
        if move_budget:
            sim.move_budget = int(move_budget)
        # Zero-out/weight ablation: override the navigator score weights
        # (w_hot/w_exp/w_stale). The base class reads self.score_weights in
        # _compute_room_scores; the NavSplit family reads the
        # self.nav_score_weights tuple — set both so either consumer sees
        # the override. Applied post-construction, works on any strategy.
        if score_weights is not None:
            w = dict(score_weights)
            if hasattr(sim, "score_weights"):
                sim.score_weights = w
            if hasattr(sim, "nav_score_weights"):
                _cur = tuple(getattr(sim, "nav_score_weights", ()) or ())
                _cur_int = float(_cur[3]) if len(_cur) > 3 else 0.0
                sim.nav_score_weights = (float(w.get("w_hot", 1.0)),
                                         float(w.get("w_exp", 1.0)),
                                         float(w.get("w_stale", 0.5)),
                                         float(w.get("w_int", _cur_int)))

    # Step through actual timestamps in the dataset
    sim_start = time.time()
    total_steps = 0
    for current_time in log.unique_timestamps:
        total_steps += 1

        # Pre-step heartbeat: tell the UI we are about to process this step
        # so slow strategies (e.g. v6) don’t appear stuck while the VLM call is in flight.
        if progress_callback is not None:
            progress_callback({
                "type": "progress",
                "strategy": strategy,
                "steps_processed": total_steps,
                "total_steps": len(log.unique_timestamps),
                "current_room": sim.current_room,
                "changes_so_far": sim.observed_changes,
                "step": None,
            })

        sim.step(current_time)

        if progress_callback is not None:
            latest_step = sim.minute_trace[-1] if sim.minute_trace else None
            progress_callback({
                "type": "progress",
                "strategy": strategy,
                "steps_processed": total_steps,
                "total_steps": len(log.unique_timestamps),
                "current_room": sim.current_room,
                "changes_so_far": sim.observed_changes,
                "step": latest_step,
            })

    result = sim.finalize()
    result.elapsed_seconds = round(time.time() - sim_start, 1)
    # Reproduction snapshot (§0.3.3): enough to re-run this result bit-for-bit.
    result.config_snapshot = {
        "strategy": strategy_label,
        "seed": seed,
        "no_tools": no_tools,
        "dwell_config": getattr(sim, "dwell_config", None),
        "detector_type": getattr(sim, "detector_type", None),
        "policy_controls_action": getattr(sim, "policy_controls_action", False),
        "hazard_fix": getattr(sim, "hazard_fix", False),
        "initial_pick": getattr(sim, "initial_pick", None),
        "signal_source": getattr(sim, "signal_source", None),
        "mask_tools": sorted(getattr(sim, "mask_tools", frozenset())),
        "anti_camping": getattr(sim, "anti_camping", False),
        "move_budget": getattr(sim, "move_budget", 0),
        "score_weights": getattr(sim, "score_weights", None),
        "staleness_deadband": getattr(sim, "staleness_deadband", None),
        "nav_walk_times": getattr(sim, "nav_walk_times", None),
        "nav_scene_context": getattr(sim, "nav_scene_context", None),
        "nav_score_weights": list(getattr(sim, "nav_score_weights", [])),
        "skip_on_detection": getattr(sim, "skip_on_detection", None),
        "skip_mode": getattr(sim, "skip_mode", None),
        "exclude_current_room_from_ranking": getattr(sim, "exclude_current_room_from_ranking", False),
        "obs_history_size": getattr(sim, "obs_history_size", 0),
        "obs_history_include_caption": getattr(sim, "obs_history_include_caption", False),
        "fta_unblock": getattr(sim, "fta_unblock", False),
        "cross_room_obs": getattr(sim, "cross_room_obs", False),
        "argmax_nav": getattr(sim, "argmax_nav", False),
        "skip_nav_policy": getattr(sim, "_skip_policy_name", None),
        "fta_mode": getattr(sim, "fta_mode", None),
        "detector_max_tokens": getattr(sim, "detector_max_tokens", None),
        "disable_thinking": os.environ.get("VLM_DISABLE_THINKING", "").lower() in ("1", "true", "yes"),
        "data_dir": str(log.data_dir),
        "simulator_sha256": _simulator_sha256(),
    }
    return result


def run_scene_change_round_robin(
    strategies: list[str] | None = None,
    seed: int | None = 42,
    no_tools: bool = False,
) -> dict[str, SceneChangeSimulationResult]:
    """Run multiple strategies and return comparison dict."""
    if strategies is None:
        strategies = ["fixed_10min", "random", "frequency", "greedy_oracle_1step", "agent_scene_change_v8", "agent_scene_change_v8_short", "agent_scene_change_v8_quick", "agent_scene_change_v8_clean", "agent_scene_change_v8_active", "agent_scene_change_v8_smart", "agent_scene_change_v9", "agent_scene_change_v10", "agent_scene_change_v11"]
    return {
        s: run_scene_change_simulation(s, seed=seed, no_tools=no_tools)
        for s in strategies
    }


# ---------------------------------------------------------------------------
# CLI / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Scene Change Simulator — Quick Test")
    print("=" * 60)

    strategies = ["fixed_10min", "random", "frequency", "greedy_oracle_1step", "agent_scene_change_v8", "agent_scene_change_v8_short", "agent_scene_change_v8_quick", "agent_scene_change_v8_clean", "agent_scene_change_v8_active", "agent_scene_change_v8_smart", "agent_scene_change_v9", "agent_scene_change_v10", "agent_scene_change_v11"]
    results = run_scene_change_round_robin(strategies, seed=42)

    print(f"\n{'Strategy':25s} | {'ChgRecall':9s} | {'ChgAcc':6s} | {'ActAcc':6s} | {'NavPrec':7s} | {'Visits':6s} | {'Coverage':8s}")
    print("-" * 75)
    for s in strategies:
        r = results[s]
        print(f"{s:25s} | {r.change_recall:9.4f} | {r.scene_change_accuracy:6.4f} | {r.activity_change_accuracy:6.4f} | {r.navigation_precision:7.4f} | {r.total_visits:6d} | {r.exploration_coverage:8.4f}")

    print("\nScene-change detection confusion matrices:")
    for s in strategies:
        r = results[s]
        print(f"  {s:25s} : TP={r.scene_change_tp:3d} TN={r.scene_change_tn:3d} FP={r.scene_change_fp:3d} FN={r.scene_change_fn:3d}")
