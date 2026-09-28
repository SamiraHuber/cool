#!/usr/bin/env python3
"""
Run scene-change ablation experiments overnight.

For each detector, pairs it with every baseline navigator (including its own
native navigation) so you can see how much navigation policy matters vs the
detector itself.

Usage:
    python scripts/run_scene_change_ablations.py --seed 42

    # Run in background overnight:
    nohup python scripts/run_scene_change_ablations.py --seed 42 > ablation.log 2>&1 &

Output:
    Saves a timestamped JSON file to ./reports/ (host) / /app/reports (container).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Detectors to ablate
# ---------------------------------------------------------------------------

DETECTORS: list[tuple[str, str]] = [
    ("agent_scene_change_v5", "V5 (Strict Entity)"),
    ("agent_scene_change_v6", "V6 (Strict Entity + Free Nav)"),
    ("agent_scene_change_history", "History-Guided"),
    ("agent_scene_change_split", "Split Agent (Detector + Navigator)"),
]

# ---------------------------------------------------------------------------
# Navigators to pair with each detector
# ---------------------------------------------------------------------------

NAVIGATORS: list[tuple[str, str]] = [
    ("native", "Agent Native Navigation"),
    ("round_robin", "Round Robin"),
    ("greedy", "Greedy"),
    ("frequency", "Frequency"),
    ("entropy", "Entropy (Info-Gain)"),
    ("random", "Random"),
    ("perfect_oracle", "Perfect Oracle"),
]


# ---------------------------------------------------------------------------
# Result serialisation (mirrors the web UI _to_dict)
# ---------------------------------------------------------------------------

def _result_to_dict(r, theoretical_max_recall: float) -> dict:
    return {
        "strategy": r.strategy,
        "total_changes": r.total_changes,
        "total_major_changes": r.total_major_changes,
        "observed_changes": r.observed_changes,
        "total_visits": r.total_visits,
        "visits_with_changes": r.visits_with_changes,
        "change_recall": round(r.change_recall, 4),
        "change_precision": round(r.change_precision, 4),
        "change_f1": round(r.change_f1, 4),
        "hit_rate": round(r.hit_rate, 4),
        "presence_recall": round(getattr(r, "presence_recall", 0.0), 4),
        "detected_recall": round(getattr(r, "detected_recall", 0.0), 4),
        "detection_basis": getattr(r, "detection_basis", ""),
        "theoretical_max_recall": theoretical_max_recall,
        "scene_change_tp": r.scene_change_tp,
        "scene_change_tn": r.scene_change_tn,
        "scene_change_fp": r.scene_change_fp,
        "scene_change_fn": r.scene_change_fn,
        "scene_change_accuracy": round(r.scene_change_accuracy, 4),
        "no_change_accuracy": round(getattr(r, "no_change_accuracy", 0.0), 4),
        "change_accuracy": round(getattr(r, "change_accuracy", 0.0), 4),
        "minor_tp": getattr(r, "minor_tp", 0),
        "minor_fp": getattr(r, "minor_fp", 0),
        "minor_fn": getattr(r, "minor_fn", 0),
        "major_tp": getattr(r, "major_tp", 0),
        "major_fp": getattr(r, "major_fp", 0),
        "major_fn": getattr(r, "major_fn", 0),
        "minor_recall": round(getattr(r, "minor_recall", 0.0), 4),
        "minor_precision": round(getattr(r, "minor_precision", 0.0), 4),
        "major_recall": round(getattr(r, "major_recall", 0.0), 4),
        "major_precision": round(getattr(r, "major_precision", 0.0), 4),
        "activity_change_tp": getattr(r, "activity_change_tp", 0),
        "activity_change_tn": getattr(r, "activity_change_tn", 0),
        "activity_change_fp": getattr(r, "activity_change_fp", 0),
        "activity_change_fn": getattr(r, "activity_change_fn", 0),
        "activity_change_accuracy": round(getattr(r, "activity_change_accuracy", 0.0), 4),
        "navigation_to_changed_room": r.navigation_to_changed_room,
        "total_moves": r.total_moves,
        "navigation_precision": round(r.navigation_precision, 4),
        "exploration_coverage": round(r.exploration_coverage, 4),
        "error_steps": getattr(r, "error_steps", 0),
        "error_rate": round(getattr(r, "error_rate", 0.0), 4),
        "overridden_moves": getattr(r, "overridden_moves", 0),
        "vlm_calls_made": r.vlm_calls_made,
        "elapsed_seconds": r.elapsed_seconds,
        "cross_room_misses": getattr(r, "cross_room_misses", 0),
        "cross_room_miss_rate": round(getattr(r, "cross_room_miss_rate", 0.0), 4),
        "total_event_timesteps": getattr(r, "total_event_timesteps", 0),
        "event_timesteps_caught": getattr(r, "event_timesteps_caught", 0),
        "event_timestep_rate": round(getattr(r, "event_timestep_rate", 0.0), 4),
        "avg_detection_latency_minutes": getattr(r, "avg_detection_latency_minutes", 0.0),
        "median_detection_latency_minutes": getattr(r, "median_detection_latency_minutes", 0.0),
        "max_detection_latency_minutes": getattr(r, "max_detection_latency_minutes", 0.0),
        "changes_never_detected": getattr(r, "changes_never_detected", 0),
        "latency_histogram": getattr(r, "latency_histogram", {}),
        "detection_latency_histogram": getattr(r, "detection_latency_histogram", {}),
        "avg_detection_latency_steps": getattr(r, "avg_detection_latency_steps", 0.0),
        "median_detection_latency_steps": getattr(r, "median_detection_latency_steps", 0.0),
        "max_detection_latency_steps": getattr(r, "max_detection_latency_steps", 0),
        "changes_detected_immediately": getattr(r, "changes_detected_immediately", 0),
        "per_room_latency": getattr(r, "per_room_latency", {}),
        "cumulative_detection_curve": getattr(r, "cumulative_detection_curve", {}),
        "catchable_changes": getattr(r, "catchable_changes", 0),
        "detected_among_catchable": getattr(r, "detected_among_catchable", 0.0),
        "absent_changes": getattr(r, "absent_changes", 0),
        "blind_changes": getattr(r, "blind_changes", 0),
        "conditional_latency_histogram": getattr(r, "conditional_latency_histogram", {}),
        "visit_opportunity_histogram": getattr(r, "visit_opportunity_histogram", {}),
        "path_max_changes": getattr(r, "path_max_changes", 0),
        "theoretical_max_changes": getattr(r, "theoretical_max_changes", 0),
        "detection_efficiency": round(getattr(r, "detection_efficiency", 0.0), 4),
        "navigation_efficiency": round(getattr(r, "navigation_efficiency", 0.0), 4),
        "normalized_recall": round(getattr(r, "normalized_recall", 0.0), 4),
        "visits": [
            {
                "visit": v.visit_number,
                "room": v.room,
                "start": v.start.strftime("%H:%M"),
                "end": v.end.strftime("%H:%M"),
                "changes_observed": v.changes_observed,
                "dwell_seconds": v.dwell_seconds,
            }
            for v in r.visits
        ],
        "config_snapshot": getattr(r, "config_snapshot", {}),
    }


# ---------------------------------------------------------------------------
# Reports directory (same logic as the web UI)
# ---------------------------------------------------------------------------

def _get_reports_dir() -> Path:
    """Resolve the reports directory for both host and container environments."""
    file_parent = Path(__file__).resolve().parent
    if file_parent.name == "scripts":
        # Host layout: scripts/ -> project root is 1 level up
        return file_parent.parent / "reports"
    # Container layout fallback
    return Path("/app/reports")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run scene-change ablation experiments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--detectors",
        type=str,
        default="all",
        help='Comma-separated detector keys, or "all" (default: all)',
    )
    parser.add_argument(
        "--navigators",
        type=str,
        default="all",
        help='Comma-separated navigator keys, or "all" (default: all)',
    )
    parser.add_argument(
        "--include-standalone",
        action="store_true",
        help="Also run the standalone (full-agent) versions as baselines",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for result files",
    )
    args = parser.parse_args()

    # Determine detectors to run
    if args.detectors == "all":
        detectors = DETECTORS
    else:
        wanted = {k.strip() for k in args.detectors.split(",")}
        detectors = [(k, label) for k, label in DETECTORS if k in wanted]

    # Determine navigators to run
    if args.navigators == "all":
        navigators = NAVIGATORS
    else:
        wanted = {k.strip() for k in args.navigators.split(",")}
        navigators = [(k, label) for k, label in NAVIGATORS if k in wanted]

    # Build strategy list
    strategies: list[str] = []
    strategy_labels: dict[str, str] = {}

    if args.include_standalone:
        for det_key, det_label in detectors:
            strategies.append(det_key)
            strategy_labels[det_key] = det_label

    for det_key, det_label in detectors:
        for nav_key, nav_label in navigators:
            strat = f"ablation:{det_key}:{nav_key}"
            strategies.append(strat)
            strategy_labels[strat] = f"{det_label} + {nav_label}"

    print(f"Running {len(strategies)} strategies with seed={args.seed}")
    print(f"Detectors: {[l for _, l in detectors]}")
    print(f"Navigators: {[l for _, l in navigators]}")
    print("-" * 60)

    # Ensure exploration module is on path
    project_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        project_root / "curiosity",
        Path("/app") / "curiosity",
        Path("/workspace") / "curiosity",
    ]
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from scene_change_simulator import run_scene_change_simulation, SceneChangeTimeLog

    # Theoretical max recall (same for all strategies)
    try:
        _sc_log = SceneChangeTimeLog()
        theoretical_max_recall = round(_sc_log.theoretical_max_recall, 4)
    except Exception:
        theoretical_max_recall = 0.0

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    start_time = time.time()

    for idx, strat in enumerate(strategies, 1):
        label = strategy_labels.get(strat, strat)
        print(f"[{idx}/{len(strategies)}] {label} ... ", end="", flush=True)
        t0 = time.time()
        try:
            result = run_scene_change_simulation(strat, seed=args.seed)
            results[strat] = _result_to_dict(result, theoretical_max_recall)
            elapsed = time.time() - t0
            print(
                f"done in {elapsed:.1f}s | "
                f"TP={result.scene_change_tp} FN={result.scene_change_fn} "
                f"Recall={result.change_recall:.3f}"
            )
        except Exception as exc:
            elapsed = time.time() - t0
            errors[strat] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR after {elapsed:.1f}s: {errors[strat]}")

    total_elapsed = time.time() - start_time

    # Assemble output
    output_data = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "strategies": strategies,
        "strategy_labels": strategy_labels,
        "theoretical_max_recall": theoretical_max_recall,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "results": results,
        "errors": errors,
    }

    # Save
    reports_dir = Path(args.output_dir) if args.output_dir else _get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    file_path = reports_dir / f"scene_change_ablation_{timestamp}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print("-" * 60)
    print(f"Saved results to {file_path}")
    print(f"Total time: {total_elapsed:.1f}s")
    if errors:
        print(f"Errors: {len(errors)} / {len(strategies)}")
    else:
        print("All strategies completed successfully.")


if __name__ == "__main__":
    main()
