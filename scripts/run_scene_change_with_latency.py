#!/usr/bin/env python3
"""Run scene-change strategies with full latency histogram output.

Usage:
    python3 scripts/run_scene_change_with_latency.py \
        --dataset-dir data/curiosity/datasets/adversarial \
        --strategies ablation:agent_scene_change_v8:frequency ablation:agent_scene_change_v8:native \
        --seed 45 \
        --output reports/adversarial_with_latency_seed45.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "curiosity"))

from scene_change_simulator import run_scene_change_simulation, SceneChangeTimeLog


def _result_to_full_dict(result, theoretical_max_recall: float) -> dict:
    """Serialize full result including all latency fields."""
    r = result
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
        "config_snapshot": getattr(r, "config_snapshot", {}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory")
    parser.add_argument("--strategies", nargs="+", required=True, help="Strategies to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir)
    if not data_dir.exists():
        print(f"ERROR: Dataset dir not found: {data_dir}")
        sys.exit(1)

    # Temporarily override the default data dir
    import scene_change_simulator as sim_mod
    original_dir = sim_mod.SCENE_CHANGE_DATA_DIR
    sim_mod.SCENE_CHANGE_DATA_DIR = data_dir

    log = SceneChangeTimeLog(data_dir)
    theoretical_max_recall = round(log.theoretical_max_recall, 4)
    print(f"Dataset: {data_dir}")
    print(f"Total changes: {log.total_changes}")
    print(f"Theoretical max recall: {theoretical_max_recall}")
    print(f"Strategies: {args.strategies}")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    results = {}
    errors = {}
    import time
    start = time.time()

    for strat in args.strategies:
        t0 = time.time()
        print(f"Running {strat} ... ", end="", flush=True)
        try:
            result = run_scene_change_simulation(strat, seed=args.seed)
            results[strat] = _result_to_full_dict(result, theoretical_max_recall)
            elapsed = time.time() - t0
            print(f"done in {elapsed:.1f}s | recall={result.change_recall:.3f} tp={result.scene_change_tp}")
        except Exception as exc:
            elapsed = time.time() - t0
            errors[strat] = f"{type(exc).__name__}: {exc}"
            print(f"ERROR after {elapsed:.1f}s: {errors[strat]}")

    total_elapsed = time.time() - start

    output = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_dir": str(data_dir),
        "seed": args.seed,
        "strategies": args.strategies,
        "theoretical_max_recall": theoretical_max_recall,
        "total_elapsed_seconds": round(total_elapsed, 2),
        "results": results,
        "errors": errors,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Total time: {total_elapsed:.1f}s")

    # Restore original dir
    sim_mod.SCENE_CHANGE_DATA_DIR = original_dir


if __name__ == "__main__":
    main()
