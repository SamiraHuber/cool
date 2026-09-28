#!/usr/bin/env python3
"""
Seed-sweep runner with per-run checkpointing.

Usage:
    python scripts/run_seed_sweep.py --seeds 0 1 2 3 4

    # Background:
    nohup python scripts/run_seed_sweep.py --seeds 0 1 2 3 4 > seed_sweep.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


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
        # Agent-relative state-diff metrics (visible/expired changes, latency)
        "sd_visible_instances": getattr(r, "sd_visible_instances", 0),
        "sd_detected": getattr(r, "sd_detected", 0),
        "sd_visible_recall": round(getattr(r, "sd_visible_recall", 0.0), 4),
        "sd_blind": getattr(r, "sd_blind", 0),
        "sd_unvisited": getattr(r, "sd_unvisited", 0),
        "sd_expired_changes": getattr(r, "sd_expired_changes", 0),
        "sd_false_positives": getattr(r, "sd_false_positives", 0),
        "sd_fp_rate_per_step": round(getattr(r, "sd_fp_rate_per_step", 0.0), 4),
        "sd_avg_latency_steps": getattr(r, "sd_avg_latency_steps", 0.0),
        "sd_median_latency_steps": getattr(r, "sd_median_latency_steps", 0.0),
        "sd_max_latency_steps": getattr(r, "sd_max_latency_steps", 0),
        "sd_latency_histogram": getattr(r, "sd_latency_histogram", {}),
        "sd_cumulative_detection_curve": getattr(r, "sd_cumulative_detection_curve", {}),
        "sd_per_room": getattr(r, "sd_per_room", {}),
        # Rank-divergence tracking: how often the VLM navigator deviated
        # from plain argmax of its own ROOM RANKING table (NavSplit only).
        "rank_greedy_calls": getattr(r, "rank_greedy_calls", 0),
        "rank_greedy_divergences": getattr(r, "rank_greedy_divergences", 0),
        "rank_greedy_override_moves": getattr(r, "rank_greedy_override_moves", 0),
        "rank_divergence_rate": round(getattr(r, "rank_divergence_rate", 0.0), 4),
        "rank_divergence_log": getattr(r, "rank_divergence_log", []),
        # Top-K agreement: histogram of where the chosen target landed in the
        # ROOM RANKING table + top1/top3/top5/stay/off-table rates.
        "rank_target_histogram": getattr(r, "rank_target_histogram", {}),
        "rank_topk_rates": getattr(r, "rank_topk_rates", {}),
        "path_max_changes": getattr(r, "path_max_changes", 0),
        "theoretical_max_changes": getattr(r, "theoretical_max_changes", 0),
        "detection_efficiency": round(getattr(r, "detection_efficiency", 0.0), 4),
        "navigation_efficiency": round(getattr(r, "navigation_efficiency", 0.0), 4),
        "normalized_recall": round(getattr(r, "normalized_recall", 0.0), 4),
        "visits": [
            {"visit": v.visit_number, "room": v.room, "start": v.start.strftime("%H:%M"),
             "end": v.end.strftime("%H:%M"), "changes_observed": v.changes_observed,
             "dwell_seconds": v.dwell_seconds}
            for v in r.visits
        ],
    }


def _get_reports_dir() -> Path:
    file_parent = Path(__file__).resolve().parent
    if file_parent.name == "scripts":
        return file_parent.parent / "reports"
    return Path("/app/reports")


def _checkpoint_path(reports_dir: Path, strategy: str, seed: int) -> Path:
    safe_strategy = strategy.replace(":", "_")
    return reports_dir / f"seed_sweep_{safe_strategy}_seed{seed}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    strategies = [
        ("agent_scene_change_caption", "Caption-Aware"),
        ("agent_scene_change_visual", "Visual Comparison"),
        ("agent_scene_change_visual_no_tools", "Visual No-Tools"),
        ("agent_scene_change_v8_no_tools", "V8 No-Tools"),
        ("agent_scene_change_v8", "V8 Confidence Adaptive"),
        ("ablation:agent_scene_change_v8:round_robin", "V8 + Round-Robin (Naive)"),
    ]

    seeds = args.seeds
    total_runs = len(strategies) * len(seeds)

    print(f"Seed sweep: {len(strategies)} strategies × {len(seeds)} seeds = {total_runs} runs")
    print("-" * 60)

    project_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [project_root / "curiosity", Path("/app") / "curiosity", Path("/workspace") / "curiosity"]
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break
    # Repo root on sys.path: required so the factory can import the real
    # prompts from bordsupr.frontend.agent.prompts (see run_grid_sweep.py).
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scene_change_simulator import run_scene_change_simulation, SceneChangeTimeLog

    try:
        _sc_log = SceneChangeTimeLog()
        theoretical_max_recall = round(_sc_log.theoretical_max_recall, 4)
    except Exception:
        theoretical_max_recall = 0.0

    reports_dir = _get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    skipped = 0
    errors = 0

    for strat, label in strategies:
        for seed in seeds:
            cp = _checkpoint_path(reports_dir, strat, seed)
            if cp.exists():
                print(f"[SKIP] {label} | seed={seed} — already exists: {cp.name}")
                skipped += 1
                continue

            print(f"[RUN ] {label} | seed={seed} ... ", end="", flush=True)
            t0 = time.time()
            try:
                result = run_scene_change_simulation(strat, seed=seed)
                data = {
                    "run_timestamp": datetime.now(timezone.utc).isoformat(),
                    "strategy": strat,
                    "label": label,
                    "seed": seed,
                    "theoretical_max_recall": theoretical_max_recall,
                    "result": _result_to_dict(result, theoretical_max_recall),
                }
                with open(cp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                elapsed = time.time() - t0
                print(f"done in {elapsed:.1f}s | TP={result.scene_change_tp} Recall={result.change_recall:.3f}")
                completed += 1
            except Exception as exc:
                elapsed = time.time() - t0
                print(f"ERROR after {elapsed:.1f}s: {type(exc).__name__}: {exc}")
                errors += 1

    print("-" * 60)
    print(f"Completed: {completed} | Skipped: {skipped} | Errors: {errors} | Total: {total_runs}")
    print(f"Checkpoints saved to: {reports_dir}")


if __name__ == "__main__":
    main()
