#!/usr/bin/env python3
"""Run scene-change-aware strategy evaluation and generate a comparison report.

Usage:
    python scripts/run_scene_change_eval.py
    python scripts/run_scene_change_eval.py --strategies fixed_10min random frequency greedy agent_scene_change
    python scripts/run_scene_change_eval.py --output reports/scene_change_eval.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "curiosity"))

from scene_change_simulator import run_scene_change_round_robin, SceneChangeSimulationResult


def result_to_dict(r: SceneChangeSimulationResult) -> dict:
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
        "scene_change_tp": r.scene_change_tp,
        "scene_change_tn": r.scene_change_tn,
        "scene_change_fp": r.scene_change_fp,
        "scene_change_fn": r.scene_change_fn,
        "scene_change_accuracy": round(r.scene_change_accuracy, 4),
        "navigation_to_changed_room": r.navigation_to_changed_room,
        "total_moves": r.total_moves,
        "navigation_precision": round(r.navigation_precision, 4),
        "exploration_coverage": round(r.exploration_coverage, 4),
        "error_steps": getattr(r, "error_steps", 0),
        "error_rate": round(getattr(r, "error_rate", 0.0), 4),
        "overridden_moves": getattr(r, "overridden_moves", 0),
        "vlm_calls_made": r.vlm_calls_made,
        "elapsed_seconds": r.elapsed_seconds,
        "config_snapshot": getattr(r, "config_snapshot", {}),
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
        "minute_trace": [
            {
                "time": m.time_str,
                "room": m.room,
                "gt_change": m.gt_change_type,
                "vlm_detected": m.vlm_detected_change,
                "vlm_change": m.vlm_change,
                "action": m.action,
                "target_room": m.target_room,
                "scene": m.scene_text,
                "people": m.people_present,
                "objects": m.objects_present,
            }
            for m in r.minute_trace
        ],
    }


def print_table(results: dict[str, SceneChangeSimulationResult]):
    print("\n" + "=" * 90)
    print(f"{'Strategy':20s} | {'ChgRecall':9s} | {'ChgAcc':6s} | {'NavPrec':7s} | {'Visits':6s} | {'Coverage':8s} | {'VLMCalls':8s}")
    print("-" * 90)
    for s in sorted(results.keys()):
        r = results[s]
        print(
            f"{s:20s} | {r.change_recall:9.4f} | {r.scene_change_accuracy:6.4f} | "
            f"{r.navigation_precision:7.4f} | {r.total_visits:6d} | {r.exploration_coverage:8.4f} | {r.vlm_calls_made:8d}"
        )

    print("\nScene-change detection confusion matrices:")
    for s in sorted(results.keys()):
        r = results[s]
        print(
            f"  {s:20s}: TP={r.scene_change_tp:3d} TN={r.scene_change_tn:3d} "
            f"FP={r.scene_change_fp:3d} FN={r.scene_change_fn:3d} "
            f"Accuracy={r.scene_change_accuracy:.4f}"
        )

    print("\nNavigation quality:")
    for s in sorted(results.keys()):
        r = results[s]
        print(
            f"  {s:20s}: moves_to_changed={r.navigation_to_changed_room:3d} / {r.total_moves:3d} "
            f"({r.navigation_precision:.4f})"
        )

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Run scene-change strategy evaluation.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["fixed_10min", "random", "frequency", "greedy", "agent_scene_change"],
        help="Strategies to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    print("Running scene-change strategy evaluation...")
    print(f"Strategies: {args.strategies}")
    print(f"Seed: {args.seed}")

    results = run_scene_change_round_robin(args.strategies, seed=args.seed)
    print_table(results)

    # Save to JSON
    output_data = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "strategies": args.strategies,
        "results": {s: result_to_dict(r) for s, r in results.items()},
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, indent=2, fp=f)
        print(f"\nResults saved to {output_path}")
    else:
        default_path = PROJECT_ROOT / "reports" / f"scene_change_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        with open(default_path, "w", encoding="utf-8") as f:
            json.dump(output_data, indent=2, fp=f)
        print(f"\nResults saved to {default_path}")


if __name__ == "__main__":
    main()
