#!/usr/bin/env python3
"""Full-grid sweep: datasets x strategies x seeds with per-run checkpointing.

Good-path Step 5 driver (REBUTTAL_GOOD_PATH_CURIOSITY_GUIDE.md). Reuses
_result_to_dict from run_seed_sweep.py and adds the §2/§3 metric fields
(presence_recall/detected_recall/error_rate/overridden_moves/config_snapshot).

Usage:
    python scripts/run_grid_sweep.py --seeds 0 1 2 3 4 --workers 4
    # resume: re-run the same command — existing checkpoints are skipped
    # ablation (Step 6):
    python scripts/run_grid_sweep.py --seeds 0 1 2 --strategies agent_scene_change_v8 \
        --mask-tools get_stale_rooms
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "curiosity"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
# Repo root itself: the simulator's factory imports prompts from
# bordsupr.frontend.agent.prompts and SILENTLY falls back to stub system
# prompts when that package is not importable (historically only true inside
# docker, where bordsupr is installed). A stub navigator system prompt loses
# the JSON contract → the agent never moves. Never remove this line.
sys.path.insert(0, str(PROJECT_ROOT))

from scene_change_simulator import SceneChangeTimeLog, run_scene_change_simulation  # noqa: E402
from run_seed_sweep import _result_to_dict  # noqa: E402

DATASETS = {
    "default": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "default",
    "regime": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "regime",
    "adversarial": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "adversarial",
    # v2 generator (generate_scene_change_dataset_v2.py), 6-room worlds —
    # structurally drop-in (same 1098-entry / 3-day / 6-room shape).
    "base_v2": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "base_v2",
    "distractor": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "distractor",
    "intent_cued": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "intent_cued",
    # v2 12-room scaled worlds (room universe is log-driven, §12-room fix).
    "scaled_12room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_12room",
    "scaled_base_12room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_base_12room",
    "scaled_distractor_12room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_distractor_12room",
    # v2 24-room scaled worlds (16 offices 1-4p, 3 meeting rooms, 40 people).
    "scaled_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_24room",
    "scaled_base_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_base_24room",
    "scaled_distractor_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_distractor_24room",
    # activity-level ablations of the best-performing configs (6/24-room):
    # _calm = globally less activity; _flat = flattened room-activity
    # distribution (no dominant hotspot, utility rooms quiet).
    "distractor_calm": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "distractor_calm",
    "distractor_flat": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "distractor_flat",
    "scaled_distractor_calm_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_distractor_calm_24room",
    "scaled_distractor_flat_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_distractor_flat_24room",
    # v2 24-room with heading cues + more intent episodes.
    "scaled_heading_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_heading_24room",
    # heading-cue variants at the other two scales (v2.2).
    "intent_cued_heading": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "intent_cued_heading",
    "scaled_heading_12room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "scaled_heading_12room",
    # agent-favoring 24-room worlds: flattened priors, more/spread cues,
    # distractor churn in utility rooms (config-only tuning of scaled_heading_24room).
    "agent_cued_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "agent_cued_24room",
    "agent_cued_flat_24room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "agent_cued_flat_24room",
    # same tuning at 6-room scale (extends intent_cued_heading).
    "agent_cued_6room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "agent_cued_6room",
    "agent_cued_flat_6room": PROJECT_ROOT / "data" / "curiosity" / "datasets" / "agent_cued_flat_6room",
}

# Deployable side of the §0.1 table; oracles are ceiling rows only.
# NOTE: agent_scene_change_visual/_caption from the historical seed-sweep
# list no longer exist in the factory — replaced by v8_clean/v8_active.
STRATEGIES = [
    "agent_scene_change_v8",
    "agent_scene_change_v9_antipcamp",       # v8 detector + code-enforced anti-camping
    "agent_scene_change_v9_budget",          # v8 detector + move budget (15 steps)
    "agent_scene_change_v9_split",           # Tier-2: split detector + score-table navigator
    "agent_scene_change_v9_split_soft",      # Tier-2 soft: navigator always called, follow-the-action
    "agent_scene_change_v9_split_nocur",     # Tier-2: strict split, current room excluded from ranking table
    "agent_scene_change_v10_split",          # Tier-2: V10 confidence detector + state machine + ranking navigator
    "agent_scene_change_v11_split",          # Tier-2 synthesis: soft detector+navigator, skip when interesting
    "agent_scene_change_v12_split",          # Tier-2 synthesis: v10 confidence detector + soft interestingness, ranked-table navigator
    "agent_scene_change_v13_split",          # Tier-2 synthesis: v12 + observation-history window + structured movement (leaving/destination/trend)
    "agent_scene_change_v13_split_caption",  # ablation: v13 with scene captions included in the history window
    "agent_scene_change_v14_split",          # v13 + capture fixes: FTA unblock, injected gap, bounded stale-memory, confidence floor, 2048 tok
    "agent_scene_change_v15_split",          # v14 + trend-gated FTA (winding-down only), dest!=current, top-half follow rule
    "agent_scene_change_v16_split",          # v15 + stale-memory DECISION TABLE (fixes boundary FNs: short-gap verb demand, 2-diff dead zone)
    "agent_scene_change_v17_split",          # v16 with qualitative (non-hardcoded) stale-memory thresholds — real-world transfer variant
    "agent_scene_change_v18_split",          # v17 + CONSISTENCY CHECK: scene_changed pinned to the model's own entity_diffs value
    "agent_scene_change_v19_split",          # v18 + check mirrored into reasoning instruction (say-the-count-out-loud enforcement)
    "agent_scene_change_v20_split",          # v19 with entity_diffs=1 line softened: judged by match confidence, not gap alone
    "agent_scene_change_v21_split",          # v20 + cross-room context: detector sees last observation of EVERY room
    "agent_scene_change_v22_split",          # v20, but no departure_destination — navigator infers the destination itself
    "agent_scene_change_v11r_split",         # redesigned lean baseline (PROMPTS_V11_VS_V20.md): free judgment + history window w/ captions, no FTA
    "agent_scene_change_v20r_split",         # redesigned trimmed v20 (same doc): confidence + structured movement + checklist, no noise/stale/consistency rules
    "agent_scene_change_v11r_det_v20r_nav_split",  # cross: lean detector + strict-rule navigator (no structured movement -> no FTA)
    "agent_scene_change_v20r_det_v11r_nav_split",  # cross: structured-movement detector + prose-rule navigator
    "agent_scene_change_v11r_argmax_split",        # dwell isolation: v11r detector + skip, argmax navigation (no navigator VLM)
    "agent_scene_change_v20r_argmax_split",        # dwell isolation: v20r detector + skip, argmax navigation (no navigator VLM)
    "agent_scene_change_v20r_skip_round_robin_split",      # skip-x-policy: v20r dwell + round-robin destination
    "agent_scene_change_v20r_skip_random_split",           # skip-x-policy: v20r dwell + random destination
    "agent_scene_change_v20r_skip_frequency_split",        # skip-x-policy: v20r dwell + frequency destination
    "agent_scene_change_v20r_skip_entropy_split",          # skip-x-policy: v20r dwell + entropy destination
    "agent_scene_change_v20r_skip_beta_entropy_split",     # skip-x-policy: v20r dwell + beta-entropy destination
    "agent_scene_change_v20r_skip_beta_entropy_cost_split",# skip-x-policy: v20r dwell + cost-discounted IG destination
    "agent_scene_change_v20r_skip_greedy_hazard_split",    # skip-x-policy: v20r dwell + hazard destination
    "agent_scene_change_v20r_skip_active_mapping_split",   # skip-x-policy: v20r dwell + frontier destination
    "agent_scene_change_v20r_skip_staleness_only_split",   # skip-x-policy: v20r dwell + staleness destination
    "agent_scene_change_v10_split_notime",   # ablation: navigator without WALK TIMES table
    "agent_scene_change_v10_split_scene",    # ablation: navigator with last-observed scene memory
    "agent_scene_change_v8_no_tools",
    "agent_scene_change_v8_clean",
    "agent_scene_change_v8_active",
    "round_robin_pure",
    "random_pure",
    "frequency",
    "frequency_pure",
    "entropy",
    "entropy_pure",
    "beta_entropy",
    "beta_entropy_pure",
    "greedy_hazard",
    "greedy_hazard_pure",
    "rank_greedy_pure",                      # ablation: VLM detection + pure argmax over the ROOM RANKING score the VLM navigators see
    "active_mapping",                # frontier-based exploration (Yamauchi 1997), VLM-gated
    "active_mapping_pure",           # textbook: frontier policy owns stay/move
    "staleness_only",                # bridge: COOL minus ownership signals (staleness only)
    "staleness_only_pure",
    "beta_entropy_cost",             # cost-discounted semantic IG: exp(H)/(1+walk min)
    "beta_entropy_cost_pure",
    "agent_scene_change_v8+oracle_tools",  # signal-quality upper bound
    "greedy_oracle_1step",                 # ceiling row, not a competitor
    "perfect_oracle",                      # ceiling row, not a competitor
]


def _checkpoint_path(reports_dir: Path, dataset: str, strategy: str, seed: int, mask_tag: str) -> Path:
    safe = strategy.replace(":", "_").replace("+", "_")
    return reports_dir / f"grid_{dataset}_{safe}{mask_tag}_seed{seed}.json"


def _extra_fields(r) -> dict:
    """§2/§3 metric fields missing from run_seed_sweep._result_to_dict."""
    return {
        "presence_recall": round(getattr(r, "presence_recall", 0.0), 4),
        "detected_recall": round(getattr(r, "detected_recall", 0.0), 4),
        "detection_basis": getattr(r, "detection_basis", ""),
        "error_steps": getattr(r, "error_steps", 0),
        "error_rate": round(getattr(r, "error_rate", 0.0), 4),
        "overridden_moves": getattr(r, "overridden_moves", 0),
        "config_snapshot": getattr(r, "config_snapshot", {}),
        # N2/N3/N5 navigation-quality metrics
        "move_value_k3": round(getattr(r, "move_value_k3", 0.0), 4),
        "move_value_k6": round(getattr(r, "move_value_k6", 0.0), 4),
        "wasted_dwell_rate": round(getattr(r, "wasted_dwell_rate", 0.0), 4),
        "coverage_halftime": getattr(r, "coverage_halftime", None),
        # Map-completeness metrics (active-mapping comparison)
        "coverage_milestones": getattr(r, "coverage_milestones", {}),
        "avg_mean_staleness_minutes": getattr(r, "avg_mean_staleness_minutes", 0.0),
        "visits_per_room_gini": round(getattr(r, "visits_per_room_gini", 0.0), 4),
        # Memory-quality metrics (people/interactions witnessed)
        "people_observed_steps": getattr(r, "people_observed_steps", 0),
        "people_observation_rate": round(getattr(r, "people_observation_rate", 0.0), 4),
        "interaction_cells_witnessed": getattr(r, "interaction_cells_witnessed", 0),
        "interaction_cells_total": getattr(r, "interaction_cells_total", 0),
        "interaction_witness_rate": round(getattr(r, "interaction_witness_rate", 0.0), 4),
    }


def _run_one(dataset: str, strategy: str, seed: int, reports_dir: Path,
             theoretical_max_recall: float, hazard_fix: bool,
             mask_tools: tuple[str, ...], mask_tag: str,
             score_weights: dict | None = None) -> str:
    cp = _checkpoint_path(reports_dir, dataset, strategy, seed, mask_tag)
    if cp.exists():
        return "skip"
    t0 = time.time()
    kwargs: dict = dict(seed=seed, data_dir=DATASETS[dataset], hazard_fix=hazard_fix)
    if mask_tools:
        kwargs["mask_tools"] = mask_tools
    if score_weights is not None:
        kwargs["score_weights"] = score_weights
    result = run_scene_change_simulation(strategy, **kwargs)
    result_dict = _result_to_dict(result, theoretical_max_recall)
    result_dict.update(_extra_fields(result))
    if mask_tools:
        result_dict["mask_tools"] = list(mask_tools)
    # Wrapper format matching run_seed_sweep.py checkpoints (analyse_seed_sweep.py).
    data = {
        "strategy": result.strategy,
        "seed": seed,
        "dataset": dataset,
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "result": result_dict,
    }
    tmp = cp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.rename(cp)  # atomic checkpoint
    return f"run ({time.time() - t0:.0f}s)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--datasets", type=str, default="all",
                        help='Comma-separated dataset keys (default,regime,adversarial) or "all"')
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                        help="Override strategy list (default: full §0.1 grid)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent sim runs; the VLM server batches (default: 1)")
    parser.add_argument("--hazard-fix", action="store_true", default=True,
                        help="Use the §4.2-fixed hazard model (default: on)")
    parser.add_argument("--no-hazard-fix", dest="hazard_fix", action="store_false")
    parser.add_argument("--mask-tools", type=str, default="",
                        help="Comma-separated tool names to mask (Step 6 ablation)")
    parser.add_argument("--score-weights", type=str, default="",
                        help="Navigator score-weight override 'w_hot,w_exp,w_stale' or "
                             "'w_hot,w_exp,w_stale,w_int' (e.g. '0,1,0.5' zeroes the "
                             "hotspot term; '1,1,0.5,2' doubles the v20r3 interaction "
                             "term). Tag is added to checkpoint filenames so variants "
                             "don't collide.")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    datasets = list(DATASETS) if args.datasets == "all" else [d.strip() for d in args.datasets.split(",")]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"Unknown dataset {d!r}; choose from {list(DATASETS)}")
        if not DATASETS[d].exists():
            raise SystemExit(f"Dataset dir missing: {DATASETS[d]}")
    strategies = args.strategies or STRATEGIES
    mask_tools = tuple(t.strip() for t in args.mask_tools.split(",") if t.strip())
    mask_tag = "_mask-" + "-".join(mask_tools) if mask_tools else ""

    score_weights = None
    if args.score_weights:
        _parts = [p.strip() for p in args.score_weights.split(",") if p.strip() != ""]
        if len(_parts) not in (3, 4):
            raise SystemExit(f"--score-weights must be 'w_hot,w_exp,w_stale[,w_int]' floats, got {args.score_weights!r}")
        try:
            _vals = [float(v) for v in _parts]
        except ValueError:
            raise SystemExit(f"--score-weights must be 'w_hot,w_exp,w_stale[,w_int]' floats, got {args.score_weights!r}")
        w_hot, w_exp, w_stale = _vals[:3]
        score_weights = {"w_hot": w_hot, "w_exp": w_exp, "w_stale": w_stale}
        mask_tag += f"_w{w_hot:g}-{w_exp:g}-{w_stale:g}"
        if len(_vals) == 4:
            # Only tag w_int when given, so existing 3-weight checkpoint
            # filenames stay byte-identical and are still resumed.
            score_weights["w_int"] = _vals[3]
            mask_tag += f"-i{_vals[3]:g}"

    reports_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # theoretical_max_recall depends only on the dataset log — compute once each
    tmr = {}
    for d in datasets:
        try:
            tmr[d] = round(SceneChangeTimeLog(data_dir=DATASETS[d]).theoretical_max_recall, 4)
        except Exception:
            tmr[d] = 0.0

    tasks = [(d, s, seed) for d in datasets for s in strategies for seed in args.seeds]
    print(f"Grid: {len(datasets)} datasets x {len(strategies)} strategies x {len(args.seeds)} seeds "
          f"= {len(tasks)} runs, workers={args.workers}, hazard_fix={args.hazard_fix}"
          + (f", mask={mask_tools}" if mask_tools else ""))

    done = skipped = errors = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_run_one, d, s, seed, reports_dir, tmr[d], args.hazard_fix,
                        mask_tools, mask_tag, score_weights): (d, s, seed)
            for d, s, seed in tasks
        }
        for fut in as_completed(futures):
            d, s, seed = futures[fut]
            try:
                outcome = fut.result()
            except Exception as exc:  # keep the grid going; failed runs leave no checkpoint
                errors += 1
                print(f"[ERR ] {d} | {s} | seed={seed}: {exc!r}", flush=True)
                continue
            if outcome == "skip":
                skipped += 1
            else:
                done += 1
            total_finished = done + skipped + errors
            if outcome != "skip" or total_finished == len(tasks):
                print(f"[{total_finished}/{len(tasks)}] {d} | {s} | seed={seed} -> {outcome}", flush=True)

    elapsed = (time.time() - t_start) / 60
    print(f"Grid done: {done} run, {skipped} skipped, {errors} errors in {elapsed:.1f} min")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
