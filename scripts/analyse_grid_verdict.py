#!/usr/bin/env python3
"""§0.1 verdict table for grid-sweep checkpoints.

Per dataset x strategy: mean +/- std of the honest metrics —
detected_recall (PRIMARY), presence_recall (backward-comparable alias),
plus guardrails: total_moves, overridden_moves, error_rate, vlm_calls_made.
Paired per-seed delta of every strategy vs the COOL agent (agent_scene_change_v8).

Usage:
    python scripts/analyse_grid_verdict.py 'reports/grid_*.json' \
        --out reports/grid_verdict.md
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

PRIMARY = "detected_recall"
COOL = "agent_scene_change_v8"
ORACLES = {"greedy_oracle_1step", "perfect_oracle"}
METRICS = ["detected_recall", "presence_recall", "total_moves",
           "overridden_moves", "error_rate", "vlm_calls_made", "exploration_coverage"]


def _load(pattern: str) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(pattern)):
        d = json.load(open(p, encoding="utf-8"))
        r = d.get("result", d)
        r["_dataset"] = d.get("dataset", r.get("dataset", "?"))
        r["_seed"] = d.get("seed", r.get("seed"))
        r["_strategy"] = d.get("strategy", r.get("strategy"))
        r["_mask"] = "-".join(r.get("mask_tools") or d.get("mask_tools") or [])
        rows.append(r)
    return rows


def _mean_std(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n == 0:
        return 0.0, 0.0
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return m, var ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pattern")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = _load(args.pattern)
    if not rows:
        sys.exit(f"no files matched {args.pattern}")

    by_ds: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = r["_strategy"] + (f" [mask:{r['_mask']}]" if r["_mask"] else "")
        by_ds[r["_dataset"]][key].append(r)

    out = []
    for ds in sorted(by_ds):
        groups = by_ds[ds]
        out.append(f"\n## Dataset: {ds}  ({sum(len(v) for v in groups.values())} runs)\n")
        header = "| strategy | N | detected_recall (PRIMARY) | presence_recall | moves | overridden | err_rate | vlm_calls | coverage |"
        out.append(header)
        out.append("|" + "---|" * (header.count("|") - 1))
        cool_by_seed = {r["_seed"]: r.get(PRIMARY, 0.0) for r in groups.get(COOL, [])}
        deltas: dict[str, list[float]] = {}
        for strat in sorted(groups, key=lambda s: -_mean_std([r.get(PRIMARY, 0.0) for r in groups[s]])[0]):
            runs = groups[strat]
            cells = [strat, str(len(runs))]
            for m in METRICS:
                mean, std = _mean_std([r.get(m, 0.0) for r in runs])
                cells.append(f"{mean:.3f} ±{std:.3f}" if len(runs) > 1 else f"{mean:.3f}")
            out.append("| " + " | ".join(cells) + " |")
            base = strat.replace(" [mask:" + next((r["_mask"] for r in runs if r["_mask"]), "") + "]", "")
            if base not in ORACLES and strat != COOL:
                d = [r.get(PRIMARY, 0.0) - cool_by_seed[r["_seed"]]
                     for r in runs if r["_seed"] in cool_by_seed]
                if d:
                    deltas[strat] = d
        out.append(f"\nOracle ceiling rows (not competitors): {', '.join(sorted(ORACLES & set(groups)))}")
        out.append(f"\n### Paired per-seed delta vs {COOL} on {PRIMARY} (positive = beats COOL)\n")
        for strat, d in sorted(deltas.items(), key=lambda kv: -sum(kv[1])):
            m, s = _mean_std(d)
            out.append(f"- {strat}: {m:+.3f} ±{s:.3f}  (per-seed: {['%+.3f' % x for x in d]})")

    text = "\n".join(out)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
