#!/usr/bin/env python3
"""Detector-vs-navigator decomposition table for scene-change grid checkpoints.

Separates the agent's two jobs using only fields already present in the
checkpoints (no rerun needed):

  Job 2 (navigation)  -- opportunity recall (catchable/total), drive
                         (moves, coverage, absent), per-decision quality
                         (navigation_precision, hit_rate)
  Job 1 (detection)   -- detection given opportunity (detected_among_catchable),
                         first-visit detection rate, precision/F1, speed
                         (share detected at T=0 / <=T+3, median latency)
  Failure attribution -- blind (visited, not detected) vs absent (never
                         visited): every missed change is exactly one of these.

Forward-compatible: the N2/N3/N5 columns (move_value_k3/k6, wasted_dwell_rate,
coverage_halftime) and sd_* columns appear automatically once checkpoints
written after those metrics landed exist in the glob.

Usage:
    .venv-rebuttal/bin/python scripts/analyse_detector_navigator_split.py \
        'reports/grid_*.json' --out reports/detector_navigator_split.md
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics as st
from collections import defaultdict

# ---------------------------------------------------------------------------
# Metric definitions: (key, column header, compute(run_dict) -> float|None)
# ---------------------------------------------------------------------------

def _rate(num, den):
    return num / den if den else None


def _lat_share(hist: dict, pred) -> float | None:
    """Share of detected changes whose latency bin satisfies pred(t)."""
    total = 0
    hit = 0
    for bin_key, count in (hist or {}).items():
        m = re.match(r"T=(0)$|T\+(\d+)$", bin_key)
        if not m:
            continue
        t = int(m.group(1) or m.group(2))
        total += count
        if pred(t):
            hit += count
    return hit / total if total else None


METRICS = [
    # --- navigation: opportunity & drive ---
    ("opp_recall", "OppRec", lambda r: _rate(r.get("catchable_changes", 0), r.get("total_changes", 0))),
    ("coverage", "Cov", lambda r: r.get("exploration_coverage")),
    ("moves", "Moves", lambda r: r.get("total_moves")),
    # --- navigation: per-decision quality ---
    ("nav_precision", "NavPrec", lambda r: r.get("navigation_precision")),
    ("hit_rate", "HitR", lambda r: r.get("hit_rate")),
    ("catch_per_move", "Catch/Mv", lambda r: _rate(r.get("changes_detected_immediately", 0), r.get("total_moves", 0))),
    # --- detection given opportunity ---
    ("det_given_opp", "Det\\|Opp", lambda r: r.get("detected_among_catchable") if r.get("catchable_changes") else None),
    ("first_visit", "1stVis", lambda r: _rate(
        (r.get("visit_opportunity_histogram") or {}).get("V=0", 0),
        round((r.get("detected_among_catchable") or 0.0) * r.get("catchable_changes", 0)))),
    ("precision", "Prec", lambda r: r.get("change_precision")),
    ("f1", "F1", lambda r: r.get("change_f1")),
    ("major_recall", "MajR", lambda r: r.get("major_recall")),
    # --- speed (from latency histograms) ---
    ("lat_t0", "T=0%", lambda r: _lat_share(r.get("conditional_latency_histogram"), lambda t: t == 0)),
    ("lat_le3", "<=T+3%", lambda r: _lat_share(r.get("conditional_latency_histogram"), lambda t: t <= 3)),
    ("med_latency", "MedLat", lambda r: r.get("median_detection_latency_steps")),
    # --- old headline (presence-coupled) ---
    ("detected_recall", "DetRec", lambda r: r.get("detected_recall")),
    # --- failure attribution ---
    ("blind", "Blind", lambda r: r.get("blind_changes")),
    ("absent", "Absent", lambda r: r.get("absent_changes")),
    ("error_rate", "Err", lambda r: r.get("error_rate")),
    # --- N2/N3/N5 (present only in post-metric-landing checkpoints) ---
    ("move_value_k3", "MvVal3", lambda r: r.get("move_value_k3")),
    ("move_value_k6", "MvVal6", lambda r: r.get("move_value_k6")),
    ("wasted_dwell_rate", "WstDwl", lambda r: r.get("wasted_dwell_rate")),
    ("coverage_halftime", "CovHalf", lambda r: r.get("coverage_halftime")),
    # --- map completeness (active-mapping comparison) ---
    ("avg_mean_staleness_minutes", "MapStale", lambda r: r.get("avg_mean_staleness_minutes")),
    ("visits_per_room_gini", "VisGini", lambda r: r.get("visits_per_room_gini")),
    # --- memory quality (people/interactions witnessed) ---
    ("people_observation_rate", "PplObs", lambda r: r.get("people_observation_rate")),
    ("interaction_witness_rate", "IntWit", lambda r: r.get("interaction_witness_rate")),
    # --- sd_* (state-diff metrics) ---
    ("sd_visible_recall", "SdVisR", lambda r: r.get("sd_visible_recall")),
    ("sd_expired_changes", "SdExp", lambda r: r.get("sd_expired_changes")),
    ("sd_median_latency_steps", "SdMedL", lambda r: r.get("sd_median_latency_steps")),
]

FNAME_RE = re.compile(
    r"grid_(.+?)_((?:agent_scene_change|active_mapping|staleness_only|entropy|beta_entropy|greedy_hazard|greedy|"
    r"round_robin|frequency|perfect_oracle|heuristic)[A-Za-z0-9_-]*)_seed(\d+)$"
)


def _label(strategy: str) -> str:
    s = strategy.replace("agent_scene_change_", "")
    s = s.replace("mask-get_predicted_change_probability-get_time_since_last_change", "[-exp]")
    s = s.replace("mask-get_room_change_rates-get_room_visit_history", "[-hot]")
    s = s.replace("mask-get_stale_rooms", "[-stale]")
    s = s.replace(" [mask:", "[").replace("_[", "[")
    s = s.replace("+oracle_tools", "+oracle")
    return s


def load_rows(patterns: list[str]):
    """rows[(label, dataset)][metric_key] -> list of per-seed values."""
    rows: dict = defaultdict(lambda: defaultdict(list))
    n_files = 0
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            m = FNAME_RE.search(path.split("/")[-1].removesuffix(".json"))
            d = json.load(open(path))
            r = d.get("result", d)
            # filename is authoritative; some checkpoints lack `dataset`
            ds = m.group(1) if m else r.get("dataset")
            strat = m.group(2) if m else r.get("strategy")
            if not ds or not strat:
                continue
            label = _label(strat)
            n_files += 1
            cell = rows[(label, ds)]
            for key, _hdr, fn in METRICS:
                v = fn(r)
                if v is not None:
                    cell[key].append(v)
    return rows, n_files


def render(rows, n_files: int, datasets: list[str]) -> str:
    out = ["# Detector vs. Navigator Decomposition", ""]
    out.append(f"Source: {n_files} grid checkpoints. Means over seeds (+- std). "
               "OppRec = catchable/total (navigation gave itself the chance); "
               "Det|Opp = detected among catchable (late detection counts); "
               "1stVis = share detected on first post-change visit; "
               "T=0% / <=T+3% = share of detections within that latency; "
               "DetRec = live-catch recall (presence-coupled old headline); "
               "Blind = visited post-change but never detected (detector failure); "
               "Absent = never visited post-change (navigation failure).")
    # which optional columns actually have data anywhere
    all_keys = {k for cell in rows.values() for k in cell}
    metrics = [m for m in METRICS if m[0] in all_keys]
    for ds in datasets:
        ds_rows = [(label, cell) for (label, d), cell in rows.items() if d == ds]
        if not ds_rows:
            continue
        ds_rows.sort(key=lambda t: -st.mean(t[1].get("detected_recall", [0])))
        out += ["", f"## Dataset: {ds} ({len(ds_rows)} strategies)", ""]
        header = "| strategy |" + "|".join(h for _k, h, _f in metrics) + "|"
        sep = "|---|" + "|".join("---" for _ in metrics) + "|"
        out += [header, sep]
        for label, cell in ds_rows:
            cols = []
            for key, _h, _f in metrics:
                vals = cell.get(key)
                if not vals:
                    cols.append("")
                elif key in ("moves", "blind", "absent", "med_latency", "coverage_halftime", "sd_expired_changes"):
                    cols.append(f"{st.mean(vals):.1f}")
                else:
                    mean = st.mean(vals)
                    cols.append(f"{mean:.3f}" if len(vals) == 1 else f"{mean:.3f}±{st.stdev(vals):.3f}")
            out.append(f"| {label} |" + "|".join(cols) + "|")
        # failure attribution summary
        out += ["", f"Failure attribution ({ds}): share of missed changes that are **absent** "
                    "(navigation failure) vs **blind** (detector failure), mean over seeds:", ""]
        for label, cell in ds_rows:
            blind = st.mean(cell.get("blind", [0]))
            absent = st.mean(cell.get("absent", [0]))
            total = blind + absent
            if total > 0:
                out.append(f"- {label}: {absent / total * 100:.0f}% absent / {blind / total * 100:.0f}% blind "
                           f"({absent:.0f} absent, {blind:.0f} blind)")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("globs", nargs="+", help="checkpoint glob(s), e.g. 'reports/grid_*.json'")
    ap.add_argument("--out", default=None, help="markdown output path (default: stdout)")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated dataset filter (default: all found in the glob)")
    args = ap.parse_args()

    rows, n_files = load_rows(args.globs)
    datasets = [d for d in (args.datasets.split(",") if args.datasets else sorted({k[1] for k in rows}))
                if any(k[1] == d for k in rows)]
    text = render(rows, n_files, datasets)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out} ({n_files} checkpoints, {len(rows)} strategy×dataset cells)")
    else:
        print(text)


if __name__ == "__main__":
    main()
