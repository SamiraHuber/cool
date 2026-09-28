#!/usr/bin/env python3
"""Analyse the v20r3 interaction-term study against the v20r2 baseline.

Pairs every v20r3 cell with the v20r2 cell of the same (dataset, seed) and
reports the metrics the paper should lead with: the ownership-relevant ones
(interaction witnessing, absolute and per move), the unconditional coverage
ones (changes never detected, rooms reached), and the presence-conditional F1
that the earlier rebuttal used — in that order, deliberately.

Reads reports/, reports/grid_v20r3_4b/ and reports/grid_v20r3_skip_4b/.
No GPU, no VLM: pure JSON aggregation. Safe to run on partial results.
"""
from __future__ import annotations
import glob, json, os, re, statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DS_DIR = os.path.join(ROOT, "data", "curiosity", "datasets")
SEARCH_DIRS = ["reports", "reports/grid_v20r3_4b", "reports/grid_v20r3_skip_4b",
               "reports/grid_v20r2_skip_x_policy_4b"]
MAIN15 = ["default", "regime", "adversarial", "base_v2", "distractor", "intent_cued",
          "intent_cued_heading", "agent_cued_6room", "agent_cued_flat_6room",
          "scaled_base_24room", "scaled_24room", "scaled_distractor_24room",
          "scaled_heading_24room", "agent_cued_24room", "agent_cued_flat_24room"]
BASE, NEW = "agent_scene_change_v20r2_split", "agent_scene_change_v20r3_split"
BASE_AX, NEW_AX = "agent_scene_change_v20r2_argmax_split", "agent_scene_change_v20r3_argmax_split"


def load() -> dict[tuple[str, str, int], dict]:
    """(dataset, strategy, seed) -> result dict. Later dirs win on collision."""
    known = sorted(os.listdir(DS_DIR), key=len, reverse=True)
    out: dict[tuple[str, str, int], dict] = {}
    for d in SEARCH_DIRS:
        for f in glob.glob(os.path.join(ROOT, d, "grid_*_seed*.json")):
            b = os.path.basename(f)
            ds = next((x for x in known if b.startswith("grid_" + x + "_")), None)
            if not ds:
                continue
            m = re.match(r"^grid_" + re.escape(ds) + r"_(.+)_seed(\d+)\.json$", b)
            if not m:
                continue
            try:
                out[(ds, m.group(1), int(m.group(2)))] = json.load(open(f))["result"]
            except Exception:
                pass
    return out


def per_move(r: dict, field: str) -> float:
    mv = r.get("total_moves") or 0
    return (r.get(field) or 0) / mv if mv else float("nan")


def mean(v):
    v = [x for x in v if isinstance(x, (int, float)) and x == x]
    return st.mean(v) if v else float("nan")


def paired(R, ds_list, a, b):
    """Per-seed paired deltas for strategies a (new) and b (baseline)."""
    rows = []
    for ds in ds_list:
        for seed in range(10):
            ra, rb = R.get((ds, a, seed)), R.get((ds, b, seed))
            if ra and rb:
                rows.append((ds, seed, ra, rb))
    return rows


def report(R, ds_list, label, a=NEW, b=BASE):
    rows = paired(R, ds_list, a, b)
    if not rows:
        print(f"\n[{label}] no paired runs yet for {a} — run the study first.")
        return
    print(f"\n===== {label}: {a}  vs  {b}  ({len(rows)} paired runs) =====")
    METRICS = [
        ("interaction_witness_rate", "interaction witness rate", True, "ownership"),
        ("__iw_per_move", "interaction cells / move", True, "ownership"),
        ("__people_rate", "person-observation rate", True, "ownership"),
        ("changes_never_detected", "changes NEVER detected", False, "coverage"),
        ("exploration_coverage", "fraction of rooms reached", True, "coverage"),
        ("change_recall", "change coverage recall", True, "coverage"),
        ("total_moves", "moves per run", False, "cost"),
        ("change_f1", "F1 (given presence)", True, "conditional"),
        ("major_recall", "major recall (given presence)", True, "conditional"),
    ]
    la = a.replace("agent_scene_change_", "")[:9]
    lb = b.replace("agent_scene_change_", "")[:9]
    print(f"{'metric':32s}{la:>10s}{lb:>10s}{'delta':>10s}{'wins':>8s}  group")
    for key, name, higher_better, group in METRICS:
        if key == "__iw_per_move":
            va = [per_move(ra, "interaction_cells_witnessed") for _, _, ra, _ in rows]
            vb = [per_move(rb, "interaction_cells_witnessed") for _, _, _, rb in rows]
        elif key == "__people_rate":
            va = [ra.get("people_observation_rate") for _, _, ra, _ in rows]
            vb = [rb.get("people_observation_rate") for _, _, _, rb in rows]
        else:
            va = [ra.get(key) for _, _, ra, _ in rows]
            vb = [rb.get(key) for _, _, _, rb in rows]
        ma, mb = mean(va), mean(vb)
        pairs = [(x, y) for x, y in zip(va, vb) if isinstance(x, (int, float)) and isinstance(y, (int, float))]
        wins = sum(1 for x, y in pairs if (x > y) == higher_better and x != y)
        arrow = "+" if (ma > mb) == higher_better else "-"
        print(f"{name:32s}{ma:10.3f}{mb:10.3f}{ma - mb:+10.3f}{wins:5d}/{len(pairs):<3d}  {group} {arrow}")

    print(f"\n  per-dataset interaction witness rate / F1 / never-detected:")
    print(f"  {'dataset':26s}{'iw ' + la:>10s}{'iw ' + lb:>10s}{'F1 ' + la:>10s}{'F1 ' + lb:>10s}"
          f"{'never ' + la:>13s}{'never ' + lb:>13s}{'mv ' + la:>10s}")
    for ds in ds_list:
        rs = [(ra, rb) for d, _, ra, rb in rows if d == ds]
        if not rs:
            continue
        g = lambda i, k: mean([x[i].get(k) for x in rs])
        print(f"  {ds:26s}{g(0,'interaction_witness_rate'):10.3f}{g(1,'interaction_witness_rate'):10.3f}"
              f"{g(0,'change_f1'):10.3f}{g(1,'change_f1'):10.3f}"
              f"{g(0,'changes_never_detected'):13.0f}{g(1,'changes_never_detected'):13.0f}{g(0,'total_moves'):10.1f}")


def skip_family(R):
    """Re-run of the when/where ablation on the v20r3 table (phase 3)."""
    pols = sorted({m for (_, s, _) in R for m in [s] if s.startswith("agent_scene_change_v20r3_skip_")})
    if not pols:
        print("\n[skip family] phase 3 not run yet.")
        return
    dss = sorted({d for (d, s, _) in R if s in pols})
    print(f"\n===== same-dwell control family on the v20r3 table ({len(dss)} datasets) =====")
    print(f"{'destination policy':40s}{'F1':>8s}{'iw':>8s}{'iw/mv':>8s}{'never':>9s}{'mv':>7s}")
    for s in [NEW] + pols:
        rs = [R[(d, s, x)] for d in dss for x in range(10) if (d, s, x) in R]
        if not rs:
            continue
        print(f"{s.replace('agent_scene_change_v20r3_',''):40s}"
              f"{mean([r.get('change_f1') for r in rs]):8.3f}"
              f"{mean([r.get('interaction_witness_rate') for r in rs]):8.3f}"
              f"{mean([per_move(r,'interaction_cells_witnessed') for r in rs]):8.2f}"
              f"{mean([r.get('changes_never_detected') for r in rs]):9.0f}"
              f"{mean([r.get('total_moves') for r in rs]):7.1f}")
    print("  -> if the LLM navigator now leads on iw and iw/mv, the term is being USED,")
    print("     not just present in the table. That is the claim worth making.")


def weight_sweep(R):
    """Phase 4: w_int sensitivity. Sweep cells carry a -i<w> checkpoint tag."""
    tags = sorted({s for (_, s, _) in R if "_w1-1-0.5-i" in s})
    if not tags:
        print("\n[w_int sweep] phase 4 not run yet.")
        return
    print("\n===== w_int sensitivity =====")
    for s in tags:
        rs = [R[k] for k in R if k[1] == s]
        print(f"  {s:56s} F1 {mean([r.get('change_f1') for r in rs]):.3f}  "
              f"iw {mean([r.get('interaction_witness_rate') for r in rs]):.3f}  "
              f"mv {mean([r.get('total_moves') for r in rs]):5.1f}  (n={len(rs)})")


if __name__ == "__main__":
    R = load()
    six = [d for d in MAIN15 if "24room" not in d]
    big = [d for d in MAIN15 if "24room" in d]
    report(R, six, "6-ROOM")
    report(R, big, "24-ROOM")
    report(R, MAIN15, "ALL 15 — argmax pair (table without navigator)", a=NEW_AX, b=BASE_AX)
    skip_family(R)
    weight_sweep(R)
    print("\nRead the ownership group first, coverage second, conditional F1 last —")
    print("F1 alone rewards camping (see CAMERA_READY_RESULTS_SUMMARY.md §2).")
