#!/usr/bin/env python3
"""Compute weighted averages (with std dev) from compact_results.csv."""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

OVERALL_QS = 59
NAV_QS = 20
TOTAL_QS = OVERALL_QS + NAV_QS

METRICS = ["Accuracy", "Interaction", "Location", "Lookup", "Ownership", "Navigation", "function calls", "time"]


def base_name(variant: str) -> str:
    v = variant
    if v.endswith("_navigation"):
        v = v.replace("_navigation", "")
    elif v.endswith("_overall"):
        v = v.replace("_overall", "")
    if v.startswith("baseline_"):
        v = v.replace("baseline_", "", 1)
    return v


def mean_std(vals: list[float]) -> tuple[float, float]:
    if not vals:
        return "", ""
    m = sum(vals) / len(vals)
    if len(vals) < 2:
        return m, 0.0
    s = statistics.stdev(vals)
    return m, s


def main() -> int:
    compact_path = Path("evaluation_outputs/compact_results.csv")
    if not compact_path.exists():
        print(f"File not found: {compact_path}")
        return 1

    rows = list(csv.DictReader(compact_path.open()))

    # Group by variant
    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    # Compute per-variant mean/std
    variant_stats = {}
    for variant, runs in by_variant.items():
        record = {"variant": variant, "runs": len(runs), "question_count": NAV_QS if "navigation" in variant else OVERALL_QS}
        for col in METRICS:
            vals = [float(r[col]) for r in runs if r.get(col)]
            m, s = mean_std(vals)
            record[f"{col}_mean"] = m
            record[f"{col}_std"] = s
        variant_stats[variant] = record

    # Group by base name for weighted average
    by_base = defaultdict(list)
    for rec in variant_stats.values():
        by_base[base_name(rec["variant"])].append(rec)

    # Build weighted summary rows
    weighted_rows = []
    for base in sorted(by_base.keys()):
        parts = by_base[base]
        overall_rec = next((p for p in parts if p["question_count"] == OVERALL_QS), None)
        nav_rec = next((p for p in parts if p["question_count"] == NAV_QS), None)

        def weighted(col: str):
            o_mean = overall_rec.get(f"{col}_mean") if overall_rec else ""
            n_mean = nav_rec.get(f"{col}_mean") if nav_rec else ""
            if o_mean != "" and n_mean != "":
                return (float(o_mean) * OVERALL_QS + float(n_mean) * NAV_QS) / TOTAL_QS
            elif o_mean != "":
                return float(o_mean)
            elif n_mean != "":
                return float(n_mean)
            return ""

        def weighted_std(col: str):
            """Propagate variance: Var(aX + bY) = a^2 Var(X) + b^2 Var(Y)"""
            o_std = overall_rec.get(f"{col}_std") if overall_rec else ""
            n_std = nav_rec.get(f"{col}_std") if nav_rec else ""
            w_o = OVERALL_QS / TOTAL_QS
            w_n = NAV_QS / TOTAL_QS
            if o_std != "" and n_std != "":
                return ((w_o ** 2) * (float(o_std) ** 2) + (w_n ** 2) * (float(n_std) ** 2)) ** 0.5
            elif o_std != "":
                return float(o_std)
            elif n_std != "":
                return float(n_std)
            return ""

        row = {
            "variant": base,
            "runs": (overall_rec["runs"] if overall_rec else 0) + (nav_rec["runs"] if nav_rec else 0),
            "overall_runs": overall_rec["runs"] if overall_rec else 0,
            "nav_runs": nav_rec["runs"] if nav_rec else 0,
        }
        for col in METRICS:
            row[f"{col}_mean"] = weighted(col)
            row[f"{col}_std"] = weighted_std(col)
        weighted_rows.append(row)

    out_path = Path("evaluation_outputs/compact_results_weighted.csv")
    fieldnames = ["variant", "runs", "overall_runs", "nav_runs"]
    for col in METRICS:
        fieldnames.append(f"{col}_mean")
        fieldnames.append(f"{col}_std")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in weighted_rows:
            formatted = {}
            for k, v in r.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.4f}"
                else:
                    formatted[k] = v
            writer.writerow(formatted)

    print(f"Saved weighted summary to {out_path}")
    print()
    print("Question counts: Overall = 59, Navigation = 20, Total = 79")
    print()

    # Print per-variant breakdown
    print("Per-variant mean ± std:")
    hdr = f"{'Variant':<28} {'Runs':>4} {'Q':>3}"
    for col in METRICS:
        hdr += f" {col[:10]:>12}"
    print(hdr)
    print("-" * len(hdr))
    for variant in sorted(variant_stats.keys()):
        rec = variant_stats[variant]
        line = f"{rec['variant']:<28} {rec['runs']:>4} {rec['question_count']:>3}"
        for col in METRICS:
            m = rec[f"{col}_mean"]
            s = rec[f"{col}_std"]
            if m != "":
                line += f" {m:.4f}±{s:.4f}"
            else:
                line += " " + " " * 11
        print(line)

    print()
    print("Weighted combined mean ± std (overall*59 + nav*20) / 79:")
    hdr2 = f"{'Variant':<18} {'Runs':>4}"
    for col in METRICS:
        hdr2 += f" {col[:10]:>14}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in weighted_rows:
        line = f"{r['variant']:<18} {r['runs']:>4}"
        for col in METRICS:
            m = r[f"{col}_mean"]
            s = r[f"{col}_std"]
            if m != "":
                line += f" {m:>6.4f}±{s:<6.4f}"
            else:
                line += " " + " " * 13
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
