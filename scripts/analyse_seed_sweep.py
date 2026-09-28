#!/usr/bin/env python3
"""
Analyse seed-sweep checkpoint files and produce comparison tables + plots.

Usage:
    python scripts/analyse_seed_sweep.py "reports/seed_sweep_*.json"
    python scripts/analyse_seed_sweep.py "reports/seed_sweep_*.json" --out report.md

    # Or analyse existing sweep while it is still running:
    python scripts/analyse_seed_sweep.py "reports/seed_sweep_*.json" --out reports/seed_analysis.md
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Computed helpers
# ---------------------------------------------------------------------------

def _mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews Correlation Coefficient."""
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return (tp * tn - fp * fn) / denom


def _compute_derived(result: dict) -> dict:
    """Add computed metrics (MCC, etc.) to a result dict copy."""
    r = dict(result)
    tp = r.get("scene_change_tp", 0)
    tn = r.get("scene_change_tn", 0)
    fp = r.get("scene_change_fp", 0)
    fn = r.get("scene_change_fn", 0)
    r["mcc"] = _mcc(tp, tn, fp, fn)
    return r


# ---------------------------------------------------------------------------
# Metrics we care about
# ---------------------------------------------------------------------------

# UI-matching metrics (what you see on :8080/scene-change-strategy)
UI_METRICS = [
    # ── Event / Scene-Change Detection ──
    ("change_recall", "Evt Detection Recall", ".3f"),
    ("scene_change_accuracy", "Scene-Chg Class. Acc", ".3f"),
    ("no_change_accuracy", "No-Chg Class. Acc", ".3f"),
    ("change_accuracy", "Change Detection Acc", ".3f"),
    ("activity_change_accuracy", "Activity Change Acc", ".3f"),
    ("detection_efficiency", "Detection Efficiency", ".3f"),
    ("event_timesteps_caught", "Event Timesteps Caught", ".0f"),
    ("total_event_timesteps", "Total Event Timesteps", ".0f"),
    # ── Navigation ──
    ("navigation_precision", "Navigation Precision", ".3f"),
    ("navigation_efficiency", "Navigation Efficiency", ".3f"),
    ("normalized_recall", "Normalized Recall", ".3f"),
    ("path_max_changes", "Max Changes on Path", ".0f"),
    ("theoretical_max_changes", "Theoretical Max Changes", ".0f"),
    ("total_visits", "Room Visits", ".0f"),
    ("vlm_calls_made", "VLM Calls Made", ".0f"),
]

CORE_METRICS = [
    # ── Event / Scene-Change Detection ──
    ("change_recall", "Evt Detection Recall", ".3f"),
    ("change_precision", "Evt Detection Precision", ".3f"),
    ("change_f1", "Evt Detection F1", ".3f"),
    ("mcc", "MCC (Detection)", ".3f"),
    ("scene_change_accuracy", "Scene-Chg Accuracy", ".3f"),
    ("event_timestep_rate", "Event Timestep Rate", ".3f"),
    ("detection_efficiency", "Detection Efficiency", ".3f"),
    # ── Navigation ──
    ("navigation_precision", "Navigation Precision", ".3f"),
    ("navigation_efficiency", "Navigation Efficiency", ".3f"),
    ("exploration_coverage", "Exploration Coverage", ".3f"),
    ("cross_room_miss_rate", "Cross-Room Miss Rate", ".3f"),
]

LATENCY_METRICS = [
    ("avg_detection_latency_minutes", "Avg Detection Latency (min)", ".1f"),
    ("median_detection_latency_minutes", "Median Detection Latency (min)", ".1f"),
    ("max_detection_latency_minutes", "Max Detection Latency (min)", ".0f"),
]

COUNT_METRICS = [
    # ── Detection Confusion ──
    ("scene_change_tp", "True Positives (TP)", ".0f"),
    ("scene_change_fp", "False Positives (FP)", ".0f"),
    ("scene_change_fn", "False Negatives (FN)", ".0f"),
    ("observed_changes", "Observed Changes", ".0f"),
    ("event_timesteps_caught", "Event Timesteps Caught", ".0f"),
    # ── Navigation / Cost ──
    ("total_visits", "Room Visits", ".0f"),
    ("vlm_calls_made", "VLM Calls Made", ".0f"),
]

ALL_METRICS = UI_METRICS + CORE_METRICS + LATENCY_METRICS + COUNT_METRICS


def _load_runs(pattern: str) -> list[dict]:
    files = glob.glob(pattern)
    if not files:
        print(f"No files matched pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    runs = []
    for p in sorted(files):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "result" not in data and "strategy" in data:
            # Flat checkpoint format (run_grid_sweep.py before the wrapper
            # fix): the result fields live at top level.
            flat = dict(data)
            data = {"strategy": flat.get("strategy"), "seed": flat.get("seed"),
                    "dataset": flat.get("dataset"), "result": flat}
        # Inject computed metrics into the result
        data["result"] = _compute_derived(data.get("result", {}))
        runs.append(data)
    return runs


def _group_by_strategy(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        key = r.get("label") or r.get("strategy", "unknown")
        groups[key].append(r)
    return dict(groups)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _build_table(groups: dict[str, list[dict]], metrics: list[tuple[str, str, str]]) -> list[list[str]]:
    headers = ["Strategy", "N"] + [label for _, label, _ in metrics]
    rows = [headers]
    for label in sorted(groups.keys()):
        runs = groups[label]
        n = len(runs)
        row = [label, str(n)]
        for key, _, fmt in metrics:
            vals = [r["result"][key] for r in runs if key in r.get("result", {})]
            mean, std = _mean_std(vals)
            if std > 0:
                cell = f"{mean:{fmt}} ±{std:{fmt}}"
            else:
                cell = f"{mean:{fmt}}"
            row.append(cell)
        rows.append(row)
    return rows


def _format_table(rows: list[list[str]]) -> str:
    if len(rows) < 2:
        return ""
    col_widths = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    lines = []
    for idx, row in enumerate(rows):
        line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if idx == 0:
            lines.append("-" * len(line))
    return "\n".join(lines)


def _markdown_table(rows: list[list[str]]) -> str:
    if len(rows) < 2:
        return ""
    lines = []
    header = " | ".join(rows[0])
    lines.append(header)
    lines.append(" | ".join("---" for _ in rows[0]))
    for row in rows[1:]:
        lines.append(" | ".join(row))
    return "\n".join(lines)


def _build_report(groups: dict[str, list[dict]], pattern: str) -> str:
    sections: list[str] = []
    sections.append("# Seed-Sweep Analysis Report")
    sections.append(f"\nGenerated: {datetime.now().isoformat()}")
    sections.append(f"Pattern: `{pattern}`")
    sections.append(f"Total runs: {sum(len(v) for v in groups.values())}")
    sections.append(f"Strategies: {len(groups)}")

    # Seeds per strategy
    sections.append("\n## Seeds per Strategy")
    for label in sorted(groups.keys()):
        seeds = sorted([r["seed"] for r in groups[label]])
        sections.append(f"- **{label}**: seeds {seeds}")

    # UI-matching metrics table
    sections.append("\n## UI Metrics (:8080/scene-change-strategy) — mean ± std")
    ui_rows = _build_table(groups, UI_METRICS)
    sections.append(_markdown_table(ui_rows))

    # Core metrics table
    sections.append("\n## Core Metrics (mean ± std)")
    core_rows = _build_table(groups, CORE_METRICS)
    sections.append(_markdown_table(core_rows))

    # Latency table
    sections.append("\n## Detection Latency (mean ± std)")
    lat_rows = _build_table(groups, LATENCY_METRICS)
    sections.append(_markdown_table(lat_rows))

    # Counts table
    sections.append("\n## Counts (mean ± std)")
    cnt_rows = _build_table(groups, COUNT_METRICS)
    sections.append(_markdown_table(cnt_rows))

    # Best per metric
    sections.append("\n## Best Strategy per Metric")
    for key, label, fmt in ALL_METRICS:
        best_strategy = None
        best_val = None
        best_std = None
        for strat in sorted(groups.keys()):
            vals = [r["result"][key] for r in groups[strat] if key in r.get("result", {})]
            if not vals:
                continue
            mean, std = _mean_std(vals)
            lower_is_better = key in {
                "avg_detection_latency_minutes",
                "median_detection_latency_minutes",
                "max_detection_latency_minutes",
                "cross_room_miss_rate",
                "scene_change_fp",
                "scene_change_fn",
                "total_visits",  # fewer visits = more efficient
            }
            if best_val is None or (mean < best_val if lower_is_better else mean > best_val):
                best_val = mean
                best_std = std
                best_strategy = strat
        if best_strategy is not None:
            std_str = f" ±{best_std:{fmt}}" if best_std and best_std > 0 else ""
            sections.append(f"- **{label}**: {best_strategy} ({best_val:{fmt}}{std_str})")

    # Rankings
    sections.append("\n## Rankings by Key Metrics")
    for key, label, fmt in [
        ("change_recall", "Recall", ".3f"),
        ("change_f1", "F1", ".3f"),
        ("mcc", "MCC", ".3f"),
        ("event_timestep_rate", "Event Rate", ".3f"),
        ("avg_detection_latency_minutes", "Avg Latency", ".1f"),
    ]:
        sections.append(f"\n### {label}")
        ranked = []
        for strat in sorted(groups.keys()):
            vals = [r["result"][key] for r in groups[strat] if key in r.get("result", {})]
            if vals:
                mean, std = _mean_std(vals)
                ranked.append((strat, mean, std))
        lower_is_better = key in {
            "avg_detection_latency_minutes",
            "median_detection_latency_minutes",
            "max_detection_latency_minutes",
        }
        ranked.sort(key=lambda x: x[1], reverse=not lower_is_better)
        for i, (s, m, st) in enumerate(ranked, 1):
            std_str = f" ±{st:{fmt}}" if st and st > 0 else ""
            sections.append(f"{i}. {s}: {m:{fmt}}{std_str}")

    return "\n".join(sections)


def _plot(groups: dict[str, list[dict]], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategies = sorted(groups.keys())
    colors = plt.cm.tab10(range(len(strategies)))
    saved: list[Path] = []

    def _bar_plot(ax, metric_key: str, title: str, ylabel: str, lower_better: bool = False):
        means = []
        stds = []
        for s in strategies:
            vals = [r["result"][metric_key] for r in groups[s] if metric_key in r.get("result", {})]
            m, st = _mean_std(vals)
            means.append(m)
            stds.append(st)
        bars = ax.bar(range(len(strategies)), means, yerr=stds, color=colors, capsize=4)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if lower_better:
            ax.invert_yaxis()
        for bar, m in zip(bars, means):
            height = bar.get_height()
            ax.annotate(f"{m:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)

    # Plot 1: UI-matching metrics
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("UI Metrics (mean across seeds)", fontsize=14)
    plot_specs = [
        ("change_recall", "Evt Detection Recall", "Recall"),
        ("scene_change_accuracy", "Scene-Chg Accuracy", "Accuracy"),
        ("change_accuracy", "Change Detection Acc", "Accuracy"),
        ("activity_change_accuracy", "Activity Change Acc", "Accuracy"),
        ("navigation_precision", "Navigation Precision", "Precision"),
        ("normalized_recall", "Normalized Recall", "Recall"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, plot_specs):
        _bar_plot(ax, key, title, ylabel)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p1 = out_dir / "seed_sweep_ui_metrics.png"
    fig.savefig(p1, dpi=150)
    saved.append(p1)
    plt.close(fig)

    # Plot 2: Core detection metrics (with MCC)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Core Detection Metrics (mean across seeds)", fontsize=14)
    plot_specs = [
        ("change_recall", "Evt Detection Recall", "Recall"),
        ("change_precision", "Evt Detection Precision", "Precision"),
        ("change_f1", "Evt Detection F1", "F1"),
        ("mcc", "MCC (Detection)", "MCC"),
        ("event_timestep_rate", "Event Timestep Rate", "Rate"),
        ("exploration_coverage", "Exploration Coverage", "Coverage"),
    ]
    for ax, (key, title, ylabel) in zip(axes.flat, plot_specs):
        _bar_plot(ax, key, title, ylabel)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p2 = out_dir / "seed_sweep_core_metrics.png"
    fig.savefig(p2, dpi=150)
    saved.append(p2)
    plt.close(fig)

    # Plot 3: Latency
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Detection Latency (minutes, mean across seeds)", fontsize=14)
    for ax, (key, title, ylabel) in zip(axes.flat, [
        ("avg_detection_latency_minutes", "Average", "Min"),
        ("median_detection_latency_minutes", "Median", "Min"),
        ("max_detection_latency_minutes", "Max", "Min"),
    ]):
        _bar_plot(ax, key, title, ylabel, lower_better=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    p3 = out_dir / "seed_sweep_latency.png"
    fig.savefig(p3, dpi=150)
    saved.append(p3)
    plt.close(fig)

    # Plot 4: Confusion counts
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(strategies))
    width = 0.25
    for offset, key, label, color in [
        (-width, "scene_change_tp", "TP", "green"),
        (0, "scene_change_fp", "FP", "orange"),
        (width, "scene_change_fn", "FN", "red"),
    ]:
        means = []
        stds = []
        for s in strategies:
            vals = [r["result"][key] for r in groups[s] if key in r.get("result", {})]
            m, st = _mean_std(vals)
            means.append(m)
            stds.append(st)
        ax.bar([i + offset for i in x], means, width, yerr=stds, label=label, color=color, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Count")
    ax.set_title("Confusion Matrix Counts (mean across seeds)")
    ax.legend()
    plt.tight_layout()
    p4 = out_dir / "seed_sweep_confusion.png"
    fig.savefig(p4, dpi=150)
    saved.append(p4)
    plt.close(fig)

    # Plot 5: Efficiency scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for s, c in zip(strategies, colors):
        vals_det = [r["result"]["detection_efficiency"] for r in groups[s] if "detection_efficiency" in r.get("result", {})]
        vals_nav = [r["result"]["navigation_efficiency"] for r in groups[s] if "navigation_efficiency" in r.get("result", {})]
        if vals_det and vals_nav:
            ax.scatter(vals_nav, vals_det, color=c, label=s, s=80, alpha=0.7)
            m_det, _ = _mean_std(vals_det)
            m_nav, _ = _mean_std(vals_nav)
            ax.scatter([m_nav], [m_det], color=c, s=200, marker="X", edgecolors="black", linewidths=1)
    ax.set_xlabel("Navigation Efficiency")
    ax.set_ylabel("Detection Efficiency")
    ax.set_title("Detection vs Navigation Efficiency (per seed + mean)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    p5 = out_dir / "seed_sweep_efficiency.png"
    fig.savefig(p5, dpi=150)
    saved.append(p5)
    plt.close(fig)

    # Plot 6: Seed-variation box plot for key metric
    fig, ax = plt.subplots(figsize=(10, 5))
    data = []
    labels = []
    for s in strategies:
        vals = [r["result"]["change_recall"] for r in groups[s] if "change_recall" in r.get("result", {})]
        if vals:
            data.append(vals)
            labels.append(s)
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors[:len(data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Recall")
    ax.set_title("Recall Distribution Across Seeds")
    plt.tight_layout()
    p6 = out_dir / "seed_sweep_recall_boxplot.png"
    fig.savefig(p6, dpi=150)
    saved.append(p6)
    plt.close(fig)

    # Plot 7: MCC comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    _bar_plot(ax, "mcc", "MCC (Matthews Correlation Coefficient)", "MCC")
    plt.tight_layout()
    p7 = out_dir / "seed_sweep_mcc.png"
    fig.savefig(p7, dpi=150)
    saved.append(p7)
    plt.close(fig)

    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pattern", help="Glob pattern for checkpoint JSON files (e.g. 'reports/seed_sweep_*.json')")
    parser.add_argument("--out", "-o", default=None, help="Output markdown file path (default: print to stdout only)")
    parser.add_argument("--plots", "-p", default="reports", help="Directory to save plots (default: reports/)")
    args = parser.parse_args()

    runs = _load_runs(args.pattern)
    groups = _group_by_strategy(runs)

    print(f"Loaded {len(runs)} runs across {len(groups)} strategies")
    print()

    ui_rows = _build_table(groups, UI_METRICS)
    print("=" * 100)
    print("UI METRICS (:8080/scene-change-strategy) — mean ± std")
    print("=" * 100)
    print(_format_table(ui_rows))
    print()

    core_rows = _build_table(groups, CORE_METRICS)
    print("=" * 100)
    print("CORE METRICS (mean ± std)")
    print("=" * 100)
    print(_format_table(core_rows))
    print()

    lat_rows = _build_table(groups, LATENCY_METRICS)
    print("=" * 100)
    print("DETECTION LATENCY (mean ± std)")
    print("=" * 100)
    print(_format_table(lat_rows))
    print()

    cnt_rows = _build_table(groups, COUNT_METRICS)
    print("=" * 100)
    print("COUNTS (mean ± std)")
    print("=" * 100)
    print(_format_table(cnt_rows))
    print()

    plot_dir = Path(args.plots)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = _plot(groups, plot_dir)
    print("Plots saved:")
    for p in plot_paths:
        print(f"  {p}")
    print()

    report = _build_report(groups, args.pattern)
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"Report written to: {out_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
