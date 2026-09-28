#!/usr/bin/env python3
"""Run all baselines plus the real website agent (Qwen) and merge summary rows into one CSV.

Usage:
    python3 ownership_reasoning/run_all_evaluations.py
    python3 ownership_reasoning/run_all_evaluations.py --building gpt_generated_office_robot_map_2026_04_27
    python3 ownership_reasoning/run_all_evaluations.py --limit 10
    python3 ownership_reasoning/run_all_evaluations.py --db postgresql://postgres:postgres@localhost:35432/bordsupr
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "evaluation_outputs"


def _build_common_args(args) -> list[str]:
    """Build argument list shared by all evaluators."""
    out: list[str] = []
    if args.building:
        out.extend(["--building", args.building])
    if args.limit is not None:
        out.extend(["--limit", str(args.limit)])
    if args.questions_file:
        out.extend(["--questions-file", str(args.questions_file)])
    return out


def _run(cmd: list[str], label: str) -> int:
    """Run a subprocess and stream output. Returns exit code."""
    print(f"\n{'=' * 60}")
    print(f"Running: {label}")
    print(f"{'=' * 60}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    # evaluate_baselines.py returns 1 when not all questions are correct,
    # which is expected. Don't treat it as an error.
    if result.returncode not in (0, 1):
        print(f"WARNING: {label} exited with code {result.returncode}")
    return result.returncode


def _read_last_row(csv_path: Path) -> tuple[list[str], dict[str, str]] | None:
    """Read the last data row from a CSV file."""
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    return list(reader.fieldnames or []), rows[-1]


def _merge_rows(rows_data: list[tuple[str, tuple[list[str], dict[str, str]] | None]]) -> None:
    """Merge the latest row from each evaluator into a single combined CSV."""
    # Collect all headers
    all_headers: list[str] = []
    seen: set[str] = set()
    for _label, data in rows_data:
        if data is None:
            continue
        headers, _ = data
        for h in headers:
            if h not in seen:
                seen.add(h)
                all_headers.append(h)

    combined_path = EVAL_DIR / "combined_results.csv"
    write_header = not combined_path.exists() or combined_path.stat().st_size == 0

    with combined_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_headers)
        if write_header:
            writer.writeheader()
        for label, data in rows_data:
            if data is None:
                print(f"Skipping {label}: no data row found.")
                continue
            _headers, row = data
            # Ensure every column exists (fill missing with empty string)
            full_row = {h: row.get(h, "") for h in all_headers}
            writer.writerow(full_row)

    print(f"\nCombined results appended to: {combined_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all baselines + real agent and merge summaries into one CSV.",
    )
    parser.add_argument(
        "--building",
        default=None,
        help="Building/map name to scope queries to.",
    )
    parser.add_argument(
        "--db",
        dest="database_url",
        default="postgresql://postgres:postgres@localhost:35432/bordsupr",
        help="PostgreSQL connection string for DB baselines.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions asked.",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=REPO_ROOT / "data" / "evaluation" / "overall_questions.json",
        help="Path to the JSON question file.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Timeout per question in seconds.",
    )
    parser.add_argument(
        "--inter-question-delay-sec",
        type=float,
        default=2.0,
        help="Delay between questions in seconds.",
    )
    parser.add_argument(
        "--skip-real-agent",
        action="store_true",
        help="Skip the real website agent evaluation.",
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip all baseline evaluations.",
    )
    args = parser.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    common = _build_common_args(args)

    runs: list[tuple[str, list[str], Path]] = []

    if not args.skip_real_agent:
        real_csv = EVAL_DIR / "combined_run_real_agent.csv"
        real_cmd = [
            sys.executable,
            "ownership_reasoning/evaluate_agent.py",
            "--model", "qwen",
            "--results-csv", str(real_csv),
            "--question-answers-csv", str(EVAL_DIR / "combined_run_real_agent_answers.csv"),
            "--reasoning-log-txt", str(EVAL_DIR / "combined_run_real_agent_reasoning.txt"),
            "--timeout-sec", str(args.timeout_sec),
            "--inter-question-delay-sec", str(args.inter_question_delay_sec),
        ] + common
        runs.append(("Real Agent (Qwen)", real_cmd, real_csv))

    if not args.skip_baselines:
        for baseline in ["direct_llm", "rule_based", "no_interaction", "heuristic"]:
            # evaluate_baselines.py prefixes the baseline name to the csv filename
            raw_baseline_csv = EVAL_DIR / f"combined_run_{baseline}.csv"
            baseline_csv = EVAL_DIR / f"{baseline}_combined_run_{baseline}.csv"
            baseline_cmd = [
                sys.executable,
                "ownership_reasoning/evaluate_baselines.py",
                "--baseline", baseline,
                "--results-csv", str(raw_baseline_csv),
                "--question-answers-csv", str(EVAL_DIR / f"combined_run_{baseline}_answers.csv"),
                "--reasoning-log-txt", str(EVAL_DIR / f"combined_run_{baseline}_reasoning.txt"),
                "--timeout-sec", str(args.timeout_sec),
                "--inter-question-delay-sec", str(args.inter_question_delay_sec),
                "--db", args.database_url,
            ] + common
            runs.append((f"Baseline: {baseline}", baseline_cmd, baseline_csv))

    results: list[tuple[str, tuple[list[str], dict[str, str]] | None]] = []
    for label, cmd, csv_path in runs:
        exit_code = _run(cmd, label)
        # Baseline evaluators return 1 when not all questions are correct,
        # but they still write the summary CSV. Always try to read it.
        row_data = _read_last_row(csv_path)
        if row_data is None:
            print(f"ERROR: {label} produced no CSV row (exit {exit_code}). It will be omitted from the combined CSV.")
            results.append((label, None))
        else:
            results.append((label, row_data))

    _merge_rows(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
