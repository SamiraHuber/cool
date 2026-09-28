#!/usr/bin/env python3
"""Build a compact method-section table from agent/baseline evaluation CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


HEADERS = [
    "Type",
    "Accuracy",
    "Lookup",
    "Location",
    "Interaction",
    "Ownership",
    "Navigation",
    "Function Calls",
    "Time",
]

LOOKUP_CATEGORIES = {"entity_lookup"}
LOCATION_CATEGORIES = {"object_location", "person_location", "time_location"}
INTERACTION_CATEGORIES = {
    "object_interaction",
    "person_interaction",
    "person_interaction_summary",
    "interaction_location_time",
    "object_interaction_summary",
    "scene_summary",
    "recent_interactions",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _resolve_output_path(value: str, base: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if path.is_absolute():
        return path
    return base / path


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "pass"}


def _format_pct(correct: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{(correct / total) * 100:.1f}"


def _format_decimal(value: str, suffix: str = "") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}{suffix}"


def _type_label(model_used: str, summary_csv: Path) -> str:
    label = str(model_used or "").strip()
    if label.startswith("baseline_"):
        return label.removeprefix("baseline_")
    if label:
        return "full_agent"
    stem = summary_csv.stem
    return stem.removesuffix("_test_results")


def _group_for_question(row: dict[str, str]) -> str | None:
    question = str(row.get("question") or "").strip().lower()
    category = str(row.get("category") or "").strip()
    if category in LOOKUP_CATEGORIES:
        return "Lookup"
    if category in LOCATION_CATEGORIES:
        return "Location"
    if "ownership" in category or "owner" in category:
        return "Ownership"
    if category in INTERACTION_CATEGORIES or "interaction" in category:
        return "Interaction"
    if question.startswith("navigate"):
        return "Navigation"
    return None


def _summarize_question_rows(rows: list[dict[str, str]]) -> tuple[dict[str, str], int, int]:
    totals = {name: 0 for name in HEADERS[2:7]}
    correct = {name: 0 for name in HEADERS[2:7]}
    overall_total = 0
    overall_correct = 0
    for row in rows:
        overall_total += 1
        if _is_true(row.get("correct")):
            overall_correct += 1
        group = _group_for_question(row)
        if group is None:
            continue
        totals[group] += 1
        if _is_true(row.get("correct")):
            correct[group] += 1
    return (
        {group: _format_pct(correct[group], totals[group]) for group in totals},
        overall_correct,
        overall_total,
    )


def _summary_to_rows(summary_csv: Path) -> list[dict[str, str]]:
    repo_root = _repo_root()
    output_rows = []
    for summary in _read_csv(summary_csv):
        answers_path = _resolve_output_path(summary.get("question_answers_csv", ""), repo_root)
        question_rows = _read_csv(answers_path) if answers_path and answers_path.exists() else []
        group_scores, row_correct, row_total = _summarize_question_rows(question_rows)
        amount_correct = row_correct if question_rows else int(float(summary.get("amount_correct") or 0))
        amount_questions = row_total if question_rows else int(float(summary.get("amount_questions_asked") or 0))
        output_rows.append(
            {
                "Type": _type_label(summary.get("model_used", ""), summary_csv),
                "Accuracy": _format_pct(amount_correct, amount_questions),
                "Lookup": group_scores["Lookup"],
                "Location": group_scores["Location"],
                "Interaction": group_scores["Interaction"],
                "Ownership": group_scores["Ownership"],
                "Navigation": group_scores["Navigation"],
                "Function Calls": _format_decimal(summary.get("avg_function_calls_per_question", "")),
                "Time": _format_decimal(summary.get("avg_duration_per_question_sec", ""), suffix="s"),
            }
        )
    return output_rows


def _print_tsv(rows: list[dict[str, str]]) -> None:
    print("\t".join(HEADERS))
    for row in rows:
        print("\t".join(row.get(header, "") for header in HEADERS))


def _print_markdown(rows: list[dict[str, str]]) -> None:
    print("| " + " | ".join(HEADERS) + " |")
    print("| " + " | ".join(["---"] * len(HEADERS)) + " |")
    for row in rows:
        print("| " + " | ".join(row.get(header, "") for header in HEADERS) + " |")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADERS)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_csv",
        nargs="+",
        type=Path,
        help="One or more summary CSVs produced by ownership_reasoning/evaluate_agent.py or ownership_reasoning/evaluate_baselines.py.",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "markdown"),
        default="tsv",
        help="Table format to print to stdout.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the compact table.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    rows: list[dict[str, str]] = []
    for summary_csv in args.summary_csv:
        rows.extend(_summary_to_rows(summary_csv))
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    if args.format == "markdown":
        _print_markdown(rows)
    else:
        _print_tsv(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
