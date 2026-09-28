#!/usr/bin/env python3
"""Evaluate robot-perception baselines against the same question sets as the full agent.

Usage:
    python3 ownership_reasoning/evaluate_baselines.py --baseline direct_llm --building gpt_generated_office_robot_map_2026_04_27
    python3 ownership_reasoning/evaluate_baselines.py --baseline rule_based --questions-file data/evaluation/navigation_questions.json
    python3 ownership_reasoning/evaluate_baselines.py --baseline no_interaction
    python3 ownership_reasoning/evaluate_baselines.py --baseline heuristic

Baselines:
    direct_llm   – Direct LLM with flattened DB context, no tools.
    rule_based   – Deterministic SQL-based retrieval without LLM.
    no_interaction – Agent with interaction-related tools removed.
    heuristic    – Latest-observation heuristic for navigation/QA.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Re-use grading and I/O utilities from the main agent evaluator.
# We append the repo root so the import works when run as a script.
_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root))

from ownership_reasoning.evaluate_agent import (  # noqa: E402
    CSV_HEADERS,
    CATEGORY_GROUPS,
    COMPACT_RESULTS_HEADERS,
    QUESTION_ANSWER_HEADERS,
    EvaluationCase,
    QuestionResult,
    _append_compact_results_row,
    _append_question_rows,
    _append_rows_with_schema_upgrade,
    _append_summary_row,
    _category_group_for_case,
    _category_percent_column,
    _default_compact_results_csv,
    _default_navigation_question_answers_csv,
    _default_navigation_reasoning_log_txt,
    _default_navigation_results_csv,
    _default_question_answers_csv,
    _default_reasoning_log_txt,
    _default_results_csv,
    _display_path,
    _format_accuracy_decimal,
    _http_json,
    _is_case_correct,
    _load_test_cases,
    _normalize_case,
    _resolve_question_answers_csv,
    _resolve_reasoning_log_txt,
    _resolve_results_csv,
    _select_vlm_option,
    _write_reasoning_log,
)

from ownership_reasoning.baseline_engines import (  # noqa: E402
    DirectLLMBaseline,
    HeuristicBaseline,
    NoInteractionBaseline,
    RuleBasedBaseline,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a baseline against the website agent question set.",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=["direct_llm", "rule_based", "no_interaction", "heuristic"],
        help="Which baseline to evaluate.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8080",
        help="Base URL of the web API (used for direct_llm and no_interaction baselines).",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=_repo_root / "data" / "evaluation" / "overall_questions.json",
        help="Path to the JSON question file.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=_default_results_csv(),
        help="CSV file to append the summary row to.",
    )
    parser.add_argument(
        "--compact-results-csv",
        type=Path,
        default=_default_compact_results_csv(),
        help="Compact CSV to append Accuracy, Accuracy per category, function calls, and time.",
    )
    parser.add_argument(
        "--question-answers-csv",
        type=Path,
        default=_default_question_answers_csv(),
        help="CSV file to append per-question answers to.",
    )
    parser.add_argument(
        "--reasoning-log-txt",
        type=Path,
        default=_default_reasoning_log_txt(),
        help="TXT file to write the reasoning trace to.",
    )
    parser.add_argument(
        "--building",
        default=None,
        help="Building/map name to scope queries to.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of questions asked.",
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
        "--model",
        default="qwen",
        help="Model label used for VLM selection (affects only no_interaction baseline).",
    )
    parser.add_argument(
        "--db", "--database-url",
        dest="database_url",
        default=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:35432/bordsupr"),
        help="PostgreSQL connection string for baselines that query the DB directly. Examples: postgresql://postgres:postgres@localhost:35432/bordsupr or postgresql://postgres:postgres@localhost:35432/bordsupr",
    )
    return parser


def _resolve_paths(args, cases: list[EvaluationCase]) -> tuple[Path, Path, Path]:
    results_csv = _resolve_results_csv(args.results_csv, cases)
    question_answers_csv = _resolve_question_answers_csv(args.question_answers_csv, cases)
    reasoning_log_txt = _resolve_reasoning_log_txt(args.reasoning_log_txt, cases)
    return results_csv, question_answers_csv, reasoning_log_txt


def _stamp_paths(paths: tuple[Path, Path, Path]) -> tuple[Path, Path, Path]:
    results_csv, question_answers_csv, reasoning_log_txt = paths
    timestamp_token = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    if "timestamp" in question_answers_csv.name:
        question_answers_csv = question_answers_csv.with_name(
            question_answers_csv.name.replace("timestamp", timestamp_token)
        )
    if "timestamp" in reasoning_log_txt.name:
        reasoning_log_txt = reasoning_log_txt.with_name(
            reasoning_log_txt.name.replace("timestamp", timestamp_token)
        )
    return results_csv, question_answers_csv, reasoning_log_txt


def _make_baseline_engine(baseline: str, base_url: str, database_url: str):
    # Inject the user-supplied database URL into the baseline engines module
    import ownership_reasoning.baseline_engines as _be
    _be.DATABASE_URL = database_url

    if baseline == "direct_llm":
        return DirectLLMBaseline(base_url=base_url)
    if baseline == "rule_based":
        return RuleBasedBaseline()
    if baseline == "no_interaction":
        return NoInteractionBaseline(base_url=base_url)
    if baseline == "heuristic":
        return HeuristicBaseline()
    raise ValueError(f"Unknown baseline: {baseline}")


def _run_baseline_question(
    engine,
    case: EvaluationCase,
    building: str | None,
    timeout: float,
) -> QuestionResult:
    start = time.perf_counter()
    try:
        answer, tool_log = engine.answer(case.prompt, building=building)
        error = None
    except Exception as exc:
        answer = ""
        tool_log = []
        error = str(exc)

    duration_sec = time.perf_counter() - start
    result = QuestionResult(
        question=case.prompt,
        answer_text=answer,
        correct=False,
        duration_sec=duration_sec,
        total_tokens=0,
        function_call_count=len(tool_log),
        model="baseline",
        tool_log=tool_log,
        trace=[],
        system_prompt=None,
        error=error,
    )
    result.correct = _is_case_correct(case, result)
    return result


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    cases = [_normalize_case(case) for case in _load_test_cases(args.questions_file)]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise ValueError("No questions selected for evaluation")

    results_csv, question_answers_csv, reasoning_log_txt = _stamp_paths(
        _resolve_paths(args, cases)
    )

    # Prefix baseline name into output filenames so different baselines don't overwrite each other.
    baseline_label = args.baseline
    results_csv = results_csv.with_name(f"{baseline_label}_{results_csv.name}")
    question_answers_csv = question_answers_csv.with_name(
        f"{baseline_label}_{question_answers_csv.name}"
    )
    reasoning_log_txt = reasoning_log_txt.with_name(
        f"{baseline_label}_{reasoning_log_txt.name}"
    )

    # Ensure output directories exist.
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    question_answers_csv.parent.mkdir(parents=True, exist_ok=True)
    reasoning_log_txt.parent.mkdir(parents=True, exist_ok=True)

    # For no_interaction baseline, make sure the web API is set up.
    if args.baseline == "no_interaction":
        try:
            _http_json(f"{args.base_url.rstrip('/')}/api/vlm/model", timeout=10.0)
        except Exception:
            pass
        requested_option_id = "local" if args.model == "qwen" else "gemini"
        try:
            _select_vlm_option(args.base_url, requested_option_id, timeout=10.0)
        except Exception:
            pass

    engine = _make_baseline_engine(args.baseline, args.base_url, args.database_url)

    results: list[QuestionResult] = []
    category_order: list[str] = []
    category_totals: dict[str, int] = {}
    category_correct: dict[str, int] = {}

    for index, case in enumerate(cases, start=1):
        category = case.category
        if category not in category_totals:
            category_order.append(category)
            category_totals[category] = 0
            category_correct[category] = 0

        result = _run_baseline_question(
            engine,
            case,
            building=args.building,
            timeout=args.timeout_sec,
        )
        result.correct = _is_case_correct(case, result)
        results.append(result)
        category_totals[category] += 1
        if result.correct:
            category_correct[category] += 1

        status = "PASS" if result.correct else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} {result.duration_sec:.2f}s :: {case.prompt}"
        )
        if result.error:
            print(f"  error: {result.error}")
        else:
            print(f"  answer: {result.answer_text}, correct: {result.correct}")

        if args.inter_question_delay_sec > 0 and index < len(cases):
            time.sleep(args.inter_question_delay_sec)

    asked = len(results)
    amount_correct = sum(1 for r in results if r.correct)
    avg_duration = sum(r.duration_sec for r in results) / asked
    avg_tokens = 0.0
    total_function_calls = sum(r.function_call_count for r in results)
    avg_function_calls = total_function_calls / asked
    max_function_calls = max(r.function_call_count for r in results)

    summary_row = {
        "model_used": f"baseline_{baseline_label}",
        "avg_duration_per_question_sec": f"{avg_duration:.4f}",
        "avg_tokens_per_question": f"{avg_tokens:.2f}",
        "avg_function_calls_per_question": f"{avg_function_calls:.2f}",
        "max_function_calls_on_question": max_function_calls,
        "amount_correct": amount_correct,
        "amount_questions_asked": asked,
        "total_function_calls": total_function_calls,
        "current_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question_answers_csv": _display_path(question_answers_csv),
        "reasoning_log_txt": _display_path(reasoning_log_txt),
    }
    summary_headers = list(CSV_HEADERS)
    for category in category_order:
        header = _category_percent_column(category)
        summary_headers.append(header)
        total = category_totals[category]
        correct = category_correct[category]
        summary_row[header] = f"{(correct / total) * 100:.2f}" if total else ""

    grouped_totals = {group: 0 for group in CATEGORY_GROUPS}
    grouped_correct = {group: 0 for group in CATEGORY_GROUPS}
    for case, result in zip(cases, results):
        group = _category_group_for_case(case)
        grouped_totals.setdefault(group, 0)
        grouped_correct.setdefault(group, 0)
        grouped_totals[group] += 1
        if result.correct:
            grouped_correct[group] += 1

    compact_row = {
        "variant": f"baseline_{baseline_label}",
        "Accuracy": _format_accuracy_decimal(amount_correct, asked),
        **{
            group: _format_accuracy_decimal(grouped_correct[group], grouped_totals[group])
            if grouped_totals[group] > 0
            else ""
            for group in CATEGORY_GROUPS
        },
        "function calls": f"{avg_function_calls:.4f}",
        "time": f"{avg_duration:.4f}",
    }

    _append_summary_row(results_csv, summary_row, summary_headers)
    _append_compact_results_row(args.compact_results_csv, compact_row)
    _append_question_rows(
        question_answers_csv,
        [
            {
                "question": result.question,
                "category": case.category,
                "amount_function_calls": result.function_call_count,
                "note": result.note or "",
                "answer": result.answer_text if not result.error else f"ERROR: {result.error}",
                "correct": result.correct,
                "expected": json.dumps(case.expected_answer, ensure_ascii=False),
            }
            for result, case in zip(results, cases)
        ],
    )
    _write_reasoning_log(
        reasoning_log_txt,
        results,
        [
            {"question": case.prompt, "answer": case.expected_answer, "category": case.category}
            for case in cases
        ],
        summary_row,
    )

    print(
        f"\nSummary ({baseline_label}): "
        f"{amount_correct}/{asked} correct, "
        f"avg {avg_duration:.2f}s/question, "
        f"avg {avg_function_calls:.2f} function calls/question, "
        f"max {max_function_calls} function calls on one question"
    )
    print(f"Appended summary row to {results_csv}")
    print(f"Appended compact results row to {args.compact_results_csv}")
    print(f"Appended per-question rows to {question_answers_csv}")
    print(f"Wrote reasoning trace to {reasoning_log_txt}")
    return 0 if amount_correct == asked else 1


if __name__ == "__main__":
    sys.exit(main())
