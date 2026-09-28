#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from math import isclose
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:8080"
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:35432/bordsupr"
DEFAULT_INTER_QUESTION_DELAY_SEC = 2.0
DEFAULT_RETRY_COUNT = 2
DEFAULT_RETRY_DELAY_SEC = 2.0
BASELINE_VARIANTS = ("direct_llm", "rule_based", "no_interaction", "heuristic")
EVALUATION_VARIANTS = ("qwen", *BASELINE_VARIANTS)
STOPWORDS = {"a", "an", "and", "or", "the", "to", "of", "at", "in", "on", "for", "by"}
IGNORED_EXPECTED_KEYS = {
    "action",
    "interaction_id",
    "interaction_ids",
    "object_id",
    "observation_id",
    "person_id",
    "scene_id",
    "start_interaction_id",
    "end_interaction_id",
    "time",
    "start_time",
    "end_time",
    "note",
}
CSV_HEADERS = [
    "model_used",
    "avg_duration_per_question_sec",
    "avg_tokens_per_question",
    "avg_function_calls_per_question",
    "max_function_calls_on_question",
    "amount_correct",
    "amount_questions_asked",
    "total_function_calls",
    "current_time",
    "question_answers_csv",
    "reasoning_log_txt",
]
CATEGORY_GROUPS = ("Interaction", "Location", "Lookup", "Ownership", "Navigation")
COMPACT_RESULTS_HEADERS = [
    "variant",
    "Accuracy",
    *CATEGORY_GROUPS,
    "function calls",
    "time",
]
CATEGORY_TO_GROUP = {
    "entity_lookup": "Lookup",
    "object_location": "Location",
    "person_location": "Location",
    "time_location": "Location",
    "object_interaction": "Interaction",
    "person_interaction": "Interaction",
    "person_interaction_summary": "Interaction",
    "object_interaction_summary": "Interaction",
    "interaction_location_time": "Interaction",
    "scene_summary": "Interaction",
    "recent_interactions": "Interaction",
    "ownership": "Ownership",
    "ownership_reasoning": "Ownership",
    "ownership_and_transport": "Ownership",
    "primary_user": "Ownership",
}
QUESTION_ANSWER_HEADERS = [
    "question",
    "category",
    "amount_function_calls",
    "note",
    "answer",
    "correct",
    "expected",
]


@dataclass
class QuestionResult:
    question: str
    answer_text: str
    correct: bool
    duration_sec: float
    total_tokens: int
    function_call_count: int
    model: str
    tool_log: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    system_prompt: str | None
    error: str | None = None
    note: str | None = None


@dataclass
class EvaluationCase:
    prompt: str
    expected_answer: Any
    category: str
    mode: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_results_csv() -> Path:
    return _repo_root() / "evaluation_outputs" / "test_results.csv"


def _default_navigation_results_csv() -> Path:
    return _repo_root() / "evaluation_outputs" / "test_results_navigation.csv"


def _default_compact_results_csv() -> Path:
    return _repo_root() / "evaluation_outputs" / "compact_results.csv"


def _default_question_answers_csv() -> Path:
    return _repo_root() / "evaluation_outputs" / "questions" / "question_answers_timestamp.csv"


def _default_navigation_question_answers_csv() -> Path:
    return _repo_root() / "evaluation_outputs" / "questions" / "navigation" / "question_answers_timestamp.csv"


def _default_reasoning_log_txt() -> Path:
    return _repo_root() / "evaluation_outputs" / "questions" / "question_reasoning_timestamp.txt"


def _default_navigation_reasoning_log_txt() -> Path:
    return _repo_root() / "evaluation_outputs" / "questions" / "navigation" / "question_reasoning_timestamp.txt"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(_repo_root()))
    except ValueError:
        return str(path)


def _normalize_text(value: Any) -> str:
    text = str(value or "").lower().replace("_", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _tokenize(value: Any) -> list[str]:
    return [token for token in _normalize_text(value).split() if token]


def _string_match(answer_norm: str, expected: str) -> bool:
    expected_norm = _normalize_text(expected)
    if not expected_norm:
        return False
    if expected_norm in answer_norm:
        return True

    answer_tokens = set(answer_norm.split())
    expected_tokens = [token for token in expected_norm.split() if token not in STOPWORDS]
    if not expected_tokens:
        expected_tokens = expected_norm.split()
    if not expected_tokens:
        return False

    matched = sum(1 for token in expected_tokens if token in answer_tokens)
    if len(expected_tokens) == 1:
        return matched == 1
    return (matched / len(expected_tokens)) >= 0.5


def _question_mentions(question: str, phrase: str) -> bool:
    question_norm = _normalize_text(question)
    return _string_match(question_norm, phrase)


def _supports_null_answer(question: str, answer_norm: str) -> bool:
    phrases = [
        "unknown",
        "not known",
        "no owner",
        "no single owner",
        "does not have an owner",
        "doesnt have an owner",
        "not owned by anyone",
        "shared",
        "public",
        "cannot determine",
        "not enough information",
        "none",
    ]
    if "own" not in question.lower():
        phrases.extend(["not found", "no record"])
    return any(phrase in answer_norm for phrase in phrases)


def _contains_answer_marker(answer_norm: str, markers: list[str]) -> bool:
    answer_tokens = set(answer_norm.split())
    for marker in markers:
        marker_norm = _normalize_text(marker)
        if not marker_norm:
            continue
        if " " in marker_norm:
            if marker_norm in answer_norm:
                return True
            continue
        if marker_norm in answer_tokens:
            return True
    return False


def _supports_bool_answer(answer_norm: str, expected: bool) -> bool:
    error_markers = [
        "internal assistant error",
        "connection error",
        "request failed",
        "timed out",
        "timeout",
        "exception",
    ]
    positive_markers = ["yes", "true", "shared public", "shared", "public"]
    negative_markers = ["no", "false", "not shared", "not public", "private", "personal"]
    if _contains_answer_marker(answer_norm, error_markers):
        return False
    if expected:
        return _contains_answer_marker(answer_norm, positive_markers)
    return _contains_answer_marker(answer_norm, negative_markers)


def _collect_salient_phrases(question: str, expected: Any) -> list[str]:
    question_lower = question.lower()
    wants_person = question_lower.startswith("who") or " which people" in question_lower
    wants_location = "where" in question_lower or "which room" in question_lower
    wants_object = "interacting with" in question_lower or "what was" in question_lower
    wants_duration = "how long" in question_lower
    wants_count = "how many" in question_lower
    wants_relationship = "relationship" in question_lower
    wants_shared_state = "shared public object" in question_lower
    phrases: list[str] = []

    def add_phrase(value: Any) -> None:
        text = str(value).strip()
        if not text:
            return
        if text not in phrases:
            phrases.append(text)

    def visit(node: Any, parent_key: str | None = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in IGNORED_EXPECTED_KEYS:
                    continue
                if key in {"name", "owner_name", "other_name"}:
                    if wants_person and not _question_mentions(question, str(value)):
                        add_phrase(value)
                    continue
                if key == "people" and isinstance(value, list):
                    if wants_person:
                        for item in value:
                            if not _question_mentions(question, str(item)):
                                add_phrase(item)
                    continue
                if key in {"room", "from_room", "to_room"}:
                    if wants_location:
                        add_phrase(value)
                    continue
                if key == "object_name":
                    if wants_object and not _question_mentions(question, str(value)):
                        add_phrase(value)
                    continue
                if key == "duration_minutes":
                    if wants_duration:
                        add_phrase(str(value))
                        add_phrase(f"{value} minutes")
                    continue
                if key == "count":
                    if wants_count:
                        add_phrase(str(value))
                    continue
                if key == "relationship_inference":
                    if wants_relationship:
                        add_phrase(value)
                    continue
                if key == "ownership_type":
                    if wants_shared_state:
                        add_phrase(value)
                    continue
                visit(value, key)
            return

        if isinstance(node, list):
            for item in node:
                visit(item, parent_key)
            return

        if isinstance(node, str) and parent_key not in IGNORED_EXPECTED_KEYS:
            if parent_key in {"room", "from_room", "to_room"} and wants_location:
                add_phrase(node)
            elif parent_key == "object_name" and wants_object and not _question_mentions(question, node):
                add_phrase(node)
            elif parent_key in {"name", "owner_name", "other_name"} and wants_person and not _question_mentions(question, node):
                add_phrase(node)

    visit(expected)

    if not phrases:
        string_values = []

        def collect_strings(node: Any, parent_key: str | None = None) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    if key in IGNORED_EXPECTED_KEYS:
                        continue
                    collect_strings(value, key)
                return
            if isinstance(node, list):
                for item in node:
                    collect_strings(item, parent_key)
                return
            if isinstance(node, str) and parent_key not in IGNORED_EXPECTED_KEYS:
                if not _question_mentions(question, node):
                    string_values.append(node)

        collect_strings(expected)
        for value in string_values:
            add_phrase(value)

    return phrases


def _is_answer_correct(question: str, expected: Any, answer_text: str) -> bool:
    answer_norm = _normalize_text(answer_text)
    if not answer_norm:
        return False
    if expected is None:
        return _supports_null_answer(question, answer_norm)
    if isinstance(expected, bool):
        return _supports_bool_answer(answer_norm, expected)

    phrases = _collect_salient_phrases(question, expected)
    if not phrases:
        return False
    return all(_string_match(answer_norm, phrase) for phrase in phrases)


def _http_json(url: str, *, method: str = "GET", payload: dict | None = None, timeout: float = 120.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc


def _select_vlm_option(base_url: str, option_id: str, timeout: float = 10.0) -> dict:
    response = _http_json(
        f"{base_url.rstrip('/')}/api/vlm/selection",
        method="POST",
        payload={"option_id": option_id},
        timeout=timeout,
    )
    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected VLM selection response for option '{option_id}': {response!r}")
    return response


def _load_test_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        cases = payload
    elif isinstance(payload, dict):
        cases = payload.get("test_case_answers")
    else:
        cases = None

    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain either a top-level list or a 'test_case_answers' list")
    return cases


def _normalize_case(case: dict[str, Any]) -> EvaluationCase:
    if not isinstance(case, dict):
        raise ValueError(f"Each test case must be a JSON object, got {type(case).__name__}")

    if "question" in case:
        prompt = str(case.get("question") or "").strip()
        expected_answer = case.get("answer")
        mode = "answer_only"
    elif "task" in case:
        prompt = str(case.get("task") or "").strip()
        expected_answer = case.get("expected_answer")
        mode = "navigation"
    else:
        raise ValueError("Each test case must contain either 'question' or 'task'")

    category = str(case.get("category") or "uncategorized").strip() or "uncategorized"
    if not prompt:
        raise ValueError("Each test case must contain a non-empty prompt")

    return EvaluationCase(
        prompt=prompt,
        expected_answer=expected_answer,
        category=category,
        mode=mode,
    )


def _normalize_tool_name(tool_name: Any) -> str:
    value = str(tool_name or "").strip().lower()
    if value == "navigate_to":
        return "move_to_position"
    return value


def _float_matches(actual: Any, expected: Any, tolerance: float = 0.05) -> bool:
    try:
        return isclose(float(actual), float(expected), abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _tool_call_matches(expected_call: dict[str, Any], tool_entry: dict[str, Any]) -> bool:
    expected_function = _normalize_tool_name(expected_call.get("function"))
    actual_function = _normalize_tool_name(tool_entry.get("tool"))
    if not expected_function or actual_function != expected_function:
        return False

    expected_args = expected_call.get("arguments") or {}
    actual_args = tool_entry.get("args") or {}
    if not isinstance(expected_args, dict) or not isinstance(actual_args, dict):
        return False

    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if key in {"x", "y", "z"}:
            if not _float_matches(actual_value, expected_value):
                return False
            continue
        if key == "reason":
            continue
        if actual_value != expected_value:
            return False

    return True


def _navigation_call_matches(expected: Any, tool_log: list[dict[str, Any]]) -> bool:
    if not isinstance(expected, dict):
        return False

    if expected.get("target_type") == "multiple_valid_locations":
        options = expected.get("navigation_call_options") or []
        if not isinstance(options, list) or not options:
            return False
        return any(
            _tool_call_matches(option, tool_entry)
            for option in options
            if isinstance(option, dict)
            for tool_entry in tool_log
            if isinstance(tool_entry, dict)
        )

    expected_call = expected.get("navigation_call")
    if not isinstance(expected_call, dict):
        return False
    return any(
        _tool_call_matches(expected_call, tool_entry)
        for tool_entry in tool_log
        if isinstance(tool_entry, dict)
    )


def _is_case_correct(case: EvaluationCase, result: QuestionResult) -> bool:
    if case.mode == "navigation":
        return _navigation_call_matches(case.expected_answer, result.tool_log)
    return _is_answer_correct(case.prompt, case.expected_answer, result.answer_text)


def _sanitize_column_token(value: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return token or "uncategorized"


def _category_percent_column(category: str) -> str:
    return f"category_{_sanitize_column_token(category)}_correct_pct"


def _expected_contains_any_key(value: Any, keys: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in keys or _expected_contains_any_key(child, keys) for key, child in value.items())
    if isinstance(value, list):
        return any(_expected_contains_any_key(item, keys) for item in value)
    return False


def _question_refers_to_owned_object(case: EvaluationCase) -> bool:
    if re.search(r"\b[\w-]+(?:'s|’s)\s+[\w-]+", case.prompt):
        return _expected_contains_any_key(
            case.expected_answer,
            {"object_id", "object_name", "resolved_owner", "owner_name", "ownership", "ownership_basis"},
        )
    return False


def _category_group_for_case(case: EvaluationCase) -> str:
    category_token = _sanitize_column_token(case.category)
    if category_token in CATEGORY_TO_GROUP:
        return CATEGORY_TO_GROUP[category_token]
    if _question_refers_to_owned_object(case):
        return "Ownership"
    if case.mode == "navigation":
        return "Navigation"
    question = case.prompt.lower()
    if any(phrase in question for phrase in ("go to", "navigate to", "move to")):
        return "Navigation"
    if "own" in question or "primary user" in question:
        return "Ownership"
    return case.category or "uncategorized"


def _format_accuracy(correct: int, total: int) -> str:
    pct = (correct / total * 100.0) if total else 0.0
    return f"{correct}/{total} ({pct:.2f}%)"


def _format_accuracy_decimal(correct: int, total: int) -> str:
    accuracy = (correct / total) if total else 0.0
    return f"{accuracy:.4f}"


def _evaluation_type(cases: list[EvaluationCase]) -> str:
    return "navigation" if cases and all(case.mode == "navigation" for case in cases) else "overall"


def _resolve_results_csv(results_csv: Path, cases: list[EvaluationCase]) -> Path:
    if results_csv != _default_results_csv():
        return results_csv
    if cases and all(case.mode == "navigation" for case in cases):
        return _default_navigation_results_csv()
    return results_csv


def _resolve_question_answers_csv(question_answers_csv: Path, cases: list[EvaluationCase]) -> Path:
    if question_answers_csv != _default_question_answers_csv():
        return question_answers_csv
    if cases and all(case.mode == "navigation" for case in cases):
        return _default_navigation_question_answers_csv()
    return question_answers_csv


def _resolve_reasoning_log_txt(reasoning_log_txt: Path, cases: list[EvaluationCase]) -> Path:
    if reasoning_log_txt != _default_reasoning_log_txt():
        return reasoning_log_txt
    if cases and all(case.mode == "navigation" for case in cases):
        return _default_navigation_reasoning_log_txt()
    return reasoning_log_txt


def _read_existing_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return [], []

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def _append_rows_with_schema_upgrade(
    csv_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_fieldnames, existing_rows = _read_existing_csv(csv_path)

    effective_fieldnames = list(existing_fieldnames)
    if not effective_fieldnames:
        effective_fieldnames = list(fieldnames)
    else:
        for fieldname in fieldnames:
            if fieldname not in effective_fieldnames:
                effective_fieldnames.append(fieldname)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=effective_fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            writer.writerow({field: existing_row.get(field, "") for field in effective_fieldnames})
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in effective_fieldnames})


def _append_summary_row(csv_path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    _append_rows_with_schema_upgrade(csv_path, [row], fieldnames)


def _append_question_rows(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    _append_rows_with_schema_upgrade(csv_path, rows, QUESTION_ANSWER_HEADERS)


def _append_compact_results_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(COMPACT_RESULTS_HEADERS)
    _, existing_rows = _read_existing_csv(csv_path)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for existing_row in existing_rows:
            if existing_row.get("Accuracy") == "Accuracy":
                continue
            writer.writerow({field: existing_row.get(field, "") for field in fieldnames})
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def _format_trace_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_reasoning_log(txt_path: Path, results: list[QuestionResult], cases: list[dict[str, Any]], summary_row: dict[str, Any]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    system_prompt = next((result.system_prompt for result in results if result.system_prompt), None)
    lines = [
        "Website agent evaluation trace",
        f"Generated at: {summary_row['current_time']}",
        f"Model: {summary_row['model_used']}",
        f"Questions asked: {summary_row['amount_questions_asked']}",
        f"Correct answers: {summary_row['amount_correct']}",
        f"Total function calls: {summary_row['total_function_calls']}",
        f"Average function calls per question: {summary_row['avg_function_calls_per_question']}",
        f"Max function calls on a question: {summary_row['max_function_calls_on_question']}",
        "",
    ]
    if system_prompt:
        lines.extend(
            [
                "System prompt:",
                system_prompt,
                "",
            ]
        )

    for index, (result, case) in enumerate(zip(results, cases), start=1):
        category = str(case.get("category") or "uncategorized")
        lines.extend(
            [
                f"Question {index}",
                f"Category: {category}",
                f"Correct: {result.correct}",
                f"Function calls: {result.function_call_count}",
                f"Note: {result.note or ''}",
                f"Question text: {result.question}",
                "Expected:",
                _format_trace_block(case.get("answer")),
            ]
        )
        if result.trace:
            lines.append("Model trace:")
            for step in result.trace:
                lines.extend(
                    [
                        f"  Step {step.get('iteration', '?')}",
                        "  Assistant reasoning before tool calls:",
                        _format_trace_block(step.get("assistant_content") or ""),
                    ]
                )
                planned_tool_calls = step.get("planned_tool_calls") or []
                if planned_tool_calls:
                    lines.append("  Planned tool calls:")
                    for call_index, planned in enumerate(planned_tool_calls, start=1):
                        lines.extend(
                            [
                                f"    Planned call {call_index}: {planned.get('tool', 'unknown_tool')}",
                                "    Args:",
                                _format_trace_block(planned.get("args") or {}),
                            ]
                        )
                tool_results = step.get("tool_results") or []
                if tool_results:
                    lines.append("  Tool results:")
                    for call_index, tool_entry in enumerate(tool_results, start=1):
                        lines.extend(
                            [
                                f"    Call {call_index}: {tool_entry.get('tool', 'unknown_tool')}",
                                "    Args:",
                                _format_trace_block(tool_entry.get("args") or {}),
                                "    Result:",
                                _format_trace_block(tool_entry.get("result") or {}),
                            ]
                        )
                if step.get("final_answer"):
                    lines.append("  Final answer step.")
        else:
            lines.extend(
                [
                    "Answer / reasoning:",
                    result.answer_text if not result.error else f"ERROR: {result.error}",
                    "Queries and tool results:",
                ]
            )
            if result.tool_log:
                for call_index, tool_entry in enumerate(result.tool_log, start=1):
                    lines.extend(
                        [
                            f"  Call {call_index}: {tool_entry.get('tool', 'unknown_tool')}",
                            "  Args:",
                            _format_trace_block(tool_entry.get("args") or {}),
                            "  Result:",
                            _format_trace_block(tool_entry.get("result") or {}),
                        ]
                    )
            else:
                lines.append("  No function calls recorded.")
        lines.extend(
            [
                "Final answer returned to evaluator:",
                result.answer_text if not result.error else f"ERROR: {result.error}",
            ]
        )
        lines.append("")

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def _run_question(
    base_url: str,
    question: str,
    expected: Any,
    timeout: float,
    fallback_model: str,
    building: str | None = None,
    mode: str = "answer_only",
    retries: int = DEFAULT_RETRY_COUNT,
    retry_delay_sec: float = DEFAULT_RETRY_DELAY_SEC,
) -> QuestionResult:
    start = time.perf_counter()
    error: str | None = None
    answer_text = ""
    total_tokens = 0
    tool_log: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    system_prompt: str | None = None
    model = fallback_model
    note: str | None = None

    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            response = _http_json(
                f"{base_url.rstrip('/')}/api/chat",
                method="POST",
                payload={
                    "question": question,
                    "history": [],
                    "building": building,
                    "mode": "navigation" if mode == "navigation" else None,
                },
                timeout=timeout,
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                response = None
                break
            time.sleep(max(0.0, retry_delay_sec))

    try:
        if response is None:
            raise last_error or RuntimeError("Request failed")
        answer_text = str(response.get("answer") or "")
        raw_tool_log = response.get("tool_log") or []
        if isinstance(raw_tool_log, list):
            tool_log = [item for item in raw_tool_log if isinstance(item, dict)]
        raw_trace = response.get("trace") or []
        if isinstance(raw_trace, list):
            trace = [item for item in raw_trace if isinstance(item, dict)]
        response_note = response.get("note")
        if response_note is not None:
            note = str(response_note).strip() or None
        response_system_prompt = response.get("system_prompt")
        if response_system_prompt is not None:
            system_prompt = str(response_system_prompt)
        usage = response.get("usage") or {}
        total_tokens = int(usage.get("total_tokens") or 0)
        model = str(response.get("model") or fallback_model or "unknown")
        correct = False
    except Exception as exc:
        error = str(exc)
        correct = False

    duration_sec = time.perf_counter() - start
    return QuestionResult(
        question=question,
        answer_text=answer_text,
        correct=correct,
        duration_sec=duration_sec,
        total_tokens=total_tokens,
        function_call_count=len(tool_log),
        model=model or "unknown",
        tool_log=tool_log,
        trace=trace,
        system_prompt=system_prompt,
        error=error,
        note=note,
    )


def _baseline_engine(variant: str, base_url: str) -> Any:
    os.environ.setdefault("DATABASE_URL", DEFAULT_DATABASE_URL)
    from baseline_engines import DirectLLMBaseline, HeuristicBaseline, NoInteractionBaseline, RuleBasedBaseline

    if variant == "direct_llm":
        return DirectLLMBaseline(base_url=base_url)
    if variant == "rule_based":
        return RuleBasedBaseline()
    if variant == "no_interaction":
        return NoInteractionBaseline(base_url=base_url)
    if variant == "heuristic":
        return HeuristicBaseline()
    raise ValueError(f"Unknown baseline variant: {variant}")


def _run_baseline_question(
    engine: Any,
    variant: str,
    question: str,
    building: str | None = None,
) -> QuestionResult:
    start = time.perf_counter()
    answer_text = ""
    tool_log: list[dict[str, Any]] = []
    error: str | None = None
    try:
        answer_text, raw_tool_log = engine.answer(question, building=building)
        if isinstance(raw_tool_log, list):
            tool_log = [item for item in raw_tool_log if isinstance(item, dict)]
    except Exception as exc:
        error = str(exc)
        answer_text = f"ERROR: {error}"

    return QuestionResult(
        question=question,
        answer_text=str(answer_text or ""),
        correct=False,
        duration_sec=time.perf_counter() - start,
        total_tokens=0,
        function_call_count=len(tool_log),
        model=variant,
        tool_log=tool_log,
        trace=[],
        system_prompt=None,
        error=error,
        note=variant,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask the website agent a fixed question set and append summary metrics to a results CSV.",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the website to test (default: %(default)s)",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=_repo_root() / "data" / "evaluation" / "overall_questions.json",
        help="Path to the JSON file containing either a top-level question list or a 'test_case_answers' list.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=_default_results_csv(),
        help=(
            "CSV file to append the summary row to. "
            "When omitted, navigation-only runs use test_results_navigation.csv."
        ),
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
        help=(
            "CSV file to append per-question answers to. "
            "When omitted, navigation-only runs use questions/navigation/."
        ),
    )
    parser.add_argument(
        "--reasoning-log-txt",
        type=Path,
        default=_default_reasoning_log_txt(),
        help=(
            "Text file to write per-question answer reasoning and tool queries to. "
            "When omitted, navigation-only runs use questions/navigation/."
        ),
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=120.0,
        help="Per-question HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--inter-question-delay-sec",
        type=float,
        default=DEFAULT_INTER_QUESTION_DELAY_SEC,
        help="Seconds to wait between asking consecutive questions.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRY_COUNT,
        help="Number of retries for transient per-question /api/chat failures.",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=DEFAULT_RETRY_DELAY_SEC,
        help="Seconds to wait before retrying a transient per-question failure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the number of questions to ask.",
    )
    parser.add_argument(
        "--building",
        default=None,
        help="Optional building/map name to scope the website agent to for every question.",
    )
    parser.add_argument(
        "--model",
        choices=("qwen", "gemini"),
        default="qwen",
        help="VLM backend to use for evaluation. Defaults to qwen, which selects the local Qwen backend.",
    )
    parser.add_argument(
        "--variant",
        choices=(*EVALUATION_VARIANTS, "all"),
        default="qwen",
        help=(
            "Evaluation variant to run. Use qwen for the website agent, one baseline name, "
            "or all to run qwen plus every baseline."
        ),
    )
    return parser


def _qwen_fallback_model(args: argparse.Namespace) -> str:
    model_response = {}
    try:
        model_response = _http_json(f"{args.base_url.rstrip('/')}/api/vlm/model", timeout=10.0)
    except Exception:
        model_response = {}
    requested_option_id = "local" if args.model == "qwen" else "gemini"
    selection_state = _select_vlm_option(args.base_url, requested_option_id, timeout=10.0)
    active_option = selection_state.get("active_option") if isinstance(selection_state, dict) else {}
    fallback_model = str(
        (active_option or {}).get("model")
        or model_response.get("model")
        or args.model
        or "unknown"
    )
    return fallback_model


def _variant_output_path(path: Path, variant_label: str) -> Path:
    if "timestamp" in path.name:
        timestamp_token = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        return path.with_name(path.name.replace("timestamp", f"{timestamp_token}_{variant_label}"))
    return path


def _run_evaluation_variant(
    args: argparse.Namespace,
    cases: list[EvaluationCase],
    variant: str,
    results_csv: Path,
    question_answers_csv_template: Path,
    reasoning_log_txt_template: Path,
) -> bool:
    eval_type = _evaluation_type(cases)
    variant_label = f"{variant}_{eval_type}"
    fallback_model = _qwen_fallback_model(args) if variant == "qwen" else variant
    baseline_engine = None if variant == "qwen" else _baseline_engine(variant, args.base_url)

    results: list[QuestionResult] = []
    category_order: list[str] = []
    category_totals: dict[str, int] = {}
    category_correct: dict[str, int] = {}
    for index, case in enumerate(cases, start=1):
        question = case.prompt
        expected = case.expected_answer
        category = case.category
        if category not in category_totals:
            category_order.append(category)
            category_totals[category] = 0
            category_correct[category] = 0

        if baseline_engine is None:
            result = _run_question(
                args.base_url,
                question,
                expected,
                args.timeout_sec,
                fallback_model,
                building=args.building,
                mode=case.mode,
                retries=max(0, args.retries),
                retry_delay_sec=args.retry_delay_sec,
            )
        else:
            result = _run_baseline_question(
                baseline_engine,
                variant,
                question,
                building=args.building,
            )
        result.correct = _is_case_correct(case, result)
        results.append(result)
        category_totals[category] += 1
        if result.correct:
            category_correct[category] += 1

        status = "PASS" if result.correct else "FAIL"
        token_label = result.total_tokens if result.total_tokens else "n/a"
        print(
            f"[{variant_label} {index}/{len(cases)}] "
            f"{status} {result.duration_sec:.2f}s {token_label} tokens :: {question}"
        )
        if result.note:
            print(f"  note: {result.note}")
        if result.error:
            print(f"  error: {result.error}")
        else:
            print(f"  answer: {result.answer_text}, correct: {result.correct}, expected: {expected}")

        if args.inter_question_delay_sec > 0 and index < len(cases):
            print(f"  waiting {args.inter_question_delay_sec:.2f}s before next question")
            time.sleep(args.inter_question_delay_sec)

    asked = len(results)
    amount_correct = sum(1 for result in results if result.correct)
    avg_duration = sum(result.duration_sec for result in results) / asked
    avg_tokens = sum(result.total_tokens for result in results) / asked
    total_function_calls = sum(result.function_call_count for result in results)
    avg_function_calls = total_function_calls / asked
    max_function_calls = max(result.function_call_count for result in results)
    models_seen = sorted({result.model for result in results if result.model and result.model != "unknown"})
    model_used = ", ".join(models_seen) if models_seen else fallback_model
    summary_row = {
        "model_used": model_used or "unknown",
        "avg_duration_per_question_sec": f"{avg_duration:.4f}",
        "avg_tokens_per_question": f"{avg_tokens:.2f}",
        "avg_function_calls_per_question": f"{avg_function_calls:.2f}",
        "max_function_calls_on_question": max_function_calls,
        "amount_correct": amount_correct,
        "amount_questions_asked": asked,
        "total_function_calls": total_function_calls,
        "current_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question_answers_csv": "",
        "reasoning_log_txt": "",
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
    grouped_accuracy = {
        group: _format_accuracy_decimal(grouped_correct[group], grouped_totals[group])
        for group in grouped_totals
        if grouped_totals[group] > 0
    }
    compact_row = {
        "variant": variant_label,
        "Accuracy": _format_accuracy_decimal(amount_correct, asked),
        **{group: grouped_accuracy.get(group, "") for group in CATEGORY_GROUPS},
        "function calls": f"{avg_function_calls:.4f}",
        "time": f"{avg_duration:.4f}",
    }
    question_answers_csv = _variant_output_path(question_answers_csv_template, variant_label)
    reasoning_log_txt = _variant_output_path(reasoning_log_txt_template, variant_label)
    summary_row["question_answers_csv"] = _display_path(question_answers_csv)
    summary_row["reasoning_log_txt"] = _display_path(reasoning_log_txt)
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
        "\nSummary: "
        f"{amount_correct}/{asked} correct, "
        f"avg {avg_duration:.2f}s/question, "
        f"avg {avg_tokens:.2f} tokens/question, "
        f"avg {avg_function_calls:.2f} function calls/question, "
        f"max {max_function_calls} function calls on one question, "
        f"variant={variant_label}, "
        f"model={summary_row['model_used']}"
    )
    print(f"Appended summary row to {results_csv}")
    print(f"Appended compact results row to {args.compact_results_csv}")
    print(f"Appended per-question rows to {question_answers_csv}")
    print(f"Wrote reasoning and query trace to {reasoning_log_txt}")
    return amount_correct == asked


def main() -> int:
    args = build_arg_parser().parse_args()
    cases = [_normalize_case(case) for case in _load_test_cases(args.questions_file)]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise ValueError("No questions selected for evaluation")

    results_csv = _resolve_results_csv(args.results_csv, cases)
    question_answers_csv = _resolve_question_answers_csv(args.question_answers_csv, cases)
    reasoning_log_txt = _resolve_reasoning_log_txt(args.reasoning_log_txt, cases)
    variants = list(EVALUATION_VARIANTS) if args.variant == "all" else [args.variant]
    all_correct = True
    for variant in variants:
        all_correct = (
            _run_evaluation_variant(
                args,
                cases,
                variant,
                results_csv,
                question_answers_csv,
                reasoning_log_txt,
            )
            and all_correct
        )
    return 0 if all_correct else 1


if __name__ == "__main__":
    sys.exit(main())
