#!/usr/bin/env python3

import unittest

from pathlib import Path

from evaluate_agent import (
    _default_question_answers_csv,
    _default_reasoning_log_txt,
    _default_results_csv,
    _is_case_correct,
    _navigation_call_matches,
    _normalize_case,
    _resolve_question_answers_csv,
    _resolve_reasoning_log_txt,
    _resolve_results_csv,
    QuestionResult,
)


def _result_with_tool_log(tool_log):
    return QuestionResult(
        question="test",
        answer_text="test",
        correct=False,
        duration_sec=0.1,
        total_tokens=0,
        function_call_count=len(tool_log),
        model="test",
        tool_log=tool_log,
        trace=[],
        system_prompt=None,
    )


class TestEvaluateAgent(unittest.TestCase):
    @staticmethod
    def _navigation_case():
        return _normalize_case(
            {
                "task": "Navigate to Simba.",
                "expected_answer": {"navigation_call": {"function": "navigate_to", "arguments": {"x": 1.0, "y": 2.0}}},
                "category": "person_location",
            }
        )

    def test_navigation_runs_default_to_navigation_results_csv(self):
        navigation_case = self._navigation_case()

        results_csv = _resolve_results_csv(_default_results_csv(), [navigation_case])

        self.assertEqual(results_csv.name, "test_results_navigation.csv")

    def test_explicit_results_csv_is_preserved_for_navigation_runs(self):
        navigation_case = self._navigation_case()

        explicit_path = Path("custom_navigation_results.csv")
        results_csv = _resolve_results_csv(explicit_path, [navigation_case])

        self.assertEqual(results_csv, explicit_path)

    def test_navigation_runs_default_to_navigation_question_artifacts(self):
        navigation_case = self._navigation_case()

        question_answers_csv = _resolve_question_answers_csv(_default_question_answers_csv(), [navigation_case])
        reasoning_log_txt = _resolve_reasoning_log_txt(_default_reasoning_log_txt(), [navigation_case])

        self.assertEqual(question_answers_csv.parts[-2:], ("navigation", "question_answers_timestamp.csv"))
        self.assertEqual(reasoning_log_txt.parts[-2:], ("navigation", "question_reasoning_timestamp.txt"))

    def test_explicit_question_artifact_paths_are_preserved_for_navigation_runs(self):
        navigation_case = self._navigation_case()

        explicit_csv = Path("custom/navigation_answers.csv")
        explicit_txt = Path("custom/navigation_reasoning.txt")

        question_answers_csv = _resolve_question_answers_csv(explicit_csv, [navigation_case])
        reasoning_log_txt = _resolve_reasoning_log_txt(explicit_txt, [navigation_case])

        self.assertEqual(question_answers_csv, explicit_csv)
        self.assertEqual(reasoning_log_txt, explicit_txt)

    def test_normalize_case_supports_question_answer_schema(self):
        case = _normalize_case(
            {
                "question": "Who owns bottle_001?",
                "answer": {"owner_name": "Simba"},
                "category": "ownership",
            }
        )

        self.assertEqual(case.prompt, "Who owns bottle_001?")
        self.assertEqual(case.expected_answer, {"owner_name": "Simba"})
        self.assertEqual(case.category, "ownership")
        self.assertEqual(case.mode, "answer_only")

    def test_normalize_case_supports_navigation_schema(self):
        case = _normalize_case(
            {
                "task": "Navigate to Simba.",
                "expected_answer": {"navigation_call": {"function": "navigate_to", "arguments": {"x": 1.0, "y": 2.0}}},
                "category": "person_location",
            }
        )

        self.assertEqual(case.prompt, "Navigate to Simba.")
        self.assertEqual(case.category, "person_location")
        self.assertEqual(case.mode, "navigation")

    def test_navigation_call_matches_move_to_position_alias(self):
        expected = {
            "navigation_call": {
                "function": "navigate_to",
                "arguments": {
                    "x": 4.65,
                    "y": 8.30,
                    "reason": "Last known observation of Simba.",
                },
            }
        }
        tool_log = [
            {
                "tool": "move_to_position",
                "args": {
                    "x": 4.651,
                    "y": 8.299,
                    "reason": "Last known observation of Simba.",
                },
                "result": {"ok": True},
            }
        ]

        self.assertTrue(_navigation_call_matches(expected, tool_log))

    def test_navigation_call_ignores_reason_text(self):
        expected = {
            "navigation_call": {
                "function": "navigate_to",
                "arguments": {
                    "x": 2.38,
                    "y": 2.42,
                    "reason": "Mufasa is the most recent person observed using coffee_machine_001.",
                },
            }
        }
        tool_log = [
            {
                "tool": "move_to_position",
                "args": {
                    "x": 2.38,
                    "y": 2.42,
                    "reason": "location of the person who most recently used the coffee machine",
                },
                "result": {"ok": True},
            }
        ]

        self.assertTrue(_navigation_call_matches(expected, tool_log))

    def test_navigation_call_matches_any_valid_option(self):
        expected = {
            "target_type": "multiple_valid_locations",
            "navigation_call_options": [
                {"function": "navigate_to", "arguments": {"x": 8.2, "y": 3.4, "reason": "One valid location where Nala and Scar talked."}},
                {"function": "navigate_to", "arguments": {"x": 6.1, "y": 9.1, "reason": "Another valid location where Nala and Scar talked."}},
            ],
        }
        tool_log = [
            {
                "tool": "move_to_position",
                "args": {"x": 6.1, "y": 9.1, "reason": "Another valid location where Nala and Scar talked."},
                "result": {"ok": True},
            }
        ]

        self.assertTrue(_navigation_call_matches(expected, tool_log))

    def test_is_case_correct_uses_text_for_standard_questions(self):
        case = _normalize_case(
            {
                "question": "Who owns bottle_001?",
                "answer": {"owner_name": "Simba"},
                "category": "ownership",
            }
        )

        result = QuestionResult(
            question=case.prompt,
            answer_text="Simba owns bottle_001.",
            correct=False,
            duration_sec=0.1,
            total_tokens=0,
            function_call_count=0,
            model="test",
            tool_log=[],
            trace=[],
            system_prompt=None,
        )

        self.assertTrue(_is_case_correct(case, result))

    def test_is_case_correct_uses_tool_log_for_navigation(self):
        case = _normalize_case(
            {
                "task": "Navigate to Simba.",
                "expected_answer": {
                    "navigation_call": {
                        "function": "navigate_to",
                        "arguments": {"x": 4.65, "y": 8.3, "reason": "Last known observation of Simba."},
                    }
                },
                "category": "person_location",
            }
        )

        result = _result_with_tool_log(
            [
                {
                    "tool": "move_to_position",
                    "args": {"x": 4.65, "y": 8.3, "reason": "Last known observation of Simba."},
                    "result": {"ok": True},
                }
            ]
        )
        result.answer_text = "I can move there."

        self.assertTrue(_is_case_correct(case, result))


if __name__ == "__main__":
    unittest.main()