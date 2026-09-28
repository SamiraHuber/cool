import types
import unittest
from unittest.mock import MagicMock, patch

import sys

if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = MagicMock()
    sys.modules["psycopg2"] = psycopg2_stub

if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")
    openai_stub.BadRequestError = RuntimeError
    openai_stub.OpenAI = MagicMock()
    sys.modules["openai"] = openai_stub

import app  # noqa: E402
from agent import agent as agent_module  # noqa: E402
from agent import tools as agent_tools  # noqa: E402


class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)
        self._index = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self.last_query = query
        self.last_params = params

    def fetchone(self):
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    def __init__(self, rows):
        self._rows = rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return _FakeCursor(self._rows)

    def commit(self):
        return None


class TestRoomNavigationTool(unittest.TestCase):
    def test_missing_position_source_is_navigation_usable_for_legacy_data(self):
        self.assertTrue(agent_tools._is_navigation_usable_position_source(None))

    def test_yolo_position_source_is_not_navigation_usable(self):
        self.assertFalse(agent_tools._is_navigation_usable_position_source("yolo"))

    def test_move_validation_rejects_explicit_yolo_source(self):
        error = agent_module._validate_move_to_position_args(
            {"x": 5.0, "y": 6.0},
            [
                {
                    "tool": "search_objects_by_class_id",
                    "result": {"results": [{"x": 5.0, "y": 6.0, "position_source": "yolo"}]},
                }
            ],
        )

        self.assertIn("YOLO-only", error)

    def test_move_validation_allows_missing_position_source_for_legacy_data(self):
        error = agent_module._validate_move_to_position_args(
            {"x": 5.0, "y": 6.0},
            [
                {
                    "tool": "get_object_last_location",
                    "result": {"x": 5.0, "y": 6.0, "position_source": None, "navigation_usable": True},
                }
            ],
        )

        self.assertIsNone(error)

    def test_get_room_navigation_target_returns_room_center(self):
        row = (523, "Room 523", 10.0, 20.0, 14.0, 20.0, 14.0, 26.0, 10.0, 26.0)
        with patch.object(agent_tools, "_get_conn", return_value=_FakeConn([row])):
            result = agent_tools.get_room_navigation_target("Room 523")

        self.assertTrue(result["found"])
        self.assertEqual(result["room_id"], 523)
        self.assertEqual(result["room_name"], "Room 523")
        self.assertAlmostEqual(result["x"], 12.0)
        self.assertAlmostEqual(result["y"], 23.0)

    def test_get_robot_location_returns_live_pose_and_room(self):
        payload = {
            "robot": {
                "x": 12.0,
                "y": 23.0,
                "z": 0.5,
                "yaw": 1.2,
            }
        }
        row = ("Room 523",)
        with patch.object(
            agent_tools, "_load_json_path", return_value=payload
        ), patch.object(agent_tools, "_get_conn", return_value=_FakeConn([row])):
            result = agent_tools.get_robot_location()

        self.assertTrue(result["found"])
        self.assertEqual(result["room"], "Room 523")
        self.assertAlmostEqual(result["x"], 12.0)
        self.assertAlmostEqual(result["y"], 23.0)
        self.assertAlmostEqual(result["z"], 0.5)
        self.assertAlmostEqual(result["yaw"], 1.2)

    def test_get_latest_observation_respects_active_map_override(self):
        rows = [
            (7,),
            (
                101,
                "55555555-5555-5555-5555-555555555555",
                24,
                12.0,
                23.0,
                0.0,
                "2026-04-28T09:31:00",
                88,
                "A backpack rests near a desk.",
            ),
        ]
        with patch.object(agent_tools, "_get_conn", return_value=_FakeConn(rows)):
            agent_tools.set_active_map_override("office_3")
            try:
                result = agent_tools.get_latest_observation()
            finally:
                agent_tools.set_active_map_override(None)

        self.assertTrue(result["found"])
        self.assertEqual(result["observation_id"], 101)
        self.assertEqual(result["object_id"], "55555555-5555-5555-5555-555555555555")
        self.assertEqual(result["class_name"], "backpack")
        self.assertEqual(result["scene_id"], 88)


class TestDuetNavigationFallback(unittest.TestCase):
    def test_room_navigation_fallback_records_move_when_agent_only_replies_in_text(self):
        fake_room_lookup = {
            "found": True,
            "room_id": 523,
            "room_name": "Room 523",
            "x": 12.0,
            "y": 23.0,
            "x1": 10.0,
            "y1": 20.0,
            "x2": 14.0,
            "y2": 26.0,
        }
        fake_move = {
            "ok": True,
            "executed": False,
            "action": "move_to_position",
            "x": 12.0,
            "y": 23.0,
            "reason": "fallback_room_navigation:Room 523",
            "status": "dummy_only",
        }

        with patch("agent.agent.run_custom_agent", return_value=(
            "I have started exploring the immediate surroundings of Room 523. What would you like to do next?",
            [],
        )), patch("agent.tools.get_room_navigation_target", return_value=fake_room_lookup) as room_lookup, patch(
            "agent.tools.move_to_position", return_value=fake_move
        ) as move_to_position:
            transcript = list(
                app.iterate_agent_duet(
                    agent_a_system_prompt="Robot agent",
                    agent_b_system_prompt="Support agent",
                    opening_message="Please explore Room 523.",
                    turns=2,
                )
            )

        first_turn = transcript[0]
        self.assertEqual(first_turn["speaker"], "agent_a")
        self.assertTrue(
            any(item["tool"] == "get_room_navigation_target" for item in first_turn["tool_log"])
        )
        self.assertTrue(any(item["tool"] == "move_to_position" for item in first_turn["tool_log"]))
        room_lookup.assert_called_once_with("Room 523")
        move_to_position.assert_called_once_with(
            x=12.0,
            y=23.0,
            reason="fallback_room_navigation:Room 523",
        )


class TestHistoryMessageFormatting(unittest.TestCase):
    def test_history_to_messages_summarizes_tool_log_without_raw_json_blob(self):
        history = [
            {
                "role": "assistant",
                "content": "I will head there.",
                "tool_log": [
                    {
                        "tool": "get_room_navigation_target",
                        "args": {"room_name": "Room 523"},
                        "result": {"found": True, "room_name": "Room 523", "x": 12.0, "y": 23.0},
                    },
                    {
                        "tool": "move_to_position",
                        "args": {"x": 12.0, "y": 23.0, "reason": "fallback_room_navigation:Room 523"},
                        "result": {"ok": True, "executed": False, "status": "dummy_only"},
                    },
                ],
            }
        ]

        messages = agent_module._history_to_messages(history)

        self.assertEqual(len(messages), 1)
        content = messages[0]["content"]
        self.assertIn("Previous tool results for context:", content)
        self.assertIn("- get_room_navigation_target", content)
        self.assertIn("- move_to_position", content)
        self.assertNotIn('{"tool"', content)













class TestRunCustomAgentFailures(unittest.TestCase):
    def test_run_custom_agent_raises_on_backend_error(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = RuntimeError("backend unavailable")

        with patch.object(agent_module, "get_vlm_client", return_value=fake_client):
            with self.assertRaises(RuntimeError) as ctx:
                agent_module.run_custom_agent(
                    "Tell me something unusual about the map.",
                    system_prompt="Test prompt",
                )

        self.assertIn("backend unavailable", str(ctx.exception))




class TestRobotLocationDirectTool(unittest.TestCase):
    def test_run_agent_uses_robot_location_direct_tool_when_allowed(self):
        with patch.object(agent_module, "_run_tool") as run_tool:
            run_tool.return_value = {
                "found": True,
                "room": "Room 523",
                "map_name": "office_3",
                "x": 12.0,
                "y": 23.0,
                "z": 0.0,
                "yaw": 0.0,
            }

            answer, tool_log = agent_module.run_agent(
                "Where is the robot right now?",
                allowed_tool_names={"get_robot_location"},
            )

        self.assertEqual(tool_log[0]["tool"], "get_robot_location")
        self.assertIn("Room 523", answer)
        self.assertIn("office_3", answer)


class TestChatApiMetadata(unittest.TestCase):
    def test_chat_returns_model_and_usage_metadata(self):
        metadata = {
            "model": "Qwen/Test-Model",
            "used_model": True,
            "iterations": 2,
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        }

        with patch("agent.agent.run_agent", return_value=("Mufasa", [], metadata)):
            result = app.chat(app.ChatRequest(question="Who used the coffee machine most recently?"))

        self.assertEqual(result["answer"], "Mufasa")
        self.assertEqual(result["tool_log"], [])
        self.assertEqual(result["model"], "Qwen/Test-Model")
        self.assertTrue(result["used_model"])
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 20)

    def test_chat_passes_building_scope_to_agent_tools(self):
        metadata = {
            "model": "Qwen/Test-Model",
            "used_model": False,
            "iterations": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

        with patch("agent.tools.set_active_map_override") as set_active_map_override, patch(
            "agent.agent.run_agent", return_value=("Scoped answer", [], metadata)
        ):
            result = app.chat(app.ChatRequest(question="Where is the backpack?", building="office_3"))

        self.assertEqual(result["answer"], "Scoped answer")
        set_active_map_override.assert_called_once_with("office_3")


class TestVlmSelectionApi(unittest.TestCase):
    def test_get_vlm_options_returns_registry_state(self):
        state = {
            "active_option": {"id": "gemini", "label": "Gemini", "source": "gemini", "model": "gemini-3-flash-preview"},
            "options": [
                {"id": "local", "label": "Local Qwen", "source": "local", "model": "Qwen/Qwen3-VL-4B-Instruct", "enabled": True, "reason": "", "active": False},
                {"id": "gemini", "label": "Gemini", "source": "gemini", "model": "gemini-3-flash-preview", "enabled": True, "reason": "", "active": True},
            ],
        }

        with patch("agent.vlm_client.get_vlm_selection_state", return_value=state):
            result = app.get_vlm_options()

        self.assertEqual(result["active_option"]["id"], "gemini")
        self.assertEqual(len(result["options"]), 2)

    def test_set_vlm_selection_updates_active_option(self):
        state = {
            "active_option": {"id": "kimi", "label": "Kimi", "source": "kimi", "model": "kimi-k2.6"},
            "options": [
                {"id": "local", "label": "Local Qwen", "source": "local", "model": "Qwen/Qwen3-VL-4B-Instruct", "enabled": True, "reason": "", "active": False},
                {"id": "kimi", "label": "Kimi", "source": "kimi", "model": "kimi-k2.6", "enabled": True, "reason": "", "active": True},
            ],
        }

        with patch("agent.vlm_client.set_active_vlm_option", return_value=state) as set_active_vlm_option:
            result = app.set_vlm_selection(app.VlmSelectionRequest(option_id="kimi"))

        set_active_vlm_option.assert_called_once_with("kimi")
        self.assertEqual(result["active_option"]["id"], "kimi")

    def test_set_vlm_selection_returns_http_400_for_invalid_option(self):
        with patch("agent.vlm_client.set_active_vlm_option", side_effect=ValueError("Unknown VLM option: bad")):
            with self.assertRaises(app.HTTPException) as ctx:
                app.set_vlm_selection(app.VlmSelectionRequest(option_id="bad"))

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail, "Unknown VLM option: bad")


class TestModelChatApi(unittest.TestCase):
    def test_model_chat_returns_usage_metadata(self):
        metadata = {
            "model": "gemini-3-flash-preview",
            "used_model": True,
            "iterations": 1,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 14,
                "total_tokens": 24,
            },
        }

        with patch("agent.agent.run_model_chat", return_value=("Hello there", [], metadata)) as run_model_chat:
            result = app.model_chat(app.ModelChatRequest(question="Hello"))

        run_model_chat.assert_called_once_with("Hello", history=None, return_metadata=True)
        self.assertEqual(result["answer"], "Hello there")
        self.assertEqual(result["model"], "gemini-3-flash-preview")
        self.assertEqual(result["usage"]["total_tokens"], 24)


if __name__ == "__main__":
    unittest.main()
