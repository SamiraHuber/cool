"""
Tests for Nav2 planning request generation in the frontend.
"""
import json
import pathlib
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Minimal patches so that importing app.py doesn't need a live DB / filesystem
# ---------------------------------------------------------------------------
import sys

# Stub psycopg2 before importing app so it doesn't fail at import time when
# the real package is absent.
if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = MagicMock()
    sys.modules["psycopg2"] = psycopg2_stub

import app  # noqa: E402  (bordsupr/frontend/app.py)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _fake_toolbox_map(robot_x=1.5, robot_y=2.5, robot_z=0.0, robot_yaw=0.0):
    """Return a toolbox_map payload dict with a robot pose."""
    return {
        "available": True,
        "source": "toolbox_map_snapshot.json",
        "robot": {"x": robot_x, "y": robot_y, "z": robot_z, "yaw": robot_yaw},
        "map": {},
    }


def _fake_plan():
    return {
        "request_id": "plan-999-abc",
        "path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
        "path_length_m": 1.414,
        "num_waypoints": 2,
        "start": {"planned_world": {"x": 0.0, "y": 0.0}},
        "goal": {"planned_world": {"x": 1.0, "y": 1.0}},
        "planner": {"name": "nav2_smac_2d"},
    }


# ---------------------------------------------------------------------------
# _get_robot_start_pose
# ---------------------------------------------------------------------------

class TestGetRobotStartPose(unittest.TestCase):

    def test_returns_none_when_no_map_file(self):
        with patch.object(app, "load_map_payload_from_path", return_value=None):
            result = app._get_robot_start_pose()
        self.assertIsNone(result)

    def test_returns_none_when_map_has_no_robot(self):
        with patch.object(app, "load_map_payload_from_path", return_value={"available": True}):
            result = app._get_robot_start_pose()
        self.assertIsNone(result)

    def test_returns_none_when_robot_coords_missing(self):
        payload = {"robot": {"x": None, "y": 2.0}}
        with patch.object(app, "load_map_payload_from_path", return_value=payload):
            result = app._get_robot_start_pose()
        self.assertIsNone(result)

    def test_returns_pose_from_toolbox_map(self):
        with patch.object(app, "load_map_payload_from_path",
                          return_value=_fake_toolbox_map(1.5, 2.5, 0.1, 0.78)):
            result = app._get_robot_start_pose()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["x"], 1.5)
        self.assertAlmostEqual(result["y"], 2.5)
        self.assertAlmostEqual(result["z"], 0.0)
        self.assertAlmostEqual(result["yaw"], 0.78)

    def test_defaults_z_and_yaw_to_zero_when_absent(self):
        payload = {"robot": {"x": 3.0, "y": 4.0}}  # no z / yaw keys
        with patch.object(app, "load_map_payload_from_path", return_value=payload):
            result = app._get_robot_start_pose()
        self.assertEqual(result["z"], 0.0)
        self.assertEqual(result["yaw"], 0.0)


# ---------------------------------------------------------------------------
# request_nav2_plan
# ---------------------------------------------------------------------------

class TestRequestNav2Plan(unittest.TestCase):

    def _write_response_later(self, request_path, response_path, plan_response):
        """Patch atomic_write_json so that after writing the request the
        response file already contains the right request_id."""
        original_write = app.atomic_write_json

        def side_effect(path, payload):
            original_write(path, payload)
            # After the request is written, place a matching response immediately.
            if path == app.NAV2_PLAN_REQUEST_PATH:
                response = dict(plan_response)
                response["request_id"] = payload["request_id"]
                original_write(app.NAV2_PLAN_RESPONSE_PATH, response)

        return side_effect

    def test_start_pose_included_in_request_when_provided(self):
        """request_nav2_plan writes 'start' in the request file when start_pose supplied."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = pathlib.Path(tmpdir) / "req.json"
            resp_path = pathlib.Path(tmpdir) / "resp.json"
            plan = _fake_plan()

            with patch.object(app, "NAV2_PLAN_REQUEST_PATH", req_path), \
                 patch.object(app, "NAV2_PLAN_RESPONSE_PATH", resp_path), \
                 patch.object(app, "NAV2_PLAN_TIMEOUT_SEC", 5.0):

                def fake_write(path, payload):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(payload, f)
                    if path == req_path:
                        resp = dict(plan)
                        resp["request_id"] = payload["request_id"]
                        with open(resp_path, "w") as f:
                            json.dump(resp, f)

                with patch.object(app, "atomic_write_json", side_effect=fake_write):
                    start = {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.5}
                    app.request_nav2_plan({"x": 5.0, "y": 6.0}, start_pose=start)

                with open(req_path) as f:
                    written = json.load(f)

        self.assertIn("start", written)
        self.assertAlmostEqual(written["start"]["x"], 1.0)
        self.assertAlmostEqual(written["start"]["y"], 2.0)
        self.assertAlmostEqual(written["start"]["yaw"], 0.5)

    def test_start_pose_absent_in_request_when_not_provided(self):
        """request_nav2_plan writes no 'start' key when start_pose=None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            req_path = pathlib.Path(tmpdir) / "req.json"
            resp_path = pathlib.Path(tmpdir) / "resp.json"
            plan = _fake_plan()

            with patch.object(app, "NAV2_PLAN_REQUEST_PATH", req_path), \
                 patch.object(app, "NAV2_PLAN_RESPONSE_PATH", resp_path), \
                 patch.object(app, "NAV2_PLAN_TIMEOUT_SEC", 5.0):

                def fake_write(path, payload):
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "w") as f:
                        json.dump(payload, f)
                    if path == req_path:
                        resp = dict(plan)
                        resp["request_id"] = payload["request_id"]
                        with open(resp_path, "w") as f:
                            json.dump(resp, f)

                with patch.object(app, "atomic_write_json", side_effect=fake_write):
                    app.request_nav2_plan({"x": 5.0, "y": 6.0}, start_pose=None)

                with open(req_path) as f:
                    written = json.load(f)

        self.assertNotIn("start", written)


# ---------------------------------------------------------------------------
# plan_lidar_detection_navigation — uses same call convention as plan_custom_path
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def execute(self, *a, **kw):
        pass

    def fetchone(self):
        return self._row

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def cursor(self):
        return _FakeCursor(self._row)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _make_db_row(obs_id=1, object_id="obj-1", class_id=0, yolo_track_id=None,
         position_source="depth", goal_x=5.0, goal_y=6.0, goal_z=0.0,
         created_at=None, scene_timestamp=None):
    return (obs_id, object_id, class_id, yolo_track_id,
        position_source, goal_x, goal_y, goal_z, created_at, scene_timestamp)


class TestPlanLidarDetectionNavigation(unittest.TestCase):

    def _run_endpoint(self, db_row, robot_pose_result, plan_result):
        with patch.object(app, "get_conn", return_value=_FakeConn(db_row)), \
             patch.object(app, "_get_robot_start_pose", return_value=robot_pose_result) as mock_robot, \
             patch.object(app, "request_nav2_plan", return_value=plan_result) as mock_plan:
            result = app.plan_lidar_detection_navigation(observation_id=1)
        return result, mock_robot, mock_plan

    def test_calls_request_nav2_plan_with_none_start_pose_even_when_robot_pose_file_exists(self):
        row = _make_db_row(goal_x=5.0, goal_y=6.0, goal_z=0.5)
        robot_pose = {"x": 1.5, "y": 2.5, "z": 0.0, "yaw": 0.0}
        plan = _fake_plan()

        _, mock_robot, mock_plan = self._run_endpoint(row, robot_pose, plan)

        mock_robot.assert_not_called()
        mock_plan.assert_called_once()
        call_args = mock_plan.call_args
        self.assertAlmostEqual(call_args.args[0]["x"], 5.0)
        self.assertAlmostEqual(call_args.args[0]["y"], 6.0)
        # Nav2 should always use TF for this endpoint so UI-only pose overrides
        # cannot send it an explicit start pose that disagrees with localization.
        self.assertIsNone(call_args.kwargs["start_pose"])

    def test_calls_request_nav2_plan_with_none_start_pose_when_map_unavailable(self):
        row = _make_db_row(goal_x=5.0, goal_y=6.0)
        plan = _fake_plan()

        _, mock_robot, mock_plan = self._run_endpoint(row, None, plan)

        mock_robot.assert_not_called()
        mock_plan.assert_called_once()
        call_args = mock_plan.call_args
        # start_pose must be None so the Nav2 planner can fall back to TF
        self.assertIsNone(call_args.kwargs["start_pose"])

    def test_response_includes_plan_and_observation(self):
        row = _make_db_row(obs_id=42, class_id=5)
        robot_pose = {"x": 1.0, "y": 1.0, "z": 0.0, "yaw": 0.0}
        plan = _fake_plan()

        result, _, _ = self._run_endpoint(row, robot_pose, plan)

        self.assertTrue(result["ok"])
        self.assertEqual(result["observation"]["id"], 42)
        self.assertEqual(result["observation"]["position_source"], "depth")
        self.assertEqual(result["plan"], plan)

    def test_rejects_yolo_only_positions_for_navigation(self):
        row = _make_db_row(position_source="yolo", goal_x=5.0, goal_y=6.0)

        with patch.object(app, "get_conn", return_value=_FakeConn(row)), \
             patch.object(app, "request_nav2_plan") as mock_plan:
            with self.assertRaises(app.HTTPException) as ctx:
                app.plan_lidar_detection_navigation(observation_id=1)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("navigation-safe coordinates", ctx.exception.detail)
        mock_plan.assert_not_called()


# ---------------------------------------------------------------------------
# plan_custom_path — unchanged but verified to match same interface
# ---------------------------------------------------------------------------

class TestPlanCustomPath(unittest.TestCase):

    def test_rejects_planning_while_frontend_set_pose_is_pending(self):
        from fastapi import HTTPException

        payload = {"goal": {"x": 5.0, "y": 6.0}}
        status = {
            "localization_pending": True,
            "message": "Set pose applied at (1.00, 2.00, 0.0°); waiting for localization to converge.",
        }

        with patch.object(app, "load_json_file", return_value=status):
            with self.assertRaises(HTTPException) as ctx:
                app.plan_custom_path(payload)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("Wait for set-pose confirmation", ctx.exception.detail)

    def test_allows_stale_pending_plan_using_slam_pose_as_start(self):
        payload = {"goal": {"x": 5.0, "y": 6.0}}
        status = {
            "localization_pending": True,
            "message": "Set pose applied at (1.00, 2.00, 0.0°); waiting for localization to converge.",
            "updated_at": 100.0,
            "localization_pending_timeout_sec": 12.0,
        }
        provisional_start = {"x": 1.0, "y": 2.0, "z": 0.0, "yaw": 0.2}
        plan = _fake_plan()

        with patch.object(app.time, "time", return_value=120.0), \
             patch.object(app, "load_json_file", return_value=status), \
             patch.object(app, "load_map_payload_from_path", return_value=None), \
             patch.object(app, "_get_robot_start_pose", return_value=provisional_start), \
             patch.object(app, "request_nav2_plan", return_value=plan) as mock_plan:
            result = app.plan_custom_path(payload)

        self.assertTrue(result["ok"])
        mock_plan.assert_called_once_with({"x": 5.0, "y": 6.0, "z": 0.0}, start_pose=provisional_start)

    def test_snaps_explicit_start_and_goal_to_nav2_costmap_before_request(self):
        payload = {"start": {"x": 1.1, "y": 1.1}, "goal": {"x": 3.1, "y": 1.1}}
        plan = _fake_plan()
        nav2_costmap = {
            "available": True,
            "map": {
                "resolution": 1.0,
                "width": 5,
                "height": 4,
                "origin": {"x": 0.0, "y": 0.0},
                "data": [
                    100, 100, 100, 100, 100,
                    0, 100, 100, 100, 0,
                    100, 100, 100, 100, 100,
                    0, 0, 0, 0, 0,
                ],
            },
        }

        with patch.object(app, "load_map_payload_from_path", return_value=nav2_costmap), \
             patch.object(app, "request_nav2_plan", return_value=plan) as mock_plan:
            app.plan_custom_path(payload)

        mock_plan.assert_called_once()
        call_args = mock_plan.call_args
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["x"], 0.5)
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["y"], 1.5)
        self.assertAlmostEqual(call_args.args[0]["x"], 4.5)
        self.assertAlmostEqual(call_args.args[0]["y"], 1.5)

    def test_allows_unknown_start_cell_on_nav2_costmap(self):
        payload = {"start": {"x": 1.5, "y": 1.5}, "goal": {"x": 3.1, "y": 1.1}}
        plan = _fake_plan()
        nav2_costmap = {
            "available": True,
            "map": {
                "resolution": 1.0,
                "width": 5,
                "height": 4,
                "origin": {"x": 0.0, "y": 0.0},
                "data": [
                    -1, -1, -1, -1, -1,
                    -1, -1, -1, 0, 0,
                    -1, -1, -1, 0, 0,
                    -1, -1, -1, 0, 0,
                ],
            },
        }

        with patch.object(app, "load_map_payload_from_path", return_value=nav2_costmap), \
             patch.object(app, "request_nav2_plan", return_value=plan) as mock_plan:
            app.plan_custom_path(payload)

        mock_plan.assert_called_once()
        call_args = mock_plan.call_args
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["x"], 1.5)
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["y"], 1.5)

    def test_calls_request_nav2_plan_with_explicit_start_pose(self):
        payload = {"start": {"x": 1.0, "y": 2.0}, "goal": {"x": 5.0, "y": 6.0}}
        plan = _fake_plan()

        with patch.object(app, "load_map_payload_from_path", return_value=None), \
             patch.object(app, "request_nav2_plan", return_value=plan) as mock_plan:
            result = app.plan_custom_path(payload)

        mock_plan.assert_called_once()
        call_args = mock_plan.call_args
        self.assertAlmostEqual(call_args.args[0]["x"], 5.0)
        self.assertAlmostEqual(call_args.args[0]["y"], 6.0)
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["x"], 1.0)
        self.assertAlmostEqual(call_args.kwargs["start_pose"]["y"], 2.0)

    def test_calls_request_nav2_plan_with_none_start_pose_when_start_is_missing(self):
        payload = {"goal": {"x": 5.0, "y": 6.0}}  # no start
        plan = _fake_plan()

        with patch.object(app, "_get_robot_start_pose", return_value=None) as mock_robot, \
             patch.object(app, "load_map_payload_from_path", return_value=None), \
             patch.object(app, "request_nav2_plan", return_value=plan) as mock_plan:
            result = app.plan_custom_path(payload)

        mock_robot.assert_not_called()
        mock_plan.assert_called_once()
        self.assertIsNone(mock_plan.call_args.kwargs["start_pose"])
        self.assertTrue(result["ok"])

    def test_raises_when_nav2_planning_fails_without_start_pose(self):
        from fastapi import HTTPException

        payload = {"goal": {"x": 5.0, "y": 6.0}}  # no start

        with patch.object(app, "request_nav2_plan", side_effect=HTTPException(status_code=503, detail="Nav2 planner unavailable")), \
             patch.object(app, "_get_robot_start_pose", return_value=None) as mock_robot, \
             patch.object(app, "load_map_payload_from_path", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                app.plan_custom_path(payload)

        mock_robot.assert_not_called()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Nav2 planner unavailable", ctx.exception.detail)

    def test_raises_400_for_missing_goal(self):
        from fastapi import HTTPException
        payload = {"start": {"x": 1.0, "y": 2.0}}  # no goal
        with self.assertRaises(HTTPException) as ctx:
            app.plan_custom_path(payload)
        self.assertEqual(ctx.exception.status_code, 400)

    def test_response_includes_plan_and_source(self):
        payload = {"start": {"x": 1.0, "y": 2.0}, "goal": {"x": 5.0, "y": 6.0}}
        plan = _fake_plan()

        with patch.object(app, "load_map_payload_from_path", return_value=None), \
             patch.object(app, "request_nav2_plan", return_value=plan):
            result = app.plan_custom_path(payload)

        self.assertTrue(result["ok"])
        self.assertEqual(result["source"], "custom")
        self.assertEqual(result["plan"], plan)

    def test_raises_when_nav2_planning_aborts_for_explicit_start(self):
        from fastapi import HTTPException

        payload = {"start": {"x": 1.0, "y": 2.0}, "goal": {"x": 5.0, "y": 6.0}}

        with patch.object(
            app,
            "request_nav2_plan",
            side_effect=HTTPException(status_code=503, detail="Nav2 planner failed"),
        ) as mock_nav2:
            with self.assertRaises(HTTPException) as ctx:
                app.plan_custom_path(payload)

        mock_nav2.assert_called_once()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Nav2 planner failed", ctx.exception.detail)


class TestExecuteNavigationPlan(unittest.TestCase):

    def test_rejects_execution_while_frontend_set_pose_is_pending(self):
        from fastapi import HTTPException

        payload = {
            "plan": {
                "path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                "goal": {"planned_world": {"x": 1.0, "y": 0.0}},
            }
        }
        status = {
            "localization_pending": True,
            "message": "Set pose applied at (1.00, 2.00, 0.0°); waiting for localization to converge.",
        }

        with patch.object(app, "load_json_file", return_value=status):
            with self.assertRaises(HTTPException) as ctx:
                app.execute_navigation_plan(payload)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("Wait for set-pose confirmation", ctx.exception.detail)

    def test_rejects_execution_when_set_pose_pending_is_stale(self):
        from fastapi import HTTPException

        payload = {
            "plan": {
                "path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                "goal": {"planned_world": {"x": 1.0, "y": 0.0}},
            }
        }
        status = {
            "localization_pending": True,
            "message": "Set pose applied at (1.00, 2.00, 0.0°); waiting for localization to converge.",
            "updated_at": 100.0,
            "localization_pending_timeout_sec": 12.0,
        }

        with patch.object(app.time, "time", return_value=120.0), \
             patch.object(app, "load_json_file", return_value=status):
            with self.assertRaises(HTTPException) as ctx:
                app.execute_navigation_plan(payload)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("navigation remains blocked", ctx.exception.detail)

    def test_rejects_preview_only_astar_plan_execution(self):
        from fastapi import HTTPException

        payload = {
            "plan": {
                "path": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}],
                "planner": {"name": "astar"},
            }
        }

        with patch.object(app, "load_json_file", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                app.execute_navigation_plan(payload)

        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("preview-only", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
