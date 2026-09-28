import json
import pathlib
import tempfile
import time
import types
import unittest
from unittest.mock import MagicMock, patch

import sys

if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = MagicMock()
    sys.modules["psycopg2"] = psycopg2_stub

import app  # noqa: E402


class _HistoricalObsCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def execute(self, query, params=None):
        self.executed.append((query, params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _HistoricalObsConn:
    def __init__(self, rows):
        self.cursor_obj = _HistoricalObsCursor(rows)

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSlamStatus(unittest.TestCase):
    def test_get_slam_status_includes_robot_connection_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = pathlib.Path(tmpdir) / "status.json"
            connected_last_seen_at = time.time()
            status_path.write_text(
                json.dumps(
                    {
                        "state": "recording",
                        "message": "Recording new map.",
                        "saved_maps": [],
                        "robot_connection": {
                            "connected": True,
                            "message": "Robot connected.",
                            "last_seen_at": connected_last_seen_at,
                            "age_sec": 0.8,
                            "topic": "/spot/odometry",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(app, "SLAM_TAB_STATUS_PATH", status_path), \
                 patch.object(app, "_touch_slam_tab_claim", return_value=None):
                payload = app.get_slam_status()

        self.assertEqual(payload["state"], "recording")
        self.assertIn("robot_connection", payload)
        self.assertTrue(payload["robot_connection"]["connected"])

    def test_get_slam_status_returns_default_when_manager_not_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = pathlib.Path(tmpdir) / "missing-status.json"

            with patch.object(app, "SLAM_TAB_STATUS_PATH", status_path), \
                 patch.object(app, "_touch_slam_tab_claim", return_value=None):
                payload = app.get_slam_status()

        self.assertEqual(payload["state"], "idle")
        self.assertEqual(payload["saved_maps"], [])

    def test_get_slam_status_refreshes_stale_robot_connection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = pathlib.Path(tmpdir) / "status.json"
            stale_last_seen_at = time.time() - (app.ROBOT_ODOM_STALE_SEC + 5.0)
            status_path.write_text(
                json.dumps(
                    {
                        "state": "idle",
                        "message": "SLAM manager ready.",
                        "saved_maps": [],
                        "robot_connection": {
                            "connected": True,
                            "message": "Robot connected.",
                            "last_seen_at": stale_last_seen_at,
                            "age_sec": 0.1,
                            "topic": "/spot/odometry",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(app, "SLAM_TAB_STATUS_PATH", status_path), \
                 patch.object(app, "_touch_slam_tab_claim", return_value=None):
                payload = app.get_slam_status()

        self.assertFalse(payload["robot_connection"]["connected"])
        self.assertEqual(payload["robot_connection"]["message"], "Robot connection stale.")
        self.assertGreater(payload["robot_connection"]["age_sec"], app.ROBOT_ODOM_STALE_SEC)

    def test_get_slam_status_preserves_explicit_disconnected_payload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            status_path = pathlib.Path(tmpdir) / "status.json"
            recent_last_seen_at = time.time()
            status_path.write_text(
                json.dumps(
                    {
                        "state": "idle",
                        "message": "SLAM manager ready.",
                        "saved_maps": [],
                        "robot_connection": {
                            "connected": False,
                            "message": "No /spot/odometry received yet.",
                            "last_seen_at": recent_last_seen_at,
                            "age_sec": 0.1,
                            "topic": "/spot/odometry",
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(app, "SLAM_TAB_STATUS_PATH", status_path), \
                 patch.object(app, "_touch_slam_tab_claim", return_value=None):
                payload = app.get_slam_status()

        self.assertFalse(payload["robot_connection"]["connected"])
        self.assertEqual(payload["robot_connection"]["message"], "No /spot/odometry received yet.")
        self.assertLess(payload["robot_connection"]["age_sec"], app.ROBOT_ODOM_STALE_SEC)


class TestSlamHistoricalObservations(unittest.TestCase):
    def test_filters_by_selected_map(self):
        fake_conn = _HistoricalObsConn([])

        with patch.object(app, "_resolve_map_id", return_value=42), \
             patch.object(app, "get_conn", return_value=fake_conn):
            response = app.get_slam_historical_observations(minutes=1440, building="office_3")

        self.assertEqual(response, [])
        self.assertEqual(len(fake_conn.cursor_obj.executed), 1)
        query, params = fake_conn.cursor_obj.executed[0]
        self.assertIn("oo.map_id = %s", query)
        self.assertIn("s2.id = oo.scene_id AND s2.map_id = %s", query)
        self.assertEqual(params, [1440, 42, 42])


if __name__ == "__main__":
    unittest.main()
