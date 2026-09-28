"""
Bridge between the web frontend and Nav2's ComputePathToPose action.

The web server writes a JSON request to NAV2_PLAN_REQUEST_PATH, polls
NAV2_PLAN_RESPONSE_PATH for a response with the matching request_id, and
reads the resulting path.  This node watches the request file, calls
ComputePathToPose (using an explicit map-frame start pose when the web UI
provides one, otherwise letting Nav2 read the start pose from TF), and
writes the path back to the response file.
"""
import json
import math
import os
import tempfile
import time
from pathlib import Path

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


REQUEST_PATH = Path(os.getenv("NAV2_PLAN_REQUEST_PATH", "/shared/nav2_plan_request.json"))
RESPONSE_PATH = Path(os.getenv("NAV2_PLAN_RESPONSE_PATH", "/shared/nav2_plan_response.json"))
POLL_PERIOD_SEC = float(os.getenv("NAV2_PLAN_BRIDGE_POLL_SEC", "0.2"))
SERVER_WAIT_TIMEOUT_SEC = float(os.getenv("NAV2_PLAN_SERVER_WAIT_TIMEOUT_SEC", "3.0"))
RESULT_TIMEOUT_SEC = float(os.getenv("NAV2_PLAN_RESULT_TIMEOUT_SEC", "8.0"))
# Timeout for the action server to accept/reject a goal after we send it.
GOAL_RESPONSE_TIMEOUT_SEC = float(os.getenv("NAV2_PLAN_GOAL_RESPONSE_TIMEOUT_SEC", "6.0"))
# After a timeout-triggered cancel, wait this long before sending a new goal so Nav2
# has time to finish processing the cancellation (it rejects incoming goals while
# is_cancel_requested_ is set).
CANCEL_COOLDOWN_SEC = float(os.getenv("NAV2_PLAN_CANCEL_COOLDOWN_SEC", "2.0"))
MAX_REJECTION_RETRIES = int(os.getenv("NAV2_PLAN_MAX_REJECTION_RETRIES", "3"))
REJECTION_RETRY_DELAY_SEC = float(os.getenv("NAV2_PLAN_REJECTION_RETRY_DELAY_SEC", "1.0"))
PLANNER_ID = os.getenv("NAV2_PLANNER_ID", "GridBased")
PATH_WAYPOINT_MIN_SPACING_M = float(os.getenv("NAV2_PLAN_WAYPOINT_MIN_SPACING_M", "0.30"))
PATH_CORNER_KEEP_ANGLE_RAD = float(os.getenv("NAV2_PLAN_CORNER_KEEP_ANGLE_RAD", "0.35"))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def _write_error(request_id: str, detail: str) -> None:
    atomic_write_json(RESPONSE_PATH, {"request_id": request_id, "error": detail})


def _goal_status_name(status: int) -> str:
    names = {
        GoalStatus.STATUS_UNKNOWN: "unknown",
        GoalStatus.STATUS_ACCEPTED: "accepted",
        GoalStatus.STATUS_EXECUTING: "executing",
        GoalStatus.STATUS_CANCELING: "canceling",
        GoalStatus.STATUS_SUCCEEDED: "succeeded",
        GoalStatus.STATUS_CANCELED: "canceled",
        GoalStatus.STATUS_ABORTED: "aborted",
    }
    return names.get(status, f"status_{status}")


def _planner_error_detail(result_wrapper) -> str:
    result = getattr(result_wrapper, "result", None)
    detail_parts = [
        f"Nav2 planner failed (status={result_wrapper.status}, state={_goal_status_name(result_wrapper.status)})."
    ]

    error_code = getattr(result, "error_code", None)
    if error_code is not None:
        detail_parts.append(f"planner_error_code={error_code}.")

    error_msg = getattr(result, "error_msg", None)
    if error_msg:
        detail_parts.append(str(error_msg).strip())
    else:
        detail_parts.append("Goal may be unreachable, blocked by the costmap, or outside the mapped area.")

    return " ".join(part for part in detail_parts if part)


def _yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def _path_length_m(path: list[dict]) -> float:
    path_length_m = 0.0
    for i in range(1, len(path)):
        path_length_m += math.hypot(
            path[i]["x"] - path[i - 1]["x"],
            path[i]["y"] - path[i - 1]["y"],
        )
    return path_length_m


def _heading_delta_rad(a: dict, b: dict, c: dict) -> float:
    heading_in = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
    heading_out = math.atan2(c["y"] - b["y"], c["x"] - b["x"])
    delta = abs((heading_out - heading_in + math.pi) % (2.0 * math.pi) - math.pi)
    return delta


def _thin_path_waypoints(
    path: list[dict],
    min_spacing_m: float = PATH_WAYPOINT_MIN_SPACING_M,
    corner_keep_angle_rad: float = PATH_CORNER_KEEP_ANGLE_RAD,
) -> list[dict]:
    """Reduce dense planner output while preserving endpoints and turns."""
    if len(path) <= 2 or min_spacing_m <= 0.0:
        return list(path)

    thinned = [path[0]]
    for i in range(1, len(path) - 1):
        point = path[i]
        last_kept = thinned[-1]
        distance_from_last = math.hypot(point["x"] - last_kept["x"], point["y"] - last_kept["y"])
        is_corner = _heading_delta_rad(path[i - 1], point, path[i + 1]) >= corner_keep_angle_rad
        if is_corner or distance_from_last >= min_spacing_m:
            thinned.append(point)

    if thinned[-1] != path[-1]:
        thinned.append(path[-1])
    return thinned


class Nav2PlannerBridge(Node):
    def __init__(self) -> None:
        super().__init__("nav2_planner_bridge")

        self.client = ActionClient(self, ComputePathToPose, "compute_path_to_pose")

        self.last_handled_request_id: str | None = None
        self.active_request_id: str | None = None
        self.active_goal_handle = None
        self.active_goal_started_at: float | None = None
        self.goal_pending = False
        self.goal_sent_at: float | None = None
        self._active_send_future = None
        # Gate that prevents new goals from being sent before this timestamp.
        self._accept_not_before: float = 0.0
        # Consecutive rejection counter for the request currently being retried.
        self._rejection_count: int = 0
        self._rejection_request_id: str | None = None

        self.start_time = time.time()

        self._clear_stale_response()
        self.timer = self.create_timer(POLL_PERIOD_SEC, self._poll)
        self.get_logger().info(
            f"Nav2 planner bridge watching {REQUEST_PATH}, responses at {RESPONSE_PATH}"
        )
        self._mark_preexisting_request_stale()

    # ------------------------------------------------------------------ #
    # Startup                                                              #
    # ------------------------------------------------------------------ #

    def _clear_stale_response(self) -> None:
        """Remove any leftover response file so the web backend doesn't read stale errors."""
        if RESPONSE_PATH.exists():
            try:
                RESPONSE_PATH.unlink()
                self.get_logger().info(f"Removed stale response file {RESPONSE_PATH}")
            except Exception as exc:
                self.get_logger().warning(f"Could not remove stale response file: {exc}")

    def _mark_preexisting_request_stale(self) -> None:
        """Ignore any request file that was written before this process started."""
        if not REQUEST_PATH.exists():
            return
        try:
            with REQUEST_PATH.open("r", encoding="utf-8") as f:
                request = json.load(f)
        except Exception:
            return
        if not isinstance(request, dict):
            return
        request_id = request.get("request_id")
        if not request_id:
            return
        try:
            created_at = float(request.get("created_at") or 0.0)
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at > self.start_time:
            return  # written after us — process normally
        self.last_handled_request_id = request_id
        self.get_logger().info(
            f"Ignoring stale nav2_plan request {request_id} left over from before startup."
        )

    # ------------------------------------------------------------------ #
    # Poll                                                                 #
    # ------------------------------------------------------------------ #

    def _poll(self) -> None:
        # Timeout for an accepted goal that hasn't produced a result yet.
        if self.active_request_id is not None and self.active_goal_started_at is not None:
            if (time.time() - self.active_goal_started_at) > RESULT_TIMEOUT_SEC:
                if self.active_goal_handle is not None:
                    try:
                        self.active_goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                _write_error(
                    self.active_request_id,
                    "Nav2 planner did not finish in time. "
                    "Check that the planner is active and that TF includes the required map/odom/base frames.",
                )
                self._reset(self.active_request_id)
                # Wait for Nav2 to finish processing the cancellation before
                # sending a new goal (it rejects goals while is_cancel_requested_).
                self._accept_not_before = time.time() + CANCEL_COOLDOWN_SEC
            return

        # Timeout for a goal that was sent but never accepted/rejected.
        if self.goal_pending and self.active_request_id is not None and self.goal_sent_at is not None:
            if (time.time() - self.goal_sent_at) > GOAL_RESPONSE_TIMEOUT_SEC:
                _write_error(
                    self.active_request_id,
                    "Nav2 planner did not accept the goal in time. "
                    "The planner server may be unresponsive or not active.",
                )
                self._reset(self.active_request_id)
                self._accept_not_before = time.time() + CANCEL_COOLDOWN_SEC
            return

        if self.active_request_id is not None or self.goal_pending:
            return
        if time.time() < self._accept_not_before:
            return
        if not REQUEST_PATH.exists():
            return
        try:
            with REQUEST_PATH.open("r", encoding="utf-8") as f:
                request = json.load(f)
        except Exception:
            return
        if not isinstance(request, dict):
            return
        request_id = request.get("request_id")
        if not request_id or request_id == self.last_handled_request_id:
            return
        self._handle_request(request_id, request)

    # ------------------------------------------------------------------ #
    # Request handling                                                     #
    # ------------------------------------------------------------------ #

    def _handle_request(self, request_id: str, request: dict) -> None:
        goal_data = request.get("goal")
        if not isinstance(goal_data, dict):
            _write_error(request_id, "Request is missing 'goal'.")
            self.last_handled_request_id = request_id
            return

        if not self.client.server_is_ready():
            created_at = request.get("created_at")
            request_age_sec = None
            try:
                request_age_sec = time.time() - float(created_at)
            except (TypeError, ValueError):
                pass
            if request_age_sec is not None and request_age_sec > SERVER_WAIT_TIMEOUT_SEC:
                _write_error(
                    request_id,
                    "Nav2 planner server is not available. "
                    "Check Nav2 lifecycle bringup and TF/map availability.",
                )
                self.last_handled_request_id = request_id
            return

        goal_msg = ComputePathToPose.Goal()

        goal_pose = PoseStamped()
        goal_pose.header.frame_id = "map"
        goal_pose.header.stamp = self.get_clock().now().to_msg()
        try:
            goal_pose.pose.position.x = float(goal_data["x"])
            goal_pose.pose.position.y = float(goal_data["y"])
        except (KeyError, TypeError, ValueError) as exc:
            _write_error(request_id, f"Invalid goal coordinates: {exc}")
            self.last_handled_request_id = request_id
            return
        goal_pose.pose.position.z = 0.0
        goal_pose.pose.orientation.w = 1.0
        goal_msg.goal = goal_pose

        start_data = request.get("start")
        use_explicit_start = isinstance(start_data, dict)
        if use_explicit_start:
            start_pose = PoseStamped()
            start_pose.header.frame_id = "map"
            start_pose.header.stamp = goal_pose.header.stamp
            try:
                start_pose.pose.position.x = float(start_data["x"])
                start_pose.pose.position.y = float(start_data["y"])
                start_pose.pose.position.z = float(start_data.get("z", 0.0) or 0.0)
                start_yaw = float(start_data.get("yaw", 0.0) or 0.0)
            except (KeyError, TypeError, ValueError) as exc:
                _write_error(request_id, f"Invalid start coordinates: {exc}")
                self.last_handled_request_id = request_id
                return
            qx, qy, qz, qw = _yaw_to_quaternion(start_yaw)
            start_pose.pose.orientation.x = qx
            start_pose.pose.orientation.y = qy
            start_pose.pose.orientation.z = qz
            start_pose.pose.orientation.w = qw
            goal_msg.start = start_pose

        goal_msg.use_start = use_explicit_start
        goal_msg.planner_id = PLANNER_ID

        self.active_request_id = request_id
        self.goal_pending = True
        self.goal_sent_at = time.time()
        send_future = self.client.send_goal_async(goal_msg)
        # Keep a strong reference so rclpy cannot drop the future before the callback fires.
        self._active_send_future = send_future
        send_future.add_done_callback(
            lambda f: self._goal_response_callback(f, request_id, request)
        )
        start_summary = "tf_start"
        if use_explicit_start:
            start_summary = f"start=({start_pose.pose.position.x:.2f}, {start_pose.pose.position.y:.2f})"
        self.get_logger().info(
            f"Sent ComputePathToPose for request {request_id}: "
            f"{start_summary}, goal=({goal_data['x']:.2f}, {goal_data['y']:.2f})"
        )

    def _goal_response_callback(self, future, request_id: str, request: dict) -> None:
        self.goal_pending = False
        self.goal_sent_at = None
        self._active_send_future = None
        if request_id != self.active_request_id:
            # A timeout already reset this request; discard the late response.
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            _write_error(request_id, f"Failed to send goal: {exc}")
            self._reset(request_id)
            return
        if not goal_handle.accepted:
            # Track consecutive rejections for this specific request.
            if self._rejection_request_id != request_id:
                self._rejection_count = 0
                self._rejection_request_id = request_id
            self._rejection_count += 1
            if self._rejection_count <= MAX_REJECTION_RETRIES:
                self.get_logger().warning(
                    f"ComputePathToPose goal rejected by Nav2 "
                    f"(attempt {self._rejection_count}/{MAX_REJECTION_RETRIES}), "
                    f"retrying in {REJECTION_RETRY_DELAY_SEC:.1f}s..."
                )
                # Clear active state without marking the request as handled so the
                # poll loop will re-send the same request after the retry delay.
                self.active_request_id = None
                self.active_goal_handle = None
                self.active_goal_started_at = None
                self.goal_pending = False
                self._accept_not_before = time.time() + REJECTION_RETRY_DELAY_SEC
            else:
                _write_error(
                    request_id,
                    f"ComputePathToPose goal was rejected by Nav2 after "
                    f"{MAX_REJECTION_RETRIES} attempts. Nav2 may still be "
                    "processing a previous cancellation or the planner is not active.",
                )
                self._reset(request_id)
            return
        self.active_goal_handle = goal_handle
        self.active_goal_started_at = time.time()
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            lambda f: self._result_callback(f, request_id, request)
        )

    def _result_callback(self, future, request_id: str, request: dict) -> None:
        if request_id != self.active_request_id:
            # A timeout already reset this request; discard the late result.
            return
        try:
            result_wrapper = future.result()
        except Exception as exc:
            _write_error(request_id, f"Planning result error: {exc}")
            self._reset(request_id)
            return

        if result_wrapper.status != GoalStatus.STATUS_SUCCEEDED:
            _write_error(request_id, _planner_error_detail(result_wrapper))
            self._reset(request_id)
            return

        nav_path = result_wrapper.result.path
        path = []
        for pose_stamped in nav_path.poses:
            path.append(
                {
                    "x": float(pose_stamped.pose.position.x),
                    "y": float(pose_stamped.pose.position.y),
                    "z": float(pose_stamped.pose.position.z),
                    # "map" frame is "world" in the frontend's naming convention
                    "frame_id": "world",
                }
            )

        raw_num_waypoints = len(path)
        original_path_length_m = _path_length_m(path)
        path = _thin_path_waypoints(path)
        path_length_m = _path_length_m(path)

        goal_data = request.get("goal") or {}
        requested_start = request.get("start") if isinstance(request.get("start"), dict) else None
        start_world = path[0] if path else {"x": 0.0, "y": 0.0, "z": 0.0, "frame_id": "world"}
        goal_world = path[-1] if path else {"x": float(goal_data.get("x", 0)), "y": float(goal_data.get("y", 0)), "z": 0.0, "frame_id": "world"}

        response = {
            "request_id": request_id,
            "path": path,
            "path_length_m": path_length_m,
            "num_waypoints": len(path),
            "start": {
                "requested_world": {
                    "x": float(requested_start.get("x", start_world["x"])) if requested_start else float(start_world["x"]),
                    "y": float(requested_start.get("y", start_world["y"])) if requested_start else float(start_world["y"]),
                    "z": float(requested_start.get("z", start_world["z"])) if requested_start else float(start_world["z"]),
                    "yaw": float(requested_start.get("yaw", 0.0)) if requested_start else 0.0,
                    "frame_id": "world",
                },
                "planned_world": start_world,
            },
            "goal": {"planned_world": goal_world},
            "planner": {"name": "nav2_smac_2d"},
            "raw_num_waypoints": raw_num_waypoints,
            "raw_path_length_m": original_path_length_m,
        }
        atomic_write_json(RESPONSE_PATH, response)
        self.get_logger().info(
            f"Plan ready for request {request_id}: "
            f"{len(path)} waypoints ({raw_num_waypoints} raw), {path_length_m:.2f} m"
        )
        try:
            REQUEST_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        self._reset(request_id)

    def _reset(self, request_id: str) -> None:
        self.active_request_id = None
        self.active_goal_handle = None
        self.active_goal_started_at = None
        self.goal_pending = False
        self.goal_sent_at = None
        self._active_send_future = None
        self.last_handled_request_id = request_id
        self._rejection_count = 0
        self._rejection_request_id = None
        self._accept_not_before = 0.0


def main() -> None:
    rclpy.init()
    node = Nav2PlannerBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
