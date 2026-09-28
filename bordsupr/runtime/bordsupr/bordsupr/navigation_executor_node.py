import json
import math
import os
import tempfile
import time
from pathlib import Path

import rclpy
import tf2_ros
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Time as RosTimeMsg
from geometry_msgs.msg import Pose, PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from nav2_msgs.action import FollowPath, NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from spot_msgs.msg import EStopStateArray, Feedback, LeaseArray, PowerState
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose


# The frontend A* planner uses "world" as the frame name for the slam_toolbox
# map frame.  Nav2 knows this frame as "map".
GOAL_FRAME = "map"
FOLLOW_PATH_CONTROLLER_ID = "FollowPath"
FOLLOW_PATH_GOAL_CHECKER_ID = "general_goal_checker"
FOLLOW_PATH_EXECUTION_MODES = {"follow_path", "nav2_follow_path"}
ROBOT_FRAME_STALE_SEC = float(os.getenv("ROBOT_ODOM_STALE_SEC", "3.0"))
ROBOT_STATUS_STALE_SEC = float(os.getenv("ROBOT_STATUS_STALE_SEC", "3.0"))
ESTOP_STATE_ESTOPPED = 1
POWER_STATE_ON = 2
LEASE_CLIENT_NAME_HINTS = tuple(
    hint.strip().lower()
    for hint in os.getenv("SPOT_LEASE_CLIENT_NAME_HINTS", "ros_spot,spot_ros2").split(",")
    if hint.strip()
)
ALWAYS_DEBUG_CMD_VEL_FOR_WEB_UI = os.getenv("NAVIGATION_ALWAYS_DEBUG_CMD_VEL_FOR_WEB_UI", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
SLAM_TAB_ROBOT_POSE_PATH = Path(os.getenv("SLAM_TAB_ROBOT_POSE_PATH", "/shared/slam_tab/robot_pose.json"))
SLAM_TAB_STATUS_PATH = Path(os.getenv("SLAM_TAB_STATUS_PATH", "/shared/slam_tab/status.json"))
DEBUG_CMD_VEL_POSE_TOLERANCE_M = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_POSE_TOLERANCE_M", "0.15"))
DEBUG_CMD_VEL_WAYPOINT_TOLERANCE_M = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_WAYPOINT_TOLERANCE_M", "0.20"))
DEBUG_CMD_VEL_YAW_TOLERANCE_RAD = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_YAW_TOLERANCE_RAD", "0.20"))
DEBUG_CMD_VEL_PROGRESS_TIMEOUT_SEC = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_PROGRESS_TIMEOUT_SEC", "8.0"))
DEBUG_CMD_VEL_MAX_TOTAL_DURATION_SEC = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_MAX_TOTAL_DURATION_SEC", "45.0"))
DEBUG_CMD_VEL_MIN_LINEAR_SPEED_MPS = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_MIN_LINEAR_SPEED_MPS", "0.25"))
DEBUG_CMD_VEL_LINEAR_KP = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_LINEAR_KP", "0.9"))
DEBUG_CMD_VEL_ANGULAR_KP = float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_ANGULAR_KP", "1.5"))
DEBUG_CMD_VEL_POSE_FEEDBACK_STALE_SEC = float(
    os.getenv("NAVIGATION_DEBUG_CMD_VEL_POSE_FEEDBACK_STALE_SEC", str(ROBOT_FRAME_STALE_SEC))
)
SPOT_CLAIM_SERVICE = os.getenv("SPOT_CLAIM_SERVICE", "/spot/claim")
SPOT_POWER_ON_SERVICE = os.getenv("SPOT_POWER_ON_SERVICE", "/spot/power_on")
SPOT_STAND_SERVICE = os.getenv("SPOT_STAND_SERVICE", "/spot/stand")
SPOT_TRIGGER_SERVICE_TIMEOUT_SEC = float(os.getenv("SPOT_TRIGGER_SERVICE_TIMEOUT_SEC", "2.0"))
SPOT_TRIGGER_RESULT_TIMEOUT_SEC = float(os.getenv("SPOT_TRIGGER_RESULT_TIMEOUT_SEC", "20.0"))
AUTO_PREPARE_SPOT_FOR_NAVIGATION = os.getenv("NAVIGATION_AUTO_PREPARE_SPOT", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
AUTO_PREPARE_WEB_UI_REQUESTS = os.getenv("NAVIGATION_AUTO_PREPARE_WEB_UI", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
NAVIGATION_COMPLETION_TOLERANCE_M = float(os.getenv("NAVIGATION_COMPLETION_TOLERANCE_M", "1.00"))
NAVIGATION_FALSE_SUCCESS_MAX_RETRIES = int(os.getenv("NAVIGATION_FALSE_SUCCESS_MAX_RETRIES", "2"))
NAVIGATION_ABORT_MAX_RETRIES = int(os.getenv("NAVIGATION_ABORT_MAX_RETRIES", "2"))


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def stamp_to_sec(stamp) -> float:
    if stamp is None:
        return 0.0
    try:
        sec = float(getattr(stamp, "sec", 0.0) or 0.0)
        nanosec = float(getattr(stamp, "nanosec", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return sec + (nanosec / 1_000_000_000.0)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class Nav2NavigationExecutor(Node):
    def __init__(self) -> None:
        super().__init__("navigation_executor_node")

        self.declare_parameter(
            "request_path",
            os.getenv("NAVIGATION_REQUEST_PATH", "/shared/navigation_request.json"),
        )
        self.declare_parameter(
            "status_path",
            os.getenv("NAVIGATION_STATUS_PATH", "/shared/navigation_status.json"),
        )
        self.declare_parameter(
            "cancel_request_path",
            os.getenv("NAVIGATION_CANCEL_REQUEST_PATH", "/shared/navigation_cancel_request.json"),
        )
        self.declare_parameter("poll_period_sec", float(os.getenv("NAV_POLL_PERIOD_SEC", "0.5")))
        self.declare_parameter(
            "debug_cmd_vel_rate_hz",
            float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_RATE_HZ", "10.0")),
        )
        self.declare_parameter(
            "debug_cmd_vel_linear_speed_mps",
            float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_LINEAR_SPEED_MPS", "0.5")),
        )
        self.declare_parameter(
            "debug_cmd_vel_angular_speed_rps",
            float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_ANGULAR_SPEED_RPS", "0.6")),
        )
        self.declare_parameter(
            "debug_cmd_vel_max_segment_duration_sec",
            float(os.getenv("NAVIGATION_DEBUG_CMD_VEL_MAX_SEGMENT_DURATION_SEC", "4.0")),
        )

        self.request_path = Path(self.get_parameter("request_path").value)
        self.status_path = Path(self.get_parameter("status_path").value)
        self.cancel_request_path = Path(self.get_parameter("cancel_request_path").value)
        self.poll_period_sec = max(0.2, float(self.get_parameter("poll_period_sec").value))
        self.debug_cmd_vel_rate_hz = max(1.0, float(self.get_parameter("debug_cmd_vel_rate_hz").value))
        self.debug_cmd_vel_linear_speed_mps = max(
            0.01, float(self.get_parameter("debug_cmd_vel_linear_speed_mps").value)
        )
        self.debug_cmd_vel_angular_speed_rps = max(
            0.05, float(self.get_parameter("debug_cmd_vel_angular_speed_rps").value)
        )
        self.debug_cmd_vel_max_segment_duration_sec = max(
            0.25, float(self.get_parameter("debug_cmd_vel_max_segment_duration_sec").value)
        )

        self.nav2_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.follow_path_client = ActionClient(self, FollowPath, "follow_path")
        self.cmd_vel_publisher = self.create_publisher(Twist, "/spot/cmd_vel", 10)
        self.body_pose_publisher = self.create_publisher(Pose, "/spot/body_pose", 10)
        self.claim_client = self.create_client(Trigger, SPOT_CLAIM_SERVICE)
        self.power_on_client = self.create_client(Trigger, SPOT_POWER_ON_SERVICE)
        self.stand_client = self.create_client(Trigger, SPOT_STAND_SERVICE)
        self.estop_state_msg = None
        self.estop_state_received_at = 0.0
        self.power_state_msg = None
        self.power_state_received_at = 0.0
        self.feedback_msg = None
        self.feedback_received_at = 0.0
        self.lease_state_msg = None
        self.lease_state_received_at = 0.0
        self.create_subscription(EStopStateArray, "/spot/status/estop", self._estop_state_callback, 10)
        self.create_subscription(PowerState, "/spot/status/power_states", self._power_state_callback, 10)
        self.create_subscription(Feedback, "/spot/status/feedback", self._feedback_callback_status, 10)
        self.create_subscription(LeaseArray, "/spot/status/leases", self._lease_state_callback, 10)

        self.active_request: dict | None = None
        self.active_request_id: str | None = None
        self.active_action_name: str | None = None
        self.last_completed_request_id: str | None = None
        self.current_goal_handle = None
        self.goal_request_pending = False
        self.false_success_retries = 0
        self.abort_retries = 0
        self.prep_queue: list[tuple[object, str, str]] = []
        self.prep_future = None
        self.prep_service_name: str | None = None
        self.prep_action_label: str | None = None
        self.start_time = time.time()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(self.poll_period_sec, self._poll_request)

        self.get_logger().info(
            f"Nav2 navigation executor: watching {self.request_path}, "
            f"status at {self.status_path}, "
            f"cancel requests at {self.cancel_request_path}"
        )
        self._mark_preexisting_request_stale()

    def _estop_state_callback(self, msg: EStopStateArray) -> None:
        self.estop_state_msg = msg
        self.estop_state_received_at = time.monotonic()

    def _power_state_callback(self, msg: PowerState) -> None:
        self.power_state_msg = msg
        self.power_state_received_at = time.monotonic()

    def _feedback_callback_status(self, msg: Feedback) -> None:
        self.feedback_msg = msg
        self.feedback_received_at = time.monotonic()

    def _lease_state_callback(self, msg: LeaseArray) -> None:
        self.lease_state_msg = msg
        self.lease_state_received_at = time.monotonic()

    def _check_cancel_request(self) -> bool:
        """Check for a cancel request file and abort the active Nav2 goal if it matches."""
        if self.active_request_id is None or self.current_goal_handle is None:
            return False
        if not self.cancel_request_path.exists():
            return False
        try:
            with self.cancel_request_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if payload.get("request_id") != self.active_request_id:
            return False
        # Matching cancel request — send cancel to Nav2
        try:
            self.current_goal_handle.cancel_goal_async()
        except Exception as exc:
            self.get_logger().warning(f"Failed to send cancel request to Nav2: {exc}")
            return False
        self.get_logger().info(f"Cancelled active navigation goal {self.active_request_id}")
        # Remove the cancel file so we don't re-process it
        try:
            self.cancel_request_path.unlink()
        except Exception:
            pass
        self._write_status("stopping", "Nav2 cancel request sent — waiting for confirmation…")
        return True

    # ------------------------------------------------------------------ #
    # Request file helpers                                                 #
    # ------------------------------------------------------------------ #

    def _load_request(self) -> dict | None:
        if not self.request_path.exists():
            return None
        try:
            with self.request_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            self._write_status("failed", f"Failed to read navigation request: {exc}")
            return None
        return payload if isinstance(payload, dict) else None

    def _write_status(self, state: str, message: str, **extra) -> None:
        payload: dict = {
            "available": True,
            "state": state,
            "message": message,
            "updated_at": time.time(),
        }
        if self.active_request is not None:
            payload["request_id"] = self.active_request_id
            payload["created_at"] = self.active_request.get("created_at")
            payload["observation"] = self.active_request.get("observation")
            payload["plan"] = self.active_request.get("plan")
            payload["execution"] = self.active_request.get("execution")
        payload.update(extra)
        atomic_write_json(self.status_path, payload)

    def _mark_preexisting_request_stale(self) -> None:
        request = self._load_request()
        if not request:
            return
        request_id = request.get("request_id")
        if not request_id:
            return
        created_at = request.get("created_at")
        try:
            created_at = float(created_at) if created_at is not None else 0.0
        except (TypeError, ValueError):
            created_at = 0.0
        if created_at > self.start_time:
            return
        self.last_completed_request_id = request_id
        atomic_write_json(
            self.status_path,
            {
                "available": True,
                "state": "idle",
                "message": "Ignored stale navigation request left over from before this startup.",
                "updated_at": time.time(),
                "request_id": request_id,
                "created_at": request.get("created_at"),
                "observation": request.get("observation"),
                "plan": request.get("plan"),
                "execution": request.get("execution"),
                "ignored_as_stale": True,
            },
        )
        self.get_logger().info(f"Ignored stale navigation request on startup: {request_id}")

    # ------------------------------------------------------------------ #
    # Poll timer                                                           #
    # ------------------------------------------------------------------ #

    def _poll_request(self) -> None:
        # Check for stop requests even when we have an active goal.
        self._check_cancel_request()

        if self.prep_future is not None:
            self._advance_robot_preparation()
            return

        if self.active_request_id is not None or self.goal_request_pending:
            return

        request = self._load_request()
        if not request:
            return

        request_id = request.get("request_id")
        if not request_id or request_id == self.last_completed_request_id:
            return

        self.active_request = request
        self.active_request_id = request_id
        self.false_success_retries = 0
        self.abort_retries = 0

        self._process_active_request()

    def _process_active_request(self, *, skip_prepare: bool = False) -> None:
        request = self.active_request or {}
        request_id = self.active_request_id
        if request_id is None:
            return

        if not skip_prepare and self._maybe_start_robot_preparation(request):
            return

        blocker_reason, force_debug_fallback = self._debug_cmd_vel_blocker(request)
        if blocker_reason is not None:
            if self._run_debug_cmd_vel_fallback(blocker_reason, force=force_debug_fallback):
                return

        # Only use FollowPath when the caller explicitly asks for it.
        # The SLAM tab queues requests as NavigateToPose, and routing those
        # preview paths through FollowPath can produce jerky stop/start motion.
        follow_path_goal = self._build_follow_path_goal(request)
        if self._should_use_follow_path(request) and follow_path_goal is not None and self.follow_path_client.server_is_ready():
            self._send_follow_path_goal(follow_path_goal)
            return

        # Fallback to NavigateToPose if no valid previewed path is available.
        goal_pose = self._extract_goal_pose(request)
        if goal_pose is None:
            self._write_status("failed", "Cannot extract a valid goal pose from the navigation plan.")
            self.last_completed_request_id = request_id
            self.active_request = None
            self.active_request_id = None
            return

        if not self.nav2_client.server_is_ready():
            # Keep the request queued until Nav2 is ready instead of wedging
            # this executor instance in a pseudo-active state.
            self.active_request = None
            self.active_request_id = None
            return

        if AUTO_PREPARE_SPOT_FOR_NAVIGATION:
            ready, ready_message = self._ensure_robot_ready_for_navigation()
            if not ready:
                if self._run_debug_cmd_vel_fallback(ready_message, force=True):
                    return
                self._finish("failed", ready_message)
                return

        self._send_nav2_goal(goal_pose)

    # ------------------------------------------------------------------ #
    # Goal extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_goal_pose(self, request: dict) -> dict | None:
        plan = request.get("plan") or {}
        path = plan.get("path") or []

        # prefer the A*-snapped planned_world point, then requested_world, then last path point
        goal_data = plan.get("goal") or {}
        goal_world = goal_data.get("planned_world") or goal_data.get("requested_world")

        x: float | None = None
        y: float | None = None
        if isinstance(goal_world, dict):
            try:
                x = float(goal_world["x"])
                y = float(goal_world["y"])
            except (KeyError, TypeError, ValueError):
                pass

        if x is None and path:
            try:
                last = path[-1]
                x = float(last["x"])
                y = float(last["y"])
            except (KeyError, TypeError, ValueError):
                return None

        if x is None:
            return None

        # compute approach yaw from the final path segment
        yaw = 0.0
        if len(path) >= 2:
            try:
                p2 = path[-1]
                p1 = path[-2]
                dx = float(p2["x"]) - float(p1["x"])
                dy = float(p2["y"]) - float(p1["y"])
                if math.hypot(dx, dy) > 1e-6:
                    yaw = math.atan2(dy, dx)
            except (KeyError, TypeError, ValueError):
                pass

        return {"x": x, "y": y, "yaw": yaw}

    def _should_use_follow_path(self, request: dict) -> bool:
        execution = request.get("execution") or {}
        mode = execution.get("mode")
        if isinstance(mode, str):
            return mode.strip().lower() in FOLLOW_PATH_EXECUTION_MODES
        return False

    def _debug_cmd_vel_requested(self, request: dict) -> bool:
        execution = request.get("execution") or {}
        return bool(execution.get("debug_publish_cmd_vel_on_failure"))

    def _always_debug_cmd_vel_for_request(self, request: dict) -> bool:
        if not self._debug_cmd_vel_requested(request):
            return False
        if not ALWAYS_DEBUG_CMD_VEL_FOR_WEB_UI:
            return False
        execution = request.get("execution") or {}
        accepted_from = str(execution.get("accepted_from") or "").strip().lower()
        return accepted_from == "web_ui"

    def _is_web_ui_request(self, request: dict) -> bool:
        execution = request.get("execution") or {}
        accepted_from = str(execution.get("accepted_from") or "").strip().lower()
        return accepted_from == "web_ui"

    def _should_auto_prepare_request(self, request: dict) -> bool:
        if self._debug_cmd_vel_requested(request):
            return False
        if self._is_web_ui_request(request):
            return AUTO_PREPARE_WEB_UI_REQUESTS
        return AUTO_PREPARE_SPOT_FOR_NAVIGATION

    def _robot_frame_available(self) -> bool:
        try:
            transform = self.tf_buffer.lookup_transform(
                GOAL_FRAME,
                "spot/body",
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
            transform_stamp = stamp_to_sec(getattr(transform, "header", None).stamp if getattr(transform, "header", None) else None)
            if transform_stamp <= 0.0:
                return False
            return (time.time() - transform_stamp) <= ROBOT_FRAME_STALE_SEC
        except Exception:
            return False

    def _status_sample_is_fresh(self, received_at: float | None) -> bool:
        try:
            timestamp = float(received_at or 0.0)
        except (TypeError, ValueError):
            return False
        return timestamp > 0.0 and (time.monotonic() - timestamp) <= ROBOT_STATUS_STALE_SEC

    def _lease_owned_by_driver(self) -> bool:
        lease_msg = getattr(self, "lease_state_msg", None)
        resources = getattr(lease_msg, "resources", None) or []
        saw_relevant_resource = False
        for resource in resources:
            resource_name = str(getattr(resource, "resource", "") or "").lower()
            if resource_name not in {"all-leases", "body", "mobility"}:
                continue
            saw_relevant_resource = True
            lease_owner = getattr(resource, "lease_owner", None)
            client_name = str(getattr(lease_owner, "client_name", "") or "").lower()
            if any(hint in client_name for hint in LEASE_CLIENT_NAME_HINTS):
                return True
        return not saw_relevant_resource

    def _robot_motion_blocker_reason(self) -> str | None:
        if self._status_sample_is_fresh(getattr(self, "estop_state_received_at", 0.0)):
            estop_states = getattr(getattr(self, "estop_state_msg", None), "estop_states", None) or []
            if any(int(getattr(state, "state", 0) or 0) == ESTOP_STATE_ESTOPPED for state in estop_states):
                return "Spot is estopped"

        if self._status_sample_is_fresh(getattr(self, "power_state_received_at", 0.0)):
            motor_power_state = int(getattr(getattr(self, "power_state_msg", None), "motor_power_state", 0) or 0)
            if motor_power_state != POWER_STATE_ON:
                return "Spot motor power is not on"

        if self._status_sample_is_fresh(getattr(self, "feedback_received_at", 0.0)):
            feedback_msg = getattr(self, "feedback_msg", None)
            if feedback_msg is not None and not bool(getattr(feedback_msg, "standing", False)):
                if bool(getattr(feedback_msg, "sitting", False)):
                    return "Spot is sitting"
                return "Spot is not standing"

        if self._status_sample_is_fresh(getattr(self, "lease_state_received_at", 0.0)) and not self._lease_owned_by_driver():
            return "Spot body or mobility lease is not held by ROS"

        return None

    def _debug_cmd_vel_blocker(self, request: dict) -> tuple[str | None, bool]:
        if not self._debug_cmd_vel_requested(request):
            return None, False
        if self._always_debug_cmd_vel_for_request(request):
            return "Skipped Nav2 execution because the SLAM tab requested direct debug cmd_vel publishing.", True
        if not self._robot_frame_available():
            return "Skipped Nav2 execution because live spot/body TF is unavailable.", False
        readiness_reason = self._robot_motion_blocker_reason()
        if readiness_reason is not None:
            return f"Skipped Nav2 execution because {readiness_reason}.", True
        return None, False

    def _build_robot_preparation_steps(self, request: dict) -> tuple[tuple[object, str, str], ...]:
        if not self._should_auto_prepare_request(request):
            return ()

        blocker = self._robot_motion_blocker_reason()
        if blocker == "Spot is estopped":
            return ()

        steps: list[tuple[object, str, str]] = []
        if self._is_web_ui_request(request):
            steps.append((self.claim_client, SPOT_CLAIM_SERVICE, "claim lease"))
        elif blocker == "Spot body or mobility lease is not held by ROS":
            steps.append((self.claim_client, SPOT_CLAIM_SERVICE, "claim lease"))

        if blocker == "Spot motor power is not on":
            steps.append((self.power_on_client, SPOT_POWER_ON_SERVICE, "power on"))
            steps.append((self.stand_client, SPOT_STAND_SERVICE, "stand"))
        elif blocker in {"Spot is sitting", "Spot is not standing"}:
            steps.append((self.stand_client, SPOT_STAND_SERVICE, "stand"))

        deduped_steps: list[tuple[object, str, str]] = []
        seen_service_names: set[str] = set()
        for client, service_name, action_label in steps:
            if service_name in seen_service_names:
                continue
            seen_service_names.add(service_name)
            deduped_steps.append((client, service_name, action_label))
        return tuple(deduped_steps)

    def _maybe_start_robot_preparation(self, request: dict) -> bool:
        steps = self._build_robot_preparation_steps(request)
        if not steps:
            return False
        self.prep_queue = list(steps)
        self._start_next_robot_preparation_step()
        return self.prep_future is not None

    def _start_next_robot_preparation_step(self) -> None:
        if not self.prep_queue:
            self.prep_future = None
            self.prep_service_name = None
            self.prep_action_label = None
            return

        client, service_name, action_label = self.prep_queue.pop(0)
        if client is None:
            self._finish("failed", f"Unable to {action_label} before navigation: {service_name} client is unavailable.")
            return
        try:
            if not client.wait_for_service(timeout_sec=SPOT_TRIGGER_SERVICE_TIMEOUT_SEC):
                self._finish("failed", f"Unable to {action_label} before navigation: {service_name} service is unavailable.")
                return
            self.prep_future = client.call_async(Trigger.Request())
            self.prep_service_name = service_name
            self.prep_action_label = action_label
            self._write_status("running", f"Preparing Spot for navigation: attempting to {action_label}…")
        except Exception as exc:
            self._finish("failed", f"Unable to {action_label} before navigation: {service_name} failed: {exc}")

    def _advance_robot_preparation(self) -> None:
        future = self.prep_future
        if future is None:
            return
        if not future.done():
            return

        service_name = self.prep_service_name or "service"
        action_label = self.prep_action_label or "prepare Spot"
        self.prep_future = None
        self.prep_service_name = None
        self.prep_action_label = None

        try:
            response = future.result()
        except Exception as exc:
            self.prep_queue = []
            self._finish("failed", f"Unable to {action_label} before navigation: {service_name} failed: {exc}")
            return

        if response is None:
            self.prep_queue = []
            self._finish("failed", f"Unable to {action_label} before navigation: {service_name} returned no response.")
            return

        if not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", "") or "").strip() or "request was rejected"
            self.prep_queue = []
            self._finish("failed", f"Unable to {action_label} before navigation: {service_name} failed: {message}")
            return

        message = str(getattr(response, "message", "") or "").strip()
        if message:
            self.get_logger().info(f"{service_name}: {message}")

        if self.prep_queue:
            self._start_next_robot_preparation_step()
            return

        self._process_active_request(skip_prepare=True)

    def _call_trigger_service(self, client, service_name: str) -> tuple[bool, str]:
        if client is None:
            return False, f"{service_name} client is unavailable."
        try:
            if not client.wait_for_service(timeout_sec=SPOT_TRIGGER_SERVICE_TIMEOUT_SEC):
                return False, f"{service_name} service is unavailable."
            future = client.call_async(Trigger.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=SPOT_TRIGGER_RESULT_TIMEOUT_SEC)
            if not future.done():
                return False, f"{service_name} timed out."
            response = future.result()
        except Exception as exc:
            return False, f"{service_name} failed: {exc}"
        if response is None:
            return False, f"{service_name} returned no response."
        if bool(getattr(response, "success", False)):
            return True, str(getattr(response, "message", "") or "")
        message = str(getattr(response, "message", "") or "").strip()
        if not message:
            message = "request was rejected"
        return False, f"{service_name} failed: {message}"

    def _ensure_robot_ready_for_navigation(self) -> tuple[bool, str]:
        blocker = self._robot_motion_blocker_reason()
        if blocker is None:
            return True, "Spot is ready for navigation."

        if blocker == "Spot is estopped":
            return False, "Cannot navigate while Spot is estopped."

        if blocker == "Spot body or mobility lease is not held by ROS":
            steps = (
                (self.claim_client, SPOT_CLAIM_SERVICE, "claim lease"),
            )
        elif blocker == "Spot motor power is not on":
            steps = (
                (self.power_on_client, SPOT_POWER_ON_SERVICE, "power on"),
                (self.stand_client, SPOT_STAND_SERVICE, "stand"),
            )
        elif blocker in {"Spot is sitting", "Spot is not standing"}:
            steps = (
                (self.stand_client, SPOT_STAND_SERVICE, "stand"),
            )
        else:
            return False, f"Unable to prepare Spot for navigation: {blocker}"

        for client, service_name, action_label in steps:
            ok, message = self._call_trigger_service(client, service_name)
            if not ok:
                return False, f"Unable to {action_label} before navigation: {message}"
            if message:
                self.get_logger().info(f"{service_name}: {message}")

        return True, "Spot is ready for navigation."

    def _build_follow_path_goal(self, request: dict) -> FollowPath.Goal | None:
        plan = request.get("plan") or {}
        raw_path = plan.get("path") or []
        if not isinstance(raw_path, list) or len(raw_path) < 2:
            return None

        points: list[tuple[float, float, float]] = []
        for point in raw_path:
            if not isinstance(point, dict):
                continue
            try:
                points.append(
                    (
                        float(point["x"]),
                        float(point["y"]),
                        float(point.get("z") or 0.0),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        if len(points) < 2:
            return None

        goal_yaw = None
        goal_data = plan.get("goal") or {}
        try:
            goal_yaw = float(goal_data.get("yaw")) if goal_data.get("yaw") is not None else None
        except (TypeError, ValueError):
            goal_yaw = None

        # Use a zero stamp so Nav2 treats the path as "latest available TF"
        # instead of forcing transforms against a fresh wall-clock timestamp.
        stamp = RosTimeMsg()
        path_msg = NavPath()
        path_msg.header.frame_id = GOAL_FRAME
        path_msg.header.stamp = stamp

        previous_yaw = 0.0
        for index, (x, y, z) in enumerate(points):
            if index < len(points) - 1:
                next_x, next_y, _ = points[index + 1]
                dx = next_x - x
                dy = next_y - y
                if math.hypot(dx, dy) > 1e-6:
                    previous_yaw = math.atan2(dy, dx)
            elif goal_yaw is not None:
                previous_yaw = goal_yaw

            pose = PoseStamped()
            pose.header.frame_id = GOAL_FRAME
            pose.header.stamp = stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            qx, qy, qz, qw = yaw_to_quaternion(previous_yaw)
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)

        goal_msg = FollowPath.Goal()
        goal_msg.path = path_msg
        goal_msg.controller_id = FOLLOW_PATH_CONTROLLER_ID
        goal_msg.goal_checker_id = FOLLOW_PATH_GOAL_CHECKER_ID
        return goal_msg

    # ------------------------------------------------------------------ #
    # Nav2 action                                                          #
    # ------------------------------------------------------------------ #

    def _prepend_robot_pose_to_path(self, path_msg: NavPath) -> None:
        """Prepend the robot's current TF pose so the path starts exactly at the robot.

        This eliminates lateral crab-walking when the A*-planned start pose differs
        from the robot's actual TF position (e.g. after set-pose or SLAM drift).
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                GOAL_FRAME,
                "spot/body",
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
        except Exception as exc:
            self.get_logger().warning(
                f"Could not look up {GOAL_FRAME} -> spot/body transform: {exc}"
            )
            return

        robot_pose = PoseStamped()
        robot_pose.header.frame_id = GOAL_FRAME
        robot_pose.header.stamp = path_msg.header.stamp
        robot_pose.pose.position.x = transform.transform.translation.x
        robot_pose.pose.position.y = transform.transform.translation.y
        robot_pose.pose.position.z = transform.transform.translation.z
        robot_pose.pose.orientation = transform.transform.rotation

        # Only prepend if the robot is meaningfully away from the first waypoint.
        if path_msg.poses:
            first = path_msg.poses[0].pose.position
            dx = first.x - robot_pose.pose.position.x
            dy = first.y - robot_pose.pose.position.y
            if math.hypot(dx, dy) < 0.05:
                return

        path_msg.poses.insert(0, robot_pose)

    def _send_follow_path_goal(self, goal_msg: FollowPath.Goal) -> None:
        self.active_action_name = "FollowPath"
        self._prepend_robot_pose_to_path(goal_msg.path)
        self._write_status(
            "running",
            f"Sending FollowPath goal to Nav2 controller: "
            f"{len(goal_msg.path.poses)} poses in frame '{GOAL_FRAME}'.",
            goal_pose={
                "x": float(goal_msg.path.poses[-1].pose.position.x),
                "y": float(goal_msg.path.poses[-1].pose.position.y),
                "yaw": 0.0,
            },
        )
        self.goal_request_pending = True
        send_future = self.follow_path_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_callback
        )
        send_future.add_done_callback(self._goal_response_callback)

    def _send_nav2_goal(self, goal_pose: dict) -> None:
        self.active_action_name = "NavigateToPose"
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = PoseStamped()
        goal_msg.pose.header.frame_id = GOAL_FRAME
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = float(goal_pose["x"])
        goal_msg.pose.pose.position.y = float(goal_pose["y"])
        goal_msg.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quaternion(float(goal_pose["yaw"]))
        goal_msg.pose.pose.orientation.x = qx
        goal_msg.pose.pose.orientation.y = qy
        goal_msg.pose.pose.orientation.z = qz
        goal_msg.pose.pose.orientation.w = qw

        self._write_status(
            "running",
            f"Sending NavigateToPose goal to Nav2: "
            f"({goal_pose['x']:.2f}, {goal_pose['y']:.2f}) in frame '{GOAL_FRAME}'.",
            goal_pose=goal_pose,
        )
        self.goal_request_pending = True
        send_future = self.nav2_client.send_goal_async(
            goal_msg, feedback_callback=self._feedback_callback
        )
        send_future.add_done_callback(self._goal_response_callback)

    def _publish_twist_for_steps(self, v_x: float, v_y: float, v_rot: float, steps: int) -> None:
        msg = Twist()
        msg.linear.x = float(v_x)
        msg.linear.y = float(v_y)
        msg.angular.z = float(v_rot)
        sleep_sec = 1.0 / self.debug_cmd_vel_rate_hz
        for _ in range(max(1, steps)):
            self.cmd_vel_publisher.publish(msg)
            time.sleep(sleep_sec)

    def _publish_zero_twist(self) -> None:
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)

    def _publish_body_pose(self, pitch: float) -> None:
        """Publish a body pose with the given pitch around the Y axis.

        Negative pitch looks UP; positive looks DOWN.
        """
        half_pitch = pitch * 0.5
        # Quaternion for rotation around Y axis (pitch): [0, sin(half), 0, cos(half)]
        qx = 0.0
        qy = math.sin(half_pitch)
        qz = 0.0
        qw = math.cos(half_pitch)
        msg = Pose()
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw
        self.body_pose_publisher.publish(msg)
        direction = "up" if pitch < 0 else "down"
        self.get_logger().info(f"Published look-{direction} body pose (pitch={pitch:.2f} rad).")

    def _publish_look_up_pose(self) -> None:
        """Publish a body pose with negative pitch so Spot tilts its cameras upward."""
        self._publish_body_pose(-0.35)

    def _publish_look_straight_pose(self) -> None:
        """Publish a body pose with zero pitch so Spot looks straight ahead while walking."""
        self._publish_body_pose(0.0)

    def _load_slam_pose_feedback(self) -> dict | None:
        try:
            with SLAM_TAB_ROBOT_POSE_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        position = payload.get("position") or {}
        orientation = payload.get("orientation") or {}
        try:
            x = float(position["x"])
            y = float(position["y"])
            yaw = quaternion_to_yaw(
                float(orientation.get("x", 0.0) or 0.0),
                float(orientation.get("y", 0.0) or 0.0),
                float(orientation.get("z", 0.0) or 0.0),
                float(orientation.get("w", 1.0) or 1.0),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return {"x": x, "y": y, "yaw": yaw, "updated_at": time.time()}

    def _slam_pose_feedback_is_fresh(self) -> bool:
        try:
            age_sec = max(0.0, time.time() - SLAM_TAB_ROBOT_POSE_PATH.stat().st_mtime)
        except Exception:
            return False
        return age_sec <= DEBUG_CMD_VEL_POSE_FEEDBACK_STALE_SEC

    def _slam_robot_connection_available(self) -> bool:
        try:
            with SLAM_TAB_STATUS_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return True
        if not isinstance(payload, dict):
            return True
        robot_connection = payload.get("robot_connection")
        if not isinstance(robot_connection, dict):
            return True
        return bool(robot_connection.get("connected"))

    @staticmethod
    def _angle_error(target: float, current: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _publish_twist_once(self, v_x: float, v_y: float, v_rot: float) -> None:
        msg = Twist()
        msg.linear.x = float(v_x)
        msg.linear.y = float(v_y)
        msg.angular.z = float(v_rot)
        self.cmd_vel_publisher.publish(msg)

    def _run_pose_feedback_cmd_vel_fallback(
        self,
        points: list[tuple[float, float]],
        desired_yaw: float | None,
        rejection_reason: str,
    ) -> bool:
        if not self._slam_robot_connection_available():
            self.get_logger().info(
                "Skipping SLAM pose feedback cmd_vel fallback because "
                "the SLAM status reports the robot connection is disconnected."
            )
            return False
        if not self._slam_pose_feedback_is_fresh():
            self.get_logger().info(
                "Skipping SLAM pose feedback cmd_vel fallback because "
                f"{SLAM_TAB_ROBOT_POSE_PATH} is stale."
            )
            return False
        current_pose = self._load_slam_pose_feedback()
        if current_pose is None:
            return False

        self._write_status(
            "running",
            "Nav2 could not execute because live spot/body TF is unavailable. "
            "Publishing debug cmd_vel with SLAM pose feedback.",
            debug_cmd_vel_fallback=True,
            nav2_rejection_reason=rejection_reason,
            feedback_pose_source=str(SLAM_TAB_ROBOT_POSE_PATH),
        )

        rate_hz = self.debug_cmd_vel_rate_hz
        sleep_sec = 1.0 / rate_hz
        started_at = time.time()
        last_progress_at = started_at
        waypoint_index = 1 if len(points) > 1 else 0
        previous_distance = None

        while waypoint_index < len(points):
            pose = self._load_slam_pose_feedback()
            if pose is None:
                self._publish_zero_twist()
                self._finish("failed", "Lost SLAM pose feedback while publishing debug cmd_vel fallback.")
                return True

            target_x, target_y = points[waypoint_index]
            dx = target_x - pose["x"]
            dy = target_y - pose["y"]
            distance = math.hypot(dx, dy)
            waypoint_tolerance = DEBUG_CMD_VEL_WAYPOINT_TOLERANCE_M
            if waypoint_index == len(points) - 1:
                waypoint_tolerance = DEBUG_CMD_VEL_POSE_TOLERANCE_M

            if distance <= waypoint_tolerance:
                if previous_distance is None or distance < previous_distance - 1e-3:
                    last_progress_at = time.time()
                previous_distance = None
                waypoint_index += 1
                continue

            now = time.time()
            if previous_distance is None or distance < previous_distance - 0.01:
                last_progress_at = now
            previous_distance = distance

            if (now - started_at) > DEBUG_CMD_VEL_MAX_TOTAL_DURATION_SEC:
                self._publish_zero_twist()
                self._finish(
                    "failed",
                    "Timed out while publishing debug cmd_vel fallback with SLAM pose feedback.",
                )
                return True

            if (now - last_progress_at) > DEBUG_CMD_VEL_PROGRESS_TIMEOUT_SEC:
                self._publish_zero_twist()
                self._finish(
                    "failed",
                    "No progress detected from SLAM pose feedback while publishing debug cmd_vel fallback.",
                )
                return True

            speed = min(
                self.debug_cmd_vel_linear_speed_mps,
                max(DEBUG_CMD_VEL_MIN_LINEAR_SPEED_MPS, DEBUG_CMD_VEL_LINEAR_KP * distance),
            )
            direction_x = dx / max(distance, 1e-6)
            direction_y = dy / max(distance, 1e-6)
            self._publish_twist_once(speed * direction_x, speed * direction_y, 0.0)
            self._write_status(
                "running",
                "Nav2 could not execute because live spot/body TF is unavailable. "
                "Publishing debug cmd_vel with SLAM pose feedback.",
                debug_cmd_vel_fallback=True,
                nav2_rejection_reason=rejection_reason,
                feedback_pose_source=str(SLAM_TAB_ROBOT_POSE_PATH),
                distance_remaining_m=distance,
                current_waypoint_index=waypoint_index,
                num_waypoints=len(points),
            )
            time.sleep(sleep_sec)

        if desired_yaw is not None:
            rotate_started_at = time.time()
            while True:
                pose = self._load_slam_pose_feedback()
                if pose is None:
                    self._publish_zero_twist()
                    self._finish("failed", "Lost SLAM pose feedback while rotating to final yaw.")
                    return True
                yaw_error = self._angle_error(float(desired_yaw), float(pose["yaw"]))
                if abs(yaw_error) <= DEBUG_CMD_VEL_YAW_TOLERANCE_RAD:
                    break
                if (time.time() - rotate_started_at) > self.debug_cmd_vel_max_segment_duration_sec:
                    self._publish_zero_twist()
                    self._finish("failed", "Timed out while rotating to the final yaw in debug cmd_vel fallback.")
                    return True
                angular_speed = min(
                    self.debug_cmd_vel_angular_speed_rps,
                    max(0.1, abs(yaw_error) * DEBUG_CMD_VEL_ANGULAR_KP),
                )
                self._publish_twist_once(0.0, 0.0, math.copysign(angular_speed, yaw_error))
                time.sleep(sleep_sec)

        self._publish_zero_twist()
        self._finish(
            "completed",
            "Published debug cmd_vel fallback from the SLAM tab plan using SLAM pose feedback.",
        )
        return True

    def _run_open_loop_debug_cmd_vel_fallback(
        self,
        points: list[tuple[float, float]],
        desired_yaw: float | None,
        rejection_reason: str,
    ) -> bool:
        self._write_status(
            "running",
            "Nav2 could not execute because live spot/body TF is unavailable. "
            "Publishing debug cmd_vel directly from the planned path.",
            debug_cmd_vel_fallback=True,
            nav2_rejection_reason=rejection_reason,
        )

        rate_hz = self.debug_cmd_vel_rate_hz
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            dx = x2 - x1
            dy = y2 - y1
            distance = math.hypot(dx, dy)
            if distance <= 1e-6:
                continue
            v_x = (dx / distance) * self.debug_cmd_vel_linear_speed_mps
            v_y = (dy / distance) * self.debug_cmd_vel_linear_speed_mps
            duration = min(
                self.debug_cmd_vel_max_segment_duration_sec,
                max(0.3, distance / self.debug_cmd_vel_linear_speed_mps),
            )
            steps = max(1, int(math.ceil(duration * rate_hz)))
            self._publish_twist_for_steps(v_x, v_y, 0.0, steps)

        if desired_yaw is not None and len(points) >= 2:
            try:
                last_dx = points[-1][0] - points[-2][0]
                last_dy = points[-1][1] - points[-2][1]
                if math.hypot(last_dx, last_dy) > 1e-6:
                    path_yaw = math.atan2(last_dy, last_dx)
                    yaw_error = self._angle_error(float(desired_yaw), path_yaw)
                    if abs(yaw_error) > 0.1:
                        v_rot = math.copysign(self.debug_cmd_vel_angular_speed_rps, yaw_error)
                        duration = min(
                            self.debug_cmd_vel_max_segment_duration_sec,
                            max(0.25, abs(yaw_error) / self.debug_cmd_vel_angular_speed_rps),
                        )
                        steps = max(1, int(math.ceil(duration * rate_hz)))
                        self._publish_twist_for_steps(0.0, 0.0, v_rot, steps)
            except (TypeError, ValueError):
                pass

        self._publish_zero_twist()
        self._finish(
            "completed",
            "Published debug cmd_vel fallback from the SLAM tab plan because live spot/body TF was unavailable.",
        )
        return True

    def _run_assumed_pose_cmd_vel_fallback(
        self,
        points: list[tuple[float, float]],
        desired_yaw: float | None,
        rejection_reason: str,
    ) -> bool:
        assumed_pose = self._load_slam_pose_feedback()
        if assumed_pose is None:
            return False

        self._write_status(
            "running",
            "Nav2 could not execute because live spot/body TF is unavailable. "
            "Publishing debug cmd_vel using the last known SLAM pose while the displayed robot pose stays frozen.",
            debug_cmd_vel_fallback=True,
            nav2_rejection_reason=rejection_reason,
            feedback_pose_source=str(SLAM_TAB_ROBOT_POSE_PATH),
            feedback_pose_assumed_static=True,
        )

        rate_hz = self.debug_cmd_vel_rate_hz
        sleep_sec = 1.0 / rate_hz
        started_at = time.time()
        waypoint_index = 1 if len(points) > 1 else 0

        while waypoint_index < len(points):
            target_x, target_y = points[waypoint_index]
            dx = target_x - assumed_pose["x"]
            dy = target_y - assumed_pose["y"]
            distance = math.hypot(dx, dy)
            waypoint_tolerance = DEBUG_CMD_VEL_WAYPOINT_TOLERANCE_M
            if waypoint_index == len(points) - 1:
                waypoint_tolerance = DEBUG_CMD_VEL_POSE_TOLERANCE_M

            if distance <= waypoint_tolerance:
                waypoint_index += 1
                continue

            now = time.time()
            if (now - started_at) > DEBUG_CMD_VEL_MAX_TOTAL_DURATION_SEC:
                self._publish_zero_twist()
                self._finish(
                    "failed",
                    "Timed out while publishing debug cmd_vel fallback using the last known SLAM pose assumption.",
                )
                return True

            speed = min(
                self.debug_cmd_vel_linear_speed_mps,
                max(DEBUG_CMD_VEL_MIN_LINEAR_SPEED_MPS, DEBUG_CMD_VEL_LINEAR_KP * distance),
            )
            direction_x = dx / max(distance, 1e-6)
            direction_y = dy / max(distance, 1e-6)
            v_x = speed * direction_x
            v_y = speed * direction_y
            self._publish_twist_once(v_x, v_y, 0.0)
            assumed_pose["x"] += v_x * sleep_sec
            assumed_pose["y"] += v_y * sleep_sec
            self._write_status(
                "running",
                "Nav2 could not execute because live spot/body TF is unavailable. "
                "Publishing debug cmd_vel using the last known SLAM pose while the displayed robot pose stays frozen.",
                debug_cmd_vel_fallback=True,
                nav2_rejection_reason=rejection_reason,
                feedback_pose_source=str(SLAM_TAB_ROBOT_POSE_PATH),
                feedback_pose_assumed_static=True,
                distance_remaining_m=max(0.0, distance - (speed * sleep_sec)),
                current_waypoint_index=waypoint_index,
                num_waypoints=len(points),
            )
            time.sleep(sleep_sec)

        if desired_yaw is not None:
            rotate_started_at = time.time()
            while True:
                yaw_error = self._angle_error(float(desired_yaw), float(assumed_pose["yaw"]))
                if abs(yaw_error) <= DEBUG_CMD_VEL_YAW_TOLERANCE_RAD:
                    break
                if (time.time() - rotate_started_at) > self.debug_cmd_vel_max_segment_duration_sec:
                    self._publish_zero_twist()
                    self._finish(
                        "failed",
                        "Timed out while rotating to the final yaw using the last known SLAM pose assumption.",
                    )
                    return True
                angular_speed = min(
                    self.debug_cmd_vel_angular_speed_rps,
                    max(0.1, abs(yaw_error) * DEBUG_CMD_VEL_ANGULAR_KP),
                )
                v_rot = math.copysign(angular_speed, yaw_error)
                self._publish_twist_once(0.0, 0.0, v_rot)
                assumed_pose["yaw"] += v_rot * sleep_sec
                time.sleep(sleep_sec)

        self._publish_zero_twist()
        self._finish(
            "completed",
            "Published debug cmd_vel fallback from the SLAM tab plan using the last known SLAM pose assumption.",
        )
        return True

    def _run_debug_cmd_vel_fallback(self, rejection_reason: str, force: bool = False) -> bool:
        request = self.active_request or {}
        if not self._debug_cmd_vel_requested(request):
            return False
        if not force and self._robot_frame_available():
            return False

        plan = request.get("plan") or {}
        path = plan.get("path") or []
        points: list[tuple[float, float]] = []
        for point in path:
            if not isinstance(point, dict):
                continue
            try:
                points.append((float(point["x"]), float(point["y"])))
            except (KeyError, TypeError, ValueError):
                continue
        if len(points) < 2:
            return False

        goal = plan.get("goal") or {}
        try:
            goal_yaw = float(goal.get("yaw")) if goal.get("yaw") is not None else None
        except (TypeError, ValueError):
            goal_yaw = None

        self._publish_look_straight_pose()
        if self._run_pose_feedback_cmd_vel_fallback(points, goal_yaw, rejection_reason):
            return True
        if self._run_assumed_pose_cmd_vel_fallback(points, goal_yaw, rejection_reason):
            return True
        return self._run_open_loop_debug_cmd_vel_fallback(points, goal_yaw, rejection_reason)

    def _feedback_callback(self, feedback_msg) -> None:
        if self.active_request_id is None:
            return
        feedback = feedback_msg.feedback
        if hasattr(feedback, "distance_to_goal"):
            distance = float(getattr(feedback, "distance_to_goal", 0.0))
            speed = float(getattr(feedback, "speed", 0.0))
            self._write_status(
                "running",
                f"Following path: {distance:.2f} m to goal at {speed:.2f} m/s.",
                distance_remaining_m=distance,
                current_speed_mps=speed,
            )
            return

        distance = float(getattr(feedback, "distance_remaining", 0.0))
        recoveries = int(getattr(feedback, "number_of_recoveries", 0))
        self._write_status(
            "running",
            f"Navigating: {distance:.2f} m remaining, {recoveries} recovery(ies).",
            distance_remaining_m=distance,
            number_of_recoveries=recoveries,
        )

    def _goal_response_callback(self, future) -> None:
        self.goal_request_pending = False
        if self.active_request_id is None:
            return
        try:
            goal_handle = future.result()
        except Exception as exc:
            self._finish("failed", f"Failed to send NavigateToPose goal: {exc}")
            return
        if goal_handle is None or not goal_handle.accepted:
            if self.active_action_name == "FollowPath":
                goal_pose = self._extract_goal_pose(self.active_request or {})
                if goal_pose is not None and self.nav2_client.server_is_ready():
                    self._write_status(
                        "running",
                        "Nav2 rejected the FollowPath goal. Falling back to NavigateToPose.",
                    )
                    self._send_nav2_goal(goal_pose)
                    return
            rejection_reason = f"Nav2 rejected the {self.active_action_name or 'navigation'} goal."
            if self._run_debug_cmd_vel_fallback(rejection_reason):
                return
            self._finish("failed", rejection_reason)
            return
        self.current_goal_handle = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._goal_result_callback)
        self._publish_look_straight_pose()
        self._write_status(
            "running",
            f"Nav2 accepted the {self.active_action_name or 'navigation'} goal and is executing.",
        )

    def _goal_result_callback(self, future) -> None:
        if self.active_request_id is None:
            return
        try:
            result_wrapper = future.result()
        except Exception as exc:
            self._finish("failed", f"{self.active_action_name or 'Navigation'} result error: {exc}")
            return
        status = result_wrapper.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            reached, message = self._verify_goal_reached_after_success()
            if reached:
                self._publish_look_up_pose()
                self._finish("completed", message)
            elif self.false_success_retries < NAVIGATION_FALSE_SUCCESS_MAX_RETRIES:
                goal_pose = self._extract_goal_pose(self.active_request or {})
                if goal_pose is not None and self.nav2_client.server_is_ready():
                    self.false_success_retries += 1
                    self.current_goal_handle = None
                    self._write_status(
                        "running",
                        f"{message} Retrying from the current pose "
                        f"({self.false_success_retries}/{NAVIGATION_FALSE_SUCCESS_MAX_RETRIES}).",
                    )
                    self._send_nav2_goal(goal_pose)
                else:
                    self._finish("failed", message)
            else:
                self._finish("failed", message)
        elif status == GoalStatus.STATUS_CANCELED:
            self._finish("failed", f"Nav2 {self.active_action_name or 'navigation'} was cancelled.")
        elif status == GoalStatus.STATUS_ABORTED and self.abort_retries < NAVIGATION_ABORT_MAX_RETRIES:
            goal_pose = self._extract_goal_pose(self.active_request or {})
            if goal_pose is not None and self.nav2_client.server_is_ready():
                self.abort_retries += 1
                self.current_goal_handle = None
                self._write_status(
                    "running",
                    f"Nav2 {self.active_action_name or 'navigation'} aborted before reaching the goal; "
                    f"retrying from the current pose ({self.abort_retries}/{NAVIGATION_ABORT_MAX_RETRIES}).",
                )
                self._send_nav2_goal(goal_pose)
            else:
                self._finish("failed", f"Nav2 {self.active_action_name or 'navigation'} aborted (status={status}).")
        else:
            self._finish("failed", f"Nav2 {self.active_action_name or 'navigation'} aborted (status={status}).")

    def _verify_goal_reached_after_success(self) -> tuple[bool, str]:
        goal_pose = self._extract_goal_pose(self.active_request or {})
        if goal_pose is None:
            return False, "Nav2 reported success, but the executor could not verify the requested goal pose."

        try:
            transform = self.tf_buffer.lookup_transform(
                GOAL_FRAME,
                "spot/body",
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
        except Exception as exc:
            return False, f"Nav2 reported success, but the final robot pose could not be verified: {exc}"

        robot_x = float(transform.transform.translation.x)
        robot_y = float(transform.transform.translation.y)
        distance = math.hypot(float(goal_pose["x"]) - robot_x, float(goal_pose["y"]) - robot_y)
        if distance > NAVIGATION_COMPLETION_TOLERANCE_M:
            return (
                False,
                f"Nav2 reported success, but Spot is still {distance:.2f} m from the goal "
                f"({goal_pose['x']:.2f}, {goal_pose['y']:.2f}).",
            )

        return (
            True,
            f"Nav2 {self.active_action_name or 'navigation'} completed successfully "
            f"({distance:.2f} m from goal).",
        )

    def _finish(self, state: str, message: str) -> None:
        self._publish_look_up_pose()
        self._write_status(state, message)
        self.last_completed_request_id = self.active_request_id
        self.active_request = None
        self.active_request_id = None
        self.active_action_name = None
        self.current_goal_handle = None
        self.goal_request_pending = False
        self.false_success_retries = 0
        self.abort_retries = 0
        self.prep_queue = []
        self.prep_future = None
        self.prep_service_name = None
        self.prep_action_label = None


def main() -> None:
    rclpy.init()
    node = Nav2NavigationExecutor()
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
