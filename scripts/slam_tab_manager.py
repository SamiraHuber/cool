#!/usr/bin/env python3
"""SLAM tab manager - runs inside the spot_ros2 container.

Subscribes to /map, /global_costmap/costmap, /local_costmap/costmap,
/spot/odometry and writes JSON snapshots to /shared/slam_tab/.
Reads /shared/slam_tab/request.json for commands and writes
/shared/slam_tab/status.json.
"""

import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
try:
    from rclpy.executors import ExternalShutdownException
except Exception:
    class ExternalShutdownException(Exception):
        pass
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry

try:
    from spot_msgs.msg import BatteryStateArray
except ImportError:
    BatteryStateArray = None

try:
    from tf2_msgs.msg import TFMessage
except ImportError:
    TFMessage = None


SHARED_DIR = Path("/shared/slam_tab")
REQUEST_PATH = SHARED_DIR / "request.json"
STATUS_PATH = SHARED_DIR / "status.json"
MAP_PATH = SHARED_DIR / "map.json"
COSTMAP_PATH = SHARED_DIR / "costmap.json"
LOCAL_COSTMAP_PATH = SHARED_DIR / "local_costmap.json"
ROBOT_POSE_PATH = SHARED_DIR / "robot_pose.json"
SAVE_DIR = SHARED_DIR / "saved"
POLL_PERIOD_SEC = 1.0
ROBOT_ODOM_STALE_SEC = float(os.getenv("ROBOT_ODOM_STALE_SEC", "3.0"))
SET_POSE_TIMEOUT_SEC = float(os.getenv("SET_POSE_TIMEOUT_SEC", "12.0"))
SET_POSE_REPUBLISH_PERIOD_SEC = float(os.getenv("SET_POSE_REPUBLISH_PERIOD_SEC", "0.5"))
SET_POSE_POSITION_TOLERANCE_M = float(os.getenv("SET_POSE_POSITION_TOLERANCE_M", "1.0"))
SET_POSE_YAW_TOLERANCE_RAD = float(os.getenv("SET_POSE_YAW_TOLERANCE_RAD", "0.70"))
SLAM_TOOLBOX_WRAPPER_PIDFILE = Path(
    os.getenv("SLAM_TOOLBOX_WRAPPER_PIDFILE", "/tmp/slam_toolbox_wrapper.pid")
)
SLAM_TOOLBOX_MODE_FILE = Path(os.getenv("SLAM_TOOLBOX_MODE_FILE", "/tmp/slam_toolbox_mode.txt"))
ACTIVE_OVERRIDE_PATH = Path("/shared/maps/toolbox_saved/active.json")
DISPLAY_TF_OVERRIDE_PATH = Path("/shared/slam_tab/display_tf_override.json")


def _write_display_tf_override(tf) -> None:
    """Serialize the display_tf_override so the path recorder can use the same transform."""
    if tf is None:
        DISPLAY_TF_OVERRIDE_PATH.unlink(missing_ok=True)
        return
    try:
        payload = {
            "translation": {
                "x": float(tf.translation.x),
                "y": float(tf.translation.y),
                "z": float(getattr(tf.translation, "z", 0.0)),
            },
            "rotation": {
                "x": float(tf.rotation.x),
                "y": float(tf.rotation.y),
                "z": float(tf.rotation.z),
                "w": float(tf.rotation.w),
            },
        }
        atomic_write_json(DISPLAY_TF_OVERRIDE_PATH, payload)
    except Exception:
        pass


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def load_json_file(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def ros_time_to_sec(stamp) -> float:
    if stamp is None:
        return 0.0
    try:
        sec = float(getattr(stamp, "sec", 0.0) or 0.0)
        nanosec = float(getattr(stamp, "nanosec", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return sec + (nanosec / 1_000_000_000.0)


def _read_wrapper_pid() -> int | None:
    try:
        return int(SLAM_TOOLBOX_WRAPPER_PIDFILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _find_wrapper_pid() -> int | None:
    """Find the run_slam_toolbox.sh process by scanning /proc."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_slam_toolbox.sh"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                try:
                    return int(line.strip())
                except ValueError:
                    continue
    except Exception:
        pass
    return None


def restart_slam_toolbox() -> None:
    pid = _read_wrapper_pid()
    if pid is None:
        pid = _find_wrapper_pid()
    if pid is None:
        raise RuntimeError(
            f"slam_toolbox wrapper is not running (pid file: {SLAM_TOOLBOX_WRAPPER_PIDFILE})"
        )
    os.kill(pid, signal.SIGUSR1)
    time.sleep(1.0)


def set_slam_toolbox_mode(mode: str) -> None:
    normalized = str(mode or "").strip().lower()
    if normalized not in {"mapping", "localization"}:
        raise RuntimeError(f"Unsupported slam_toolbox mode '{mode}'")
    SLAM_TOOLBOX_MODE_FILE.write_text(normalized, encoding="utf-8")


def get_slam_toolbox_mode() -> str:
    try:
        return SLAM_TOOLBOX_MODE_FILE.read_text(encoding="utf-8").strip().lower() or "mapping"
    except Exception:
        return "mapping"


def run_service_call(service_name: str, service_type: str, request: str, timeout_sec: float = 90.0) -> str:
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /ros_ws/install/setup.bash && "
        f"ros2 service call {service_name} {service_type} '{request}'"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"{service_name} failed with exit code {result.returncode}")
    return output


def wait_for_service(service_name: str, timeout_sec: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = subprocess.run(
            [
                "bash",
                "-lc",
                "source /opt/ros/humble/setup.bash && "
                "source /ros_ws/install/setup.bash && "
                f"ros2 service list | grep -Fx {json.dumps(service_name)}",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for service '{service_name}'")


def _quaternion_to_yaw(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _angle_error(a, b):
    return abs(_normalize_angle(a - b))


def _transform_point_2d(transform, x, y):
    """Transform a point from child frame to parent frame.

    transform is the parent->child transform (e.g. map->odom).
    Returns (x, y) in parent frame.
    """
    q = transform.rotation
    # 2D rotation matrix that maps child-frame coordinates into parent frame.
    r00 = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
    r01 = 2.0 * (q.x * q.y - q.z * q.w)
    r10 = 2.0 * (q.x * q.y + q.z * q.w)
    r11 = q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z

    tx = transform.translation.x
    ty = transform.translation.y

    return (r00 * x + r01 * y + tx), (r10 * x + r11 * y + ty)


def _inverse_transform_point_2d(transform, x, y):
    """Transform a point from parent frame to child frame.

    transform is the parent->child transform (e.g. map->odom).
    Returns (x, y) in child frame.
    """
    q = transform.rotation
    r00 = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
    r01 = 2.0 * (q.x * q.y - q.z * q.w)
    r10 = 2.0 * (q.x * q.y + q.z * q.w)
    r11 = q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z

    dx = x - transform.translation.x
    dy = y - transform.translation.y

    return (r00 * dx + r10 * dy), (r01 * dx + r11 * dy)


def _transform_yaw(transform, yaw):
    """Transform yaw from child frame to parent frame."""
    q = transform.rotation
    tf_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return yaw + tf_yaw


def _inverse_transform_yaw(transform, yaw):
    """Transform yaw from parent frame to child frame."""
    q = transform.rotation
    tf_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return yaw - tf_yaw


def _transform_pose_to_map(pose, tf_map_to_odom):
    """Return a new pose dict with position/orientation in map frame."""
    x, y = _transform_point_2d(
        tf_map_to_odom, pose.position.x, pose.position.y
    )
    yaw = _quaternion_to_yaw(
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
    )
    new_yaw = _transform_yaw(tf_map_to_odom, yaw)
    half_yaw = new_yaw / 2.0
    return {
        "position": {"x": x, "y": y, "z": pose.position.z},
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(half_yaw),
            "w": math.cos(half_yaw),
        },
    }


def _make_map_to_odom_transform(odom_pose, map_x, map_y, map_yaw):
    """Build a map->odom transform that places odom_pose at the requested map pose."""
    odom_yaw = _quaternion_to_yaw(
        odom_pose.orientation.x,
        odom_pose.orientation.y,
        odom_pose.orientation.z,
        odom_pose.orientation.w,
    )
    # We need a transform T_map_odom such that:
    #   map_pose = T_map_odom * odom_pose
    # therefore the transform yaw is the delta from odom into map.
    tf_yaw = map_yaw - odom_yaw
    cos_yaw = math.cos(tf_yaw)
    sin_yaw = math.sin(tf_yaw)

    # T_map_odom maps odom-frame coordinates into the map frame.
    tx = map_x - (cos_yaw * odom_pose.position.x - sin_yaw * odom_pose.position.y)
    ty = map_y - (sin_yaw * odom_pose.position.x + cos_yaw * odom_pose.position.y)
    return SimpleNamespace(
        translation=SimpleNamespace(x=tx, y=ty, z=0.0),
        rotation=SimpleNamespace(
            x=0.0,
            y=0.0,
            z=math.sin(tf_yaw / 2.0),
            w=math.cos(tf_yaw / 2.0),
        ),
    )


def occupancy_grid_to_json(msg: OccupancyGrid, tf_map_to_odom=None) -> dict:
    origin = msg.info.origin
    if tf_map_to_odom is not None:
        # Local costmap origin is in the child frame (spot/vision).
        # Transform it to the parent frame (map) so it aligns with the
        # robot pose and global map on the SLAM tab canvas.
        ox, oy = _transform_point_2d(
            tf_map_to_odom, origin.position.x, origin.position.y
        )
        yaw = _quaternion_to_yaw(
            origin.orientation.x, origin.orientation.y, origin.orientation.z, origin.orientation.w
        )
        new_yaw = _transform_yaw(tf_map_to_odom, yaw)
        half_yaw = new_yaw / 2.0
        origin_json = {
            "position": {"x": ox, "y": oy, "z": origin.position.z},
            "orientation": {
                "x": 0.0,
                "y": 0.0,
                "z": math.sin(half_yaw),
                "w": math.cos(half_yaw),
            },
        }
    else:
        origin_json = {
            "position": {
                "x": origin.position.x,
                "y": origin.position.y,
                "z": origin.position.z,
            },
            "orientation": {
                "x": origin.orientation.x,
                "y": origin.orientation.y,
                "z": origin.orientation.z,
                "w": origin.orientation.w,
            },
        }

    return {
        "header": {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
        },
        "info": {
            "map_load_time": {
                "sec": msg.info.map_load_time.sec,
                "nanosec": msg.info.map_load_time.nanosec,
            },
            "resolution": msg.info.resolution,
            "width": msg.info.width,
            "height": msg.info.height,
            "origin": origin_json,
        },
        "data": list(msg.data),
    }


def pose_to_json(pose) -> dict:
    return {
        "position": {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
        },
        "orientation": {
            "x": pose.orientation.x,
            "y": pose.orientation.y,
            "z": pose.orientation.z,
            "w": pose.orientation.w,
        },
    }


def _save_occupancy_grid(map_msg: OccupancyGrid, base: Path) -> None:
    """Write OccupancyGrid to PGM/YAML files matching nav2_map_server output."""
    width = int(map_msg.info.width)
    height = int(map_msg.info.height)
    resolution = float(map_msg.info.resolution)
    origin = map_msg.info.origin

    yaw = _quaternion_to_yaw(
        origin.orientation.x,
        origin.orientation.y,
        origin.orientation.z,
        origin.orientation.w,
    )

    pgm_path = base.with_suffix(".pgm")
    yaml_path = base.with_suffix(".yaml")

    # Build PGM pixel data (row-major, ROS standard: row 0 is bottom)
    pixels = bytearray(width * height)
    data = map_msg.data
    idx = 0
    for y in range(height):
        row_start = y * width
        for x in range(width):
            val = int(data[row_start + x])
            if val == -1 or val == 255:
                pixels[idx] = 205
            elif val == 0:
                pixels[idx] = 254
            else:
                pixels[idx] = 0
            idx += 1

    pgm_path.write_bytes(
        f"P5\n{width} {height}\n255\n".encode("ascii") + bytes(pixels)
    )

    yaml_content = (
        f"image: {pgm_path.name}\n"
        f"mode: trinary\n"
        f"resolution: {resolution}\n"
        f"origin: [{origin.position.x}, {origin.position.y}, {yaw}]\n"
        f"negate: 0\n"
        f"occupied_thresh: 0.65\n"
        f"free_thresh: 0.25\n"
        f"slam_tab_manager_version: 2\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")


class SlamTabManager(Node):
    def __init__(self):
        super().__init__("slam_tab_manager")
        self.map_msg = None
        self.map_msg_received_at = 0.0
        self.local_costmap_msg = None
        self.robot_pose = None
        self.robot_pose_received_at = 0.0
        self.robot_pose_updated_at = 0.0
        self.robot_pose_message_stamp = 0.0
        self.latest_battery = None
        self.latest_battery_received_at = 0.0
        self.tf_map_to_odom = None
        self.frozen_tf = None
        self.map_sub = self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)
        self.local_costmap_sub = self.create_subscription(
            OccupancyGrid, "/local_costmap/costmap", self._local_costmap_callback, 10
        )
        self.odom_sub = self.create_subscription(Odometry, "/spot/odometry", self._odom_callback, 10)
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/initialpose", 10)

        if BatteryStateArray is not None:
            self.create_subscription(BatteryStateArray, "/spot/status/battery_states", self._battery_callback, 10)
        else:
            self.get_logger().warn("spot_msgs not available; battery monitoring disabled.")

        if TFMessage is not None:
            self.tf_sub = self.create_subscription(TFMessage, "/tf", self._tf_callback, 10)
        else:
            self.get_logger().warn("tf2_msgs not available; TF-based alignment disabled.")

        self.export_timer = self.create_timer(2.0, self._export_callback)
        self.poll_timer = self.create_timer(POLL_PERIOD_SEC, self._poll_requests)

        self.last_request_id = None
        self.state = "idle"
        self.message = "SLAM manager ready."
        self._lock = threading.Lock()
        self._worker = None
        self.frozen_tf = None
        self.display_tf_override = None
        _write_display_tf_override(None)
        self.pending_pose_target = None
        self.set_pose_generation = 0
        self._pose_match_consecutive_count = 0
        self._pose_match_consecutive_count = 0

        self._write_status()
        self.get_logger().info("SLAM tab manager started.")

    def _map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.map_msg_received_at = time.monotonic()

    def _local_costmap_callback(self, msg: OccupancyGrid):
        self.local_costmap_msg = msg

    def _odom_callback(self, msg: Odometry):
        self.robot_pose = msg.pose.pose
        self.robot_pose_received_at = time.monotonic()
        self.robot_pose_updated_at = time.time()
        self.robot_pose_message_stamp = ros_time_to_sec(getattr(msg, "header", None).stamp if getattr(msg, "header", None) else None)

    def _battery_callback(self, msg):
        with open("/tmp/slam_battery.log", "a") as f: f.write("battery callback fired\n")
        if not msg.battery_states:
            return
        battery = msg.battery_states[0]
        self.latest_battery = {
            "charge_percentage": float(battery.charge_percentage),
            "status": int(battery.status),
        }
        self.latest_battery_received_at = time.monotonic()

    def _tf_callback(self, msg):
        for tf in msg.transforms:
            if tf.header.frame_id == "map" and tf.child_frame_id == "spot/vision":
                self.tf_map_to_odom = tf.transform
                self._maybe_finalize_pose_override()

    def _export_callback(self):
        # Only update the map snapshot while recording so the display
        # does not build up new data when the user has stopped recording.
        if self.state == "recording" and self.map_msg is not None:
            atomic_write_json(MAP_PATH, occupancy_grid_to_json(self.map_msg))

        self._maybe_finalize_pose_override()

        # Use the frozen TF when idle (map stopped) so the robot position stays
        # consistent with the frozen map. While recording or actively localizing
        # on a loaded map we use the live TF so pose updates are reflected.
        tf = self._display_tf()

        if self.local_costmap_msg is not None:
            atomic_write_json(
                LOCAL_COSTMAP_PATH,
                occupancy_grid_to_json(self.local_costmap_msg, tf),
            )

        pose_json = self._current_robot_pose_json()
        if pose_json is not None:
            atomic_write_json(ROBOT_POSE_PATH, pose_json)

        self._refresh_robot_connection_status()

    def _robot_pose_is_fresh(self) -> bool:
        if self.robot_pose is None or self.robot_pose_received_at <= 0.0:
            return False
        receive_age_sec = max(0.0, time.monotonic() - self.robot_pose_received_at)
        return receive_age_sec <= ROBOT_ODOM_STALE_SEC

    def _current_robot_pose_json(self) -> dict | None:
        tf = self._display_tf()
        if self.robot_pose is not None:
            return _transform_pose_to_map(self.robot_pose, tf) if tf is not None else pose_to_json(self.robot_pose)
        payload = load_json_file(ROBOT_POSE_PATH)
        return payload if isinstance(payload, dict) else None

    def _refresh_robot_connection_status(self):
        payload = load_json_file(STATUS_PATH)
        if not isinstance(payload, dict):
            self._write_status()
            return

        payload["robot_connection"] = self._robot_connection_payload()
        payload["battery"] = self._battery_payload()
        atomic_write_json(STATUS_PATH, payload)

    def _wait_for_new_map_message(self, previous_received_at: float, timeout_sec: float = 12.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline:
            if self.map_msg is not None and self.map_msg_received_at > previous_received_at:
                return True
            time.sleep(0.1)
        return self.map_msg is not None and self.map_msg_received_at > previous_received_at

    def _display_tf(self):
        if self.display_tf_override is not None:
            return self.display_tf_override
        return (
            self.tf_map_to_odom
            if self.state in ("recording", "loaded")
            else (self.frozen_tf or self.tf_map_to_odom)
        )

    def _live_robot_pose_in_map(self) -> dict | None:
        if self.robot_pose is None or self.tf_map_to_odom is None:
            return None
        try:
            return _transform_pose_to_map(self.robot_pose, self.tf_map_to_odom)
        except Exception:
            return None

    def _pose_matches_target(self, pose_json: dict | None, target_x: float, target_y: float, target_theta: float) -> bool:
        if pose_json is None:
            return False

        position = pose_json.get("position") or {}
        orientation = pose_json.get("orientation") or {}
        pose_x = position.get("x")
        pose_y = position.get("y")
        if pose_x is None or pose_y is None:
            return False

        yaw = _quaternion_to_yaw(
            orientation.get("x", 0.0),
            orientation.get("y", 0.0),
            orientation.get("z", 0.0),
            orientation.get("w", 1.0),
        )
        distance = math.hypot(float(pose_x) - target_x, float(pose_y) - target_y)
        yaw_error = _angle_error(yaw, target_theta)
        return distance <= SET_POSE_POSITION_TOLERANCE_M and yaw_error <= SET_POSE_YAW_TOLERANCE_RAD

    def _maybe_finalize_pose_override(self):
        # Auto-clear is disabled because slam_toolbox with slow Velodyne never
        # converges reliably — it oscillates forever. The override stays active
        # until the user explicitly resets, loads a map, or does a new set-pose.
        pass

    def _wait_for_pose_update(
        self,
        target_x: float,
        target_y: float,
        target_theta: float,
        initial_pose_msg=None,
        set_pose_generation: int | None = None,
    ) -> dict | None:
        """Republish /initialpose for SET_POSE_TIMEOUT_SEC but do NOT auto-clear
        the display_tf_override when "converged" — the override stays active
        until the user explicitly resets, loads, or re-sets pose.
        """
        deadline = time.monotonic() + SET_POSE_TIMEOUT_SEC
        next_publish_at = time.monotonic() + SET_POSE_REPUBLISH_PERIOD_SEC
        while time.monotonic() < deadline:
            if set_pose_generation is not None and set_pose_generation != self.set_pose_generation:
                return None
            now = time.monotonic()
            if initial_pose_msg is not None and now >= next_publish_at:
                initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
                self.initial_pose_pub.publish(initial_pose_msg)
                next_publish_at = now + SET_POSE_REPUBLISH_PERIOD_SEC
            time.sleep(0.1)
        return None

    def _make_initial_pose_msg(self, x: float, y: float, theta: float) -> PoseWithCovarianceStamped:
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.pose.orientation.w = math.cos(theta / 2.0)
        # Small covariance indicates high confidence.
        msg.pose.covariance[0] = 0.1
        msg.pose.covariance[7] = 0.1
        msg.pose.covariance[35] = 0.05
        return msg

    def _battery_payload(self):
        with open("/tmp/slam_battery_payload.log", "a") as f: f.write(f"payload: latest_battery={self.latest_battery}\n")
        if self.latest_battery is None:
            return None
        receive_age_sec = max(0.0, time.monotonic() - self.latest_battery_received_at)
        payload = dict(self.latest_battery)
        payload["age_sec"] = round(receive_age_sec, 1)
        payload["stale"] = receive_age_sec > ROBOT_ODOM_STALE_SEC
        return payload

    def _write_status(self, **extra):
        payload = {
            "state": self.state,
            "message": self.message,
            "updated_at": time.time(),
            "saved_maps": self._list_saved_maps(),
            "robot_connection": self._robot_connection_payload(),
            "battery": self._battery_payload(),
        }
        payload.update(extra)
        atomic_write_json(STATUS_PATH, payload)

    def _robot_connection_payload(self):
        if self.robot_pose is None or self.robot_pose_received_at <= 0:
            return {
                "connected": False,
                "message": "No /spot/odometry received yet.",
                "last_seen_at": None,
                "age_sec": None,
                "topic": "/spot/odometry",
            }

        receive_age_sec = max(0.0, time.monotonic() - self.robot_pose_received_at)
        message_age_sec = None
        if self.robot_pose_message_stamp > 0.0:
            message_age_sec = max(0.0, time.time() - self.robot_pose_message_stamp)

        connected = receive_age_sec <= ROBOT_ODOM_STALE_SEC
        age_sec = receive_age_sec
        last_seen_at = self.robot_pose_updated_at or None
        return {
            "connected": connected,
            "message": "Robot connected." if connected else "Robot connection stale.",
            "last_seen_at": last_seen_at,
            "age_sec": age_sec,
            "topic": "/spot/odometry",
            "message_stamp": self.robot_pose_message_stamp or None,
            "message_age_sec": message_age_sec,
        }

    def _list_saved_maps(self):
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        maps = []
        for path in sorted(SAVE_DIR.glob("*.posegraph")):
            name = path.stem
            maps.append({"name": name, "updated_at": path.stat().st_mtime})
        return maps

    def _poll_requests(self):
        request = load_json_file(REQUEST_PATH)
        if not isinstance(request, dict):
            return

        request_id = str(request.get("request_id") or "")
        if request_id and request_id == self.last_request_id:
            return

        self.last_request_id = request_id
        action = str(request.get("action") or "").strip().lower()

        try:
            if action == "start":
                self._do_start()
            elif action == "save":
                self._run_in_worker(self._do_save, request.get("name", "map"))
            elif action == "load":
                self._run_in_worker(
                    self._do_load,
                    request.get("name"),
                    request.get("x"),
                    request.get("y"),
                    request.get("theta"),
                )
            elif action == "set_pose":
                self._run_in_worker(
                    self._do_set_pose,
                    request.get("x"), request.get("y"), request.get("theta")
                )
            elif action == "mark_start":
                self._do_mark_start(request.get("name"))
            elif action == "stop":
                self._do_stop()
            else:
                raise RuntimeError(f"Unknown action: {action}")
        except Exception as exc:
            self.state = "error"
            self.message = str(exc)
            self._write_status(error=str(exc))
            self.get_logger().error(f"Action {action} failed: {exc}")
        finally:
            try:
                REQUEST_PATH.unlink(missing_ok=True)
            except Exception:
                pass

    def _run_in_worker(self, target, *args):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("Another operation is already in progress. Please wait.")
            self._worker = threading.Thread(target=target, args=args, daemon=True)
            self._worker.start()

    def _do_start(self):
        self.get_logger().info("Starting fresh SLAM recording...")
        self.set_pose_generation += 1
        self.state = "starting"
        self.message = "Clearing old map and restarting slam_toolbox..."
        self._write_status()

        # Wipe old snapshot files so the frontend never shows stale data
        for p in (MAP_PATH, LOCAL_COSTMAP_PATH, ROBOT_POSE_PATH):
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        self.map_msg = None
        self.local_costmap_msg = None
        self.robot_pose = None
        self.robot_pose_received_at = 0.0
        self.robot_pose_updated_at = 0.0
        self.robot_pose_message_stamp = 0.0
        self.tf_map_to_odom = None
        self.frozen_tf = None
        self.display_tf_override = None
        _write_display_tf_override(None)
        self.pending_pose_target = None
        self._pose_match_consecutive_count = 0

        # Tell publish_toolbox_map_snapshot.py to stop republishing old data
        # so we only see the fresh map from the restarted slam_toolbox node.
        try:
            ACTIVE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(ACTIVE_OVERRIDE_PATH, {"mode": "recording", "updated_at": time.time()})
        except Exception:
            pass

        set_slam_toolbox_mode("mapping")
        restart_slam_toolbox()
        # Give the new node a moment to initialise before we accept map data
        time.sleep(3.0)

        self.state = "recording"
        self.message = "Recording new map. Drive the robot to explore."
        self._write_status()
        self.get_logger().info("Recording started.")

    def _do_stop(self):
        self.get_logger().info("Stopping SLAM recording...")
        self.state = "stopping"
        self.message = "Stopping slam_toolbox..."
        self._write_status()

        # Remove the recording flag so snapshot publishing resumes normally
        try:
            ACTIVE_OVERRIDE_PATH.unlink(missing_ok=True)
        except Exception:
            pass

        self.frozen_tf = self.tf_map_to_odom
        self.display_tf_override = None
        _write_display_tf_override(None)
        self.pending_pose_target = None
        self._pose_match_consecutive_count = 0
        set_slam_toolbox_mode("mapping")
        restart_slam_toolbox()
        self.state = "idle"
        self.message = "Recording stopped."
        self._write_status()
        self.get_logger().info("Stopped.")

    def _do_save(self, name: str):
        try:
            name = str(name).strip() or "map"
            base = SAVE_DIR / name
            SAVE_DIR.mkdir(parents=True, exist_ok=True)

            self.get_logger().info(f"Saving map as '{name}'...")
            self.state = "saving"
            self.message = f"Saving map as '{name}'..."
            self._write_status()

            # Save YAML/PGM directly from the live OccupancyGrid to avoid
            # the map_saver DDS subscription timeout that plagues Fast DDS.
            local_map = self.map_msg
            if local_map is not None:
                try:
                    _save_occupancy_grid(local_map, base)
                    self.get_logger().info(f"Saved map PGM/YAML from live OccupancyGrid to {base}")
                except Exception as exc:
                    self.get_logger().warning(
                        f"Direct map save failed ({exc}); falling back to save_map service."
                    )
                    run_service_call(
                        "/slam_toolbox/save_map",
                        "slam_toolbox/srv/SaveMap",
                        f'{{name: {{data: "{base}"}}}}',
                    )
            else:
                run_service_call(
                    "/slam_toolbox/save_map",
                    "slam_toolbox/srv/SaveMap",
                    f'{{name: {{data: "{base}"}}}}',
                )
            # Save posegraph (only possible in mapping mode). In localization
            # mode slam_toolbox cannot serialize the posegraph.
            if get_slam_toolbox_mode() != "localization":
                posegraph_path = base.with_suffix(".posegraph")
                data_path = base.with_suffix(".data")
                try:
                    run_service_call(
                        "/slam_toolbox/serialize_map",
                        "slam_toolbox/srv/SerializePoseGraph",
                        f'{{filename: "{base}"}}',
                        timeout_sec=300.0,
                    )
                except subprocess.TimeoutExpired:
                    self.get_logger().warning(
                        "serialize_map service call timed out; polling for output files..."
                    )
                    deadline = time.monotonic() + 60.0
                    while time.monotonic() < deadline:
                        if posegraph_path.exists() and data_path.exists():
                            if posegraph_path.stat().st_size > 0 and data_path.stat().st_size > 0:
                                break
                        time.sleep(1.0)
                    if not (posegraph_path.exists() and data_path.exists()):
                        raise RuntimeError(
                            "serialize_map timed out and output files were not created."
                        )
                    self.get_logger().info(
                        "Output files detected after timeout; treating save as successful."
                    )
            else:
                self.get_logger().warning(
                    "Skipping posegraph serialization because slam_toolbox is in localization mode. "
                    "Only PGM/YAML map image will be saved."
                )

            self.state = "saved"
            self.message = f"Map '{name}' saved."
            self._write_status(saved_name=name)
            self.get_logger().info(f"Map '{name}' saved.")
        except Exception as exc:
            self.state = "error"
            self.message = str(exc)
            self._write_status(error=str(exc))
            self.get_logger().error(f"Save failed: {exc}")

    def _do_load(self, name, x=None, y=None, theta=None):
        try:
            if not name:
                raise RuntimeError("Map name is required.")
            base = SAVE_DIR / str(name)
            posegraph_path = base.with_suffix(".posegraph")
            if not posegraph_path.exists():
                raise RuntimeError(f"Saved map '{name}' not found.")

            self.get_logger().info(f"Loading map '{name}'...")
            self.state = "loading"
            self.message = f"Loading map '{name}'..."
            self._write_status()

            previous_map_received_at = self.map_msg_received_at
            try:
                MAP_PATH.unlink(missing_ok=True)
            except Exception:
                pass

            # Use explicit pose, saved start position, or fallback to origin
            if x is not None or y is not None or theta is not None:
                initial_x = float(x) if x is not None else 0.0
                initial_y = float(y) if y is not None else 0.0
                initial_theta = float(theta) if theta is not None else 0.0
            else:
                start_path = SAVE_DIR / f"{name}.start.json"
                if start_path.exists():
                    try:
                        start_data = load_json_file(start_path)
                        initial_x = float(start_data.get("x", 0.0))
                        initial_y = float(start_data.get("y", 0.0))
                        initial_theta = float(start_data.get("theta", 0.0))
                        self.get_logger().info(
                            f"Using saved start position for '{name}': "
                            f"({initial_x}, {initial_y}, {initial_theta})"
                        )
                    except Exception:
                        initial_x = 0.0
                        initial_y = 0.0
                        initial_theta = 0.0
                else:
                    initial_x = 0.0
                    initial_y = 0.0
                    initial_theta = 0.0

            set_slam_toolbox_mode("localization")
            restart_slam_toolbox()
            wait_for_service("/slam_toolbox/deserialize_map", timeout_sec=100.0)
            deserialize_timed_out = False
            try:
                run_service_call(
                    "/slam_toolbox/deserialize_map",
                    "slam_toolbox/srv/DeserializePoseGraph",
                    (
                        f'{{filename: "{base}", match_type: 3, '
                        f'initial_pose: {{x: {initial_x}, y: {initial_y}, theta: {initial_theta}}}}}'
                    ),
                    timeout_sec=500.0,
                )
            except subprocess.TimeoutExpired:
                self.get_logger().warning(
                    "deserialize_map service call timed out; will wait for map publication..."
                )
                deserialize_timed_out = True

            # slam_toolbox republishes /map after deserialization, but the
            # publication can land a few seconds later. Wait for a fresh map
            # callback so we do not freeze the previous map into MAP_PATH.
            # If the service call itself timed out (response dropped), give the
            # node extra time to finish deserialization and publish the map.
            map_wait_timeout = 60.0 if deserialize_timed_out else 12.0
            if not self._wait_for_new_map_message(previous_map_received_at, timeout_sec=map_wait_timeout):
                raise RuntimeError(
                    f"Timed out waiting for slam_toolbox to publish the loaded map for '{name}'."
                )
            atomic_write_json(MAP_PATH, occupancy_grid_to_json(self.map_msg))
            self.get_logger().info("Loaded map snapshot exported.")

            self.frozen_tf = self.tf_map_to_odom
            self.display_tf_override = None
            _write_display_tf_override(None)
            self.pending_pose_target = None
            self._pose_match_consecutive_count = 0
            self.state = "loaded"
            self.message = f"Map '{name}' loaded."
            self._write_status(loaded_name=name)
            self.get_logger().info(f"Map '{name}' loaded.")

            # Keep the web frontend and snapshot publisher in sync
            try:
                ACTIVE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(
                    ACTIVE_OVERRIDE_PATH,
                    {"mode": "frozen", "name": name, "updated_at": time.time()},
                )
            except Exception:
                pass
        except Exception as exc:
            self.state = "error"
            self.message = str(exc)
            self._write_status(error=str(exc))
            self.get_logger().error(f"Load failed: {exc}")

    def _do_set_pose(self, x, y, theta):
        try:
            x = float(x)
            y = float(y)
            theta = float(theta)

            if self.initial_pose_pub.get_subscription_count() <= 0:
                raise RuntimeError(
                    "Set pose failed: no localization node is subscribed to /initialpose, "
                    "so the live map pose cannot update."
                )
            set_pose_generation = self.set_pose_generation
            self._pose_match_consecutive_count = 0

            msg = self._make_initial_pose_msg(x, y, theta)

            self.initial_pose_pub.publish(msg)
            provisional_pose_json = None
            if self.robot_pose is not None:
                self.display_tf_override = _make_map_to_odom_transform(self.robot_pose, x, y, theta)
                _write_display_tf_override(self.display_tf_override)
                self.pending_pose_target = {"x": x, "y": y, "theta": theta}
                provisional_pose_json = _transform_pose_to_map(self.robot_pose, self.display_tf_override)
                atomic_write_json(ROBOT_POSE_PATH, provisional_pose_json)
                if self.local_costmap_msg is not None:
                    atomic_write_json(
                        LOCAL_COSTMAP_PATH,
                        occupancy_grid_to_json(self.local_costmap_msg, self.display_tf_override),
                    )
                self.message = (
                    f"Set pose applied at ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°); "
                    "waiting for localization to converge."
                )
                self._write_status(
                    localization_pending=True,
                    localization_pending_since=time.time(),
                    localization_pending_timeout_sec=SET_POSE_TIMEOUT_SEC,
                )

            pose_json = self._wait_for_pose_update(
                x,
                y,
                theta,
                initial_pose_msg=msg,
                set_pose_generation=set_pose_generation,
            )
            if set_pose_generation != self.set_pose_generation:
                self.get_logger().info(
                    "Set pose confirmation cancelled because a new recording/reset started."
                )
                return
            # Override stays active indefinitely — do not auto-clear.
            # We still republished /initialpose for 12 s above so slam_toolbox
            # has a chance to converge, but with 1 Hz Velodyne it oscillates
            # forever. The UI uses display_tf_override to stay stable.
            if pose_json is None:
                if provisional_pose_json is None:
                    raise RuntimeError(
                        "Set pose failed: localization did not update the live map pose after publishing /initialpose."
                    )
                self.message = (
                    f"Set pose applied at ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°); "
                    "using provisional pose (localization did not fully converge)."
                )
                self._write_status(localization_unconfirmed=True)
                self.get_logger().info(self.message)
                return

            # Even when pose "matches" we keep the override active.
            self.message = f"Set pose applied at ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)."
            self._write_status()
            self.get_logger().info(self.message)
        except Exception as exc:
            self.get_logger().error(f"Set pose failed: {exc}")
            raise

    def _do_mark_start(self, name):
        try:
            if not name:
                raise RuntimeError("Map name is required.")

            tf = self._display_tf()
            if self.robot_pose is None:
                raise RuntimeError("Robot pose not available yet.")

            pose_in_map = (
                _transform_pose_to_map(self.robot_pose, tf)
                if tf is not None
                else pose_to_json(self.robot_pose)
            )
            x = pose_in_map["position"]["x"]
            y = pose_in_map["position"]["y"]
            yaw = _quaternion_to_yaw(
                pose_in_map["orientation"]["x"],
                pose_in_map["orientation"]["y"],
                pose_in_map["orientation"]["z"],
                pose_in_map["orientation"]["w"],
            )

            start_path = SAVE_DIR / f"{name}.start.json"
            atomic_write_json(
                start_path,
                {"x": x, "y": y, "theta": yaw, "updated_at": time.time()},
            )

            self.message = (
                f"Start position for '{name}' marked at "
                f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°)."
            )
            self._write_status()
            self.get_logger().info(
                f"Marked start for '{name}': ({x}, {y}, {yaw})"
            )
        except Exception as exc:
            self.state = "error"
            self.message = str(exc)
            self._write_status(error=str(exc))
            self.get_logger().error(f"Mark start failed: {exc}")


def main():
    rclpy.init()
    node = SlamTabManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
