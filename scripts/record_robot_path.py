#!/usr/bin/env python3
"""Record the robot's path as a stream of map-frame poses.

Subscribes to /spot/odometry and (optionally) looks up the map->odom TF to
store poses in the map frame.  Appends one line per sample as JSON to
/shared/robot_path.jsonl so the file grows monotonically and can be replayed
or plotted later.

Intended to run inside the spot_ros2 container alongside the other
navigation/SLAM helpers.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


DEFAULT_OUTPUT_PATH = Path(os.getenv("ROBOT_PATH_OUTPUT", "/shared/robot_path.jsonl"))
DEFAULT_SAMPLE_PERIOD_SEC = float(os.getenv("ROBOT_PATH_SAMPLE_PERIOD_SEC", "1.0"))
RECORDING_SIGNAL_PATH = Path(os.getenv("ROBOT_PATH_RECORDING_SIGNAL", "/shared/robot_path_recording.signal"))
DISPLAY_TF_OVERRIDE_PATH = Path(os.getenv("ROBOT_PATH_DISPLAY_TF_OVERRIDE", "/shared/slam_tab/display_tf_override.json"))
ODOM_TOPIC = os.getenv("ROBOT_PATH_ODOM_TOPIC", "/spot/odometry")
MAP_FRAME = os.getenv("ROBOT_PATH_MAP_FRAME", "map")
ODOM_FRAME = os.getenv("ROBOT_PATH_ODOM_FRAME", "spot/vision")
ROBOT_FRAME = os.getenv("ROBOT_PATH_ROBOT_FRAME", "spot/body")


def _transform_point_2d(tf: TransformStamped, x: float, y: float) -> tuple[float, float]:
    """Apply a 2-D transform to a point."""
    q = tf.transform.rotation
    # 2-D rotation matrix
    r00 = q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z
    r01 = 2.0 * (q.x * q.y - q.z * q.w)
    r10 = 2.0 * (q.x * q.y + q.z * q.w)
    r11 = q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z
    tx = tf.transform.translation.x
    ty = tf.transform.translation.y
    return r00 * x + r01 * y + tx, r10 * x + r11 * y + ty


def _quat_to_yaw(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _load_display_tf_override(path: Path) -> TransformStamped | None:
    """Read the SLAM tab's display_tf_override so the recorder matches the map display."""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        from geometry_msgs.msg import TransformStamped as _TS, Transform as _T
        from geometry_msgs.msg import Vector3 as _V3, Quaternion as _Q

        tx = float(data.get("translation", {}).get("x", 0.0))
        ty = float(data.get("translation", {}).get("y", 0.0))
        tz = float(data.get("translation", {}).get("z", 0.0))
        rx = float(data.get("rotation", {}).get("x", 0.0))
        ry = float(data.get("rotation", {}).get("y", 0.0))
        rz = float(data.get("rotation", {}).get("z", 0.0))
        rw = float(data.get("rotation", {}).get("w", 1.0))
        tf = _TS()
        tf.transform = _T(
            translation=_V3(x=tx, y=ty, z=tz),
            rotation=_Q(x=rx, y=ry, z=rz, w=rw),
        )
        return tf
    except Exception:
        return None


class PathRecorder(Node):
    def __init__(self) -> None:
        super().__init__("robot_path_recorder")

        self.output_path = Path(os.getenv("ROBOT_PATH_OUTPUT", str(DEFAULT_OUTPUT_PATH)))
        self.sample_period_sec = max(0.1, float(os.getenv("ROBOT_PATH_SAMPLE_PERIOD_SEC", str(DEFAULT_SAMPLE_PERIOD_SEC))))
        self.recording_signal_path = Path(os.getenv("ROBOT_PATH_RECORDING_SIGNAL", str(RECORDING_SIGNAL_PATH)))
        self.odom_topic = str(os.getenv("ROBOT_PATH_ODOM_TOPIC", ODOM_TOPIC))
        self.map_frame = str(os.getenv("ROBOT_PATH_MAP_FRAME", MAP_FRAME))
        self.odom_frame = str(os.getenv("ROBOT_PATH_ODOM_FRAME", ODOM_FRAME))

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_odom: Odometry | None = None
        self._odom_sub = self.create_subscription(Odometry, self.odom_topic, self._odom_cb, 10)

        self._timer = self.create_timer(self.sample_period_sec, self._sample)

        self._samples_written = 0
        self._last_position: tuple[float, float] | None = None
        self._min_movement_m = float(os.getenv("ROBOT_PATH_MIN_MOVEMENT_M", "0.0"))
        self._was_recording = False

        self.get_logger().info(
            f"Recording robot path to {self.output_path} "
            f"every {self.sample_period_sec:.1f}s (map_frame={self.map_frame})"
        )

    def _odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg

    def _sample(self) -> None:
        is_recording = self.recording_signal_path.exists()
        if is_recording and not self._was_recording:
            self.get_logger().info("Path recording resumed (signal file detected).")
        if not is_recording and self._was_recording:
            self.get_logger().info("Path recording paused (signal file removed).")
        self._was_recording = is_recording
        if not is_recording:
            return
        if self.latest_odom is None:
            return

        odom_pose = self.latest_odom.pose.pose
        ox = odom_pose.position.x
        oy = odom_pose.position.y
        oz = odom_pose.position.z
        oyaw = _quat_to_yaw(odom_pose.orientation)
        odom_stamp = self.latest_odom.header.stamp.sec + self.latest_odom.header.stamp.nanosec * 1e-9

        # Try to get map-frame coordinates via TF or display override
        mx, my, myaw = ox, oy, oyaw
        in_map_frame = False
        tf_map_to_odom = None
        try:
            # Prefer the SLAM tab's display_tf_override so the recorded path
            # aligns with what the user sees on the map.
            tf_map_to_odom = _load_display_tf_override(DISPLAY_TF_OVERRIDE_PATH)
            if tf_map_to_odom is not None:
                mx, my = _transform_point_2d(tf_map_to_odom, ox, oy)
                tf_yaw = _quat_to_yaw(tf_map_to_odom.transform.rotation)
                myaw = oyaw + tf_yaw
                in_map_frame = True
            else:
                tf_map_to_odom = self.tf_buffer.lookup_transform(
                    self.map_frame, self.odom_frame, rclpy.time.Time()
                )
                mx, my = _transform_point_2d(tf_map_to_odom, ox, oy)
                tf_yaw = _quat_to_yaw(tf_map_to_odom.transform.rotation)
                myaw = oyaw + tf_yaw
                in_map_frame = True
        except Exception:
            pass  # fall back to odom-frame coordinates

        # Deduplicate: skip if robot hasn't moved meaningfully
        if self._last_position is not None:
            dx = mx - self._last_position[0]
            dy = my - self._last_position[1]
            if (dx * dx + dy * dy) ** 0.5 < self._min_movement_m:
                return

        self._last_position = (mx, my)

        record = {
            "t": odom_stamp,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(odom_stamp)),
            "map": in_map_frame,
            "x": round(mx, 4),
            "y": round(my, 4),
            "z": round(oz, 4),
            "yaw": round(myaw, 4),
            "odom": {"x": round(ox, 4), "y": round(oy, 4), "yaw": round(oyaw, 4)},
        }

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._samples_written += 1
        if self._samples_written % 60 == 0:
            self.get_logger().info(
                f"Path recorder: {self._samples_written} samples written "
                f"(latest {mx:.2f}, {my:.2f})"
            )


def main() -> None:
    rclpy.init()
    node = PathRecorder()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.get_logger().info(f"Path recorder shutting down. Total samples: {node._samples_written}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
