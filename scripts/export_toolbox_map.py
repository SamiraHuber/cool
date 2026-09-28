import json
import math
import os
import tempfile
import time
from pathlib import Path

import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from spot_msgs.msg import BatteryStateArray


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class ToolboxMapExporter(Node):
    def __init__(self) -> None:
        super().__init__("toolbox_map_exporter")

        self.declare_parameter(
            "output_path",
            os.getenv("SLAM_TOOLBOX_MAP_PATH", "/shared/toolbox_map_snapshot.json"),
        )
        self.declare_parameter("map_topic", os.getenv("SLAM_TOOLBOX_MAP_TOPIC", "/map"))
        self.declare_parameter("odometry_topic", os.getenv("SLAM_TOOLBOX_ODOMETRY_TOPIC", "/spot/odometry"))
        self.declare_parameter("publish_period_sec", float(os.getenv("SLAM_TOOLBOX_EXPORT_PERIOD_SEC", "1.0")))
        self.declare_parameter("occupied_threshold", float(os.getenv("SLAM_TOOLBOX_OCCUPIED_THRESHOLD", "0.68")))
        self.declare_parameter("free_threshold", float(os.getenv("SLAM_TOOLBOX_FREE_THRESHOLD", "0.35")))

        self.output_path = Path(self.get_parameter("output_path").value)
        self.map_topic = str(self.get_parameter("map_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.publish_period_sec = max(0.2, float(self.get_parameter("publish_period_sec").value))
        self.occupied_threshold = max(0.0, min(1.0, float(self.get_parameter("occupied_threshold").value)))
        self.free_threshold = max(0.0, min(1.0, float(self.get_parameter("free_threshold").value)))

        self.latest_map = None
        self.latest_robot_pose = None
        self.latest_battery = None
        self.dirty = False
        self.last_written_signature = None
        # Fingerprint of the last map data written to JSON (width, height, len, first 64 cells).
        # Used to ignore re-publications of our own snapshot by publish_toolbox_map_snapshot.py,
        # which would otherwise overwrite a newer slam_toolbox map still waiting in self.latest_map.
        self._last_written_data_fingerprint: tuple | None = None

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, map_qos)
        self.create_subscription(Odometry, self.odometry_topic, self._odometry_callback, odom_qos)
        self.create_subscription(BatteryStateArray, "/spot/status/battery_states", self._battery_callback, odom_qos)
        self.create_timer(self.publish_period_sec, self._publish_snapshot_if_needed)

        self.get_logger().info(
            f"Exporting slam_toolbox map from {self.map_topic} with odometry from {self.odometry_topic} to {self.output_path}"
        )

    def _map_callback(self, msg: OccupancyGrid) -> None:
        # publish_toolbox_map_snapshot.py runs alongside slam_toolbox and re-publishes
        # our last-written JSON back onto /map (with an updated header stamp).  If we
        # accepted that message it would overwrite a newer slam_toolbox map that is
        # already in self.latest_map but not yet flushed to disk, causing the exported
        # snapshot to regress.  Detect re-publications by comparing the raw cell data
        # against the fingerprint of the map we most recently wrote.
        if self._last_written_data_fingerprint is not None:
            raw = msg.data
            n = len(raw)
            fingerprint = (int(msg.info.width), int(msg.info.height), n) + tuple(raw[: min(n, 2048)])
            if fingerprint == self._last_written_data_fingerprint:
                return  # our own previous export re-broadcast — ignore
        self.latest_map = msg
        self.dirty = True

    def _odometry_callback(self, msg: Odometry) -> None:
        orientation = msg.pose.pose.orientation
        self.latest_robot_pose = {
            "frame_id": "world" if msg.header.frame_id in {"odom", "spot/odom", "spot/vision"} else msg.header.frame_id,
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "z": float(msg.pose.pose.position.z),
            "yaw": quaternion_to_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        }
        self.dirty = True

    def _battery_callback(self, msg: BatteryStateArray) -> None:
        if not msg.battery_states:
            return
        battery = msg.battery_states[0]
        self.latest_battery = {
            "charge_percentage": float(battery.charge_percentage),
            "status": int(battery.status),
        }
        self.dirty = True

    def _publish_snapshot_if_needed(self) -> None:
        if self.latest_map is None:
            return
        if not self.dirty and self.output_path.exists():
            return

        msg = self.latest_map
        data = [int(v) for v in msg.data]
        width = int(msg.info.width)
        height = int(msg.info.height)
        if width <= 0 or height <= 0 or len(data) != width * height:
            return

        observed_cells = 0
        free_cells = 0
        occupied_cells = 0
        occupied_threshold_value = int(round(self.occupied_threshold * 100.0))
        free_threshold_value = int(round(self.free_threshold * 100.0))

        for value in data:
            if value < 0:
                continue
            observed_cells += 1
            if value >= occupied_threshold_value:
                occupied_cells += 1
            elif value <= free_threshold_value:
                free_cells += 1

        signature = (
            int(msg.header.stamp.sec),
            int(msg.header.stamp.nanosec),
            width,
            height,
            int(self.latest_robot_pose is not None),
            round(float(self.latest_robot_pose["x"]), 3) if self.latest_robot_pose else None,
            round(float(self.latest_robot_pose["y"]), 3) if self.latest_robot_pose else None,
            round(float(self.latest_battery["charge_percentage"]), 1) if self.latest_battery else None,
        )
        if signature == self.last_written_signature and self.output_path.exists():
            self.dirty = False
            return

        payload = {
            "generated_at": time.time(),
            "map": {
                "frame_id": msg.header.frame_id or "map",
                "resolution": float(msg.info.resolution),
                "width": width,
                "height": height,
                "origin": {
                    "x": float(msg.info.origin.position.x),
                    "y": float(msg.info.origin.position.y),
                },
                "data": data,
            },
            "robot": self.latest_robot_pose,
            "battery": self.latest_battery,
            "detections": [],
            "stats": {
                "observed_cells": observed_cells,
                "free_cells": free_cells,
                "occupied_cells": occupied_cells,
                "unknown_cells": int((width * height) - observed_cells),
                "active_detections": 0,
                "thresholds": {
                    "free_probability_leq": self.free_threshold,
                    "occupied_probability_geq": self.occupied_threshold,
                },
                "bounds": {
                    "min_x": float(msg.info.origin.position.x),
                    "max_x": float(msg.info.origin.position.x) + width * float(msg.info.resolution),
                    "min_y": float(msg.info.origin.position.y),
                    "max_y": float(msg.info.origin.position.y) + height * float(msg.info.resolution),
                },
            },
        }
        atomic_write_json(self.output_path, payload)
        self.last_written_signature = signature
        self.dirty = False
        # Update fingerprint so _map_callback can identify and ignore the
        # re-publication of this exact map by publish_toolbox_map_snapshot.py.
        n = len(data)
        self._last_written_data_fingerprint = (
            width, height, n
        ) + tuple(data[: min(n, 2048)])


def main() -> None:
    rclpy.init()
    node = ToolboxMapExporter()
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
