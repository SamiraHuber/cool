import json
import math
import os
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster


SNAPSHOT_PATH = Path(os.getenv("SLAM_TOOLBOX_MAP_PATH", "/shared/toolbox_map_snapshot.json"))
ACTIVE_SELECTION_PATH = Path(
    os.getenv("TOOLBOX_MAP_ACTIVE_PATH", "/shared/maps/toolbox_saved/active.json")
)
SLAM_TAB_CLAIM_PATH = Path(os.getenv("SLAM_TAB_CLAIM_PATH", "/shared/slam_tab/claim.json"))
SLAM_TAB_MAP_PATH = Path(os.getenv("SLAM_TAB_MAP_PATH", "/shared/slam_tab/map.json"))
SLAM_TAB_ROBOT_POSE_PATH = Path(os.getenv("SLAM_TAB_ROBOT_POSE_PATH", "/shared/slam_tab/robot_pose.json"))
FROZEN_SNAPSHOT_PATH = Path(
    os.getenv("TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH", "/shared/maps/toolbox_saved/active_snapshot.json")
)
SLAM_TOOLBOX_MODE_FILE = Path(os.getenv("SLAM_TOOLBOX_MODE_FILE", "/tmp/slam_toolbox_mode.txt"))
TOPIC_NAME = os.getenv("SLAM_TOOLBOX_MAP_TOPIC", "/map")
PUBLISH_PERIOD_SEC = float(os.getenv("TOOLBOX_MAP_PUBLISH_PERIOD_SEC", "1.0"))
FALLBACK_TF_PERIOD_SEC = float(os.getenv("MAP_FALLBACK_TF_PERIOD_SEC", "0.05"))
SLAM_TAB_CLAIM_TTL_SEC = float(os.getenv("SLAM_TAB_CLAIM_TTL_SEC", "5.0"))
OCCUPIED_THRESHOLD = int(os.getenv("TOOLBOX_MAP_OCCUPIED_THRESHOLD", "65"))
DESPECKLE_MAX_CELLS = int(os.getenv("TOOLBOX_MAP_DESPECKLE_MAX_CELLS", "6"))
DESPECKLE_MIN_NEIGHBORS = int(os.getenv("TOOLBOX_MAP_DESPECKLE_MIN_NEIGHBORS", "2"))

# Publish only the missing fallback map → odom transform. The real Spot driver
# should remain the single source of truth for odom → base.
FALLBACK_TF_MAP_FRAME = os.getenv("MAP_FALLBACK_MAP_FRAME", "map")
FALLBACK_TF_ODOM_FRAME = os.getenv("MAP_FALLBACK_ODOM_FRAME", "spot/vision")
PUBLISH_FALLBACK_TF = os.getenv("MAP_PUBLISH_FALLBACK_TF", "true").lower() not in ("false", "0", "no")


def _slam_tab_claim_is_active() -> bool:
    if not SLAM_TAB_CLAIM_PATH.exists():
        return False
    try:
        payload = json.loads(SLAM_TAB_CLAIM_PATH.read_text(encoding="utf-8"))
    except Exception:
        return False
    updated_at = float(payload.get("updated_at") or 0.0)
    return updated_at > 0.0 and (time.time() - updated_at) <= SLAM_TAB_CLAIM_TTL_SEC


def _normalize_robot_pose(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None
    position = payload.get("position")
    if not isinstance(position, dict):
        return None
    if position.get("x") is None or position.get("y") is None:
        return None
    return {
        "x": float(position.get("x") or 0.0),
        "y": float(position.get("y") or 0.0),
        "z": float(position.get("z") or 0.0),
    }


def _slam_toolbox_mode() -> str:
    try:
        return SLAM_TOOLBOX_MODE_FILE.read_text(encoding="utf-8").strip().lower() or "mapping"
    except Exception:
        return "mapping"


def _load_map_snapshot_payload(snapshot_path: Path) -> tuple[dict | None, dict | None]:
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    map_payload = payload.get("map") if isinstance(payload, dict) else None
    robot_payload = payload.get("robot") if isinstance(payload, dict) else None
    if isinstance(map_payload, dict):
        return payload, robot_payload if isinstance(robot_payload, dict) else None

    info = payload.get("info") if isinstance(payload, dict) else None
    data = payload.get("data") if isinstance(payload, dict) else None
    header = payload.get("header") if isinstance(payload, dict) else None
    if not isinstance(info, dict) or not isinstance(data, list):
        return None, None

    origin = info.get("origin") or {}
    origin_position = origin.get("position") or {}
    frame_id = "map"
    if isinstance(header, dict):
        frame_id = str(header.get("frame_id") or frame_id)

    normalized = {
        "map": {
            "frame_id": frame_id,
            "resolution": float(info.get("resolution", 0.0) or 0.0),
            "width": int(info.get("width", 0) or 0),
            "height": int(info.get("height", 0) or 0),
            "origin": {
                "x": float(origin_position.get("x", 0.0) or 0.0),
                "y": float(origin_position.get("y", 0.0) or 0.0),
            },
            "data": data,
        }
    }

    robot_payload = None
    if SLAM_TAB_ROBOT_POSE_PATH.exists():
        try:
            robot_payload = _normalize_robot_pose(
                json.loads(SLAM_TAB_ROBOT_POSE_PATH.read_text(encoding="utf-8"))
            )
        except Exception:
            robot_payload = None

    return normalized, robot_payload


def _despeckle_occupied_cells(data: list[int], width: int, height: int) -> list[int]:
    if DESPECKLE_MAX_CELLS <= 0:
        return data

    filtered = [int(v) for v in data]
    visited = [False] * len(filtered)

    def occupied(index: int) -> bool:
        return filtered[index] >= OCCUPIED_THRESHOLD

    def neighbors(index: int) -> list[int]:
        x = index % width
        y = index // width
        adjacent = []
        for dy in (-1, 0, 1):
            ny = y + dy
            if ny < 0 or ny >= height:
                continue
            for dx in (-1, 0, 1):
                nx = x + dx
                if dx == 0 and dy == 0:
                    continue
                if nx < 0 or nx >= width:
                    continue
                adjacent.append(ny * width + nx)
        return adjacent

    for start in range(len(filtered)):
        if visited[start] or not occupied(start):
            continue

        stack = [start]
        component = []
        visited[start] = True
        occupied_neighbor_links = 0

        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in neighbors(current):
                if occupied(neighbor):
                    occupied_neighbor_links += 1
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)

        # Remove tiny isolated islands but keep thin real structures that have
        # enough occupied connectivity to look intentional.
        if len(component) <= DESPECKLE_MAX_CELLS and occupied_neighbor_links <= (
            len(component) * DESPECKLE_MIN_NEIGHBORS
        ):
            for index in component:
                filtered[index] = 0

    return filtered


class ToolboxMapSnapshotPublisher(Node):
    def __init__(self) -> None:
        super().__init__("toolbox_map_snapshot_publisher")

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(OccupancyGrid, TOPIC_NAME, qos)
        self.last_signature = None
        self.last_message = None
        self.last_robot_pose = None

        if PUBLISH_FALLBACK_TF:
            self.tf_broadcaster = TransformBroadcaster(self)
            self.get_logger().info(
                f"Fallback TF enabled: {FALLBACK_TF_MAP_FRAME} → {FALLBACK_TF_ODOM_FRAME}"
            )
        else:
            self.tf_broadcaster = None

        self.create_timer(max(0.2, PUBLISH_PERIOD_SEC), self._publish_snapshot)
        if self.tf_broadcaster is not None:
            self.create_timer(max(0.02, FALLBACK_TF_PERIOD_SEC), self._broadcast_fallback_tf)
        self.get_logger().info(f"Publishing toolbox snapshot {SNAPSHOT_PATH} to {TOPIC_NAME}")

    def _snapshot_path(self) -> Path:
        try:
            if ACTIVE_SELECTION_PATH.exists():
                payload = json.loads(ACTIVE_SELECTION_PATH.read_text(encoding="utf-8"))
                if str(payload.get("mode") or "").strip().lower() == "frozen" and FROZEN_SNAPSHOT_PATH.exists():
                    return FROZEN_SNAPSHOT_PATH
        except Exception as exc:
            self.get_logger().warning(f"Failed to read active toolbox map selection: {exc}")
        return SNAPSHOT_PATH

    def _publish_snapshot(self) -> None:
        slam_tab_active = _slam_tab_claim_is_active()
        slam_mode = _slam_toolbox_mode()
        is_localization = slam_mode == "localization"

        is_recording = False
        try:
            if ACTIVE_SELECTION_PATH.exists():
                payload = json.loads(ACTIVE_SELECTION_PATH.read_text(encoding="utf-8"))
                is_recording = str(payload.get("mode") or "").strip().lower() == "recording"
        except Exception:
            pass

        # In localization mode, prefer the frozen snapshot (saved YAML/PGM) over
        # slam_tab's last-recorded map, since the loaded map may differ from the
        # last mapping session.
        if is_localization:
            snapshot_path = self._snapshot_path()
        elif slam_tab_active and not is_recording and SLAM_TAB_MAP_PATH.exists():
            snapshot_path = SLAM_TAB_MAP_PATH
        else:
            snapshot_path = self._snapshot_path()

        if not snapshot_path.exists():
            return

        payload, robot_payload = _load_map_snapshot_payload(snapshot_path)
        if not isinstance(payload, dict):
            self.get_logger().warning(f"Failed to parse map snapshot: {snapshot_path}")
            return

        map_payload = payload.get("map")
        if not isinstance(map_payload, dict):
            return

        data = map_payload.get("data")
        width = int(map_payload.get("width", 0) or 0)
        height = int(map_payload.get("height", 0) or 0)
        resolution = float(map_payload.get("resolution", 0.0) or 0.0)
        origin = map_payload.get("origin") or {}
        if not isinstance(data, list) or width <= 0 or height <= 0 or resolution <= 0.0:
            return
        if len(data) != width * height:
            return

        if isinstance(robot_payload, dict) and robot_payload.get("x") is not None:
            self.last_robot_pose = robot_payload

        signature = (
            width,
            height,
            resolution,
            float(origin.get("x", 0.0) or 0.0),
            float(origin.get("y", 0.0) or 0.0),
            len(data),
            hash(tuple(data[: min(len(data), 2048)])),
            OCCUPIED_THRESHOLD,
            DESPECKLE_MAX_CELLS,
            DESPECKLE_MIN_NEIGHBORS,
        )

        if self.last_signature != signature:
            filtered_data = _despeckle_occupied_cells(data, width, height)
            cleared_cells = sum(1 for old, new in zip(data, filtered_data) if int(old) != int(new))
            msg = OccupancyGrid()
            msg.header.frame_id = str(map_payload.get("frame_id") or "map")
            msg.info.resolution = resolution
            msg.info.width = width
            msg.info.height = height
            msg.info.origin.position.x = float(origin.get("x", 0.0) or 0.0)
            msg.info.origin.position.y = float(origin.get("y", 0.0) or 0.0)
            msg.info.origin.orientation.w = 1.0
            msg.data = filtered_data
            self.last_message = msg
            self.last_signature = signature
            self.get_logger().info(
                "Loaded map snapshot: "
                f"{width}x{height} at {resolution:.3f} m/cell "
                f"(occupied>={OCCUPIED_THRESHOLD}, despeckle<={DESPECKLE_MAX_CELLS} cells, "
                f"cleared={cleared_cells}, source={snapshot_path.name})"
            )

        # When slam_toolbox owns /map during recording, skip publishing to avoid
        # fighting the live map.
        if is_recording:
            self.get_logger().debug(
                "Skipping map publish: slam_toolbox owns /map "
                f"during recording mode ({ACTIVE_SELECTION_PATH.name})"
            )
        elif is_localization and self._snapshot_path() == FROZEN_SNAPSHOT_PATH:
            # In localization mode slam_toolbox rebuilds the map from the posegraph,
            # which can clear unknown cells behind walls. If we have a frozen snapshot
            # built from the saved YAML/PGM, publish it to /map so Nav2 sees the
            # original map with proper unknown cells.
            if self.last_message is not None:
                self.last_message.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(self.last_message)
        elif self.last_message is not None:
            self.last_message.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(self.last_message)

    def _broadcast_fallback_tf(self) -> None:
        """Publish the fallback map→odom transform Nav2 needs when slam_toolbox is
        not publishing it. The robot driver should own odom→base."""
        if _slam_toolbox_mode() == "localization":
            # In localization mode slam_toolbox owns map->odom. Publishing an
            # identity fallback here fights the localized pose and causes
            # set-pose updates to snap back immediately.
            return

        now = self.get_clock().now().to_msg()

        # map → spot/odom  (identity: odom origin at map origin as fallback)
        t_map_odom = TransformStamped()
        t_map_odom.header.stamp = now
        t_map_odom.header.frame_id = FALLBACK_TF_MAP_FRAME
        t_map_odom.child_frame_id = FALLBACK_TF_ODOM_FRAME
        t_map_odom.transform.rotation.w = 1.0

        self.tf_broadcaster.sendTransform(t_map_odom)


def main() -> None:
    rclpy.init()
    node = ToolboxMapSnapshotPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            node.destroy_node()
        except RCLError:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except RCLError:
            pass


if __name__ == "__main__":
    main()
