import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from bordsupr_interfaces.msg import YoloOutput
from dynamic_slam_interfaces.msg import ObjectOdometry


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def has_valid_relative_pose(relative_pose, tolerance: float = 1e-6, now_sec: float | None = None, ttl_sec: float | None = None) -> bool:
    if not isinstance(relative_pose, dict):
        return False

    rel_x = float(relative_pose.get("x", 0.0))
    rel_y = float(relative_pose.get("y", 0.0))
    rel_z = float(relative_pose.get("z", 0.0))
    if abs(rel_x) <= tolerance and abs(rel_y) <= tolerance and abs(rel_z) <= tolerance:
        return False

    if now_sec is not None and ttl_sec is not None and ttl_sec > 0:
        last_updated = float(relative_pose.get("last_updated_at", 0.0))
        if (now_sec - last_updated) > ttl_sec:
            return False

    return True


def probability_to_log_odds(probability: float) -> float:
    probability = min(max(float(probability), 1e-3), 1.0 - 1e-3)
    return math.log(probability / (1.0 - probability))


def log_odds_to_probability(log_odds: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-log_odds))


class LidarOccupancyMapNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_occupancy_map_node")

        self.declare_parameter("scan_snapshot_path", "/shared/velodyne_scan_snapshot.json")
        self.declare_parameter("output_path", "/shared/lidar_occupancy_map.json")
        self.declare_parameter("odometry_topic", "/spot/odometry")
        self.declare_parameter("resolution", 0.10)
        self.declare_parameter("width", 200)
        self.declare_parameter("height", 200)
        self.declare_parameter("origin_x", -10.0)
        self.declare_parameter("origin_y", -10.0)
        self.declare_parameter("poll_period_sec", 1.0)
        self.declare_parameter("publish_every_n_scans", 1)
        self.declare_parameter("hit_probability", 0.72)
        self.declare_parameter("free_probability", 0.43)
        self.declare_parameter("occupied_threshold", 0.68)
        self.declare_parameter("free_threshold", 0.35)
        self.declare_parameter("max_log_odds", 3.5)
        self.declare_parameter("min_log_odds", -2.5)
        self.declare_parameter("occupancy_increment", 8)
        self.declare_parameter("occupancy_decay", 0.0)
        self.declare_parameter("free_space_decrement", 10)
        self.declare_parameter("raytrace_free_space", True)
        self.declare_parameter("max_raytrace_range_m", 8.0)
        self.declare_parameter("robot_exclusion_radius_m", 0.75)
        self.declare_parameter("min_points_per_occupied_cell", 2)
        self.declare_parameter("isolated_occupied_min_neighbors", 1)
        self.declare_parameter("max_occupancy_increment_per_scan", 24)
        self.declare_parameter("max_hit_observations_per_cell_per_scan", 3)
        self.declare_parameter("max_free_observations_per_cell_per_scan", 6)
        self.declare_parameter("occupancy_grid_topic", "/lidar_occupancy_grid")
        self.declare_parameter("publish_occupancy_grid", True)
        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("object_odometry_topic", "/dynosam/frontend/object_odometry")
        self.declare_parameter("backend_object_odometry_topic", "/dynosam/backend/object_odometry")
        self.declare_parameter("dynosam_frontend_odom_topic", "/dynosam/frontend/odometry")
        self.declare_parameter("dynosam_backend_odom_topic", "/dynosam/backend/odometry")
        self.declare_parameter("use_backend_object_odometry", True)
        self.declare_parameter("object_position_ema_alpha", 0.4)
        self.declare_parameter("detection_marker_ttl_sec", 30.0)
        self.declare_parameter("object_odometry_ttl_sec", 5.0)
        self.declare_parameter("use_tf_for_dynosam_poses", True)
        self.declare_parameter("spot_camera_frame", "frontleft_fisheye")
        self.declare_parameter("clear_detections_signal_path", "/shared/clear_detections.signal")

        self.scan_snapshot_path = Path(self.get_parameter("scan_snapshot_path").value)
        self.output_path = Path(self.get_parameter("output_path").value)
        self.odometry_topic = self.get_parameter("odometry_topic").value
        self.resolution = float(self.get_parameter("resolution").value)
        self.width = int(self.get_parameter("width").value)
        self.height = int(self.get_parameter("height").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)
        self.poll_period_sec = max(0.1, float(self.get_parameter("poll_period_sec").value))
        self.publish_every_n_scans = max(1, int(self.get_parameter("publish_every_n_scans").value))
        self.hit_probability = float(self.get_parameter("hit_probability").value)
        self.free_probability = float(self.get_parameter("free_probability").value)
        self.occupied_threshold = float(self.get_parameter("occupied_threshold").value)
        self.free_threshold = float(self.get_parameter("free_threshold").value)
        self.max_log_odds = float(self.get_parameter("max_log_odds").value)
        self.min_log_odds = float(self.get_parameter("min_log_odds").value)
        self.occupancy_increment = max(1, int(self.get_parameter("occupancy_increment").value))
        self.occupancy_decay = max(0.0, float(self.get_parameter("occupancy_decay").value))
        self.free_space_decrement = max(1, int(self.get_parameter("free_space_decrement").value))
        self.raytrace_free_space = bool(self.get_parameter("raytrace_free_space").value)
        self.max_raytrace_range_m = max(0.5, float(self.get_parameter("max_raytrace_range_m").value))
        self.robot_exclusion_radius_m = max(0.0, float(self.get_parameter("robot_exclusion_radius_m").value))
        self.min_points_per_occupied_cell = max(
            1, int(self.get_parameter("min_points_per_occupied_cell").value)
        )
        self.isolated_occupied_min_neighbors = max(
            0, int(self.get_parameter("isolated_occupied_min_neighbors").value)
        )
        self.max_occupancy_increment_per_scan = max(
            self.occupancy_increment,
            int(self.get_parameter("max_occupancy_increment_per_scan").value),
        )
        self.max_hit_observations_per_cell_per_scan = max(
            1, int(self.get_parameter("max_hit_observations_per_cell_per_scan").value)
        )
        self.max_free_observations_per_cell_per_scan = max(
            1, int(self.get_parameter("max_free_observations_per_cell_per_scan").value)
        )
        self.occupancy_grid_topic = self.get_parameter("occupancy_grid_topic").value
        self.publish_occupancy_grid = bool(self.get_parameter("publish_occupancy_grid").value)
        self.yolo_output_topic = self.get_parameter("yolo_output_topic").value
        self.object_odometry_topic = self.get_parameter("object_odometry_topic").value
        self.backend_object_odometry_topic = self.get_parameter("backend_object_odometry_topic").value
        self.dynosam_frontend_odom_topic = self.get_parameter("dynosam_frontend_odom_topic").value
        self.dynosam_backend_odom_topic = self.get_parameter("dynosam_backend_odom_topic").value
        self.use_backend_object_odometry = bool(self.get_parameter("use_backend_object_odometry").value)
        self.object_position_ema_alpha = max(0.0, min(1.0, float(self.get_parameter("object_position_ema_alpha").value)))
        self.detection_marker_ttl_sec = max(
            0.1, float(self.get_parameter("detection_marker_ttl_sec").value)
        )
        self.object_odometry_ttl_sec = max(
            0.0, float(self.get_parameter("object_odometry_ttl_sec").value)
        )
        self.use_tf_for_dynosam_poses = bool(self.get_parameter("use_tf_for_dynosam_poses").value)
        self.spot_camera_frame = str(self.get_parameter("spot_camera_frame").value)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.hit_log_odds = probability_to_log_odds(self.hit_probability)
        self.free_log_odds = probability_to_log_odds(self.free_probability)
        self.occupied_threshold_log_odds = probability_to_log_odds(self.occupied_threshold)
        self.free_threshold_log_odds = probability_to_log_odds(self.free_threshold)

        self.log_odds_grid = np.zeros((self.height, self.width), dtype=np.float32)
        self.observed_mask = np.zeros((self.height, self.width), dtype=bool)
        self.scan_count = 0
        self.last_scan_timestamp = None
        self.last_frame_id = "world"
        self.last_robot_pose = None
        self.occupancy_grid_pub = None
        self.active_detections = {}
        self.object_relative_positions = {}
        self.backend_object_relative_positions = {}
        self.object_ema_positions = {}
        self.depth_object_positions: Dict[int, dict] = {}
        self.depth_object_odometry_ttl_sec = 2.0
        self.last_dynosam_frontend_camera_pose = None
        self.last_dynosam_backend_camera_pose = None
        self.clear_detections_signal_path = Path(self.get_parameter("clear_detections_signal_path").value)
        self._last_clear_signal_mtime = 0.0

        if self.publish_occupancy_grid:
            self.occupancy_grid_pub = self.create_publisher(OccupancyGrid, self.occupancy_grid_topic, 10)

        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            Odometry,
            self.odometry_topic,
            self.odometry_callback,
            odom_qos,
        )
        self.create_subscription(
            YoloOutput,
            self.yolo_output_topic,
            self.yolo_output_callback,
            10,
        )
        self.create_subscription(
            ObjectOdometry,
            self.object_odometry_topic,
            self._frontend_object_odometry_callback,
            50,
        )
        if self.use_backend_object_odometry:
            self.create_subscription(
                ObjectOdometry,
                self.backend_object_odometry_topic,
                self._backend_object_odometry_callback,
                50,
            )
        self.create_subscription(
            Odometry,
            self.dynosam_frontend_odom_topic,
            self.dynosam_frontend_odom_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            self.dynosam_backend_odom_topic,
            self.dynosam_backend_odom_callback,
            10,
        )
        self.create_subscription(
            ObjectOdometry,
            "/bordsupr/depth_object_odometry",
            self._depth_object_odometry_callback,
            100,
        )
        if self.use_tf_for_dynosam_poses:
            self.get_logger().info("Using TF lookups for DynoSAM object poses")
        self.get_logger().info(
            f"Spot camera frame for DynoSAM alignment: {self.spot_camera_frame}"
        )
        self.timer = self.create_timer(self.poll_period_sec, self.poll_scan_snapshot)
        self.get_logger().info(
            f"Polling {self.scan_snapshot_path} and writing occupancy map snapshots to {self.output_path}"
        )
        if self.occupancy_grid_pub is not None:
            self.get_logger().info(f"Publishing OccupancyGrid snapshots on {self.occupancy_grid_topic}")
        self.get_logger().info(
            f"Overlaying recent YOLO detections from {self.yolo_output_topic} in world/map coordinates "
            f"for {self.detection_marker_ttl_sec:.1f}s"
        )
        self.get_logger().info(
            f"Combining relative DynoSAM positions from {self.object_odometry_topic} with robot odometry"
        )
        if self.use_backend_object_odometry:
            self.get_logger().info(
                f"Also consuming backend object odometry from {self.backend_object_odometry_topic}"
            )
        self.get_logger().info(
            f"Object position EMA alpha = {self.object_position_ema_alpha}"
        )
        if self.raytrace_free_space:
            self.get_logger().info(
                f"Ray-carving free space from robot pose with free decrement {self.free_space_decrement} "
                f"and max range {self.max_raytrace_range_m:.1f} m"
            )
        if self.robot_exclusion_radius_m > 0.0:
            self.get_logger().info(
                f"Ignoring lidar obstacle endpoints within {self.robot_exclusion_radius_m:.2f} m of the robot"
            )
        self.get_logger().info(
            f"Requiring {self.min_points_per_occupied_cell} point(s) per occupied cell and "
            f"pruning occupied cells with fewer than {self.isolated_occupied_min_neighbors} occupied neighbors"
        )
        self.get_logger().info(
            f"Using probabilistic occupancy updates with free threshold {self.free_threshold:.2f} "
            f"and occupied threshold {self.occupied_threshold:.2f}"
        )

    def odometry_callback(self, msg: Odometry) -> None:
        previous_pose = self.last_robot_pose
        orientation = msg.pose.pose.orientation
        self.last_robot_pose = {
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
        if self.scan_count > 0 and (
            previous_pose is None
            or previous_pose["x"] != self.last_robot_pose["x"]
            or previous_pose["y"] != self.last_robot_pose["y"]
            or previous_pose["z"] != self.last_robot_pose["z"]
        ):
            self._write_snapshot()

    def yolo_output_callback(self, msg: YoloOutput) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        robot_pose = self.last_robot_pose
        if robot_pose is None:
            return

        for det in msg.objects:
            instance_id = int(getattr(det, "instance_id", 0))
            if instance_id < 0:
                continue

            # Prefer backend pose when available
            relative_pose = self.backend_object_relative_positions.get(instance_id)
            source = "backend"
            if relative_pose is None:
                relative_pose = self.object_relative_positions.get(instance_id)
                source = "frontend"

            has_dynosam_position = has_valid_relative_pose(
                relative_pose,
                now_sec=now_sec,
                ttl_sec=self.object_odometry_ttl_sec,
            )

            if relative_pose is not None and "robot_x" in relative_pose:
                saved_robot_pose = {
                    "x": relative_pose["robot_x"],
                    "y": relative_pose["robot_y"],
                    "z": relative_pose["robot_z"],
                    "yaw": relative_pose["robot_yaw"],
                }
                world_pose = self._compose_relative_pose_with_robot(
                    relative_pose, saved_robot_pose
                )
            else:
                world_pose = self._compose_relative_pose_with_robot(relative_pose, robot_pose)

            # ------------------------------------------------------------------
            # Fallback when no DynoSAM pose is available yet:
            #   - If we have a prior display position, freeze it (don't snap to robot).
            #   - Otherwise fall back to the robot's current position so the
            #     detection is at least registered in active_detections and can
            #     be updated once the DynoSAM pose arrives.
            # ------------------------------------------------------------------
            if world_pose is None:
                depth_pose = self.depth_object_positions.get(instance_id)
                if depth_pose is not None and (now_sec - depth_pose["timestamp"]) < self.depth_object_odometry_ttl_sec:
                    world_pose = (depth_pose["x"], depth_pose["y"], depth_pose["z"])
                    source = "depth"
                    has_dynosam_position = False
                else:
                    entry = self.active_detections.get(instance_id)
                    if entry is not None and all(
                        k in entry for k in ("x", "y", "z")
                    ):
                        world_pose = (float(entry["x"]), float(entry["y"]), float(entry["z"]))
                        source = entry.get("dynosam_source", source)
                    else:
                        world_pose = (
                            float(robot_pose["x"]),
                            float(robot_pose["y"]),
                            float(robot_pose.get("z", 0.0)),
                        )
                        source = "robot"
                        has_dynosam_position = False

            entry = self.active_detections.get(instance_id, {})
            # When TF lookups are enabled, preserve TF-derived poses and source
            if self.use_tf_for_dynosam_poses and entry.get("has_dynosam_position"):
                has_dynosam_position = True
                source = entry.get("dynosam_source", source)
                world_pose = (float(entry["x"]), float(entry["y"]), float(entry["z"]))
            entry.update(
                {
                    "instance_id": instance_id,
                    "track_id": int(getattr(det, "track_id", -1)),
                    "class_id": int(getattr(det, "class_id", -1)),
                    "class_name": str(getattr(det, "class_name", "")),
                    "score": float(getattr(det, "score", 0.0)),
                    "frame_id": robot_pose.get("frame_id", "world"),
                    "x": float(world_pose[0]),
                    "y": float(world_pose[1]),
                    "z": float(world_pose[2]),
                    "has_dynosam_position": has_dynosam_position,
                    "has_depth_position": source == "depth",
                    "position_source": source if source in ("backend", "frontend", "depth", "robot") else "yolo",
                    "last_detected_at": now_sec,
                    "dynosam_source": source,
                }
            )
            self.active_detections[instance_id] = entry

        self._prune_stale_detections(now_sec)

        if msg.objects and (self.scan_count > 0 or self.output_path.exists()):
            self._write_snapshot()

    def dynosam_frontend_odom_callback(self, msg: Odometry) -> None:
        self.last_dynosam_frontend_camera_pose = self._odom_to_pose_dict(msg)

    def dynosam_backend_odom_callback(self, msg: Odometry) -> None:
        self.last_dynosam_backend_camera_pose = self._odom_to_pose_dict(msg)

    @staticmethod
    def _odom_to_pose_dict(msg: Odometry) -> dict:
        orientation = msg.pose.pose.orientation
        return {
            "x": float(msg.pose.pose.position.x),
            "y": float(msg.pose.pose.position.y),
            "z": float(msg.pose.pose.position.z),
            "yaw": quaternion_to_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
            "frame_id": str(getattr(msg, "child_frame_id", "") or ""),
        }

    def _lookup_spot_camera_pose(self, frame_id: str | None = None) -> dict | None:
        """Look up the camera pose in Spot odom via TF."""
        camera_frame = frame_id or self.spot_camera_frame
        if not camera_frame:
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                "spot/vision", camera_frame, rclpy.time.Time()
            )
            orientation = transform.transform.rotation
            return {
                "x": float(transform.transform.translation.x),
                "y": float(transform.transform.translation.y),
                "z": float(transform.transform.translation.z),
                "yaw": quaternion_to_yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            }
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _transform_dynosam_world_to_spot_odom(
        self,
        dynosam_world_point: tuple[float, float, float],
        dynosam_camera_pose: dict,
    ) -> tuple[float, float, float]:
        """Transform a 3-D point from DynoSAM world frame to Spot odom frame.

        Uses the camera pose in both coordinate systems to compute the rigid
        alignment between DynoSAM world and Spot odom.
        """
        if dynosam_camera_pose is None:
            return dynosam_world_point

        # Look up the same physical camera in Spot odom via TF
        spot_camera_pose = self._lookup_spot_camera_pose(
            dynosam_camera_pose.get("frame_id")
        )
        if spot_camera_pose is None:
            # Fallback to robot pose (approximate camera pose)
            if self.last_robot_pose is None:
                return dynosam_world_point
            spot_camera_pose = self.last_robot_pose
            self.get_logger().warning(
                f"TF lookup failed for camera frame {dynosam_camera_pose.get('frame_id') or self.spot_camera_frame}; "
                f"falling back to robot pose for DynoSAM world→odom transform"
            )

        # Rigid alignment using the camera as the anchor point:
        #   p_spot = R(dyaw) * (p_dynosam - p_camera_dynosam) + p_camera_spot
        dx = dynosam_world_point[0] - dynosam_camera_pose["x"]
        dy = dynosam_world_point[1] - dynosam_camera_pose["y"]
        dyaw = spot_camera_pose["yaw"] - dynosam_camera_pose["yaw"]
        cos_dyaw = math.cos(dyaw)
        sin_dyaw = math.sin(dyaw)

        spot_x = spot_camera_pose["x"] + cos_dyaw * dx - sin_dyaw * dy
        spot_y = spot_camera_pose["y"] + sin_dyaw * dx + cos_dyaw * dy
        spot_z = (
            dynosam_world_point[2]
            - dynosam_camera_pose.get("z", 0.0)
            + spot_camera_pose.get("z", 0.0)
        )
        self.get_logger().debug(
            f"DynoSAM transform: dynosam=({dynosam_world_point[0]:.3f},{dynosam_world_point[1]:.3f}) "
            f"cam_dynosam=({dynosam_camera_pose['x']:.3f},{dynosam_camera_pose['y']:.3f},{dynosam_camera_pose['yaw']:.3f}) "
            f"cam_spot=({spot_camera_pose['x']:.3f},{spot_camera_pose['y']:.3f},{spot_camera_pose['yaw']:.3f}) "
            f"dyaw={dyaw:.3f} result=({spot_x:.3f},{spot_y:.3f})"
        )
        return (spot_x, spot_y, spot_z)

    def _apply_ema(
        self, instance_id: int, new_position: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Apply exponential moving average to smooth noisy re-initialisations."""
        alpha = self.object_position_ema_alpha
        if alpha <= 0.0:
            return new_position

        old = self.object_ema_positions.get(instance_id)
        if old is None:
            self.object_ema_positions[instance_id] = new_position
            return new_position

        result = (
            alpha * new_position[0] + (1.0 - alpha) * old[0],
            alpha * new_position[1] + (1.0 - alpha) * old[1],
            alpha * new_position[2] + (1.0 - alpha) * old[2],
        )
        self.object_ema_positions[instance_id] = result
        return result

    def _frontend_object_odometry_callback(self, msg: ObjectOdometry) -> None:
        self._handle_object_odometry(msg, is_backend=False)

    def _backend_object_odometry_callback(self, msg: ObjectOdometry) -> None:
        self._handle_object_odometry(msg, is_backend=True)

    def _depth_object_odometry_callback(self, msg: ObjectOdometry) -> None:
        instance_id = int(msg.object_id)
        if instance_id < 0:
            return
        self.depth_object_positions[instance_id] = {
            "x": float(msg.odom.pose.pose.position.x),
            "y": float(msg.odom.pose.pose.position.y),
            "z": float(msg.odom.pose.pose.position.z),
            "timestamp": self.get_clock().now().nanoseconds / 1e9,
        }

    def _handle_object_odometry(self, msg: ObjectOdometry, is_backend: bool = False) -> None:
        instance_id = int(msg.object_id)
        pose = msg.odom.pose.pose.position
        frame_id = str(getattr(getattr(getattr(msg, "odom", None), "header", None), "frame_id", "") or "")
        is_world_frame = frame_id in {"world", "map"}
        now_sec = self.get_clock().now().nanoseconds / 1e9

        relative_pose = {
            "x": float(pose.x),
            "y": float(pose.y),
            "z": float(pose.z),
            "is_world_frame": is_world_frame,
            "last_updated_at": now_sec,
            "source": "backend" if is_backend else "frontend",
        }

        # ------------------------------------------------------------------
        # 1) Transform DynoSAM-world → Spot-odom using the live camera pose
        # ------------------------------------------------------------------
        camera_pose = (
            self.last_dynosam_backend_camera_pose
            if is_backend else self.last_dynosam_frontend_camera_pose
        )
        if is_world_frame and camera_pose is not None:
            transformed = self._transform_dynosam_world_to_spot_odom(
                (float(pose.x), float(pose.y), float(pose.z)), camera_pose
            )
            relative_pose["x"] = transformed[0]
            relative_pose["y"] = transformed[1]
            relative_pose["z"] = transformed[2]
            relative_pose["is_world_frame"] = True  # now in Spot-odom (our display frame)
        elif is_world_frame:
            self.get_logger().warning(
                f"{'Backend' if is_backend else 'Frontend'} object {instance_id}: "
                f"cannot transform world-frame pose (no camera odometry yet)"
            )

        # ------------------------------------------------------------------
        # 2) Snapshot the robot pose ONCE per object trajectory
        # ------------------------------------------------------------------
        robot_pose = self.last_robot_pose
        target_dict = (
            self.backend_object_relative_positions
            if is_backend else self.object_relative_positions
        )
        existing = target_dict.get(instance_id)
        if robot_pose is not None:
            if existing is None or "robot_x" not in existing:
                relative_pose["robot_x"] = robot_pose["x"]
                relative_pose["robot_y"] = robot_pose["y"]
                relative_pose["robot_z"] = robot_pose.get("z", 0.0)
                relative_pose["robot_yaw"] = robot_pose.get("yaw", 0.0)
            else:
                # Preserve the original snapshot; do NOT overwrite with current pose
                relative_pose["robot_x"] = existing["robot_x"]
                relative_pose["robot_y"] = existing["robot_y"]
                relative_pose["robot_z"] = existing["robot_z"]
                relative_pose["robot_yaw"] = existing["robot_yaw"]

        target_dict[instance_id] = relative_pose

        # ------------------------------------------------------------------
        # 3) Update active detection with transformed + smoothed position
        # ------------------------------------------------------------------
        entry = self.active_detections.get(instance_id)
        if entry is not None:
            saved_robot_pose = {
                "x": relative_pose.get("robot_x", 0.0),
                "y": relative_pose.get("robot_y", 0.0),
                "z": relative_pose.get("robot_z", 0.0),
                "yaw": relative_pose.get("robot_yaw", 0.0),
            }
            world_pose = self._compose_relative_pose_with_robot(
                relative_pose, saved_robot_pose
            )
            if world_pose is not None:
                smoothed = self._apply_ema(instance_id, world_pose)
                entry["x"] = float(smoothed[0])
                entry["y"] = float(smoothed[1])
                entry["z"] = float(smoothed[2])
                entry["has_dynosam_position"] = has_valid_relative_pose(
                    relative_pose,
                    now_sec=now_sec,
                    ttl_sec=self.object_odometry_ttl_sec,
                )
                entry["dynosam_source"] = "backend" if is_backend else "frontend"

        if self.scan_count > 0 or self.output_path.exists():
            self._write_snapshot()

    def _update_object_positions_from_tf(self) -> None:
        if not self.use_tf_for_dynosam_poses:
            return
        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9

        # Gather all DynoSAM object TF frames
        tf_objects = []
        for suffix, src in [("_backend", "backend"), ("", "frontend")]:
            # TF frames are named object_1_link, object_2_link, etc.
            # We check the first 50 IDs; DynoSAM rarely tracks more.
            for dyno_id in range(1, 51):
                frame_id = f"object_{dyno_id}_link{suffix}"
                try:
                    transform = self.tf_buffer.lookup_transform(
                        "world", frame_id, rclpy.time.Time()
                    )
                    pos = (
                        float(transform.transform.translation.x),
                        float(transform.transform.translation.y),
                        float(transform.transform.translation.z),
                    )
                    # Transform DynoSAM world → Spot odom
                    camera_pose = (
                        self.last_dynosam_backend_camera_pose
                        if src == "backend" else self.last_dynosam_frontend_camera_pose
                    )
                    if camera_pose is not None:
                        pos = self._transform_dynosam_world_to_spot_odom(pos, camera_pose)
                    tf_objects.append((dyno_id, src, pos))
                except (LookupException, ConnectivityException, ExtrapolationException):
                    continue

        if not tf_objects:
            return

        # Match each TF object to the nearest active detection within 2.5 m
        matched_detections = set()
        for dyno_id, src, pos in tf_objects:
            best_id = None
            best_dist = 2.5
            for instance_id, entry in self.active_detections.items():
                if instance_id in matched_detections:
                    continue
                det_x = entry.get("x")
                det_y = entry.get("y")
                if det_x is None or det_y is None:
                    continue
                dx = det_x - pos[0]
                dy = det_y - pos[1]
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist = dist
                    best_id = instance_id

            if best_id is not None:
                matched_detections.add(best_id)
                entry = self.active_detections[best_id]
                smoothed = self._apply_ema(best_id, pos)
                entry["x"] = float(smoothed[0])
                entry["y"] = float(smoothed[1])
                entry["z"] = float(smoothed[2])
                entry["has_dynosam_position"] = True
                entry["dynosam_source"] = src
                entry["last_detected_at"] = now_sec
                entry["dynosam_object_id"] = dyno_id

    def poll_scan_snapshot(self) -> None:
        self._update_object_positions_from_tf()
        try:
            stale_removed = self._prune_stale_detections()
            signal_cleared = self._check_clear_signal()
            if not self.scan_snapshot_path.exists():
                if (stale_removed or signal_cleared) and self.output_path.exists():
                    self._write_snapshot()
                if self.scan_count > 0 and not self.output_path.exists():
                    self._write_snapshot()
                return

            with self.scan_snapshot_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)

            if not payload.get("available"):
                return

            generated_at = payload.get("generated_at")
            if generated_at == self.last_scan_timestamp:
                if (stale_removed or signal_cleared) and self.output_path.exists():
                    self._write_snapshot()
                if not self.output_path.exists() and self.scan_count > 0:
                    self._write_snapshot()
                return

            points = self._extract_points(payload.get("points"))
            self.last_frame_id = payload.get("frame_id") or self.last_frame_id
            self.last_scan_timestamp = generated_at

            if points.size == 0:
                if self.scan_count > 0 and not self.output_path.exists():
                    self._write_snapshot()
                return

            self._update_grid(points)
            self.scan_count += 1
            if self.scan_count % self.publish_every_n_scans == 0 or not self.output_path.exists():
                self._write_snapshot()
        except Exception as exc:
            self.get_logger().error(f"Failed to update lidar occupancy map: {exc}")

    def _extract_points(self, raw_points) -> np.ndarray:
        if not isinstance(raw_points, list) or not raw_points:
            return np.empty((0, 2), dtype=np.float32)

        points = np.asarray(raw_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            return np.empty((0, 2), dtype=np.float32)

        points = points[:, :2]
        finite_mask = np.isfinite(points).all(axis=1)
        return points[finite_mask]

    def _update_grid(self, points: np.ndarray) -> None:
        if self.occupancy_decay > 0:
            observed_values = self.log_odds_grid[self.observed_mask]
            decay = np.minimum(np.abs(observed_values), self.occupancy_decay)
            self.log_odds_grid[self.observed_mask] = observed_values - (np.sign(observed_values) * decay)

        points = self._filter_robot_near_field(points)
        if points.size == 0:
            return

        if self.raytrace_free_space and self.last_robot_pose is not None:
            self._raytrace_free_space(points)

        valid_points, unique_cols, unique_rows, counts = self._accumulate_hit_cells(points)
        if valid_points.size == 0:
            return

        hit_counts = np.minimum(counts, self.max_hit_observations_per_cell_per_scan).astype(np.float32)
        hit_delta = self.hit_log_odds * hit_counts
        self._apply_log_odds(unique_rows, unique_cols, hit_delta)

        if self.isolated_occupied_min_neighbors > 0:
            self._prune_isolated_occupied_cells()

    def _accumulate_hit_cells(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if points.size == 0:
            empty_points = np.empty((0, 2), dtype=np.float32)
            empty_int = np.empty((0,), dtype=np.int32)
            return empty_points, empty_int, empty_int, empty_int

        cols = np.floor((points[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        rows = np.floor((points[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        valid = (cols >= 0) & (cols < self.width) & (rows >= 0) & (rows < self.height)
        if not np.any(valid):
            empty_points = np.empty((0, 2), dtype=np.float32)
            empty_int = np.empty((0,), dtype=np.int32)
            return empty_points, empty_int, empty_int, empty_int

        valid_points = points[valid]
        cell_ids = rows[valid] * self.width + cols[valid]
        unique_ids, counts = np.unique(cell_ids, return_counts=True)

        keep = counts >= self.min_points_per_occupied_cell
        if not np.any(keep):
            empty_points = np.empty((0, 2), dtype=np.float32)
            empty_int = np.empty((0,), dtype=np.int32)
            return empty_points, empty_int, empty_int, empty_int

        kept_ids = unique_ids[keep]
        kept_counts = counts[keep].astype(np.int16)
        kept_rows = (kept_ids // self.width).astype(np.int32)
        kept_cols = (kept_ids % self.width).astype(np.int32)
        return valid_points, kept_cols, kept_rows, kept_counts

    def _filter_robot_near_field(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0 or self.last_robot_pose is None or self.robot_exclusion_radius_m <= 0.0:
            return points

        robot_x = float(self.last_robot_pose["x"])
        robot_y = float(self.last_robot_pose["y"])
        deltas = points - np.array([robot_x, robot_y], dtype=np.float32)
        distances = np.hypot(deltas[:, 0], deltas[:, 1])
        keep_mask = distances >= self.robot_exclusion_radius_m
        return points[keep_mask]

    def _apply_log_odds(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        delta: np.ndarray | float,
    ) -> None:
        if rows.size == 0 or cols.size == 0:
            return

        self.observed_mask[rows, cols] = True
        self.log_odds_grid[rows, cols] = np.clip(
            self.log_odds_grid[rows, cols] + delta,
            self.min_log_odds,
            self.max_log_odds,
        )

    def _world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        col = int(np.floor((x - self.origin_x) / self.resolution))
        row = int(np.floor((y - self.origin_y) / self.resolution))
        return col, row

    def _in_bounds(self, col: int, row: int) -> bool:
        return 0 <= col < self.width and 0 <= row < self.height

    def _grid_fractional(self, x: float, y: float) -> tuple[float, float]:
        return (
            (x - self.origin_x) / self.resolution,
            (y - self.origin_y) / self.resolution,
        )

    def _raytrace_cells(self, start_x: float, start_y: float, end_x: float, end_y: float):
        start_fx, start_fy = self._grid_fractional(start_x, start_y)
        end_fx, end_fy = self._grid_fractional(end_x, end_y)

        start_col = int(math.floor(start_fx))
        start_row = int(math.floor(start_fy))
        end_col = int(math.floor(end_fx))
        end_row = int(math.floor(end_fy))

        if not self._in_bounds(start_col, start_row) or not self._in_bounds(end_col, end_row):
            return []

        step_x = 0
        step_y = 0
        if end_fx > start_fx:
            step_x = 1
        elif end_fx < start_fx:
            step_x = -1
        if end_fy > start_fy:
            step_y = 1
        elif end_fy < start_fy:
            step_y = -1

        delta_fx = end_fx - start_fx
        delta_fy = end_fy - start_fy
        if abs(delta_fx) < 1e-9 and abs(delta_fy) < 1e-9:
            return [(start_col, start_row)]

        t_delta_x = math.inf if step_x == 0 else abs(1.0 / delta_fx)
        t_delta_y = math.inf if step_y == 0 else abs(1.0 / delta_fy)

        if step_x > 0:
            t_max_x = (math.floor(start_fx) + 1.0 - start_fx) * t_delta_x
        elif step_x < 0:
            t_max_x = (start_fx - math.floor(start_fx)) * t_delta_x
        else:
            t_max_x = math.inf

        if step_y > 0:
            t_max_y = (math.floor(start_fy) + 1.0 - start_fy) * t_delta_y
        elif step_y < 0:
            t_max_y = (start_fy - math.floor(start_fy)) * t_delta_y
        else:
            t_max_y = math.inf

        cells = []
        col = start_col
        row = start_row
        while self._in_bounds(col, row):
            cells.append((col, row))
            if col == end_col and row == end_row:
                break
            if t_max_x <= t_max_y:
                col += step_x
                t_max_x += t_delta_x
            else:
                row += step_y
                t_max_y += t_delta_y
        return cells

    def _raytrace_free_space(self, points: np.ndarray) -> None:
        robot_x = float(self.last_robot_pose["x"])
        robot_y = float(self.last_robot_pose["y"])
        robot_col, robot_row = self._world_to_grid(robot_x, robot_y)
        if not self._in_bounds(robot_col, robot_row):
            return

        free_counts: dict[int, int] = {robot_row * self.width + robot_col: 1}

        for point_x, point_y in points:
            dx = float(point_x) - robot_x
            dy = float(point_y) - robot_y
            if (dx * dx + dy * dy) > (self.max_raytrace_range_m * self.max_raytrace_range_m):
                continue

            cells = self._raytrace_cells(robot_x, robot_y, float(point_x), float(point_y))
            if len(cells) <= 1:
                continue

            for free_col, free_row in cells[:-1]:
                cell_id = free_row * self.width + free_col
                free_counts[cell_id] = free_counts.get(cell_id, 0) + 1

        if not free_counts:
            return

        free_ids = np.fromiter(free_counts.keys(), dtype=np.int32)
        free_rows = (free_ids // self.width).astype(np.int32)
        free_cols = (free_ids % self.width).astype(np.int32)
        free_hits = np.fromiter(free_counts.values(), dtype=np.int32)
        free_hits = np.minimum(free_hits, self.max_free_observations_per_cell_per_scan).astype(np.float32)
        free_delta = self.free_log_odds * free_hits
        self._apply_log_odds(free_rows, free_cols, free_delta)

    def _prune_isolated_occupied_cells(self) -> None:
        occupied = self._occupied_mask()
        if not np.any(occupied):
            return

        padded = np.pad(occupied.astype(np.uint8), 1, mode="constant")
        neighbor_count = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        )
        isolated = occupied & (neighbor_count < self.isolated_occupied_min_neighbors)
        self.log_odds_grid[isolated] = np.minimum(
            self.log_odds_grid[isolated],
            self.free_threshold_log_odds,
        )
        self.observed_mask[isolated] = True

    def _occupied_mask(self) -> np.ndarray:
        return self.observed_mask & (self.log_odds_grid >= self.occupied_threshold_log_odds)

    def _free_mask(self) -> np.ndarray:
        return self.observed_mask & (self.log_odds_grid <= self.free_threshold_log_odds)

    def _serialize_grid(self) -> np.ndarray:
        serialized = np.full((self.height, self.width), -1, dtype=np.int16)
        if not np.any(self.observed_mask):
            return serialized

        probabilities = log_odds_to_probability(self.log_odds_grid)
        serialized[self.observed_mask] = np.rint(probabilities[self.observed_mask] * 100.0).astype(np.int16)
        return serialized

    def _prune_stale_detections(self, now_sec: float | None = None) -> bool:
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9

        stale_ids = [
            instance_id
            for instance_id, entry in self.active_detections.items()
            if (now_sec - float(entry.get("last_detected_at", 0.0))) > self.detection_marker_ttl_sec
        ]
        for instance_id in stale_ids:
            self.active_detections.pop(instance_id, None)

        # Also prune stale DynoSAM object poses so they don't persist forever
        if self.object_odometry_ttl_sec > 0:
            for source_dict in (self.object_relative_positions, self.backend_object_relative_positions):
                stale_obj_ids = [
                    instance_id
                    for instance_id, pose in source_dict.items()
                    if (now_sec - float(pose.get("last_updated_at", 0.0))) > self.object_odometry_ttl_sec
                ]
                for instance_id in stale_obj_ids:
                    source_dict.pop(instance_id, None)
                if stale_obj_ids:
                    for instance_id, entry in self.active_detections.items():
                        frontend_valid = has_valid_relative_pose(
                            self.object_relative_positions.get(instance_id),
                            now_sec=now_sec,
                            ttl_sec=self.object_odometry_ttl_sec,
                        )
                        backend_valid = has_valid_relative_pose(
                            self.backend_object_relative_positions.get(instance_id),
                            now_sec=now_sec,
                            ttl_sec=self.object_odometry_ttl_sec,
                        )
                        if not frontend_valid and not backend_valid:
                            entry["has_dynosam_position"] = False
                    stale_ids.extend(stale_obj_ids)

        return bool(stale_ids)

    def _check_clear_signal(self) -> bool:
        try:
            if not self.clear_detections_signal_path.exists():
                return False
            mtime = self.clear_detections_signal_path.stat().st_mtime
            if mtime > self._last_clear_signal_mtime:
                self._last_clear_signal_mtime = mtime
                cleared_count = len(self.active_detections)
                self.active_detections.clear()
                self.object_relative_positions.clear()
                if cleared_count > 0:
                    self.get_logger().info(f"Cleared {cleared_count} active detection(s) via signal file")
                return True
        except Exception as exc:
            self.get_logger().warn(f"Failed to check clear signal: {exc}")
        return False

    @staticmethod
    def _compose_relative_pose_with_robot(relative_pose, robot_pose):
        if not isinstance(relative_pose, dict) or not isinstance(robot_pose, dict):
            return None

        rel_x = float(relative_pose.get("x", 0.0))
        rel_y = float(relative_pose.get("y", 0.0))
        rel_z = float(relative_pose.get("z", 0.0))

        if relative_pose.get("is_world_frame"):
            return (rel_x, rel_y, rel_z)

        robot_x = float(robot_pose.get("x", 0.0))
        robot_y = float(robot_pose.get("y", 0.0))
        robot_z = float(robot_pose.get("z", 0.0))
        robot_yaw = float(robot_pose.get("yaw", 0.0))

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        world_x = robot_x + (rel_x * cos_yaw) - (rel_y * sin_yaw)
        world_y = robot_y + (rel_x * sin_yaw) + (rel_y * cos_yaw)
        world_z = robot_z + rel_z
        return (world_x, world_y, world_z)

    def _transform_point_odom_to_map(
        self,
        point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """Transform a point from odom to map frame using the robot pose offset.

        Computes: obj_map = robot_map + R(map_yaw - odom_yaw) * (obj_odom - robot_odom)
        """
        robot_odom = self.last_robot_pose
        if robot_odom is None:
            return point

        try:
            with Path("/shared/slam_tab/robot_pose.json").open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return point

        pos = payload.get("position")
        orient = payload.get("orientation")
        if not isinstance(pos, dict) or not isinstance(orient, dict):
            return point

        map_x = float(pos.get("x", 0.0))
        map_y = float(pos.get("y", 0.0))
        map_z = float(pos.get("z", 0.0))
        qz = float(orient.get("z", 0.0))
        qw = float(orient.get("w", 1.0))
        map_yaw = math.atan2(2.0 * (qw * qz), 1.0 - 2.0 * (qz * qz))

        odom_x = float(robot_odom["x"])
        odom_y = float(robot_odom["y"])
        odom_z = float(robot_odom.get("z", 0.0))
        odom_yaw = float(robot_odom["yaw"])

        dx = point[0] - odom_x
        dy = point[1] - odom_y
        dz = point[2] - odom_z

        dyaw = map_yaw - odom_yaw
        cos_dyaw = math.cos(dyaw)
        sin_dyaw = math.sin(dyaw)

        return (
            map_x + cos_dyaw * dx - sin_dyaw * dy,
            map_y + sin_dyaw * dx + cos_dyaw * dy,
            map_z + dz,
        )

    def _build_detection_overlay(self):
        self._prune_stale_detections()
        overlays = []
        for instance_id, entry in sorted(self.active_detections.items()):
            x = entry.get("x")
            y = entry.get("y")
            if x is None or y is None:
                continue

            x, y, z = self._transform_point_odom_to_map(
                (float(x), float(y), float(entry.get("z", 0.0)))
            )

            overlays.append(
                {
                    "instance_id": int(instance_id),
                    "track_id": int(entry.get("track_id", -1)),
                    "class_id": int(entry.get("class_id", -1)),
                    "class_name": entry.get("class_name") or "",
                    "score": float(entry.get("score", 0.0)),
                    "frame_id": "map",
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "has_dynosam_position": bool(entry.get("has_dynosam_position", False)),
                    "has_depth_position": bool(entry.get("has_depth_position", False)),
                    "position_source": entry.get("position_source"),
                    "last_detected_at": float(entry.get("last_detected_at", 0.0)),
                    "dynosam_source": entry.get("dynosam_source"),
                }
            )
        return overlays

    def _write_snapshot(self) -> None:
        frame_id = "world" if self.last_frame_id in {"odom", "spot/odom", "spot/vision"} else self.last_frame_id
        serialized_grid = self._serialize_grid()
        occupied_cells = int(np.count_nonzero(self._occupied_mask()))
        free_cells = int(np.count_nonzero(self._free_mask()))
        observed_cells = int(np.count_nonzero(self.observed_mask))
        detections = self._build_detection_overlay()
        payload = {
            "generated_at": time.time(),
            "map": {
                "frame_id": frame_id,
                "resolution": self.resolution,
                "width": self.width,
                "height": self.height,
                "origin": {
                    "x": self.origin_x,
                    "y": self.origin_y,
                },
                "data": serialized_grid.reshape(-1).astype(int).tolist(),
            },
            "robot": self.last_robot_pose,
            "detections": detections,
            "stats": {
                "scans_processed": self.scan_count,
                "observed_cells": observed_cells,
                "free_cells": free_cells,
                "occupied_cells": occupied_cells,
                "unknown_cells": int((self.width * self.height) - observed_cells),
                "active_detections": len(detections),
                "thresholds": {
                    "free_probability_leq": self.free_threshold,
                    "occupied_probability_geq": self.occupied_threshold,
                },
                "bounds": {
                    "min_x": self.origin_x,
                    "max_x": self.origin_x + self.width * self.resolution,
                    "min_y": self.origin_y,
                    "max_y": self.origin_y + self.height * self.resolution,
                },
            },
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=self.output_path.parent, encoding="utf-8") as tmp:
            json.dump(payload, tmp)
            temp_path = tmp.name
        os.replace(temp_path, self.output_path)
        self._publish_occupancy_grid(frame_id)

    def _publish_occupancy_grid(self, frame_id: str) -> None:
        if self.occupancy_grid_pub is None:
            return

        serialized_grid = self._serialize_grid()
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.info.map_load_time = msg.header.stamp
        msg.info.resolution = float(self.resolution)
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = float(self.origin_x)
        msg.info.origin.position.y = float(self.origin_y)
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = serialized_grid.reshape(-1).astype(np.int8).tolist()
        self.occupancy_grid_pub.publish(msg)


def main() -> None:
    rclpy.init()
    node = LidarOccupancyMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
