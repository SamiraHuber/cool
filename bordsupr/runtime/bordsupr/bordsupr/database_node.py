#!/usr/bin/env python3

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import datetime
import json
import re
import threading
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SLAM_TAB_ROBOT_POSE_PATH = Path(os.getenv("SLAM_TAB_ROBOT_POSE_PATH", "/shared/slam_tab/robot_pose.json"))

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.parameter import Parameter

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage
from dynamic_slam_interfaces.msg import ObjectOdometry
from tf2_ros import Buffer, TransformListener

from bordsupr_interfaces.srv import GetImageEmbedding
from bordsupr_interfaces.msg import (
    FaceOutput,
    InteractionOutput,
    StampedString,
    YoloOutput,
)

import psycopg2
from cv_bridge import CvBridge
import cv2
import numpy as np
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data

from .text_embeddings import embed_text, get_text_embedding_dim
from .visual_attributes import AttributeConfig, extract_visual_attributes, resize_for_embedding


TOOLBOX_MAP_ACTIVE_PATH = Path(
    os.getenv("TOOLBOX_MAP_ACTIVE_PATH", "/shared/maps/toolbox_saved/active.json")
)

@dataclass
class PendingEmbeddingRequest:
    request_id: int
    image_msg: Image
    stamp_key: Tuple[int, int]
    source: str = ""
    class_id: int = 0


@dataclass
class PendingInteractionRecord:
    stamp_key: Tuple[int, int]
    action: str
    caption: str
    model_source: str = ""
    raw_response: str = ""
    refs: List[Dict[str, Any]] = field(default_factory=list)
    attempts: int = 0
    map_id: Optional[int] = None
    position_source: Optional[str] = None
    confidence: float = 0.0


@dataclass
class PendingObservationRecord:
    object_id: int
    cropped_image: bytes
    mask_image: bytes
    original_cropped_image: bytes
    embedding: list[float]
    scene_id: Optional[int]
    yolo_track_id: Optional[str]
    class_id: Optional[int]
    stamp_key: Tuple[int, int]
    dynosam_instance_id: Optional[int]
    bbox_x_min: Optional[int]
    bbox_y_min: Optional[int]
    bbox_x_max: Optional[int]
    bbox_y_max: Optional[int]
    estimated_position: Optional[Tuple[float, float, float]]
    robot_position: Optional[Tuple[float, float, float]]
    created_wall_time_sec: float
    attempts: int = 0
    map_id: Optional[int] = None
    confidence: Optional[float] = None
    attributes_json: Optional[Dict[str, Any]] = None
    quality_score: Optional[float] = None
    part_images: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    part_embeddings: Dict[str, list[float]] = field(default_factory=dict)
    part_preprocessing: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    estimated_position_frame: str = "odom"
    detection_backend: Optional[str] = None
    embedding_backend: Optional[str] = None


@dataclass
class MatchCandidate:
    object_id: int
    score: float
    avg_similarity: float
    best_similarity: float
    hit_count: int


@dataclass
class FaceMatchCandidate:
    person_id: int
    similarity: float


class DatabaseNode(Node):

    def _load_active_observation_map_name(self) -> Optional[str]:
        try:
            if not TOOLBOX_MAP_ACTIVE_PATH.exists():
                return None
            with TOOLBOX_MAP_ACTIVE_PATH.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        if not bool(payload.get("recording_enabled", True)):
            return None
        try:
            map_open_until = float(payload.get("map_open_until") or 0.0)
        except (TypeError, ValueError):
            map_open_until = 0.0
        if map_open_until > 0 and map_open_until < time.time():
            return None
        mode = str(payload.get("mode") or "recording").strip().lower()
        if mode == "frozen":
            value = payload.get("name")
        else:
            value = payload.get("observation_map_name") or payload.get("name")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._-")
        return text or None

    def _ensure_map_exists(self, map_name: Optional[str]) -> Optional[int]:
        normalized_name = str(map_name or "").strip()
        if not normalized_name:
            return None
        row = self._db_execute(
            """
            INSERT INTO maps (name)
            VALUES (%s)
            ON CONFLICT (name)
            DO UPDATE SET name = EXCLUDED.name
            RETURNING id
            """,
            (normalized_name,),
            fetchone=True,
        )
        return int(row[0]) if row is not None else None

    def _get_active_map_id(self) -> Optional[int]:
        map_name = self._load_active_observation_map_name()
        if not map_name:
            return None
        return self._ensure_map_exists(map_name)

    @staticmethod
    def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _parse_dynosam_object_id(child_frame_id: str) -> Optional[int]:
        if not child_frame_id.startswith("object_") or not child_frame_id.endswith("_link"):
            return None
        object_id_str = child_frame_id[len("object_"):-len("_link")]
        if not object_id_str.isdigit():
            return None
        return int(object_id_str)

    @staticmethod
    def _to_vector_literal(values: list[float]) -> str:
        return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"

    def _embed_scene_caption(self, caption: str) -> list[float] | None:
        return embed_text(caption)

    def _refresh_embedding_dim(self) -> None:
        row = self._db_execute(
            """
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'object_observations'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1
            """,
            fetchone=True,
        )
        if row is None:
            self._db_embedding_dim = None
            return

        type_text = str(row[0])
        match = re.match(r"vector\((\d+)\)", type_text)
        if match is None:
            self._db_embedding_dim = None
            self.get_logger().warning(
                f"Could not parse embedding column type '{type_text}'. "
                "Embedding dimension coercion disabled."
            )
            return

        self._db_embedding_dim = int(match.group(1))
        self.get_logger().info(
            f"Detected object_observations.embedding dimension: {self._db_embedding_dim}"
        )

    def _refresh_face_embedding_dim(self) -> None:
        row = self._db_execute(
            """
            SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            WHERE c.relname = 'face_observations'
              AND a.attname = 'embedding'
              AND a.attnum > 0
              AND NOT a.attisdropped
            LIMIT 1
            """,
            fetchone=True,
        )
        if row is None:
            self._db_face_embedding_dim = None
            return

        type_text = str(row[0])
        match = re.match(r"vector\((\d+)\)", type_text)
        if match is None:
            self._db_face_embedding_dim = None
            self.get_logger().warning(
                f"Could not parse face embedding column type '{type_text}'. "
                "Face embedding dimension coercion disabled."
            )
            return

        self._db_face_embedding_dim = int(match.group(1))
        self.get_logger().info(
            f"Detected face_observations.embedding dimension: {self._db_face_embedding_dim}"
        )

    def _coerce_embedding_dim(self, embedding: list[float]) -> list[float]:
        if self._db_embedding_dim is None or self._db_embedding_dim <= 0:
            return [float(v) for v in embedding]

        target_dim = int(self._db_embedding_dim)
        current_dim = len(embedding)
        vector = [float(v) for v in embedding]

        if current_dim == target_dim:
            return vector

        if not self._embedding_dim_warning_emitted:
            self.get_logger().warning(
                f"Embedding dimension mismatch: model produced {current_dim}, "
                f"database expects {target_dim}. Coercing vector length to match DB schema."
            )
            self._embedding_dim_warning_emitted = True

        if current_dim > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - current_dim)

    def _coerce_face_embedding_dim(self, embedding: list[float]) -> list[float]:
        if self._db_face_embedding_dim is None or self._db_face_embedding_dim <= 0:
            return [float(v) for v in embedding]

        target_dim = int(self._db_face_embedding_dim)
        current_dim = len(embedding)
        vector = [float(v) for v in embedding]

        if current_dim == target_dim:
            return vector

        if not self._face_embedding_dim_warning_emitted:
            self.get_logger().warning(
                f"Face embedding dimension mismatch: model produced {current_dim}, "
                f"database expects {target_dim}. Coercing vector length to match DB schema."
            )
            self._face_embedding_dim_warning_emitted = True

        if current_dim > target_dim:
            return vector[:target_dim]
        return vector + [0.0] * (target_dim - current_dim)

    def _connect_db(self) -> None:
        self.db_conn = psycopg2.connect(
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
            host=self.db_host,
            port=self.db_port,
        )
        self.db_conn.autocommit = True
        self.db_cursor = self.db_conn.cursor()
        self.get_logger().info(
            f"Connected to database {self.db_name} at {self.db_host}:{self.db_port} as {self.db_user}"
        )
        self._ensure_attribute_schema()
        self._ensure_object_embedding_schema()
        self._refresh_embedding_dim()
        self._refresh_face_embedding_dim()

    def _ensure_attribute_schema(self) -> None:
        with self.db_conn.cursor() as cur:
            cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION")
            cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION")
            cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS attributes_json JSONB")
            # 1 face = 1 person observation: link each face to its specific class-0
            # object_observations row. NULL until matched; SET NULL if the obs is deleted.
            cur.execute("ALTER TABLE face_observations ADD COLUMN IF NOT EXISTS observation_id BIGINT")
            cur.execute(
                """
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint WHERE conname = 'face_observations_observation_id_fkey'
                    ) THEN
                        ALTER TABLE face_observations
                            ADD CONSTRAINT face_observations_observation_id_fkey
                            FOREIGN KEY (observation_id) REFERENCES object_observations(id) ON DELETE SET NULL;
                    END IF;
                END$$;
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_face_observations_observation_id ON face_observations(observation_id)")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS object_observation_parts (
                    id BIGSERIAL PRIMARY KEY,
                    observation_id BIGINT REFERENCES object_observations(id) ON DELETE CASCADE,
                    object_id BIGINT REFERENCES objects(id) ON DELETE CASCADE,
                    part_name TEXT NOT NULL,
                    embedding VECTOR(384),
                    colors_json JSONB,
                    bbox_x_min BIGINT,
                    bbox_y_min BIGINT,
                    bbox_x_max BIGINT,
                    bbox_y_max BIGINT,
                    quality_score DOUBLE PRECISION,
                    preprocessing_json JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_object_observation_parts_observation
                ON object_observation_parts (observation_id)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_object_observation_parts_object_part
                ON object_observation_parts (object_id, part_name)
                """
            )
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_object_observation_parts_unique_part
                ON object_observation_parts (observation_id, part_name)
                """
            )

    def _ensure_object_embedding_schema(self) -> None:
        desired_dim = 512
        with self.db_conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
                FROM pg_attribute a
                JOIN pg_class c ON a.attrelid = c.oid
                WHERE c.relname = 'object_observations'
                  AND a.attname = 'embedding'
                  AND a.attnum > 0
                  AND NOT a.attisdropped
                LIMIT 1
                """
            )
            row = cur.fetchone()
            type_text = str(row[0]) if row is not None else ""
            if type_text == f"vector({desired_dim})":
                return

            cur.execute("SELECT COUNT(*) FROM object_observations")
            count_row = cur.fetchone()
            observation_count = int(count_row[0]) if count_row is not None else 0
            if observation_count > 0:
                self.get_logger().info(
                    f"Migrating {observation_count} existing object embeddings from "
                    f"{type_text or 'unknown'} to vector({desired_dim})"
                )
                cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS embedding_512_tmp VECTOR(512)")
                cur.execute("SELECT id, embedding::text FROM object_observations WHERE embedding_512_tmp IS NULL")
                rows = cur.fetchall()
                for observation_id, embedding_text in rows:
                    vector_text = str(embedding_text or "").strip().strip("[]")
                    values = [float(part) for part in vector_text.split(",") if part.strip()]
                    if len(values) > desired_dim:
                        values = values[:desired_dim]
                    elif len(values) < desired_dim:
                        values = values + [0.0] * (desired_dim - len(values))
                    cur.execute(
                        "UPDATE object_observations SET embedding_512_tmp = %s::vector WHERE id = %s",
                        (self._to_vector_literal(values), observation_id),
                    )
                cur.execute("DROP INDEX IF EXISTS idx_object_observations_embedding_hnsw")
                cur.execute("ALTER TABLE object_observations DROP COLUMN embedding")
                cur.execute("ALTER TABLE object_observations RENAME COLUMN embedding_512_tmp TO embedding")
                cur.execute("ALTER TABLE object_observations ALTER COLUMN embedding SET NOT NULL")
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_object_observations_embedding_hnsw
                    ON object_observations
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
                return

            self.get_logger().info(
                f"Updating empty object_observations.embedding column from {type_text or 'unknown'} "
                f"to vector({desired_dim}) for OSNet live clustering"
            )
            cur.execute("DROP INDEX IF EXISTS idx_object_observations_embedding_hnsw")
            cur.execute(
                f"""
                ALTER TABLE object_observations
                ALTER COLUMN embedding TYPE VECTOR({desired_dim})
                USING embedding::vector({desired_dim})
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_object_observations_embedding_hnsw
                ON object_observations
                USING hnsw (embedding vector_cosine_ops)
                """
            )

    def _ensure_db_connection(self) -> None:
        needs_reconnect = (
            not hasattr(self, "db_conn")
            or self.db_conn is None
            or getattr(self.db_conn, "closed", 1) != 0
            or not hasattr(self, "db_cursor")
            or self.db_cursor is None
            or getattr(self.db_cursor, "closed", True)
        )
        if needs_reconnect:
            self.get_logger().warn("Database connection not ready. Reconnecting...")
            self._connect_db()

    def _reset_db_connection_locked(self) -> None:
        """Close and drop the current connection/cursor. Caller must hold _db_lock."""
        try:
            if getattr(self, "db_cursor", None) is not None:
                self.db_cursor.close()
        except Exception:
            pass
        try:
            if getattr(self, "db_conn", None) is not None:
                self.db_conn.close()
        except Exception:
            pass
        self.db_conn = None
        self.db_cursor = None

    def _db_execute(
        self,
        sql: str,
        params=None,
        fetchone: bool = False,
        return_rowcount: bool = False,
    ):
        if fetchone and return_rowcount:
            raise ValueError("fetchone and return_rowcount cannot both be enabled")
        # Retry on connection-level failures (Postgres restart/kill) until the
        # deadline, reconnecting each time. Short restarts are ridden out with
        # no data loss; longer outages raise and are caught by the spin loop in
        # main() so the node stays alive and resumes when the DB returns.
        deadline = time.monotonic() + float(getattr(self, "_db_retry_timeout_sec", 15.0))
        last_exc = None
        with self._db_lock:
            attempt = 0
            while True:
                attempt += 1
                try:
                    self._ensure_db_connection()
                    self.db_cursor.execute(sql, params)
                    if fetchone:
                        return self.db_cursor.fetchone()
                    if return_rowcount:
                        return int(getattr(self.db_cursor, "rowcount", 0))
                    return None
                except (psycopg2.InterfaceError, psycopg2.OperationalError) as exc:
                    last_exc = exc
                    self._reset_db_connection_locked()
                    if time.monotonic() >= deadline:
                        break
                    self.get_logger().warn(
                        f"DB execute failed (attempt {attempt}): {exc}. Reconnecting..."
                    )
                    time.sleep(min(0.5 * attempt, 2.0))
        if last_exc is not None:
            raise last_exc

    def _db_fetchall(self, sql: str, params=None):
        deadline = time.monotonic() + float(getattr(self, "_db_retry_timeout_sec", 15.0))
        last_exc = None
        with self._db_lock:
            attempt = 0
            while True:
                attempt += 1
                try:
                    self._ensure_db_connection()
                    self.db_cursor.execute(sql, params)
                    return self.db_cursor.fetchall()
                except (psycopg2.InterfaceError, psycopg2.OperationalError) as exc:
                    last_exc = exc
                    self._reset_db_connection_locked()
                    if time.monotonic() >= deadline:
                        break
                    self.get_logger().warn(
                        f"DB fetchall failed (attempt {attempt}): {exc}. Reconnecting..."
                    )
                    time.sleep(min(0.5 * attempt, 2.0))
        if last_exc is not None:
            raise last_exc

    @staticmethod
    def _stamp_key(msg) -> Tuple[int, int]:
        """
        Extract (sec, nanosec) from a ROS message or Header.
        Accepts:
        - msg with .header.stamp
        - msg with .stamp
        """
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            return (int(msg.header.stamp.sec), int(msg.header.stamp.nanosec))
        elif hasattr(msg, 'stamp'):
            return (int(msg.stamp.sec), int(msg.stamp.nanosec))
        else:
            raise AttributeError("Object does not have a ROS Header or stamp")

    def _compute_scene_thumb(self, rgb_msg: Optional[Image]) -> Optional[np.ndarray]:
        if rgb_msg is None:
            return None
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
            thumb = cv2.GaussianBlur(thumb, (3, 3), 0.5)
            return thumb.astype(np.int16)
        except Exception as exc:
            self.get_logger().warning(f"Failed to compute scene thumb: {exc}")
            return None

    def _should_save_scene(
        self, thumb: Optional[np.ndarray]
    ) -> Tuple[bool, float, float, bool]:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        if self.scene_save_mode == "interval":
            if self._last_scene_save_time <= 0.0:
                return True, 0.0, 0.0, False
            should_save = (now_sec - self._last_scene_save_time) >= self.scene_force_save_interval_sec
            return should_save, 0.0, 0.0, should_save
        if not self.scene_dedup_enabled or thumb is None:
            return True, 0.0, 0.0, False
        if self._last_saved_scene_thumb is None:
            return True, 0.0, 0.0, False
        if (now_sec - self._last_scene_save_time) >= self.scene_force_save_interval_sec:
            return True, 0.0, 0.0, True
        diff = cv2.absdiff(thumb, self._last_saved_scene_thumb)
        mean_abs_diff = float(np.mean(diff))
        changed_pixels = int(np.sum(diff > self.scene_pixel_change_threshold))
        changed_pixel_ratio = changed_pixels / float(thumb.size)
        should_save = (
            mean_abs_diff >= self.scene_min_mean_abs_diff
            or changed_pixel_ratio >= self.scene_min_changed_pixel_ratio
        )
        return should_save, mean_abs_diff, changed_pixel_ratio, False

    def _compute_crop_thumb(self, crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Small grayscale thumbnail of a detection crop, for per-track diff."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
            thumb = cv2.GaussianBlur(thumb, (3, 3), 0.5)
            return thumb.astype(np.int16)
        except Exception:
            return None

    def _detect_track_content_change(
        self,
        yolo_track_id: Optional[str],
        crop_bgr: Optional[np.ndarray],
    ) -> bool:
        """Detect a sharp change in a track's crop (scene cut / person swap).

        Compares the current detection crop to the previous frame's crop of the SAME
        track. A hard video cut that swaps the tracked subject yields a large pixel
        diff; normal motion of the same subject yields a small one. Returns True when
        the content changed sharply (the track id was likely reused across a cut).

        Always refreshes the stored thumbnail for the track so the next frame compares
        against the most recent crop.
        """
        if yolo_track_id is None:
            return False
        thumb = self._compute_crop_thumb(crop_bgr)
        if thumb is None:
            return False
        prev = self._crop_thumb_by_track_id.get(yolo_track_id)
        self._crop_thumb_by_track_id[yolo_track_id] = thumb
        if prev is None:
            return False
        if not getattr(self, "track_content_change_enabled", True):
            return False
        try:
            diff = cv2.absdiff(thumb, prev)
            mean_abs_diff = float(np.mean(diff))
            changed_pixels = int(np.sum(diff > self.track_content_change_pixel_threshold))
            changed_pixel_ratio = changed_pixels / float(thumb.size)
            changed = (
                mean_abs_diff >= self.track_content_change_mean_abs_diff
                or changed_pixel_ratio >= self.track_content_change_changed_pixel_ratio
            )
            if changed:
                self.get_logger().info(
                    f"Track {yolo_track_id} content changed sharply "
                    f"(mean_abs_diff={mean_abs_diff:.2f}, changed_ratio={changed_pixel_ratio:.3f}); "
                    f"likely scene cut / subject swap"
                )
            return changed
        except Exception as exc:
            self.get_logger().warning(f"Track content-change detection failed for {yolo_track_id}: {exc}")
            return False

    def _cache_rgb_image(self, msg: Image) -> None:
        stamp_key = self._stamp_key(msg)
        self._latest_rgb_by_stamp[stamp_key] = msg
        self._latest_rgb_by_stamp.move_to_end(stamp_key)
        self._prune_old_rgb_entries(current_stamp=stamp_key)

    def _cache_corrected_rgb_image(self, msg: Image) -> None:
        stamp_key = self._stamp_key(msg)
        self._latest_corrected_rgb_by_stamp[stamp_key] = msg
        self._latest_corrected_rgb_by_stamp.move_to_end(stamp_key)
        self._prune_old_corrected_rgb_entries(current_stamp=stamp_key)

    def _cache_depth_image(self, msg: Image) -> None:
        stamp_key = self._stamp_key(msg)
        self._latest_depth_by_stamp[stamp_key] = msg
        self._prune_old_stamped_entries(self._latest_depth_by_stamp, current_stamp=stamp_key)
        self._prune_old_stamped_entries(self._cached_camera_transforms, current_stamp=stamp_key)
        self._cache_camera_transform(stamp_key, msg.header.frame_id)

    def _cache_camera_info(self, msg: CameraInfo) -> None:
        stamp_key = self._stamp_key(msg)
        self._latest_camera_info_by_stamp[stamp_key] = msg
        self._prune_old_stamped_entries(self._latest_camera_info_by_stamp, current_stamp=stamp_key)

    def _cache_camera_transform(self, stamp_key: Tuple[int, int], frame_id: str) -> None:
        """Cache the camera-to-odom transform at image capture time.

        This prevents using the robot's current position when OWLv2 inference
        takes longer than the TF buffer cache TTL.
        """
        if not frame_id:
            return
        if stamp_key in self._cached_camera_transforms:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                "spot/vision",
                frame_id,
                rclpy.time.Time(seconds=int(stamp_key[0]), nanoseconds=int(stamp_key[1])),
                timeout=Duration(seconds=0.5),
            )
            self._cached_camera_transforms[stamp_key] = transform
        except Exception:
            pass

    def _prune_old_rgb_entries(self, current_stamp: Tuple[int, int]) -> None:
        self._prune_old_stamped_entries(self._latest_rgb_by_stamp, current_stamp)

    def _prune_old_corrected_rgb_entries(self, current_stamp: Tuple[int, int]) -> None:
        self._prune_old_stamped_entries(self._latest_corrected_rgb_by_stamp, current_stamp)

    def _prune_old_stitched_rgb_entries(self, current_stamp: Tuple[int, int]) -> None:
        self._prune_old_stamped_entries(self._latest_stitched_rgb_by_stamp, current_stamp)

    def _cache_stitched_rgb_image(self, msg: Image) -> None:
        stamp_key = self._stamp_key(msg)
        self._latest_stitched_rgb_by_stamp[stamp_key] = msg
        self._latest_stitched_rgb_by_stamp.move_to_end(stamp_key)
        self._prune_old_stitched_rgb_entries(current_stamp=stamp_key)

    def _lookup_stitched_rgb_image_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[Image]:
        # Try exact/near timestamp match first
        matched = self._lookup_stamped_message(self._latest_stitched_rgb_by_stamp, stamp_key)
        if matched is not None:
            return matched
        # Fallback: return the most recent stitched image regardless of timestamp
        # (stitched and frontleft are from different nodes with different timestamps)
        if self._latest_stitched_rgb_by_stamp:
            return next(reversed(self._latest_stitched_rgb_by_stamp.values()))
        return None

    def _prune_old_stamped_entries(
        self,
        cache: Dict[Tuple[int, int], Any],
        current_stamp: Tuple[int, int],
    ) -> None:
        now_sec = self._stamp_key_to_seconds(current_stamp)
        # Snapshot keys to avoid "dictionary changed size during iteration"
        try:
            keys_snapshot = list(cache.keys())
        except RuntimeError:
            keys_snapshot = []
        stale_keys = [
            key for key in keys_snapshot
            if (now_sec - self._stamp_key_to_seconds(key)) > self._rgb_cache_ttl_sec
        ]
        for key in stale_keys:
            cache.pop(key, None)
        while len(cache) > self._rgb_cache_max_entries:
            cache.popitem(last=False)

    def _lookup_rgb_image_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[Image]:
        return self._lookup_stamped_message(self._latest_rgb_by_stamp, stamp_key)

    def _lookup_corrected_rgb_image_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[Image]:
        return self._lookup_stamped_message(self._latest_corrected_rgb_by_stamp, stamp_key)

    def _lookup_depth_image_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[Image]:
        return self._lookup_stamped_message(self._latest_depth_by_stamp, stamp_key)

    def _lookup_camera_info_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[CameraInfo]:
        return self._lookup_stamped_message(self._latest_camera_info_by_stamp, stamp_key)

    def _lookup_stamped_message(
        self,
        cache: Dict[Tuple[int, int], Any],
        stamp_key: Tuple[int, int],
    ) -> Optional[Any]:
        exact = cache.get(stamp_key)
        if exact is not None:
            return exact

        target_sec = self._stamp_key_to_seconds(stamp_key)
        best_key = None
        best_delta = None
        for key in cache:
            delta = abs(target_sec - self._stamp_key_to_seconds(key))
            if delta > self._rgb_match_window_sec:
                continue
            if best_delta is None or delta < best_delta:
                best_key = key
                best_delta = delta
        if best_key is None:
            return None
        return cache.get(best_key)

    def _extract_original_crop_bytes(
        self,
        rgb_msg: Optional[Image],
        x_min: Optional[int],
        y_min: Optional[int],
        x_max: Optional[int],
        y_max: Optional[int],
    ) -> bytes:
        if rgb_msg is None:
            return b""
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert RGB for original crop: {exc}")
            return b""
        h, w = bgr.shape[:2]
        x1 = max(0, min(int(x_min or 0), w))
        y1 = max(0, min(int(y_min or 0), h))
        x2 = max(0, min(int(x_max or 0), w))
        y2 = max(0, min(int(y_max or 0), h))
        if x2 <= x1 or y2 <= y1:
            return b""
        crop = bgr[y1:y2, x1:x2]
        ok, buf = cv2.imencode(".jpg", crop)
        return buf.tobytes() if ok else b""

    def _encode_scene_image(self, bgr) -> bytes:
        """Downscale (longest side -> scene_image_max_side, 0 = no resize) and JPEG-encode
        at scene_image_jpeg_quality, to keep stored scene images small."""
        max_side = int(getattr(self, "scene_image_max_side", 0) or 0)
        if max_side > 0 and bgr is not None:
            h, w = bgr.shape[:2]
            longest = max(h, w)
            if longest > max_side:
                scale = max_side / float(longest)
                bgr = cv2.resize(bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
        quality = int(getattr(self, "scene_image_jpeg_quality", 80) or 80)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return encoded.tobytes() if ok else b""

    def _build_original_scene_image_bytes(self, rgb_msg: Optional[Image]) -> bytes:
        """Encode the raw, unannotated camera frame as JPEG (no bbox overlay).

        This is the true "original" scene image. _build_scene_image_bytes draws
        YOLO detection boxes/labels; the original must be the clean frame.
        """
        if rgb_msg is None:
            return b""
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert RGB image for original scene image: {exc}")
            return b""
        data = self._encode_scene_image(bgr)
        if not data:
            self.get_logger().error("Failed to encode original scene image")
        return data

    def _build_stitched_scene_image_bytes(self, rgb_msg: Optional[Image]) -> bytes:
        """Encode the stitched front-middle image as JPEG (no bbox overlay)."""
        if rgb_msg is None:
            return b""
        try:
            bgr = self.cv_bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"Failed to convert stitched RGB image: {exc}")
            return b""
        # Rotate 90° clockwise to match frontleft orientation
        bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
        return self._encode_scene_image(bgr)

    def _cache_object_transform(self, msg: TFMessage) -> None:
        saw_object_transform = False
        for transform in msg.transforms:
            object_id = self._parse_dynosam_object_id(transform.child_frame_id)
            if object_id is None:
                continue

            saw_object_transform = True
            stamp_key = (
                int(transform.header.stamp.sec),
                int(transform.header.stamp.nanosec),
            )
            translation = transform.transform.translation
            parent_frame = str(getattr(getattr(transform, "header", None), "frame_id", "") or "")
            is_world_frame = parent_frame in {"world", "map"}
            entry = (
                self._stamp_key_to_seconds(stamp_key),
                stamp_key,
                (float(translation.x), float(translation.y), float(translation.z)),
                is_world_frame,
            )
            self._object_pose_history.setdefault(object_id, []).append(entry)

        if saw_object_transform:
            self._prune_old_object_pose_entries()
            self._flush_pending_observations()

    def _cache_object_odometry(self, msg: ObjectOdometry) -> None:
        stamp_key = (
            int(msg.odom.header.stamp.sec),
            int(msg.odom.header.stamp.nanosec),
        )
        pose = msg.odom.pose.pose.position
        frame_id = str(getattr(getattr(getattr(msg, "odom", None), "header", None), "frame_id", "") or "")
        is_world_frame = frame_id in {"world", "map"}
        entry = (
            self._stamp_key_to_seconds(stamp_key),
            stamp_key,
            (float(pose.x), float(pose.y), float(pose.z)),
            is_world_frame,
        )
        self._object_pose_history.setdefault(int(msg.object_id), []).append(entry)
        self._prune_old_object_pose_entries()
        self._flush_pending_observations()

    def _prune_old_object_pose_entries(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        cutoff_sec = now_sec - self._object_pose_cache_ttl_sec
        stale_object_ids = []
        for object_id, entries in self._object_pose_history.items():
            kept_entries = [entry for entry in entries if entry[0] >= cutoff_sec]
            if kept_entries:
                self._object_pose_history[object_id] = kept_entries
            else:
                stale_object_ids.append(object_id)
        for object_id in stale_object_ids:
            self._object_pose_history.pop(object_id, None)

    def _lookup_object_position_for_stamp(
        self,
        dynosam_instance_id: Optional[int],
        stamp_key: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float]]:
        if dynosam_instance_id is None:
            return None

        entries = self._object_pose_history.get(int(dynosam_instance_id))
        if not entries:
            return None

        target_sec = self._stamp_key_to_seconds(stamp_key)
        best_position = None
        best_delta = None
        best_is_world_frame = False
        for entry in entries:
            pose_sec = entry[0]
            position = entry[2]
            is_world_frame = entry[3] if len(entry) > 3 else False
            delta = abs(target_sec - pose_sec)
            if delta > self._object_pose_match_window_sec:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_position = position
                best_is_world_frame = is_world_frame

        if best_position is None:
            # No entry within the timestamp window — fall back to the most recent
            # known pose for this instance (still within the cache TTL).
            newest = max(entries, key=lambda e: e[0])
            best_position = newest[2]
            best_is_world_frame = newest[3] if len(newest) > 3 else False

        robot_position = self._lookup_robot_position_for_stamp(stamp_key)
        if robot_position is None:
            return None

        return self._compose_relative_pose_with_robot(best_position, robot_position, best_is_world_frame)

    def _cache_robot_pose(self, msg: Odometry) -> None:
        stamp_key = self._stamp_key(msg)
        pose = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        frame_id = getattr(msg.header, "frame_id", "") or "world"
        normalized_frame_id = "world" if frame_id in {"odom", "spot/odom", "spot/vision"} else frame_id
        entry = (
            self._stamp_key_to_seconds(stamp_key),
            stamp_key,
            (
                float(pose.x),
                float(pose.y),
                float(pose.z),
                normalized_frame_id,
                self._quaternion_to_yaw(
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
            ),
        )
        self._robot_pose_history.append(entry)
        self._latest_robot_pose = {
            "frame_id": normalized_frame_id,
            "x": float(pose.x),
            "y": float(pose.y),
            "z": float(pose.z),
            "yaw": self._quaternion_to_yaw(
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            ),
        }
        self._prune_old_robot_pose_entries()

    def _prune_old_robot_pose_entries(self) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        cutoff_sec = now_sec - self._robot_pose_cache_ttl_sec
        self._robot_pose_history = [
            entry for entry in self._robot_pose_history
            if entry[0] >= cutoff_sec
        ]

    def _lookup_robot_position_for_stamp(
        self,
        stamp_key: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, str, float]]:
        self._prune_old_robot_pose_entries()
        if not self._robot_pose_history:
            latest = self._latest_robot_pose
            if latest is None:
                return None
            return (
                float(latest["x"]),
                float(latest["y"]),
                float(latest.get("z", 0.0)),
                str(latest.get("frame_id", "world")),
                float(latest.get("yaw", 0.0)),
            )

        target_sec = self._stamp_key_to_seconds(stamp_key)
        best_position = None
        best_delta = None
        for pose_sec, _, position in self._robot_pose_history:
            delta = abs(target_sec - pose_sec)
            if delta > self._robot_pose_match_window_sec:
                continue
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best_position = position

        if best_position is not None:
            return best_position

        latest = self._latest_robot_pose
        if latest is None:
            return None
        return (
            float(latest["x"]),
            float(latest["y"]),
            float(latest.get("z", 0.0)),
            str(latest.get("frame_id", "world")),
            float(latest.get("yaw", 0.0)),
        )

    def _get_slam_robot_pose(self) -> Optional[Tuple[float, float, float, float]]:
        try:
            with SLAM_TAB_ROBOT_POSE_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        pos = payload.get("position")
        orient = payload.get("orientation")
        if not isinstance(pos, dict) or not isinstance(orient, dict):
            return None
        x = float(pos.get("x", 0.0))
        y = float(pos.get("y", 0.0))
        z = float(pos.get("z", 0.0))
        qx = float(orient.get("x", 0.0))
        qy = float(orient.get("y", 0.0))
        qz = float(orient.get("z", 0.0))
        qw = float(orient.get("w", 1.0))
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return (x, y, z, yaw)

    def _compose_relative_pose_with_robot(
        self,
        relative_position: Tuple[float, float, float],
        robot_position: Tuple[float, float, float, str, float],
        is_world_frame: bool = False,
    ) -> Tuple[float, float, float]:
        if is_world_frame:
            return relative_position
        rel_x, rel_y, rel_z = relative_position
        robot_x, robot_y, robot_z, _, robot_yaw = robot_position
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        world_x = robot_x + (rel_x * cos_yaw) - (rel_y * sin_yaw)
        world_y = robot_y + (rel_x * sin_yaw) + (rel_y * cos_yaw)
        world_z = robot_z + rel_z
        return (float(world_x), float(world_y), float(world_z))

    @staticmethod
    def _quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
        return np.array(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def _transform_point_to_world(
        self,
        source_frame: str,
        stamp_key: Tuple[int, int],
        point_camera: np.ndarray,
    ) -> Optional[Tuple[float, float, float]]:
        # 1. Prefer the transform cached at image capture time (before OWLv2 delay)
        transform = self._cached_camera_transforms.get(stamp_key)
        if transform is None:
            # 2. Fallback to TF buffer lookup at the exact image timestamp
            try:
                transform = self.tf_buffer.lookup_transform(
                    "spot/vision",
                    source_frame,
                    rclpy.time.Time(seconds=int(stamp_key[0]), nanoseconds=int(stamp_key[1])),
                    timeout=Duration(seconds=0.1),
                )
            except Exception as exc:
                self.get_logger().warning(
                    f"TF lookup failed for {source_frame} -> spot/odom at "
                    f"{stamp_key[0]}.{stamp_key[1]}: {exc}"
                )
                return None

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        rot = self._quaternion_to_rotation_matrix(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        )
        point_world = rot @ point_camera + np.array(
            [float(translation.x), float(translation.y), float(translation.z)],
            dtype=np.float64,
        )
        return (float(point_world[0]), float(point_world[1]), float(point_world[2]))

    def _transform_object_position_to_map_frame(
        self,
        obj_position: Tuple[float, float, float],
        stamp_key: Tuple[int, int],
        robot_map_pose: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Transform an object position from odom to map frame using the robot pose offset.

        Computes: obj_map = robot_map + R(map_yaw - odom_yaw) * (obj_odom - robot_odom)
        """
        robot_odom = self._lookup_robot_position_for_stamp(stamp_key)
        robot_map = robot_map_pose if robot_map_pose is not None else self._get_slam_robot_pose()
        if robot_odom is None or robot_map is None:
            return obj_position

        dx = obj_position[0] - robot_odom[0]
        dy = obj_position[1] - robot_odom[1]
        dz = obj_position[2] - robot_odom[2]

        dyaw = robot_map[3] - robot_odom[4]
        cos_dyaw = math.cos(dyaw)
        sin_dyaw = math.sin(dyaw)

        result = (
            robot_map[0] + cos_dyaw * dx - sin_dyaw * dy,
            robot_map[1] + sin_dyaw * dx + cos_dyaw * dy,
            robot_map[2] + dz,
        )
        self.get_logger().info(
            f"Map transform: obj_odom=({obj_position[0]:.3f},{obj_position[1]:.3f},{obj_position[2]:.3f}) "
            f"robot_odom=({robot_odom[0]:.3f},{robot_odom[1]:.3f},{robot_odom[4]:.3f}rad) "
            f"robot_map=({robot_map[0]:.3f},{robot_map[1]:.3f},{robot_map[3]:.3f}rad) "
            f"dyaw={dyaw:.3f}rad result=({result[0]:.3f},{result[1]:.3f},{result[2]:.3f})"
        )
        return result

    def _restore_camera_frame_from_rotated_image(self, point_camera: np.ndarray) -> np.ndarray:
        rotation = self._image_rotation
        if rotation == "none":
            return point_camera
        if rotation == "cw":
            return np.array(
                [point_camera[1], -point_camera[0], point_camera[2]],
                dtype=np.float64,
            )
        if rotation == "ccw":
            return np.array(
                [-point_camera[1], point_camera[0], point_camera[2]],
                dtype=np.float64,
            )
        return np.array(
            [-point_camera[0], -point_camera[1], point_camera[2]],
            dtype=np.float64,
        )

    def _estimate_detection_world_position(
        self,
        det: Any,
        stamp_key: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float]]:
        class_name = str(getattr(det, "class_name", "unknown"))
        instance_id = int(getattr(det, "instance_id", -1)) if hasattr(det, "instance_id") else -1
        track_id = int(getattr(det, "track_id", -1)) if hasattr(det, "track_id") else -1
        det_label = f"{class_name}#{instance_id}" if instance_id >= 0 else f"{class_name}#t{track_id}"

        def _throttled_log(reason: str, period_sec: float = 5.0) -> None:
            key = f"{det_label}:{reason}"
            now = time.time()
            last = self._depth_failure_log_throttle.get(key, 0.0)
            if now - last >= period_sec:
                self._depth_failure_log_throttle[key] = now
                extra = ""
                try:
                    extra = (
                        f" bbox=({x1},{y1},{x2},{y2})"
                        f" valid_pixels={valid_indices.shape[0]}"
                        f" depth_m={depth_m:.3f}"
                        f" rel_std={relative_std:.3f}"
                    )
                except Exception:
                    pass
                self.get_logger().warning(
                    f"Depth estimation failed for {det_label} at check '{reason}'{extra}"
                )

        depth_msg = self._lookup_depth_image_for_stamp(stamp_key)
        camera_info = self._lookup_camera_info_for_stamp(stamp_key)
        if depth_msg is None or camera_info is None:
            return None

        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().debug(f"Depth cv_bridge failed for {det_label}: {e}")
            return None
        if depth_image is None or depth_image.size == 0:
            return None

        x1 = max(0, int(getattr(det, "x_min", 0)))
        y1 = max(0, int(getattr(det, "y_min", 0)))
        x2 = min(int(getattr(det, "x_max", depth_msg.width)), int(depth_msg.width))
        y2 = min(int(getattr(det, "y_max", depth_msg.height)), int(depth_msg.height))
        if x2 <= x1 or y2 <= y1:
            return None

        roi_depth = depth_image[y1:y2, x1:x2]
        if roi_depth.size == 0:
            return None

        valid_mask = roi_depth > 0
        try:
            mask_image = self.cv_bridge.imgmsg_to_cv2(det.mask_image, desired_encoding="mono8")
            roi_h, roi_w = roi_depth.shape[:2]
            mask_h, mask_w = mask_image.shape[:2]
            if mask_h == roi_h and mask_w == roi_w:
                valid_mask = np.logical_and(valid_mask, mask_image > 0)
            else:
                # Resize mask to match ROI so alignment differences don't disable masking
                resized_mask = cv2.resize(
                    mask_image, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
                )
                valid_mask = np.logical_and(valid_mask, resized_mask > 0)
        except Exception:
            pass

        # Center-crop for classes that span large depth ranges (e.g. person limbs)
        if class_name in self.depth_estimation_center_crop_classes:
            crop_ratio = self.depth_estimation_center_crop_ratio
            if 0.0 < crop_ratio < 1.0:
                roi_h, roi_w = roi_depth.shape[:2]
                crop_h = max(1, int(roi_h * crop_ratio))
                crop_w = max(1, int(roi_w * crop_ratio))
                cy_start = (roi_h - crop_h) // 2
                cx_start = (roi_w - crop_w) // 2
                crop_mask = np.zeros_like(valid_mask)
                crop_mask[cy_start : cy_start + crop_h, cx_start : cx_start + crop_w] = True
                valid_mask = np.logical_and(valid_mask, crop_mask)

        valid_indices = np.argwhere(valid_mask)
        if valid_indices.size == 0:
            return None

        # Class-specific thresholds (person is typically larger and less rigid)
        is_person = class_name == "person"
        min_valid_pixels = (
            self.depth_estimation_person_min_valid_pixels
            if is_person
            else self.depth_estimation_min_valid_pixels
        )
        max_relative_std = (
            self.depth_estimation_person_max_relative_std
            if is_person
            else self.depth_estimation_max_relative_std
        )

        if valid_indices.shape[0] < min_valid_pixels:
            _throttled_log(f"too_few_pixels(min={min_valid_pixels})")
            return None

        depth_values = roi_depth[valid_mask].astype(np.float64)
        depth_m = float(np.median(depth_values)) * self._depth_scale
        if not math.isfinite(depth_m) or depth_m <= 0.0:
            _throttled_log("invalid_depth_median")
            return None

        relative_std = 0.0
        if valid_indices.shape[0] > 1:
            depth_std = float(np.std(depth_values)) * self._depth_scale
            relative_std = depth_std / depth_m if depth_m > 0.0 else float("inf")
            if relative_std > max_relative_std:
                _throttled_log(f"high_variance(max={max_relative_std:.2f},got={relative_std:.2f})")
                return None

        pixel_rc = np.median(valid_indices, axis=0)
        v = float(y1 + pixel_rc[0])
        u = float(x1 + pixel_rc[1])

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx == 0.0 or fy == 0.0:
            return None

        point_camera = np.array(
            [
                ((u - cx) * depth_m) / fx,
                ((v - cy) * depth_m) / fy,
                depth_m,
            ],
            dtype=np.float64,
        )
        point_camera_raw = point_camera.copy()
        point_camera = self._restore_camera_frame_from_rotated_image(point_camera)
        source_frame = depth_msg.header.frame_id or camera_info.header.frame_id
        if not source_frame:
            return None
        world_pos = self._transform_point_to_world(source_frame, stamp_key, point_camera)
        self.get_logger().info(
            f"Depth estimate: {det_label} bbox=({x1},{y1},{x2},{y2}) "
            f"pixel=({u:.1f},{v:.1f}) depth={depth_m:.3f}m "
            f"valid_pixels={valid_indices.shape[0]} rel_std={relative_std:.3f} "
            f"world=({world_pos[0] if world_pos else None}, "
            f"{world_pos[1] if world_pos else None}, {world_pos[2] if world_pos else None})"
        )
        return world_pos

    @staticmethod
    def _timestamp_from_header(header) -> datetime.datetime:
        sec = int(getattr(header.stamp, "sec", 0))
        nsec = int(getattr(header.stamp, "nanosec", 0))
        return datetime.datetime.utcfromtimestamp(sec + nsec * 1e-9)

    def _check_embedding_service(self) -> None:
        """
        Checks if the embedding service is ready and logs status.
        """
        object_ready = self.object_embedding_client.service_is_ready()
        person_ready = self.person_embedding_client.service_is_ready()
        if object_ready and person_ready:
            self.get_logger().info(
                "Embedding services are ready: "
                f"objects={self.object_embedding_service_name}, people={self.person_embedding_service_name}"
            )
            self.service_check_timer.cancel()
        else:
            if not object_ready:
                self.get_logger().info(f"Waiting for object embedding service {self.object_embedding_service_name} ...")
            if not person_ready:
                self.get_logger().info(f"Waiting for person embedding service {self.person_embedding_service_name} ...")

    def _embedding_client_for_class(self, class_id: Optional[int]):
        if self._is_person_class(class_id):
            return self.person_embedding_client
        return self.object_embedding_client

    def _embedding_service_name_for_class(self, class_id: Optional[int]) -> str:
        if self._is_person_class(class_id):
            return self.person_embedding_service_name
        return self.object_embedding_service_name

    def yolo_output_callback(self, msg):
        self.get_logger().info(f"Received YOLO output message with timestamp {msg.header.stamp.sec}.{msg.header.stamp.nanosec} and {len(msg.objects)} objects")
        # Latency instrumentation: frame age when the YOLO output reaches the DB node.
        _capture_ts = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) if (hasattr(msg, 'header') and hasattr(msg.header, 'stamp')) else 0.0
        _cb_entry = time.perf_counter()
        _frame_age_at_db = (time.time() - _capture_ts) if _capture_ts > 0 else -1.0
        # Match YOLO timestamp to nearest recent VLM text (exact matches are often delayed by one frame).
        stamp_key = self._stamp_key(msg)
        vlm_text = self._lookup_vlm_text_for_stamp(stamp_key)
        rgb_msg = self._lookup_rgb_image_for_stamp(stamp_key)
        corrected_rgb_msg = self._lookup_corrected_rgb_image_for_stamp(stamp_key)
        stitched_rgb_msg = self._lookup_stitched_rgb_image_for_stamp(stamp_key)
        # Build original scene image from uncorrected frame (clean, no bbox overlay)
        original_scene_image = self._build_original_scene_image_bytes(rgb_msg)
        # Build stitched scene image (no bbox overlay — YOLO boxes are from frontleft)
        stitched_scene_image = self._build_stitched_scene_image_bytes(stitched_rgb_msg)

        # Scene deduplication: skip visually unchanged frames
        thumb = self._compute_scene_thumb(corrected_rgb_msg)
        should_save, mean_abs_diff, changed_pixel_ratio, forced = self._should_save_scene(thumb)
        if not should_save:
            self._scene_dedup_skip_count += 1
            # Remember which scene this skipped frame was visually identical to,
            # so stamp-based lookups (interactions, faces) resolve it exactly
            # instead of dropping after retries when no scene is nearby.
            if self._last_saved_scene_id is not None:
                self._remember_scene_id_for_stamp(stamp_key, self._last_saved_scene_id)
            self.get_logger().debug(
                f"Dropped frame: visually unchanged "
                f"mean_abs_diff={mean_abs_diff:.2f} changed_ratio={changed_pixel_ratio:.4f}"
            )
            if self._scene_dedup_skip_count % 30 == 1:
                self.get_logger().info(
                    f"Dropped frame ({self._scene_dedup_skip_count} consecutive): visually unchanged "
                    f"mean_abs_diff={mean_abs_diff:.2f} changed_ratio={changed_pixel_ratio:.4f}"
                )
            return

        if forced:
            self.get_logger().info(
                "Saved scene despite low visual change due to force interval"
            )

        robot_position = self._lookup_robot_position_for_stamp(stamp_key)
        slam_pose = self._get_slam_robot_pose()
        scene_timestamp = None
        if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
            sec = getattr(msg.header.stamp, 'sec', None)
            nsec = getattr(msg.header.stamp, 'nanosec', None)
            if sec is not None and nsec is not None:
                import datetime
                scene_timestamp = datetime.datetime.utcfromtimestamp(sec + nsec * 1e-9)
        scene_id = None
        active_map_id = None
        try:
            active_map_id = self._get_active_map_id()
            # Use SLAM-transformed pose for scene coordinates (same as SLAM map)
            if slam_pose is not None:
                scene_x = slam_pose[0]
                scene_y = slam_pose[1]
            else:
                scene_x = robot_position[0] if robot_position is not None else None
                scene_y = robot_position[1] if robot_position is not None else None
            _t_scene = time.perf_counter()
            scene_id = self._insert_scene(
                vlm_text or "",
                scene_x,
                scene_y,
                scene_timestamp,
                str(getattr(msg.header, "frame_id", "") or ""),
                original_scene_image=original_scene_image,
                stitched_scene_image=stitched_scene_image,
                map_id=active_map_id,
            )
            _scene_insert_ms = (time.perf_counter() - _t_scene) * 1000.0
            _e2e_scene_ms = (time.time() - _capture_ts) * 1000.0 if _capture_ts > 0 else -1.0
            self.get_logger().info(
                f"[latency] db: scene_insert={_scene_insert_ms:.0f}ms "
                f"frame_age_at_db={_frame_age_at_db*1000.0:.0f}ms "
                f"capture_to_scene_saved={_e2e_scene_ms:.0f}ms scene_id={scene_id}"
            )
            self._remember_scene_id_for_stamp(stamp_key, scene_id)
            self._scene_id_by_stamp[stamp_key] = scene_id
            if scene_id is not None:
                self._last_saved_scene_id = int(scene_id)
            self._last_saved_scene_thumb = thumb
            self._last_scene_save_time = self.get_clock().now().nanoseconds / 1e9
            if self._scene_dedup_skip_count > 0:
                self.get_logger().info(
                    f"Saved scene after dropping {self._scene_dedup_skip_count} "
                    f"visually unchanged frame(s)"
                )
            self._scene_dedup_skip_count = 0
            self.get_logger().info(f"Saved scene with id {scene_id} and caption '{vlm_text}'")
        except Exception as exc:
            self.get_logger().error(f"Failed to insert scene for YOLO frame: {exc}")

        # For each detection, request embedding and resolve identity in async callback.
        
        bridge = CvBridge()
        frame_sec = self._stamp_key_to_seconds(stamp_key)
        self._prune_stale_track_mappings(frame_sec)
        for det in msg.objects:
            yolo_track_id = self._extract_track_id(det)
            yolo_id_log = yolo_track_id if yolo_track_id is not None else "none"
            self.get_logger().info(f"Processing detection for YOLO ID: {yolo_id_log}")
            class_id = det.class_id if hasattr(det, 'class_id') else None
            mask_cv = None
            cropped_img_cv = None
            try:
                cropped_img_cv = bridge.imgmsg_to_cv2(det.cropped_image, desired_encoding="bgr8")
                if hasattr(det, 'mask_image') and det.mask_image is not None and det.mask_image.width > 0 and det.mask_image.height > 0:
                    try:
                        mask_cv = bridge.imgmsg_to_cv2(det.mask_image, desired_encoding="mono8")
                    except Exception:
                        mask_cv = None
                _, cropped_img_bytes = cv2.imencode('.jpg', cropped_img_cv)
                cropped_img_bytes = cropped_img_bytes.tobytes()
            except Exception as exc:
                self.get_logger().error(f"Failed to convert cropped image for object {yolo_id_log}: {exc}")
                cropped_img_bytes = b''

            mapped_object_id = None
            if yolo_track_id is not None:
                mapped_object_id = self._lookup_track_mapping(yolo_track_id)
                if mapped_object_id is not None:
                    self.get_logger().info(
                        f"Reusing mapped object_id {mapped_object_id} for YOLO track {yolo_track_id}"
                    )

            # Scene-cut / subject-swap guard: if this track's crop changed sharply vs
            # the previous frame, the tracker likely reused the track id for a different
            # object/person. Drop the track->object mapping AND the cached embedding so
            # the detection is re-resolved and re-embedded fresh (prevents two different
            # people being merged under one reused track id).
            content_changed = self._detect_track_content_change(yolo_track_id, cropped_img_cv)
            if content_changed and yolo_track_id is not None:
                if mapped_object_id is not None:
                    self.get_logger().info(
                        f"Dropping track mapping {mapped_object_id} for track {yolo_track_id} "
                        f"due to sharp crop change (scene cut); re-resolving identity"
                    )
                    mapped_object_id = None
                self.yolo_id_to_object_id.pop(yolo_track_id, None)
                self.track_id_last_seen_sec.pop(yolo_track_id, None)
                self._embedding_by_track_id.pop(yolo_track_id, None)

            # Request embedding for cropped image
            req = GetImageEmbedding.Request()
            
            # Use the masked crop for embedding service (background pixels set to black)
            if mask_cv is not None and mask_cv.shape[:2] == cropped_img_cv.shape[:2]:
                masked_crop_cv = cropped_img_cv.copy()
                masked_crop_cv[mask_cv == 0] = [0, 0, 0]
                req.image = bridge.cv2_to_imgmsg(masked_crop_cv, encoding="bgr8")
                req.image.header = det.cropped_image.header
                self.get_logger().info(f"Using masked crop for embedding track={yolo_id_log}")
            else:
                req.image = det.cropped_image if isinstance(det.cropped_image, Image) else None
            if req.image is None:
                self.get_logger().error(f"No valid cropped image for embedding for object {yolo_id_log}")
                continue
            # Store all info needed for callback
            mask_img_bytes = b''
            if mask_cv is not None:
                try:
                    ok, mask_img_bytes = cv2.imencode('.png', mask_cv)
                    if ok:
                        mask_img_bytes = mask_img_bytes.tobytes()
                    else:
                        mask_img_bytes = b''
                except Exception as mask_exc:
                    self.get_logger().warning(f"Failed to encode mask for track={yolo_id_log}: {mask_exc}")
                    mask_img_bytes = b''

            original_crop_bytes = b''
            if hasattr(det, 'original_cropped_image') and det.original_cropped_image is not None and det.original_cropped_image.data:
                try:
                    orig_cv = self.cv_bridge.imgmsg_to_cv2(det.original_cropped_image, desired_encoding="bgr8")
                    ok, orig_buf = cv2.imencode('.jpg', orig_cv)
                    original_crop_bytes = orig_buf.tobytes() if ok else b''
                except Exception as orig_exc:
                    self.get_logger().warning(f"Failed to encode original crop for track={yolo_id_log}: {orig_exc}")
                    original_crop_bytes = b''

            attributes_json = {"enabled": False}
            part_images = {}
            quality_score = None
            if cropped_img_cv is not None:
                try:
                    attributes_json, part_images = extract_visual_attributes(
                        cropped_img_cv,
                        mask_cv,
                        class_id=class_id,
                        detection_confidence=float(det.score) if hasattr(det, 'score') else None,
                        config=self.attribute_config,
                    )
                    quality_score = attributes_json.get("quality_score")
                except Exception as attr_exc:
                    self.get_logger().warning(f"Failed to extract attributes for track={yolo_id_log}: {attr_exc}")
                    attributes_json = {"enabled": False, "error": str(attr_exc)}
            depth_position_odom = self._estimate_detection_world_position(det, stamp_key)
            estimated_position = depth_position_odom
            estimated_position_frame = "odom"
            if estimated_position is not None and slam_pose is not None:
                estimated_position = self._transform_object_position_to_map_frame(
                    estimated_position,
                    stamp_key,
                    robot_map_pose=slam_pose,
                )
                estimated_position_frame = "map"

            # Publish depth-based object odometry for map display
            if depth_position_odom is not None:
                try:
                    odom_msg = ObjectOdometry()
                    odom_msg.object_id = int(det.instance_id) if hasattr(det, 'instance_id') else -1
                    odom_msg.sequence = 0
                    odom_msg.odom.header.stamp = msg.header.stamp
                    odom_msg.odom.header.frame_id = "spot/vision"
                    odom_msg.odom.child_frame_id = f"object_{odom_msg.object_id}"
                    odom_msg.odom.pose.pose.position.x = float(depth_position_odom[0])
                    odom_msg.odom.pose.pose.position.y = float(depth_position_odom[1])
                    odom_msg.odom.pose.pose.position.z = float(depth_position_odom[2])
                    odom_msg.odom.pose.pose.orientation.w = 1.0
                    odom_msg.odom.pose.covariance = [0.0] * 36
                    odom_msg.odom.pose.covariance[0] = 0.01
                    odom_msg.odom.pose.covariance[7] = 0.01
                    odom_msg.odom.pose.covariance[14] = 0.01
                    self.depth_object_odometry_pub.publish(odom_msg)
                except Exception as pub_exc:
                    self.get_logger().warning(f"Failed to publish depth object odometry: {pub_exc}")

            pending = {
                'mapped_object_id': mapped_object_id,
                'yolo_track_id': yolo_track_id,
                'class_id': class_id,
                'confidence': float(det.score) if hasattr(det, 'score') else None,
                'dynosam_instance_id': int(det.instance_id) if hasattr(det, 'instance_id') else None,
                'bbox_x_min': int(det.x_min) if hasattr(det, 'x_min') else None,
                'bbox_y_min': int(det.y_min) if hasattr(det, 'y_min') else None,
                'bbox_x_max': int(det.x_max) if hasattr(det, 'x_max') else None,
                'bbox_y_max': int(det.y_max) if hasattr(det, 'y_max') else None,
                'estimated_position': estimated_position,
                'estimated_position_frame': estimated_position_frame,
                'cropped_image': cropped_img_bytes,
                'mask_image': mask_img_bytes,
                'original_cropped_image': original_crop_bytes,
                'scene_id': scene_id,
                'msg_header': msg.header,
                'frame_sec': frame_sec,
                'stamp_key': stamp_key,
                'robot_position': robot_position,
                'attributes_json': attributes_json,
                'quality_score': quality_score,
                'part_images': part_images,
                'part_embeddings': {},
                'part_preprocessing': {},
                'observation_id': None,
                'map_id': active_map_id,
                'detection_backend': self.detector_backend,
                'embedding_backend': None,
            }

            if self.attribute_config.part_embeddings_enabled and part_images:
                self._request_part_embeddings(pending, bridge)

            cached_embedding = None
            if mapped_object_id is not None and yolo_track_id is not None:
                cached_embedding = self._embedding_by_track_id.get(yolo_track_id)

            # Cached-embedding fast-path. For PERSONS we deliberately do NOT take this
            # path: reusing the track's first-frame embedding (a) stores bit-identical
            # vectors across the track and (b) skips _resolve_object_id_for_detection,
            # so a wrong/sink mapping is never corrected by face identity or the
            # track-content-change trigger. OSNet body embeddings are too weak to be
            # trusted without re-resolution, so persons always re-embed + re-resolve.
            # Objects keep the fast-path (their embeddings are stable and the color
            # gate already guards them).
            if (
                mapped_object_id is not None
                and cached_embedding is not None
                and not self._is_person_class(class_id)
            ):
                try:
                    self._persist_or_queue_observation(
                        object_id=mapped_object_id,
                        embedding=list(cached_embedding),
                        pending=pending,
                    )
                    continue
                except Exception as exc:
                    self.get_logger().error(
                        f"Failed to persist cached-embedding observation for track {yolo_track_id}: {exc}"
                    )

            embedding_client = self._embedding_client_for_class(class_id)
            embedding_service_name = self._embedding_service_name_for_class(class_id)
            if not embedding_client.service_is_ready():
                self.get_logger().warning(
                    f"Embedding service {embedding_service_name} not ready for class_id={class_id}; "
                    f"skipping track={yolo_id_log}"
                )
                continue
            pending["embedding_service_name"] = embedding_service_name
            future = embedding_client.call_async(req)
            future.add_done_callback(lambda fut, p=pending: self._handle_detection_embedding_response(fut, p))

    def interaction_output_callback(self, msg: InteractionOutput) -> None:
        # Prefer explicit source stamp (exact correlation ID passed through pipeline)
        if (
            hasattr(msg, "source_stamp_sec")
            and hasattr(msg, "source_stamp_nanosec")
            and int(msg.source_stamp_sec) > 0
        ):
            stamp_key = (int(msg.source_stamp_sec), int(msg.source_stamp_nanosec))
        else:
            stamp_key = self._stamp_key(msg)
        if not msg.interactions:
            self.get_logger().info(
                f"Received empty interaction output for {stamp_key[0]}.{stamp_key[1]}"
            )
            return

        for interaction in msg.interactions:
            refs = [
                {
                    "role": ref.role,
                    "track_id": ref.track_id,
                    "detection_id": int(ref.detection_id),
                    "class_id": int(ref.class_id),
                    "class_name": ref.class_name,
                    "x_min": int(ref.x_min),
                    "y_min": int(ref.y_min),
                    "x_max": int(ref.x_max),
                    "y_max": int(ref.y_max),
                }
                for ref in interaction.objects
            ]
            record = PendingInteractionRecord(
                stamp_key=stamp_key,
                action=str(interaction.action or "").strip(),
                caption=str(interaction.caption or "").strip(),
                model_source="interaction_description_node",
                raw_response=str(msg.raw_response or ""),
                refs=refs,
                map_id=self._get_active_map_id(),
                confidence=float(getattr(interaction, "confidence", 0.0) or 0.0),
            )
            self._queue_interaction_record(record)

        self._flush_pending_interactions()

    def face_output_callback(self, msg: FaceOutput) -> None:
        if not msg.faces:
            return

        stamp_key = self._stamp_key(msg)
        fallback_scene_id = self._lookup_scene_id_for_stamp(stamp_key)
        frame_sec = self._stamp_key_to_seconds(stamp_key)
        self._prune_stale_face_track_mappings(frame_sec)

        persisted = 0
        dropped = 0
        deferred = 0
        for face in msg.faces:
            embedding = self._coerce_face_embedding_dim(list(face.embedding))
            if not embedding:
                continue

            track_id: Optional[str] = None
            try:
                if int(face.track_id) >= 0:
                    track_id = str(int(face.track_id))
            except Exception:
                track_id = None

            scene_id: Optional[int] = fallback_scene_id
            try:
                if int(face.scene_id) > 0:
                    scene_id = int(face.scene_id)
            except Exception:
                pass

            person_id = self._resolve_person_id_for_face(
                embedding=embedding,
                yolo_track_id=track_id,
                frame_sec=frame_sec,
            )

            # 1 face = 1 person observation: find the specific class-0 observation this
            # face was detected on (track + scene + person bbox, else closest in time).
            # The face's object_id/scene_id derive from that observation's cluster. If
            # the person observation was not persisted, drop the face (objects have no
            # faces; every face must attach to a real person observation).
            person_bbox = None
            try:
                person_bbox = (
                    int(face.person_x_min), int(face.person_y_min),
                    int(face.person_x_max), int(face.person_y_max),
                )
            except Exception:
                person_bbox = None
            face_bbox = None
            try:
                face_bbox = (
                    int(face.face_x_min), int(face.face_y_min),
                    int(face.face_x_max), int(face.face_y_max),
                )
            except Exception:
                face_bbox = None
            matched = self._try_persist_face(
                face=face,
                embedding=embedding,
                person_id=person_id,
                track_id=track_id,
                scene_id=scene_id,
                person_bbox=person_bbox,
                face_bbox=face_bbox,
                frame_sec=frame_sec,
            )
            if matched:
                persisted += 1
            else:
                # The body observation for this face is likely not persisted yet (the
                # face pipeline currently runs ahead of body persistence). Defer and
                # retry shortly instead of dropping, so faces are not lost.
                self._queue_pending_face(
                    face=face,
                    embedding=embedding,
                    person_id=person_id,
                    track_id=track_id,
                    scene_id=scene_id,
                    person_bbox=person_bbox,
                    face_bbox=face_bbox,
                    frame_sec=frame_sec,
                )
                deferred += 1

        if persisted > 0 or dropped > 0 or deferred > 0:
            self.get_logger().info(
                f"Saved {persisted} face observations for {stamp_key[0]}.{stamp_key[1]}"
                + (f" (deferred {deferred} pending body observation)" if deferred > 0 else "")
                + (f" (dropped {dropped} with no person observation)" if dropped > 0 else "")
            )

    def _queue_pending_face(self, *, face, embedding, person_id, track_id, scene_id, person_bbox, face_bbox, frame_sec) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        self._pending_faces.append({
            "face": face,
            "embedding": embedding,
            "person_id": person_id,
            "track_id": track_id,
            "scene_id": scene_id,
            "person_bbox": person_bbox,
            "face_bbox": face_bbox,
            "frame_sec": frame_sec,
            "first_seen_wall_sec": now_sec,
        })

    def _flush_pending_faces(self) -> None:
        if not getattr(self, "_pending_faces", None):
            return
        now_sec = self.get_clock().now().nanoseconds / 1e9
        max_age = float(getattr(self, "face_pending_max_age_sec", 120.0) or 120.0)
        remaining = []
        persisted = 0
        dropped = 0
        for rec in self._pending_faces:
            matched = self._try_persist_face(
                face=rec["face"],
                embedding=rec["embedding"],
                person_id=rec["person_id"],
                track_id=rec["track_id"],
                scene_id=rec["scene_id"],
                person_bbox=rec["person_bbox"],
                face_bbox=rec["face_bbox"],
                frame_sec=rec["frame_sec"],
            )
            if matched:
                persisted += 1
                continue
            if (now_sec - rec["first_seen_wall_sec"]) > max_age:
                dropped += 1
                self.get_logger().info(
                    f"Face dropped after {max_age:.0f}s pending (no person obs): "
                    f"track={rec['track_id']} scene={rec['scene_id']} frame_sec={rec['frame_sec']}"
                )
                continue
            remaining.append(rec)
        self._pending_faces = remaining
        if persisted > 0 or dropped > 0:
            self.get_logger().info(
                f"Pending faces: persisted {persisted}, dropped {dropped}, {len(remaining)} still pending"
            )

    def _try_persist_face(self, *, face, embedding, person_id, track_id, scene_id, person_bbox, face_bbox, frame_sec) -> bool:
        """Attempt to match + persist a single face. Returns True on success, False when
        no person observation exists yet (caller may defer/retry)."""
        match = self._find_person_observation_for_face(
            yolo_track_id=track_id,
            scene_id=scene_id,
            person_bbox=person_bbox,
            frame_sec=frame_sec,
            face_bbox=face_bbox,
        )
        if match is None:
            return False
        observation_id, object_id, obs_scene_id = match
        if obs_scene_id is not None:
            scene_id = obs_scene_id

        # Enforce 1 face per observation: keep only the higher-scoring one.
        try:
            existing = self._db_execute(
                "SELECT id, COALESCE(score, 0.0) FROM face_observations WHERE observation_id = %s LIMIT 1",
                (observation_id,),
                fetchone=True,
            )
        except Exception:
            existing = None
        if existing is not None:
            new_score = float(face.face_score) if hasattr(face, "face_score") else 0.0
            if new_score > float(existing[1]):
                try:
                    self._db_execute("DELETE FROM face_observations WHERE id = %s", (int(existing[0]),))
                except Exception as exc:
                    self.get_logger().warning(f"Failed to replace lower-score face for observation {observation_id}: {exc}")
                    return True  # treat as handled (do not retry)
            else:
                return True  # existing face is better; handled

        try:
            face_img_bgr = CvBridge().imgmsg_to_cv2(face.aligned_face_image, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(".jpg", face_img_bgr)
            face_image_bytes = encoded.tobytes() if ok else b""
        except Exception:
            face_image_bytes = b""

        # Reclusters (face rebuild / object recluster) can delete objects while this node
        # is running, leaving person_id / object_id referencing rows that no longer exist
        # (FK violation on face_observations). Re-validate just before insert:
        #  - person_id is NOT NULL -> re-create the person object if it was deleted (UPSERT).
        #  - object_id is nullable -> drop to NULL if it no longer exists.
        try:
            self._ensure_person_exists(person_id, embedding)
        except Exception as exc:
            self.get_logger().warning(f"Could not ensure person {person_id} exists before face insert: {exc}")
        if object_id is not None and not self._object_exists(object_id):
            self.get_logger().info(
                f"Face insert: object_id {object_id} no longer exists (deleted by recluster); "
                f"storing face with object_id NULL for person {person_id}"
            )
            object_id = None

        try:
            self._insert_face_observation(
                person_id=person_id,
                scene_id=scene_id,
                object_id=object_id,
                observation_id=observation_id,
                yolo_track_id=track_id,
                face_image=face_image_bytes,
                person_x_min=int(face.person_x_min),
                person_y_min=int(face.person_y_min),
                person_x_max=int(face.person_x_max),
                person_y_max=int(face.person_y_max),
                face_x_min=int(face.face_x_min),
                face_y_min=int(face.face_y_min),
                face_x_max=int(face.face_x_max),
                face_y_max=int(face.face_y_max),
                score=float(face.face_score),
                embedding=embedding,
                embedding_id=str(face.embedding_id or ""),
                map_id=self._get_active_map_id(),
            )
            self._update_person_canonical_embedding(person_id)
            corrected_object_id = self._correct_body_cluster_by_face_identity(
                person_id=person_id,
                track_id=track_id,
                current_object_id=object_id,
                frame_sec=frame_sec,
            )
            if corrected_object_id is not None:
                object_id = corrected_object_id
            if object_id is not None:
                self._update_track_mapping(track_id, object_id, frame_sec)
            if track_id is not None:
                self._face_embedding_by_track[track_id] = (embedding, frame_sec)
            return True
        except Exception as exc:
            self.get_logger().error(f"Failed to persist face observation for person {person_id}: {exc}")
            return True  # do not retry on insert error

    def _lookup_face_track_mapping(self, yolo_track_id: str) -> Optional[str]:
        person_id = self.face_track_to_person_id.get(yolo_track_id)
        if person_id is None:
            return None
        self.face_track_last_seen_sec[yolo_track_id] = self.get_clock().now().nanoseconds / 1e9
        return person_id

    def _update_face_track_mapping(
        self,
        yolo_track_id: Optional[str],
        person_id: str,
        now_sec: Optional[float] = None,
    ) -> None:
        if yolo_track_id is None:
            return
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
        self.face_track_to_person_id[yolo_track_id] = person_id
        self.face_track_last_seen_sec[yolo_track_id] = now_sec

    def _prune_stale_face_track_mappings(self, now_sec: Optional[float] = None) -> None:
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
        stale_ids = [
            track_id
            for track_id, last_seen in self.face_track_last_seen_sec.items()
            if (now_sec - last_seen) > self.face_track_id_ttl_sec
        ]
        for track_id in stale_ids:
            self.face_track_last_seen_sec.pop(track_id, None)
            self.face_track_to_person_id.pop(track_id, None)

    def _is_person_class(self, class_id: Optional[int]) -> bool:
        if class_id is None:
            return False
        try:
            return int(class_id) == int(self.person_class_id)
        except Exception:
            return False

    def _is_person_object(self, object_id: Optional[str]) -> bool:
        """Return True only if the given object cluster is a person (class_id == person_class_id).

        Face observations must never be linked to non-person object clusters
        (e.g. a refrigerator). The shared yolo_id_to_object_id track mapping is
        class-agnostic and BoTSORT reuses track ids across classes, so a track id
        can map to a non-person cluster; this guard rejects such mappings.
        """
        if object_id is None:
            return False
        try:
            row = self._db_execute(
                "SELECT class_id FROM objects WHERE id = %s",
                (int(object_id),),
                fetchone=True,
            )
        except Exception:
            return False
        if row is None:
            return False
        return self._is_person_class(row[0])

    def _lookup_recent_person_id_for_track(self, yolo_track_id: Optional[str]) -> Optional[str]:
        if yolo_track_id is None:
            return None
        row = self._db_execute(
            """
            SELECT person_id
            FROM face_observations
            WHERE yolo_track_id = %s
              AND person_id IS NOT NULL
              AND created_at >= (NOW() - (%s * INTERVAL '1 second'))
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (yolo_track_id, self.face_track_db_lookup_window_sec),
            fetchone=True,
        )
        if row is None:
            return None
        return str(row[0])

    def _resolve_person_id_for_track(
        self,
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
    ) -> Optional[str]:
        if yolo_track_id is None:
            return None

        mapped_person_id = self._lookup_face_track_mapping(yolo_track_id)
        if mapped_person_id is not None:
            return mapped_person_id

        recent_person_id = self._lookup_recent_person_id_for_track(yolo_track_id)
        if recent_person_id is None:
            return None

        self._update_face_track_mapping(yolo_track_id, recent_person_id, frame_sec)
        return recent_person_id

    def _lookup_primary_object_for_person(
        self,
        person_id: str,
        class_id: Optional[int],
    ) -> tuple[Optional[str], int]:
        row = self._db_execute(
            """
            SELECT
                fo.object_id,
                COUNT(*) AS hit_count
            FROM face_observations fo
            JOIN objects o ON o.id = fo.object_id
            WHERE fo.person_id = %s
              AND fo.object_id IS NOT NULL
              AND (%s IS NULL OR o.class_id = %s)
            GROUP BY fo.object_id
            ORDER BY hit_count DESC, MAX(fo.created_at) DESC
            LIMIT 1
            """,
            (person_id, class_id, class_id),
            fetchone=True,
        )
        if row is None:
            return None, 0
        return str(row[0]), int(row[1])

    def _correct_body_cluster_by_face_identity(
        self,
        person_id: Optional[str],
        track_id: Optional[str],
        current_object_id: Optional[str],
        frame_sec: Optional[float],
    ) -> Optional[str]:
        """Reassign this track's recent body observations to the face-identified person cluster.

        Faces (ArcFace) are far more discriminative than body embeddings (OSNet).
        The body detection is often clustered before its face arrives, so a wrong
        body cluster can be assigned first. When a confident face later identifies
        the person and their primary object cluster differs from the cluster the
        body observations were assigned to, move the track's recent body
        observations to the face-identified cluster.

        Returns the corrected object_id when a correction was applied, else None.
        """
        if person_id is None or track_id is None:
            return None
        try:
            correct_object_id, hit_count = self._lookup_primary_object_for_person(
                person_id=person_id,
                class_id=self.person_class_id,
            )
        except Exception as exc:
            self.get_logger().warning(f"Face correction: primary object lookup failed for person {person_id}: {exc}")
            return None
        if correct_object_id is None:
            return None
        if current_object_id is not None and str(current_object_id) == str(correct_object_id):
            return None  # already consistent

        # Guard: do not feed an over-merged sink cluster. If the face-identified
        # person's primary cluster already holds a huge number of body observations,
        # it is almost certainly a multi-person sink; moving more observations into
        # it only reinforces the over-merge. Skip the correction in that case.
        max_obs = int(getattr(self, "face_correction_max_cluster_observations", 0) or 0)
        if max_obs > 0:
            try:
                row = self._db_execute(
                    "SELECT COUNT(*) FROM object_observations WHERE object_id = %s",
                    (int(correct_object_id),),
                    fetchone=True,
                )
                cluster_obs = int(row[0]) if row else 0
            except Exception as exc:
                self.get_logger().warning(f"Face correction: cluster size check failed for {correct_object_id}: {exc}")
                cluster_obs = 0
            if cluster_obs >= max_obs:
                self.get_logger().info(
                    f"Face correction: skipped track {track_id} -> person cluster {correct_object_id} "
                    f"(person {person_id}); target cluster too large ({cluster_obs} obs >= {max_obs}), likely a sink"
                )
                return None

        try:
            # Move this track's body observations (currently assigned to a different
            # cluster) to the face-identified person cluster. When
            # face_correction_whole_track is enabled we move ALL of the track's person
            # observations, not just the recent window: a confident face is far more
            # discriminative than the body embedding that mis-assigned them, so the
            # whole track should follow the face. Otherwise keep the legacy time window.
            whole_track = bool(getattr(self, "face_correction_whole_track", True))
            if whole_track:
                self._db_execute(
                    """
                    UPDATE object_observations
                    SET object_id = %s
                    WHERE yolo_track_id = %s
                      AND class_id = %s
                      AND object_id IS DISTINCT FROM %s
                    """,
                    (
                        int(correct_object_id),
                        track_id,
                        int(self.person_class_id),
                        int(correct_object_id),
                    ),
                )
            else:
                self._db_execute(
                    """
                    UPDATE object_observations
                    SET object_id = %s
                    WHERE yolo_track_id = %s
                      AND class_id = %s
                      AND object_id IS DISTINCT FROM %s
                      AND created_at >= NOW() - make_interval(secs => %s)
                    """,
                    (
                        int(correct_object_id),
                        track_id,
                        int(self.person_class_id),
                        int(correct_object_id),
                        float(self.face_correction_window_sec),
                    ),
                )
            self._update_track_mapping(track_id, str(correct_object_id), frame_sec)
            self.get_logger().info(
                f"Face correction: moved track {track_id} body observations to person cluster "
                f"{correct_object_id} (was {current_object_id}, person {person_id}, face hits {hit_count})"
            )
            return str(correct_object_id)
        except Exception as exc:
            self.get_logger().error(f"Face correction: failed to reassign track {track_id} observations: {exc}")
            return None

    def _resolve_object_id_from_face_identity(
        self,
        class_id: Optional[int],
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
    ) -> Optional[str]:
        if not self._is_person_class(class_id):
            return None

        person_id = self._resolve_person_id_for_track(yolo_track_id, frame_sec)
        if person_id is None:
            return None

        object_id, hit_count = self._lookup_primary_object_for_person(person_id, class_id)
        if object_id is None:
            return None

        try:
            self._ensure_object_exists(object_id, class_id)
        except Exception as exc:
            self.get_logger().error(f"Failed to ensure face-linked object exists: {exc}")
            return None

        self._update_track_mapping(yolo_track_id, object_id, frame_sec)
        self.get_logger().info(
            f"Face-first re-id matched person {person_id} to object {object_id} "
            f"(track={yolo_track_id}, hits={hit_count})"
        )
        return object_id

    def _match_person_by_embedding(self, embedding: list[float]) -> tuple[Optional[str], float, float]:
        embedding_vec = self._to_vector_literal(embedding)
        rows = self._db_fetchall(
            """
            WITH ranked AS (
                SELECT
                    o.id,
                    (1.0 - (o.canonical_embedding <=> %s::vector)) AS similarity
                FROM objects o
                WHERE o.canonical_embedding IS NOT NULL
                  AND o.class_id = %s
                ORDER BY o.canonical_embedding <=> %s::vector ASC
                LIMIT 2
            )
            SELECT id, similarity
            FROM ranked
            ORDER BY similarity DESC
            """,
            (embedding_vec, self.person_class_id, embedding_vec),
        )
        if not rows:
            return None, 0.0, 0.0

        best = FaceMatchCandidate(person_id=int(rows[0][0]), similarity=float(rows[0][1]))
        margin = best.similarity
        if len(rows) > 1:
            margin = best.similarity - float(rows[1][1])

        return best.person_id, best.similarity, margin

    def _face_embedding_similarity_to_person(self, embedding: list[float], person_id: str) -> Optional[float]:
        """Cosine similarity between a face embedding and a person's canonical embedding."""
        try:
            row = self._db_execute(
                """
                SELECT 1.0 - (canonical_embedding <=> %s::vector)
                FROM objects
                WHERE id = %s AND canonical_embedding IS NOT NULL
                """,
                (self._to_vector_literal(embedding), int(person_id)),
                fetchone=True,
            )
        except Exception:
            return None
        if row is None or row[0] is None:
            return None
        return float(row[0])

    def _resolve_person_id_for_face(
        self,
        embedding: list[float],
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
    ) -> str:
        mapped_person_id = None
        if yolo_track_id is not None:
            mapped_person_id = self._lookup_face_track_mapping(yolo_track_id)
        if mapped_person_id is not None:
            # Verify the track mapping against the actual face embedding. BoTSORT can
            # reuse a track id for a different person (e.g. one person leaves, another
            # enters); blindly trusting the track mapping would merge two different
            # people into one identity. If the face no longer matches the mapped
            # person, drop the mapping and fall through to embedding matching.
            sim_to_mapped = self._face_embedding_similarity_to_person(embedding, mapped_person_id)
            if sim_to_mapped is not None and sim_to_mapped < self.face_reid_similarity_threshold:
                self.get_logger().info(
                    f"Track {yolo_track_id} remapped: face no longer matches person "
                    f"{mapped_person_id} (sim {sim_to_mapped:.3f} < {self.face_reid_similarity_threshold}); "
                    f"resolving identity by embedding instead"
                )
                self.face_track_to_person_id.pop(yolo_track_id, None)
                mapped_person_id = None
            else:
                self._ensure_person_exists(mapped_person_id, embedding)
                self._update_face_track_mapping(yolo_track_id, mapped_person_id, frame_sec)
                return mapped_person_id

        matched_person_id, similarity, margin = self._match_person_by_embedding(embedding)
        if (
            matched_person_id is not None
            and similarity >= self.face_reid_similarity_threshold
            and margin >= self.face_reid_min_score_margin
        ):
            self._ensure_person_exists(matched_person_id, embedding)
            self._update_face_track_mapping(yolo_track_id, matched_person_id, frame_sec)
            return matched_person_id

        import uuid

        person_id = self._create_object(class_id=self.person_class_id, canonical_embedding=embedding)
        self._ensure_person_exists(person_id, embedding)
        self._update_face_track_mapping(yolo_track_id, person_id, frame_sec)
        return person_id

    def _ensure_person_exists(self, person_id: int, embedding: Optional[list[float]] = None) -> None:
        embedding_vec = self._to_vector_literal(embedding or [0.0] * (self._db_face_embedding_dim or 512))
        self._db_execute(
            """
            INSERT INTO objects (id, class_id, canonical_embedding)
            VALUES (%s, %s, %s::vector)
            ON CONFLICT (id)
            DO UPDATE SET
                class_id = EXCLUDED.class_id,
                canonical_embedding = COALESCE(objects.canonical_embedding, EXCLUDED.canonical_embedding)
            """,
            (person_id, self.person_class_id, embedding_vec),
        )

    def _object_exists(self, object_id) -> bool:
        if object_id is None:
            return False
        try:
            row = self._db_execute(
                "SELECT 1 FROM objects WHERE id = %s",
                (int(object_id),),
                fetchone=True,
            )
            return row is not None
        except Exception:
            return False

    def _insert_face_observation(
        self,
        person_id: int,
        scene_id: Optional[int],
        object_id: Optional[int],
        yolo_track_id: Optional[str],
        face_image: bytes,
        person_x_min: int,
        person_y_min: int,
        person_x_max: int,
        person_y_max: int,
        face_x_min: int,
        face_y_min: int,
        face_x_max: int,
        face_y_max: int,
        score: float,
        embedding: list[float],
        embedding_id: str,
        map_id: Optional[int] = None,
        observation_id: Optional[int] = None,
    ) -> Optional[int]:
        embedding_vec = self._to_vector_literal(embedding)
        row = self._db_execute(
            """
            INSERT INTO face_observations (
                person_id,
                scene_id,
                object_id,
                observation_id,
                yolo_track_id,
                face_image,
                person_x_min,
                person_y_min,
                person_x_max,
                person_y_max,
                face_x_min,
                face_y_min,
                face_x_max,
                face_y_max,
                score,
                embedding,
                embedding_id,
                map_id
            )
            VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s::vector, NULLIF(%s, ''), %s
            )
            ON CONFLICT (embedding_id)
            DO NOTHING
            RETURNING id
            """,
            (
                person_id,
                scene_id,
                object_id,
                observation_id,
                yolo_track_id,
                psycopg2.Binary(face_image) if face_image else None,
                person_x_min,
                person_y_min,
                person_x_max,
                person_y_max,
                face_x_min,
                face_y_min,
                face_x_max,
                face_y_max,
                score,
                embedding_vec,
                embedding_id,
                map_id,
            ),
            fetchone=True,
        )
        return int(row[0]) if row is not None else None

    def _update_person_canonical_embedding(self, person_id: int) -> None:
        self._db_execute(
            """
            UPDATE objects o
            SET
                canonical_embedding = agg.avg_embedding
            FROM (
                SELECT person_id, AVG(embedding) AS avg_embedding
                FROM face_observations
                WHERE person_id = %s
                GROUP BY person_id
            ) AS agg
            WHERE o.id = agg.person_id
              AND o.id = %s
            """,
            (person_id, person_id),
        )

    def _lookup_recent_person_id_for_track(self, yolo_track_id: Optional[str]) -> Optional[int]:
        if yolo_track_id is None:
            return None
        row = self._db_execute(
            """
            SELECT person_id
            FROM face_observations
            WHERE yolo_track_id = %s
              AND person_id IS NOT NULL
              AND created_at >= (NOW() - (%s * INTERVAL '1 second'))
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (yolo_track_id, self.face_track_db_lookup_window_sec),
            fetchone=True,
        )
        if row is None:
            return None
        return int(row[0])

    def _extract_track_id(self, det: Any) -> Optional[str]:
        if not hasattr(det, 'track_id'):
            return None
        try:
            track_id = int(det.track_id)
        except Exception:
            return None
        if track_id < 0:
            return None
        return str(track_id)

    def _lookup_track_mapping(self, yolo_track_id: str) -> Optional[str]:
        object_id = self.yolo_id_to_object_id.get(yolo_track_id)
        if object_id is None:
            return None
        self.track_id_last_seen_sec[yolo_track_id] = self.get_clock().now().nanoseconds / 1e9
        return object_id

    def _update_track_mapping(self, yolo_track_id: Optional[str], object_id: str, now_sec: Optional[float] = None) -> None:
        if yolo_track_id is None:
            return
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
        self.yolo_id_to_object_id[yolo_track_id] = str(object_id)
        self.track_id_last_seen_sec[yolo_track_id] = now_sec

    def _lookup_body_object_for_face(
        self,
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
    ) -> Optional[str]:
        """Find the correct body cluster for a face by track + time proximity.

        A scene can contain several people, so linking a face to a body cluster by
        scene alone would pick the wrong body. Instead we match on the shared
        yolo_track_id AND require the body observation to be close in time to the
        face (BoTSORT reuses track ids after a track dies, so an unbounded track
        match would link faces from a different, later use of the same id). Returns
        the object_id of the nearest-in-time person body observation for this track,
        or None when there is no suitably close match.
        """
        if yolo_track_id is None or frame_sec is None:
            return None
        window = float(getattr(self, "face_body_link_window_sec", 5.0) or 5.0)
        try:
            row = self._db_execute(
                """
                SELECT oo.object_id
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                WHERE oo.yolo_track_id = %s
                  AND o.class_id = %s
                  AND ABS(EXTRACT(EPOCH FROM (oo.created_at - to_timestamp(%s)))) <= %s
                ORDER BY ABS(EXTRACT(EPOCH FROM (oo.created_at - to_timestamp(%s)))) ASC
                LIMIT 1
                """,
                (yolo_track_id, int(self.person_class_id), frame_sec, window, frame_sec),
                fetchone=True,
            )
        except Exception as exc:
            self.get_logger().warning(f"Face->body lookup failed for track {yolo_track_id}: {exc}")
            return None
        return str(row[0]) if row is not None else None

    def _find_person_observation_for_face(
        self,
        yolo_track_id: Optional[str],
        scene_id: Optional[int],
        person_bbox: Optional[tuple],
        frame_sec: Optional[float],
        face_bbox: Optional[tuple] = None,
    ) -> Optional[tuple]:
        """Find the single person observation (class 0) this face belongs to.

        The face detector lags ~30s behind the body pipeline, so the face's header
        stamp (frame_sec, capture time) is far earlier than the body observation's
        created_at (persist time). Matching must therefore be LAG-INDEPENDENT: rely on
        track_id + scene_id (both assigned from the same source frame), NOT on a tight
        time window. bbox containment is used only to disambiguate when several people
        share a (track, scene) or a scene.

        Match priority:
          1. track + scene (lag-independent); containment tiebreak among duplicates.
          2. scene + best containment (any track) — when track is unknown/unmatched.
          3. track + closest-in-time with a wide window (last resort).

        Returns (observation_id, object_id, scene_id) of the match, or None when the
        person observation was not persisted (the face is then dropped by the caller).
        """
        def _containment_key(r, fx, fy):
            """(face_center_inside, h_off) for a candidate obs row; lower h_off is better."""
            px_min, py_min, px_max, py_max = int(r[3]), int(r[4]), int(r[5]), int(r[6])
            pw = max(1, px_max - px_min)
            ph = max(1, py_max - py_min)
            face_cx, face_cy = fx, fy
            inside = (px_min <= face_cx <= px_max and py_min <= face_cy <= py_max)
            person_cx = (px_min + px_max) / 2.0
            h_off = abs(face_cx - person_cx) / pw
            v_rel = (face_cy - py_min) / ph
            return inside, h_off, v_rel

        def _iou_with_obs(r, pbbox):
            """IoU between the face's person crop bbox and a candidate observation bbox.

            The face detector reports person_bbox = the exact person crop the face was
            detected on (full-frame coords). The observation we link the face to MUST
            be that same crop; the body/YOLO pipeline and the face pipeline can produce
            different bboxes for the 'same' person (different detector, motion, or track
            reuse), so matching by track+scene alone can link a face to an observation
            whose bbox does not actually contain it. IoU against person_bbox is the
            ground-truth check."""
            if pbbox is None:
                return 0.0
            try:
                ax1, ay1, ax2, ay2 = [float(v) for v in pbbox]
                bx1, by1, bx2, by2 = float(r[3]), float(r[4]), float(r[5]), float(r[6])
            except Exception:
                return 0.0
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0.0:
                return 0.0
            area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
            return inter / (area_a + area_b - inter)

        face_center = None
        if face_bbox is not None:
            try:
                fx_min, fy_min, fx_max, fy_max = [int(v) for v in face_bbox]
                face_center = ((fx_min + fx_max) / 2.0, (fy_min + fy_max) / 2.0)
            except Exception:
                face_center = None

        # Minimum IoU between the face's person crop and the linked observation's bbox.
        # Below this the observation is a different crop than the one the face was
        # detected on, so linking would attach the face to an observation that does not
        # contain it. 0 disables (legacy track+scene behaviour).
        min_iou = float(getattr(self, "face_body_link_min_iou", 0.0) or 0.0)

        # 0. person-crop IoU match across RECENT observations (scene/track-agnostic).
        # This is the most robust link: the face detector reports person_bbox = the
        # exact person crop the face was detected on. The persisted observation for that
        # SAME detection has (nearly) the same bbox. Matching by track+scene is UNRELIABLE
        # because (a) BoTSORT tracks drift across people sitting close, and (b) the
        # face's scene_id is stamped from a laggy lookup and can point at an adjacent
        # frame whose same-track observation is a DIFFERENT person. We therefore search
        # recent person observations for the one whose bbox best matches person_bbox and
        # require a high IoU. Observed: the correct crop matches at IoU ~0.85-1.0 while a
        # wrong same-track/same-scene crop is ~0.4-0.5, so a 0.5+ floor separates them.
        if person_bbox is not None and min_iou > 0.0:
            try:
                # Bound the search to recently-persisted person observations (the face
                # lags the body by only a few seconds, plus the pending-face retry
                # window), so we do not scan the whole table.
                lookback = max(float(getattr(self, "face_pending_max_age_sec", 120.0) or 120.0), 60.0)
                rows = self._db_fetchall(
                    """
                    SELECT oo.id, oo.object_id, oo.scene_id,
                           oo.bbox_x_min, oo.bbox_y_min, oo.bbox_x_max, oo.bbox_y_max
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE o.class_id = %s
                      AND oo.created_at >= NOW() - make_interval(secs => %s)
                    ORDER BY oo.id DESC
                    LIMIT 400
                    """,
                    (int(self.person_class_id), lookback),
                ) or []
                best = None
                best_iou = -1.0
                for r in rows:
                    iou = _iou_with_obs(r, person_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best = r
                # Use a higher floor for this high-confidence path.
                if best is not None and best_iou >= max(min_iou, 0.5):
                    return (int(best[0]), int(best[1]), int(best[2]) if best[2] is not None else None)
            except Exception as exc:
                self.get_logger().warning(f"Face->obs person-crop IoU match failed: {exc}")
            # NOTE: if no recent observation matches the person crop at high IoU, fall
            # through to the track+scene / scene-containment paths below. The face and
            # body pipelines run DIFFERENT detectors, so even a correct person crop can
            # have low IoU with the persisted observation bbox; requiring a high IoU
            # here would drop nearly every face. The high-IoU path above only fires on
            # genuinely confident matches; everything else uses the legacy linking.

        # 1. track + scene (lag-independent). This is how faces actually link to bodies.
        if yolo_track_id is not None and scene_id is not None:
            try:
                rows = self._db_fetchall(
                    """
                    SELECT oo.id, oo.object_id, oo.scene_id,
                           oo.bbox_x_min, oo.bbox_y_min, oo.bbox_x_max, oo.bbox_y_max
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE oo.yolo_track_id = %s
                      AND oo.scene_id = %s
                      AND o.class_id = %s
                    ORDER BY oo.id DESC
                    """,
                    (yolo_track_id, int(scene_id), int(self.person_class_id)),
                ) or []
                if len(rows) >= 1:
                    # Pick the observation whose bbox best matches the face's person
                    # crop (highest IoU). This is the ground-truth link: the face was
                    # detected on that exact crop. The body/YOLO pipeline and the face
                    # pipeline can produce different bboxes for the 'same' person
                    # (different detector, motion, or track reuse), so matching by
                    # track+scene alone can link a face to an observation whose bbox
                    # does not actually contain it. Containment of the face center is
                    # only a fallback tiebreak when person_bbox is unavailable.
                    best = None
                    best_iou = -1.0
                    best_key = None
                    for r in rows:
                        if person_bbox is not None:
                            iou = _iou_with_obs(r, person_bbox)
                            if iou > best_iou:
                                best_iou = iou
                                best = r
                        else:
                            if face_center is not None:
                                inside, h_off, v_rel = _containment_key(r, face_center[0], face_center[1])
                                key = (0 if inside else 1, h_off)
                            else:
                                key = (1, 0.0)
                            if best_key is None or key < best_key:
                                best_key = key
                                best = r
                    # Enforce the IoU floor: if even the best-matching observation is a
                    # different crop than the one the face came from, do NOT link here —
                    # fall through to the scene-wide search (Path 2) / drop.
                    if best is not None and person_bbox is not None and min_iou > 0.0 and best_iou < min_iou:
                        best = None
                    if best is not None:
                        return (int(best[0]), int(best[1]), int(best[2]) if best[2] is not None else None)
            except Exception as exc:
                self.get_logger().warning(f"Face->obs track+scene match failed for track {yolo_track_id}: {exc}")

        # 2a. scene + best IoU to the face's person crop (any track). This is the
        # robust fallback: find the observation in the scene whose bbox matches the
        # crop the face was actually detected on, regardless of track.
        if scene_id is not None and person_bbox is not None and min_iou > 0.0:
            try:
                rows = self._db_fetchall(
                    """
                    SELECT oo.id, oo.object_id, oo.scene_id,
                           oo.bbox_x_min, oo.bbox_y_min, oo.bbox_x_max, oo.bbox_y_max
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE oo.scene_id = %s
                      AND o.class_id = %s
                    """,
                    (int(scene_id), int(self.person_class_id)),
                ) or []
                best = None
                best_iou = -1.0
                for r in rows:
                    iou = _iou_with_obs(r, person_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best = r
                if best is not None and best_iou >= min_iou:
                    return (int(best[0]), int(best[1]), int(best[2]) if best[2] is not None else None)
            except Exception as exc:
                self.get_logger().warning(f"Face->obs scene IoU match failed for scene {scene_id}: {exc}")

        # 2b. scene + best containment (any track). Person may have moved during the
        # lag, so use a loose horizontal-center threshold (0.5) and require the face
        # in the upper 80% (head region).
        if scene_id is not None and face_center is not None:
            try:
                rows = self._db_fetchall(
                    """
                    SELECT oo.id, oo.object_id, oo.scene_id,
                           oo.bbox_x_min, oo.bbox_y_min, oo.bbox_x_max, oo.bbox_y_max
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE oo.scene_id = %s
                      AND o.class_id = %s
                    """,
                    (int(scene_id), int(self.person_class_id)),
                ) or []
                best = None
                best_h_off = None
                for r in rows:
                    inside, h_off, v_rel = _containment_key(r, face_center[0], face_center[1])
                    if not inside or v_rel > 0.80:
                        continue
                    if best_h_off is None or h_off < best_h_off:
                        best_h_off = h_off
                        best = r
                if best is not None and best_h_off is not None and best_h_off <= 0.50:
                    return (int(best[0]), int(best[1]), int(best[2]) if best[2] is not None else None)
            except Exception as exc:
                self.get_logger().warning(f"Face->obs scene containment match failed for scene {scene_id}: {exc}")

        # 3. track + closest-in-time with a WIDE window (handles the ~30s face lag).
        if yolo_track_id is not None and frame_sec is not None:
            window = max(float(getattr(self, "face_body_link_window_sec", 5.0) or 5.0), 60.0)
            try:
                row = self._db_execute(
                    """
                    SELECT oo.id, oo.object_id, oo.scene_id
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE oo.yolo_track_id = %s
                      AND o.class_id = %s
                      AND ABS(EXTRACT(EPOCH FROM (oo.created_at - to_timestamp(%s)))) <= %s
                    ORDER BY ABS(EXTRACT(EPOCH FROM (oo.created_at - to_timestamp(%s)))) ASC
                    LIMIT 1
                    """,
                    (yolo_track_id, int(self.person_class_id), frame_sec, window, frame_sec),
                    fetchone=True,
                )
                if row is not None:
                    return (int(row[0]), int(row[1]), int(row[2]) if row[2] is not None else None)
            except Exception as exc:
                self.get_logger().warning(f"Face->obs time match failed for track {yolo_track_id}: {exc}")

        # 4. Fallback: face may belong to a person that YOLO misclassified as furniture
        # (e.g. seated person labeled as couch/chair). Search non-person observations in
        # the same scene whose bbox contains the face center.
        if (
            self.reclassify_furniture_with_faces
            and scene_id is not None
            and face_center is not None
        ):
            match = self._find_furniture_observation_containing_face(
                scene_id=scene_id,
                face_center=face_center,
                yolo_track_id=yolo_track_id,
            )
            if match is not None:
                observation_id, object_id, obs_scene_id = match
                person_object_id = self._reclassify_observation_as_person(
                    observation_id=observation_id,
                    yolo_track_id=yolo_track_id,
                )
                if person_object_id is not None:
                    self.get_logger().info(
                        f"Face fallback: reclassified furniture observation {observation_id} "
                        f"to person object {person_object_id}"
                    )
                    return (observation_id, person_object_id, obs_scene_id)
        return None

    def _find_furniture_observation_containing_face(
        self,
        scene_id: int,
        face_center: Tuple[float, float],
        yolo_track_id: Optional[str],
    ) -> Optional[Tuple[int, int, int]]:
        """Find a non-person observation in the scene whose bbox contains the face center."""
        try:
            fx, fy = face_center
            rows = self._db_fetchall(
                """
                SELECT oo.id, oo.object_id, oo.scene_id,
                       oo.bbox_x_min, oo.bbox_y_min, oo.bbox_x_max, oo.bbox_y_max
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                WHERE oo.scene_id = %s
                  AND o.class_id != %s
                  AND oo.bbox_x_min <= %s
                  AND oo.bbox_y_min <= %s
                  AND oo.bbox_x_max >= %s
                  AND oo.bbox_y_max >= %s
                ORDER BY
                    CASE WHEN oo.yolo_track_id = %s THEN 0 ELSE 1 END,
                    (oo.bbox_x_max - oo.bbox_x_min) * (oo.bbox_y_max - oo.bbox_y_min) ASC
                LIMIT 1
                """,
                (
                    int(scene_id),
                    int(self.person_class_id),
                    int(fx),
                    int(fy),
                    int(fx),
                    int(fy),
                    yolo_track_id,
                ),
            ) or []
            if rows:
                r = rows[0]
                return (int(r[0]), int(r[1]), int(r[2]) if r[2] is not None else None)
        except Exception as exc:
            self.get_logger().warning(f"Face->furniture fallback search failed: {exc}")
        return None

    def _reclassify_observation_as_person(
        self,
        observation_id: int,
        yolo_track_id: Optional[str],
    ) -> Optional[int]:
        """Move a misclassified observation into a person object and update its class.

        Returns the person object_id or None.
        """
        try:
            row = self._db_execute(
                """
                SELECT object_id, scene_id
                FROM object_observations
                WHERE id = %s
                """,
                (observation_id,),
                fetchone=True,
            )
            if row is None:
                return None
            current_object_id, scene_id = int(row[0]), row[1]

            # Create a new person object for this observation.
            new_object_id = self._create_object(class_id=self.person_class_id)

            # Move the observation to the person object and update its class.
            self._db_execute(
                """
                UPDATE object_observations
                SET object_id = %s, class_id = %s
                WHERE id = %s
                """,
                (new_object_id, int(self.person_class_id), observation_id),
            )

            # Update track mapping so future detections of this track use the person object.
            if yolo_track_id is not None:
                self._update_track_mapping(yolo_track_id, str(new_object_id))

            return new_object_id
        except Exception as exc:
            self.get_logger().error(
                f"Failed to reclassify observation {observation_id} as person: {exc}"
            )
            return None

    def _prune_stale_track_mappings(self, now_sec: Optional[float] = None) -> None:
        if now_sec is None:
            now_sec = self.get_clock().now().nanoseconds / 1e9
        stale_ids = [
            track_id for track_id, last_seen in self.track_id_last_seen_sec.items()
            if (now_sec - last_seen) > self.track_id_ttl_sec
        ]
        for track_id in stale_ids:
            self.track_id_last_seen_sec.pop(track_id, None)
            self.yolo_id_to_object_id.pop(track_id, None)
            self._embedding_by_track_id.pop(track_id, None)
            self._face_embedding_by_track.pop(track_id, None)
            self._crop_thumb_by_track_id.pop(track_id, None)

    def _select_observation_position(
        self,
        depth_position: Optional[Tuple[float, float, float]],
        dynosam_position: Optional[Tuple[float, float, float]],
        depth_position_frame: str = 'odom',
    ) -> Tuple[Optional[Tuple[float, float, float]], Optional[str], str]:
        if self._position_source_preference == 'depth':
            prioritized_positions = (
                (depth_position, 'depth', depth_position_frame),
                (dynosam_position, 'dynosam', 'odom'),
            )
        else:
            prioritized_positions = (
                (dynosam_position, 'dynosam', 'odom'),
                (depth_position, 'depth', depth_position_frame),
            )

        for position, source, frame in prioritized_positions:
            if position is not None:
                return position, source, frame
        return None, None, 'odom'

    def _persist_or_queue_observation(
        self,
        object_id: str,
        embedding: list[float],
        pending: dict,
    ) -> None:
        depth_position = pending.get('estimated_position')
        dynosam_position = self._lookup_object_position_for_stamp(
            pending.get('dynosam_instance_id'),
            pending.get('stamp_key'),
        )
        position, position_source, position_frame = self._select_observation_position(
            depth_position,
            dynosam_position,
            pending.get('estimated_position_frame', 'odom'),
        )

        if position is not None and position_frame != 'map':
            position = self._transform_object_position_to_map_frame(position, pending.get('stamp_key'))

        if position is None:
            self._enqueue_pending_observation(
                PendingObservationRecord(
                    object_id=object_id,
                    cropped_image=pending['cropped_image'],
                    mask_image=pending.get('mask_image', b''),
                    original_cropped_image=pending.get('original_cropped_image', b''),
                    embedding=embedding,
                    scene_id=pending['scene_id'],
                    yolo_track_id=pending.get('yolo_track_id'),
                    class_id=pending.get('class_id'),
                    stamp_key=pending.get('stamp_key'),
                    dynosam_instance_id=pending.get('dynosam_instance_id'),
                    bbox_x_min=pending.get('bbox_x_min'),
                    bbox_y_min=pending.get('bbox_y_min'),
                    bbox_x_max=pending.get('bbox_x_max'),
                    bbox_y_max=pending.get('bbox_y_max'),
                    estimated_position=pending.get('estimated_position'),
                    robot_position=pending.get('robot_position'),
                    created_wall_time_sec=self.get_clock().now().nanoseconds / 1e9,
                    confidence=pending.get('confidence'),
                    attributes_json=pending.get('attributes_json'),
                    quality_score=pending.get('quality_score'),
                    part_images=pending.get('part_images') or {},
                    part_embeddings=pending.get('part_embeddings') or {},
                    part_preprocessing=pending.get('part_preprocessing') or {},
                    map_id=pending.get('map_id', self._get_active_map_id()),
                    estimated_position_frame=pending.get('estimated_position_frame', 'odom'),
                    detection_backend=pending.get('detection_backend'),
                    embedding_backend=pending.get('embedding_backend'),
                )
            )
            self.get_logger().info(
                f"Queued object observation for object {object_id} while waiting for "
                f"DynoSAM pose for instance {pending.get('dynosam_instance_id')}"
            )
            return

        try:
            # Consolidation may have merged/deleted the resolved object while the
            # embedding was in flight; re-create it rather than fail the FK insert.
            self._ensure_object_exists(object_id, pending.get('class_id'))
        except Exception as exc:
            self.get_logger().error(f"Failed to ensure object {object_id} exists before insert: {exc}")
        observation_id = self._insert_object_observation(
            object_id,
            pending['cropped_image'],
            pending.get('mask_image', b''),
            pending.get('original_cropped_image', b''),
            embedding,
            position[0],
            position[1],
            position[2],
            pending['scene_id'],
            pending.get('yolo_track_id'),
            pending.get('class_id'),
            pending.get('bbox_x_min'),
            pending.get('bbox_y_min'),
            pending.get('bbox_x_max'),
            pending.get('bbox_y_max'),
            pending['robot_position'][0] if pending.get('robot_position') is not None else None,
            pending['robot_position'][1] if pending.get('robot_position') is not None else None,
            pending['robot_position'][2] if pending.get('robot_position') is not None else None,
            None,
            None,
            None,
            position_source,
            self._get_active_map_id(),
            pending.get('confidence'),
            pending.get('attributes_json'),
            pending.get('quality_score'),
            pending.get('detection_backend'),
            pending.get('embedding_backend'),
        )
        pending['observation_id'] = observation_id
        if observation_id is not None:
            self._insert_pending_part_observations(
                observation_id=observation_id,
                object_id=object_id,
                pending=pending,
        )
        self.get_logger().info(
            f"Saved object observation for object {object_id} in scene {pending['scene_id']} "
            f"with coordinates ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})"
        )
        # Update spatial-temporal cache for fast re-identification of the same object
        stamp_key = pending.get('stamp_key')
        if stamp_key is not None:
            self._update_recent_detection_cache(
                int(object_id), position, stamp_key[0] + stamp_key[1] / 1e9,
                pending.get('class_id')
            )

    def _request_part_embeddings(self, pending: dict, bridge: CvBridge) -> None:
        for part_name, part in (pending.get('part_images') or {}).items():
            if part_name not in {"upper_body", "lower_body"}:
                continue
            image = part.get("image")
            if image is None:
                continue
            try:
                preprocessed, preprocessing = resize_for_embedding(
                    image,
                    enabled=self.attribute_config.super_resolution_enabled,
                    method=self.attribute_config.super_resolution_method,
                    min_side=self.attribute_config.super_resolution_min_side,
                )
                pending.setdefault('part_preprocessing', {})[part_name] = preprocessing
                req = GetImageEmbedding.Request()
                req.image = bridge.cv2_to_imgmsg(preprocessed, encoding="bgr8")
                future = self.embedding_client.call_async(req)
                future.add_done_callback(
                    lambda fut, p=pending, name=part_name: self._handle_part_embedding_response(fut, p, name)
                )
            except Exception as exc:
                self.get_logger().warning(f"Failed to request {part_name} embedding: {exc}")

    def _handle_part_embedding_response(self, future, pending: dict, part_name: str) -> None:
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().warning(f"Part embedding future failed for {part_name}: {exc}")
            return
        if result is None or not result.success:
            self.get_logger().warning(
                f"Part embedding service failed for {part_name}: {getattr(result, 'message', 'no result')}"
            )
            return
        try:
            embedding = self._coerce_embedding_dim(list(result.embedding))
            pending.setdefault('part_embeddings', {})[part_name] = embedding
            observation_id = pending.get('observation_id')
            if observation_id is not None:
                self._insert_part_observation(
                    observation_id=observation_id,
                    object_id=pending.get('object_id'),
                    part_name=part_name,
                    part_payload=(pending.get('part_images') or {}).get(part_name) or {},
                    part_attrs=((pending.get('attributes_json') or {}).get('parts') or {}).get(part_name) or {},
                    embedding=embedding,
                    preprocessing=(pending.get('part_preprocessing') or {}).get(part_name) or {},
                )
        except Exception as exc:
            self.get_logger().warning(f"Failed to persist part embedding for {part_name}: {exc}")

    def _insert_pending_part_observations(self, observation_id: int, object_id: str, pending: dict) -> None:
        pending['object_id'] = object_id
        parts = pending.get('part_images') or {}
        for part_name, part_payload in parts.items():
            part_attrs = ((pending.get('attributes_json') or {}).get('parts') or {}).get(part_name) or {}
            embedding = (pending.get('part_embeddings') or {}).get(part_name)
            if not self.attribute_config.part_embeddings_enabled and part_name not in {"upper_body", "lower_body", "feet"}:
                continue
            self._insert_part_observation(
                observation_id=observation_id,
                object_id=object_id,
                part_name=part_name,
                part_payload=part_payload,
                part_attrs=part_attrs,
                embedding=embedding,
                preprocessing=(pending.get('part_preprocessing') or {}).get(part_name) or {},
            )

    def _insert_part_observation(
        self,
        *,
        observation_id: int,
        object_id: Optional[str],
        part_name: str,
        part_payload: dict,
        part_attrs: dict,
        embedding: Optional[list[float]],
        preprocessing: dict,
    ) -> None:
        bbox = part_payload.get("bbox") or part_attrs.get("bbox") or [None, None, None, None]
        colors_json = part_attrs.get("colors")
        quality_score = part_attrs.get("quality_score")
        embedding_literal = self._to_vector_literal(embedding) if embedding else None
        self._db_execute(
            """
            INSERT INTO object_observation_parts (
                observation_id,
                object_id,
                part_name,
                embedding,
                colors_json,
                bbox_x_min,
                bbox_y_min,
                bbox_x_max,
                bbox_y_max,
                quality_score,
                preprocessing_json
            )
            VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (observation_id, part_name) DO UPDATE SET
                object_id = EXCLUDED.object_id,
                embedding = COALESCE(EXCLUDED.embedding, object_observation_parts.embedding),
                colors_json = COALESCE(EXCLUDED.colors_json, object_observation_parts.colors_json),
                bbox_x_min = EXCLUDED.bbox_x_min,
                bbox_y_min = EXCLUDED.bbox_y_min,
                bbox_x_max = EXCLUDED.bbox_x_max,
                bbox_y_max = EXCLUDED.bbox_y_max,
                quality_score = COALESCE(EXCLUDED.quality_score, object_observation_parts.quality_score),
                preprocessing_json = COALESCE(EXCLUDED.preprocessing_json, object_observation_parts.preprocessing_json)
            """,
            (
                observation_id,
                object_id,
                part_name,
                embedding_literal,
                json.dumps(colors_json) if colors_json is not None else None,
                bbox[0] if len(bbox) > 0 else None,
                bbox[1] if len(bbox) > 1 else None,
                bbox[2] if len(bbox) > 2 else None,
                bbox[3] if len(bbox) > 3 else None,
                quality_score,
                json.dumps(preprocessing or {}),
            ),
        )

    def _match_object_by_embedding(
        self,
        embedding: list[float],
        class_id: Optional[int],
    ) -> Tuple[Optional[int], float, float, int]:
        try:
            embedding_vec = self._to_vector_literal(embedding)
            # Exclude over-merged sink clusters from being match candidates. A giant
            # cluster holds many different people's embeddings, so it spuriously
            # produces high-similarity matches for almost anyone and self-reinforces.
            # 0 disables the guard.
            max_match_obs = int(getattr(self, "reid_max_match_cluster_observations", 0) or 0)
            sink_exclusion = ""
            sink_params: tuple = ()
            if max_match_obs > 0:
                sink_exclusion = (
                    "\n                  AND oo.object_id NOT IN ("
                    "\n                      SELECT object_id FROM object_observations"
                    "\n                      GROUP BY object_id HAVING COUNT(*) >= %s"
                    "\n                  )"
                )
                sink_params = (max_match_obs,)
            sql = """
            WITH nearest_neighbors AS (
                SELECT
                    oo.object_id,
                    (1.0 - (oo.embedding <=> %s::vector)) AS similarity
                FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                WHERE (%s IS NULL OR o.class_id = %s)
                  AND COALESCE(oo.quality_score, 1.0) >= %s""" + sink_exclusion + """
                ORDER BY oo.embedding <=> %s::vector ASC
                LIMIT %s
            ),
            per_object_stats AS (
                SELECT
                    object_id,
                    AVG(similarity) AS avg_similarity,
                    MAX(similarity) AS best_similarity,
                    COUNT(*) AS hit_count
                FROM nearest_neighbors
                GROUP BY object_id
            )
            SELECT
                object_id,
                (%s * avg_similarity + (1.0 - %s) * best_similarity) AS score,
                avg_similarity,
                best_similarity,
                hit_count
            FROM per_object_stats
            ORDER BY score DESC, best_similarity DESC, hit_count DESC
            LIMIT 2
            """
            rows = self._db_fetchall(
                sql,
                (
                    embedding_vec,
                    class_id,
                    class_id,
                    self.reid_min_observation_quality,
                )
                + sink_params
                + (
                    embedding_vec,
                    self.reid_knn_neighbors,
                    self.reid_average_weight,
                    self.reid_average_weight,
                ),
            )
            if not rows:
                return None, 0.0, 0.0, 0

            best = MatchCandidate(
                object_id=int(rows[0][0]),
                score=float(rows[0][1]),
                avg_similarity=float(rows[0][2]),
                best_similarity=float(rows[0][3]),
                hit_count=int(rows[0][4]),
            )
            score_margin = best.score
            if len(rows) > 1:
                score_margin = best.score - float(rows[1][1])

            return best.object_id, best.score, score_margin, best.hit_count
        except Exception as exc:
            self.get_logger().error(f"Embedding match query failed: {exc}")
            return None, 0.0, 0.0, 0

    def _match_object_by_face_embedding(
        self,
        embedding: list[float],
        class_id: Optional[int],
    ) -> Tuple[Optional[int], float, float, int]:
        try:
            embedding_vec = self._to_vector_literal(embedding)
            sql = """
            WITH nearest_neighbors AS (
                SELECT
                    fo.object_id,
                    (1.0 - (fo.embedding <=> %s::vector)) AS similarity
                FROM face_observations fo
                JOIN objects o ON o.id = fo.object_id
                WHERE (%s IS NULL OR o.class_id = %s)
                  AND fo.embedding IS NOT NULL
                  AND fo.object_id IS NOT NULL
                ORDER BY fo.embedding <=> %s::vector ASC
                LIMIT %s
            ),
            per_object_stats AS (
                SELECT
                    object_id,
                    AVG(similarity) AS avg_similarity,
                    MAX(similarity) AS best_similarity,
                    COUNT(*) AS hit_count
                FROM nearest_neighbors
                GROUP BY object_id
            )
            SELECT
                object_id,
                (%s * avg_similarity + (1.0 - %s) * best_similarity) AS score,
                avg_similarity,
                best_similarity,
                hit_count
            FROM per_object_stats
            ORDER BY score DESC, best_similarity DESC, hit_count DESC
            LIMIT 2
            """
            rows = self._db_fetchall(
                sql,
                (
                    embedding_vec,
                    class_id,
                    class_id,
                    embedding_vec,
                    self.reid_knn_neighbors,
                    self.reid_average_weight,
                    self.reid_average_weight,
                ),
            )
            if not rows:
                return None, 0.0, 0.0, 0

            best = MatchCandidate(
                object_id=int(rows[0][0]),
                score=float(rows[0][1]),
                avg_similarity=float(rows[0][2]),
                best_similarity=float(rows[0][3]),
                hit_count=int(rows[0][4]),
            )
            score_margin = best.score
            if len(rows) > 1:
                score_margin = best.score - float(rows[1][1])

            return best.object_id, best.score, score_margin, best.hit_count
        except Exception as exc:
            self.get_logger().error(f"Face embedding match query failed: {exc}")
            return None, 0.0, 0.0, 0

    def _resolve_object_id_for_detection(
        self,
        embedding: list[float],
        mapped_object_id: Optional[int],
        class_id: Optional[int],
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
        estimated_position: Optional[Tuple[float, float, float]] = None,
        detection_colors: Optional[Dict[str, float]] = None,
    ) -> int:
        # 1. Short-term tracker continuity
        if mapped_object_id is not None:
            # Anti-sink: a track still mapped to an over-merged sink cluster would
            # keep feeding the sink via continuity, bypassing the embedding guard.
            # Drop the mapping so the detection is re-resolved by embedding (which
            # excludes sink clusters and will create/find a normal-sized cluster).
            if self._is_sink_cluster(mapped_object_id):
                self.get_logger().info(
                    f"Track mapping {mapped_object_id} dropped for track {yolo_track_id}: "
                    f"mapped cluster is an over-merged sink; re-resolving by embedding"
                )
                if yolo_track_id is not None:
                    self.yolo_id_to_object_id.pop(yolo_track_id, None)
                    self.track_id_last_seen_sec.pop(yolo_track_id, None)
                mapped_object_id = None
        if mapped_object_id is not None:
            last_pos = self._get_object_last_db_position(mapped_object_id)
            dist = self._position_distance_m(estimated_position, last_pos)
            if dist is not None and dist > self.reid_max_position_distance_m and not self._is_portable_class(class_id):
                self.get_logger().info(
                    f"Track mapping {mapped_object_id} rejected for track {yolo_track_id}: "
                    f"distance {dist:.2f}m > {self.reid_max_position_distance_m}m "
                    f"(new={estimated_position}, last={last_pos})"
                )
                if yolo_track_id is not None:
                    self.yolo_id_to_object_id.pop(yolo_track_id, None)
                    self.track_id_last_seen_sec.pop(yolo_track_id, None)
                mapped_object_id = None
            else:
                try:
                    self._ensure_object_exists(mapped_object_id, class_id)
                except Exception as exc:
                    self.get_logger().error(f"Failed to ensure mapped object exists: {exc}")
                self._update_track_mapping(yolo_track_id, mapped_object_id, frame_sec)
                return mapped_object_id

        # 2. Spatial-temporal cache: if a recent detection of the same class is very close,
        # prefer it over a full embedding search (handles viewpoint/lighting changes).
        cached_object_id = self._check_recent_detection_cache(
            estimated_position, class_id, frame_sec
        )
        if cached_object_id is not None and self._is_sink_cluster(cached_object_id):
            self.get_logger().info(
                f"Recent-detection cache hit {cached_object_id} ignored for track {yolo_track_id}: "
                f"cached cluster is an over-merged sink"
            )
            cached_object_id = None
        if cached_object_id is not None:
            try:
                self._ensure_object_exists(cached_object_id, class_id)
            except Exception as exc:
                self.get_logger().error(f"Failed to ensure cached object exists: {exc}")
            self._update_track_mapping(yolo_track_id, cached_object_id, frame_sec)
            return cached_object_id

        # 3. Parallel embedding-based matching (body + optional face for people)
        body_match_id, body_sim, body_margin, body_hits = self._match_object_by_embedding(
            embedding, class_id
        )
        face_match_id, face_sim, face_margin, face_hits = None, 0.0, 0.0, 0
        if self._is_person_class(class_id) and yolo_track_id is not None:
            face_embedding = self._get_face_embedding_for_track(yolo_track_id, frame_sec)
            if face_embedding is not None:
                face_match_id, face_sim, face_margin, face_hits = self._match_object_by_face_embedding(
                    face_embedding, class_id
                )

        matched_object_id = None
        similarity = 0.0
        score_margin = 0.0
        hit_count = 0
        use_face = False

        body_reid_threshold = (
            self.embedding_reid_similarity_threshold
            if self._is_person_class(class_id)
            else self.embedding_reid_similarity_threshold_objects
        )
        body_passes = (
            body_match_id is not None
            and body_sim >= body_reid_threshold
            and body_hits >= self.reid_min_neighbor_hits
            and body_margin >= self.reid_min_score_margin
        )
        face_passes = (
            face_match_id is not None
            and face_sim >= self.face_reid_similarity_threshold
            and face_hits >= self.reid_min_neighbor_hits
            and face_margin >= self.face_reid_min_score_margin
        )

        if body_passes and face_passes:
            if face_match_id != body_match_id:
                # Face and body disagree. Face embeddings (ArcFace) are far more
                # discriminative than body embeddings (OSNet), so trust the face.
                matched_object_id, similarity, score_margin, hit_count = (
                    face_match_id, face_sim, face_margin, face_hits
                )
                use_face = True
                self.get_logger().info(
                    f"Face override: face match {face_match_id} (sim {face_sim:.3f}) "
                    f"disagrees with body match {body_match_id} (sim {body_sim:.3f}); "
                    f"using face for track {yolo_track_id}"
                )
            elif face_sim > body_sim:
                matched_object_id, similarity, score_margin, hit_count = (
                    face_match_id, face_sim, face_margin, face_hits
                )
                use_face = True
            else:
                matched_object_id, similarity, score_margin, hit_count = (
                    body_match_id, body_sim, body_margin, body_hits
                )
        elif face_passes:
            matched_object_id, similarity, score_margin, hit_count = (
                face_match_id, face_sim, face_margin, face_hits
            )
            use_face = True
        elif body_passes:
            matched_object_id, similarity, score_margin, hit_count = (
                body_match_id, body_sim, body_margin, body_hits
            )

        # Anti-sink face gate (persons only): a body-ONLY match (no confirming face)
        # must not feed an already-large person cluster. OSNet body embeddings are too
        # weak to be trusted for this — offline analysis showed cross-person best-match
        # similarities up to ~0.95, so an unguarded body merge snowballs a multi-person
        # sink. When the matched person cluster already holds >=
        # person_body_merge_face_required_obs observations and no face passed, reject
        # the body match so a new (small) cluster is created instead; a later face can
        # still merge it correctly via _correct_body_cluster_by_face_identity.
        if (
            matched_object_id is not None
            and not use_face
            and self._is_person_class(class_id)
        ):
            face_required_obs = int(getattr(self, "person_body_merge_face_required_obs", 0) or 0)
            if face_required_obs > 0:
                try:
                    row = self._db_execute(
                        "SELECT COUNT(*) FROM object_observations WHERE object_id = %s",
                        (int(matched_object_id),),
                        fetchone=True,
                    )
                    cluster_obs = int(row[0]) if row else 0
                except Exception:
                    cluster_obs = 0
                if cluster_obs >= face_required_obs:
                    self.get_logger().info(
                        f"Body-only person match {matched_object_id} rejected for track {yolo_track_id}: "
                        f"target cluster large ({cluster_obs} obs >= {face_required_obs}) and no face confirmation "
                        f"(body sim {body_sim:.3f}); creating a fresh cluster instead"
                    )
                    matched_object_id = None

        # Step B (targeted harder body-merge gate, persons only): a body-ONLY match
        # into a cluster that ALREADY has a confirmed face identity must clear a much
        # higher similarity bar. OSNet cross-person best-match reaches ~0.95-0.97 for
        # this cast, so at the default 0.85 threshold a body-only match freely merges
        # DIFFERENT people into a face-identified cluster (the residual contamination
        # Step A then has to split). Require body_sim >=
        # person_body_merge_face_identified_sim to merge into a face-identified cluster
        # without a confirming face; otherwise reject so a fresh cluster is created
        # (a later face can still merge it correctly). Only applies when the target
        # cluster has a dominant face person; clusters without face identity are
        # unaffected (they have no identity to protect yet).
        if (
            matched_object_id is not None
            and not use_face
            and self._is_person_class(class_id)
        ):
            face_ident_sim = float(getattr(self, "person_body_merge_face_identified_sim", 0.0) or 0.0)
            if face_ident_sim > 0.0:
                dominant_face = self._dominant_face_person_for_cluster(int(matched_object_id))
                if dominant_face is not None and float(body_sim) < face_ident_sim:
                    self.get_logger().info(
                        f"Body-only person match {matched_object_id} rejected for track {yolo_track_id}: "
                        f"target cluster has confirmed face identity (person {dominant_face}) and body sim "
                        f"{body_sim:.3f} < {face_ident_sim:.3f}; creating a fresh cluster instead"
                    )
                    matched_object_id = None

        if matched_object_id is not None:
            # Color-compatibility gate (non-person objects): if the detection's
            # dominant colors clearly conflict with the candidate cluster's color
            # profile (e.g. a pink laptop vs a black-laptop cluster), reject the
            # match so a new cluster is created instead of over-merging.
            if not self._colors_compatible_with_cluster(detection_colors, matched_object_id, class_id):
                self.get_logger().info(
                    f"{'Face' if use_face else 'Body'} match {matched_object_id} rejected for track {yolo_track_id}: "
                    f"color profile conflicts with cluster (class {class_id})"
                )
                matched_object_id = None
        if matched_object_id is not None:
            last_pos = self._get_object_last_db_position(matched_object_id)
            dist = self._position_distance_m(estimated_position, last_pos)
            if dist is not None and dist > self.reid_max_position_distance_m and not self._is_portable_class(class_id):
                self.get_logger().info(
                    f"{'Face' if use_face else 'Body'} match {matched_object_id} rejected for track {yolo_track_id}: "
                    f"distance {dist:.2f}m > {self.reid_max_position_distance_m}m "
                    f"(new={estimated_position}, last={last_pos})"
                )
                matched_object_id = None
            else:
                self.get_logger().info(
                    f"{'Face' if use_face else 'Body'} re-id matched object {matched_object_id} with score {similarity:.3f}, "
                    f"margin {score_margin:.3f}, neighbors {hit_count}"
                )
                try:
                    self._ensure_object_exists(matched_object_id, class_id)
                except Exception as exc:
                    self.get_logger().error(f"Failed to ensure matched object exists: {exc}")
                self._update_track_mapping(yolo_track_id, matched_object_id, frame_sec)
                return matched_object_id

        # 4. Long-term spatial match for static objects (chairs, tables, etc.):
        # if a prior depth/dynosam observation of the same class is at the same
        # position, reuse that object cluster instead of creating a duplicate.
        static_object_id = self._match_static_object_by_position(
            estimated_position, class_id
        )
        if static_object_id is not None and not self._colors_compatible_with_cluster(
            detection_colors, static_object_id, class_id
        ):
            self.get_logger().info(
                f"Static position match {static_object_id} rejected for track {yolo_track_id}: "
                f"color profile conflicts with cluster (class {class_id})"
            )
            static_object_id = None
        if static_object_id is not None:
            try:
                self._ensure_object_exists(static_object_id, class_id)
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to ensure static-matched object exists: {exc}"
                )
            self._update_track_mapping(yolo_track_id, static_object_id, frame_sec)
            self.get_logger().info(
                f"Static position match reused object {static_object_id} "
                f"instead of creating a new object"
            )
            return static_object_id

        object_id = self._create_object(class_id=class_id)
        try:
            self._ensure_object_exists(object_id, class_id)
        except Exception as exc:
            self.get_logger().error(f"Failed to ensure new object exists: {exc}")
            if body_match_id is not None:
                return body_match_id
        self._update_track_mapping(yolo_track_id, object_id, frame_sec)
        if body_match_id is not None:
            self.get_logger().info(
                f"Body embedding candidate {body_match_id} rejected "
                f"(score={body_sim:.3f}, margin={body_margin:.3f}, neighbors={body_hits}); "
                f"created new object {object_id}"
            )
        else:
            self.get_logger().info(f"No embedding neighbor found; created new object {object_id}")
        return object_id

    def _create_object(
        self,
        class_id: Optional[int] = 0,
        canonical_embedding: Optional[list[float]] = None,
    ) -> int:
        normalized_class_id = 0 if class_id is None else int(class_id)
        embedding_vec = self._to_vector_literal(canonical_embedding) if canonical_embedding is not None else None
        row = self._db_execute(
            """
            INSERT INTO objects (class_id, canonical_embedding)
            VALUES (%s, %s::vector)
            RETURNING id
            """,
            (normalized_class_id, embedding_vec),
            fetchone=True,
        )
        return int(row[0])

    def _is_sink_cluster(self, object_id) -> bool:
        """True if the cluster already holds >= reid_max_match_cluster_observations
        observations (an over-merged sink). Result cached briefly to avoid a DB hit
        per detection. Returns False when the guard is disabled (<=0)."""
        max_obs = int(getattr(self, "reid_max_match_cluster_observations", 0) or 0)
        if max_obs <= 0 or object_id is None:
            return False
        try:
            oid = int(object_id)
        except (TypeError, ValueError):
            return False
        import time as _t
        now = _t.time()
        cache = getattr(self, "_sink_cluster_cache", None)
        if cache is None:
            cache = {}
            self._sink_cluster_cache = cache
        ent = cache.get(oid)
        if ent is not None and (now - ent[1]) < 30.0:
            return ent[0]
        try:
            row = self._db_execute(
                "SELECT COUNT(*) FROM object_observations WHERE object_id = %s",
                (oid,),
                fetchone=True,
            )
            is_sink = bool(row and int(row[0]) >= max_obs)
        except Exception as exc:
            self.get_logger().warning(f"Sink cluster check failed for {oid}: {exc}")
            is_sink = False
        cache[oid] = (is_sink, now)
        return is_sink

    def _cluster_color_profile(self, object_id) -> Optional[Dict[str, float]]:
        """Aggregate color histogram for a cluster: the per-bin mean fraction across
        its observations (cached briefly). Returns None if no color data."""
        try:
            oid = int(object_id)
        except (TypeError, ValueError):
            return None
        import time as _t
        now = _t.time()
        cache = getattr(self, "_cluster_color_cache", None)
        if cache is None:
            cache = {}
            self._cluster_color_cache = cache
        ent = cache.get(oid)
        if ent is not None and (now - ent[1]) < 60.0:
            return ent[0]
        profile: Optional[Dict[str, float]] = None
        try:
            rows = self._db_fetchall(
                """
                SELECT attributes_json->'colors'->'histogram' AS hist
                FROM object_observations
                WHERE object_id = %s
                  AND attributes_json->'colors'->'histogram' IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (oid,),
            )
            sums: Dict[str, float] = {}
            n = 0
            for (hist,) in rows or []:
                if not hist:
                    continue
                if isinstance(hist, str):
                    try:
                        hist = json.loads(hist)
                    except Exception:
                        continue
                if not isinstance(hist, dict):
                    continue
                n += 1
                for name, frac in hist.items():
                    try:
                        sums[name] = sums.get(name, 0.0) + float(frac)
                    except (TypeError, ValueError):
                        continue
            if n > 0:
                profile = {name: total / n for name, total in sums.items()}
        except Exception as exc:
            self.get_logger().warning(f"Cluster color profile lookup failed for {oid}: {exc}")
            profile = None
        cache[oid] = (profile, now)
        return profile

    # Colors considered "neutral": they co-occur on many objects and are hard to
    # separate reliably, so they are treated as mutually compatible.
    _NEUTRAL_COLORS = frozenset({"black", "white", "gray"})

    @staticmethod
    def _dominant_color(hist: Optional[Dict[str, float]]) -> Optional[str]:
        if not hist:
            return None
        best_name, best_val = None, 0.0
        for name, frac in hist.items():
            try:
                v = float(frac)
            except (TypeError, ValueError):
                continue
            if v > best_val:
                best_val, best_name = v, name
        return best_name

    def _colors_compatible_with_cluster(
        self,
        detection_colors: Optional[Dict[str, float]],
        object_id,
        class_id: Optional[int],
    ) -> bool:
        """True if the detection's dominant color is compatible with the cluster's
        dominant color. Chromatic colors (pink/blue/red/...) must match; neutral
        colors (black/white/gray) are mutually compatible since they co-occur and
        are hard to separate. Only gates non-person classes (persons use face/body
        identity, and clothing color varies). Returns True (compatible) whenever
        there is insufficient color data or the gate is disabled, so it never
        blocks a match it cannot judge."""
        if not bool(getattr(self, "object_color_gate_enabled", False)):
            return True  # gate disabled
        if self._is_person_class(class_id):
            return True  # persons use face/body identity, not color
        det_dom = self._dominant_color(detection_colors)
        if det_dom is None:
            return True  # no detection color info -> cannot judge
        profile = self._cluster_color_profile(object_id)
        cluster_dom = self._dominant_color(profile)
        if cluster_dom is None:
            return True  # no cluster color info -> cannot judge
        if det_dom == cluster_dom:
            return True
        # Both neutral -> compatible (black/gray/white objects are hard to separate).
        if det_dom in self._NEUTRAL_COLORS and cluster_dom in self._NEUTRAL_COLORS:
            return True
        return False

    def _clusters_color_compatible(self, object_id_a, object_id_b, class_id: Optional[int]) -> bool:
        """True if two clusters' dominant colors are compatible for merging. Uses the
        same neutral-aware dominant-color rule as the detection-level gate. Only
        gates non-person classes; returns True whenever color data is insufficient
        or the gate is disabled, so it never blocks a merge it cannot judge."""
        if not bool(getattr(self, "object_color_gate_enabled", False)):
            return True
        if self._is_person_class(class_id):
            return True
        dom_a = self._dominant_color(self._cluster_color_profile(object_id_a))
        dom_b = self._dominant_color(self._cluster_color_profile(object_id_b))
        if dom_a is None or dom_b is None:
            return True
        if dom_a == dom_b:
            return True
        if dom_a in self._NEUTRAL_COLORS and dom_b in self._NEUTRAL_COLORS:
            return True
        return False

    def _dominant_face_person_for_cluster(self, object_id) -> Optional[str]:
        """Return the face person_id with the most face observations linked to this
        body cluster, or None if the cluster has no usable face identity."""
        try:
            row = self._db_execute(
                """
                SELECT person_id, COUNT(*) AS n
                FROM face_observations
                WHERE object_id = %s AND person_id IS NOT NULL
                GROUP BY person_id
                ORDER BY n DESC
                LIMIT 1
                """,
                (object_id,),
                fetchone=True,
            )
        except Exception:
            return None
        if not row:
            return None
        return str(row[0]) if row[0] is not None else None

    def _person_clusters_face_conflict(self, object_id_a, object_id_b, class_id: Optional[int]) -> bool:
        """True if two PERSON clusters are proven to be DIFFERENT people by their face
        identities (each has a dominant face person_id and they differ). Returns False
        whenever either cluster lacks face evidence, so the guard never blocks a merge
        it cannot judge. Only applies to person classes."""
        if not self._is_person_class(class_id):
            return False
        if not bool(getattr(self, "face_identity_consolidation_guard_enabled", True)):
            return False
        pa = self._dominant_face_person_for_cluster(object_id_a)
        pb = self._dominant_face_person_for_cluster(object_id_b)
        if pa is None or pb is None:
            return False
        return pa != pb

    def _get_object_last_db_position(self, object_id) -> Optional[Tuple[float, float, float]]:
        """Return the most recent (x, y, z) for an object from the DB."""
        row = self._db_execute(
            """
            SELECT x, y, z
            FROM object_observations
            WHERE object_id = %s AND x IS NOT NULL AND y IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (object_id,),
            fetchone=True,
        )
        if row is None or row[0] is None or row[1] is None:
            return None
        return (float(row[0]), float(row[1]), float(row[2]) if row[2] is not None else 0.0)

    def _is_static_object_class(self, class_id: Optional[int]) -> bool:
        """Return True if the class is considered static for long-term position matching."""
        if class_id is None:
            return False
        if self._is_person_class(class_id):
            return False
        if self.static_object_class_ids is None:
            return True
        return int(class_id) in self.static_object_class_ids

    def _is_portable_class(self, class_id: Optional[int]) -> bool:
        """Return True if the class is portable (can be carried far from last known position)."""
        if class_id is None:
            return False
        if self.portable_object_class_ids is None:
            return False
        return int(class_id) in self.portable_object_class_ids

    def _match_static_object_by_position(
        self,
        estimated_position: Optional[Tuple[float, float, float]],
        class_id: Optional[int],
    ) -> Optional[int]:
        """Find an existing static object of the same class near the given position.

        Queries the most recent depth/dynosam observation per object and returns
        the object_id if within static_object_max_position_distance_m.
        """
        if estimated_position is None or class_id is None:
            return None
        if not self._is_static_object_class(class_id):
            return None
        try:
            threshold = self.static_object_max_position_distance_m
            if threshold <= 0.0:
                return None
            threshold_sq = threshold * threshold
            sql = """
            WITH latest_positions AS (
                SELECT DISTINCT ON (object_id)
                    object_id,
                    x, y, z
                FROM object_observations
                WHERE class_id = %s
                  AND x IS NOT NULL AND y IS NOT NULL
                  AND position_source IN ('depth', 'dynosam')
                ORDER BY object_id, created_at DESC
            )
            SELECT object_id, x, y, z
            FROM latest_positions
            WHERE POWER(x - %s, 2) + POWER(y - %s, 2) + POWER(z - %s, 2) <= %s
            ORDER BY POWER(x - %s, 2) + POWER(y - %s, 2) + POWER(z - %s, 2) ASC
            LIMIT 1
            """
            row = self._db_execute(
                sql,
                (
                    int(class_id),
                    float(estimated_position[0]),
                    float(estimated_position[1]),
                    float(estimated_position[2]),
                    float(threshold_sq),
                    float(estimated_position[0]),
                    float(estimated_position[1]),
                    float(estimated_position[2]),
                ),
                fetchone=True,
            )
            if row is not None:
                object_id = int(row[0])
                pos = (float(row[1]), float(row[2]), float(row[3]) if row[3] is not None else 0.0)
                dist = self._position_distance_m(estimated_position, pos)
                self.get_logger().info(
                    f"Static position match: object {object_id} at {dist:.2f}m "
                    f"(threshold {threshold:.2f}m)"
                )
                return object_id
        except Exception as exc:
            self.get_logger().error(f"Static object position match query failed: {exc}")
        return None

    def _check_recent_detection_cache(
        self,
        estimated_position: Optional[Tuple[float, float, float]],
        class_id: Optional[int],
        frame_sec: Optional[float],
    ) -> Optional[int]:
        """Check if a very recent detection of the same class is within spatial threshold.

        This bypasses embedding-based matching when viewpoint/lighting changes cause
        the ConvNeXt embedding to drift, but the object is clearly still nearby.
        """
        if estimated_position is None or frame_sec is None or class_id is None:
            return None
        best_object_id = None
        best_dist = None
        now_sec = frame_sec
        # Prune stale entries first
        stale_keys = [
            oid
            for oid, (x, y, z, ts, cached_class_id) in self._recent_detection_cache.items()
            if now_sec - ts > self._recent_detection_cache_ttl_sec
        ]
        for oid in stale_keys:
            self._recent_detection_cache.pop(oid, None)
        for oid, (x, y, z, ts, cached_class_id) in self._recent_detection_cache.items():
            age = now_sec - ts
            if age > self._recent_detection_cache_ttl_sec:
                continue
            if cached_class_id != class_id:
                continue
            dist = self._position_distance_m(estimated_position, (x, y, z))
            if dist is None:
                continue
            # Use half the re-id distance threshold for cache hits (tighter)
            threshold = self.reid_max_position_distance_m * 0.5
            if dist <= threshold:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_object_id = oid
        if best_object_id is not None:
            self.get_logger().info(
                f"Recent-detection cache hit: object {best_object_id} at "
                f"{best_dist:.2f}m (threshold {self.reid_max_position_distance_m * 0.5:.2f}m)"
            )
        return best_object_id

    def _update_recent_detection_cache(
        self,
        object_id: int,
        position: Optional[Tuple[float, float, float]],
        frame_sec: Optional[float],
        class_id: Optional[int] = None,
    ) -> None:
        if position is None or frame_sec is None:
            return
        self._recent_detection_cache[object_id] = (position[0], position[1], position[2], frame_sec, class_id if class_id is not None else 0)

    def _get_face_embedding_for_track(
        self,
        yolo_track_id: Optional[str],
        frame_sec: Optional[float],
    ) -> Optional[list[float]]:
        if yolo_track_id is None:
            return None
        info = self._face_embedding_by_track.get(yolo_track_id)
        if info is None:
            return None
        embedding, ts = info
        if frame_sec is not None and (frame_sec - ts) > self.face_track_id_ttl_sec:
            return None
        return embedding

    def _position_distance_m(self, a: Optional[Tuple[float, float, float]], b: Optional[Tuple[float, float, float]]) -> Optional[float]:
        """Euclidean distance between two 3D points."""
        if a is None or b is None:
            return None
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

    def _ensure_object_exists(self, object_id, class_id=0):
        if class_id is None:
            class_id = 0
        # Check if object_id exists, insert if not
        row = self._db_execute(
            "SELECT id, class_id FROM objects WHERE id=%s",
            (object_id,),
            fetchone=True,
        )
        if row is None:
            self._db_execute(
                "INSERT INTO objects (id, class_id) VALUES (%s, %s)",
                (object_id, class_id)
            )
            self.get_logger().info(f"Inserted new object with id {object_id}")
        else:
            current_class_id = int(row[1])
            # Never downgrade a person object to a non-person class. This prevents
            # tracks that were reclassified to person (e.g. a seated person YOLO
            # labeled as couch that was corrected by face detection) from being
            # turned back into furniture by later misclassifications.
            if current_class_id == self.person_class_id and class_id != self.person_class_id:
                return
            self._db_execute(
                "UPDATE objects SET class_id = %s WHERE id = %s",
                (class_id, object_id),
            )
            self.get_logger().info(f"Object with id {object_id} already exists in database")
            
    def _insert_scene(
        self,
        caption,
        x=None,
        y=None,
        timestamp=None,
        source_frame: Optional[str] = None,
        original_scene_image: Optional[bytes] = None,
        stitched_scene_image: Optional[bytes] = None,
        map_id: Optional[int] = None,
    ):
        # Insert into scenes table and return the new id
        caption_embedding = self._embed_scene_caption(str(caption or ""))
        sql = """
        INSERT INTO scenes (caption, caption_embedding, x, y, timestamp, original_scene_image, stitched_scene_image, source_frame, map_id)
        VALUES (%s, %s::vector, %s, %s, COALESCE(%s, NOW()), %s, %s, %s, %s)
        RETURNING id
        """
        original_scene_image_param = psycopg2.Binary(original_scene_image) if original_scene_image else None
        stitched_scene_image_param = psycopg2.Binary(stitched_scene_image) if stitched_scene_image else None
        row = self._db_execute(
            sql,
            (
                caption,
                self._to_vector_literal(caption_embedding) if caption_embedding else None,
                x,
                y,
                timestamp,
                original_scene_image_param,
                stitched_scene_image_param,
                source_frame,
                map_id,
            ),
            fetchone=True,
        )
        return row[0]

    def _insert_interaction(
        self,
        action: str,
        caption: str,
        model_source: Optional[str] = None,
        subject_bbox: Optional[Dict[str, int]] = None,
        object_bbox: Optional[Dict[str, int]] = None,
        subject_id: Optional[int] = None,
        object_id: Optional[int] = None,
        scene_id: Optional[int] = None,
        map_id: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> Optional[int]:
        sql = """
        INSERT INTO interactions (
            action,
            caption,
            model_source,
            subject_bbox,
            object_bbox,
            subject_id,
            object_id,
            scene_id,
            map_id,
            confidence
        )
        VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
        RETURNING id
        """
        row = self._db_execute(
            sql,
            (
                action or None,
                caption,
                model_source or None,
                json.dumps(subject_bbox) if subject_bbox is not None else None,
                json.dumps(object_bbox) if object_bbox is not None else None,
                subject_id,
                object_id,
                scene_id,
                map_id,
                confidence,
            ),
            fetchone=True,
        )
        if row is None:
            self.get_logger().warning(
                "Interaction insert returned no id; skipping interaction linkage for this record"
            )
            return None
        return int(row[0])

    def _lookup_scene_id_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[int]:
        exact = self._scene_id_by_stamp.get(stamp_key)
        if exact is not None:
            return int(exact)

        target_sec = self._stamp_key_to_seconds(stamp_key)

        # Exact match in database by timestamp
        row = self._db_execute(
            """
            SELECT id
            FROM scenes
            WHERE timestamp = to_timestamp(%s)
            LIMIT 1
            """,
            (target_sec,),
            fetchone=True,
        )
        if row is not None:
            scene_id = int(row[0])
            self._remember_scene_id_for_stamp(stamp_key, scene_id)
            return scene_id

        # Fallback: nearest cached scene within small window
        best_key = None
        best_delta = None
        nearest_scene_id = None
        for key, scene_id in self._scene_id_by_stamp.items():
            delta = abs(target_sec - self._stamp_key_to_seconds(key))
            if delta > self._interaction_scene_match_window_sec:
                continue
            if best_delta is None or delta < best_delta:
                best_key = key
                best_delta = delta
                nearest_scene_id = scene_id
        if nearest_scene_id is not None:
            self.get_logger().warning(
                f"Scene exact match missed for stamp {stamp_key}; "
                f"falling back to cached scene {nearest_scene_id} at delta={best_delta:.3f}s"
            )
            return int(nearest_scene_id)

        # Fallback: nearest scene in database within small window
        row = self._db_execute(
            """
            SELECT id
            FROM scenes
            WHERE ABS(EXTRACT(EPOCH FROM (timestamp - to_timestamp(%s)))) <= %s
            ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - to_timestamp(%s)))) ASC
            LIMIT 1
            """,
            (target_sec, self._interaction_scene_match_window_sec, target_sec),
            fetchone=True,
        )
        if row is not None:
            scene_id = int(row[0])
            self.get_logger().warning(
                f"Scene exact match missed for stamp {stamp_key}; "
                f"falling back to DB scene {scene_id}"
            )
            self._remember_scene_id_for_stamp(stamp_key, scene_id)
            return scene_id
        return None

    def _lookup_observation_id_for_ref(self, scene_id: int, ref: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        track_id = str(ref.get("track_id") or "").strip()
        detection_id = ref.get("detection_id")
        class_id = ref.get("class_id")
        x_min = ref.get("x_min")
        y_min = ref.get("y_min")
        x_max = ref.get("x_max")
        y_max = ref.get("y_max")

        # Only trust track_id for matching when it is a genuine tracker id.
        # interaction_description_node sets track_id = str(det.track_id) when the
        # tracker assigned a non-negative id, otherwise it falls back to
        # str(det.instance_id) (untracked detection). instance_id values are reused
        # across frames/scenes and are NOT stable observation keys, so matching on
        # them can bind the interaction to a stale observation. When track_id equals
        # detection_id we cannot distinguish a real tracker id from the instance_id
        # fallback, so we treat it as untrusted and fall through to the bbox match.
        track_id_reliable = bool(track_id) and (
            detection_id is None or track_id != str(detection_id)
        )
        if track_id and track_id_reliable:
            row = self._db_execute(
                """
                                SELECT id, object_id
                FROM object_observations
                WHERE scene_id = %s
                  AND yolo_track_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (scene_id, track_id),
                fetchone=True,
            )
            if row is not None:
                return int(row[0]), int(row[1])

        if (
            class_id is not None
            and None not in (x_min, y_min, x_max, y_max)
        ):
            row = self._db_execute(
                """
                                SELECT id, object_id
                FROM object_observations
                WHERE scene_id = %s
                  AND class_id = %s
                ORDER BY
                    ABS(COALESCE(bbox_x_min, 0) - %s)
                  + ABS(COALESCE(bbox_y_min, 0) - %s)
                  + ABS(COALESCE(bbox_x_max, 0) - %s)
                  + ABS(COALESCE(bbox_y_max, 0) - %s),
                    created_at DESC
                LIMIT 1
                """,
                (scene_id, class_id, x_min, y_min, x_max, y_max),
                fetchone=True,
            )
            if row is not None:
                return int(row[0]), int(row[1])

        if detection_id is None or class_id is None:
            return None
        row = self._db_execute(
            """
                        SELECT id, object_id
            FROM object_observations
            WHERE scene_id = %s
              AND class_id = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (scene_id, class_id),
            fetchone=True,
        )
        if row is None:
            return None
        return int(row[0]), int(row[1])

    @staticmethod
    def _interaction_bbox_for_role(
        refs: List[Dict[str, Any]],
        role: str,
    ) -> Optional[Dict[str, int]]:
        for ref in refs:
            if str(ref.get("role") or "").strip().lower() != role:
                continue
            coords = {
                "x_min": int(ref.get("x_min") or 0),
                "y_min": int(ref.get("y_min") or 0),
                "x_max": int(ref.get("x_max") or 0),
                "y_max": int(ref.get("y_max") or 0),
            }
            if coords["x_max"] <= coords["x_min"] or coords["y_max"] <= coords["y_min"]:
                continue
            return coords
        return None

    _INTERACTION_COLOR_WORDS = {
        "red", "green", "blue", "yellow", "orange", "purple", "pink", "black",
        "white", "gray", "grey", "brown", "cyan", "magenta", "teal", "beige",
        "maroon", "olive", "navy", "turquoise", "violet", "indigo", "lime",
        "coral", "salmon", "gold", "silver", "bronze", "tan", "khaki",
    }

    def _extract_color_words(self, text: str) -> set[str]:
        words = set(re.findall(r"\b[a-z]+\b", str(text or "").lower()))
        return words & self._INTERACTION_COLOR_WORDS

    def _get_observation_colors(self, observation_id: int) -> list[tuple[str, float]]:
        try:
            row = self._db_execute(
                "SELECT attributes_json FROM object_observations WHERE id = %s",
                (observation_id,),
                fetchone=True,
            )
            if not row or not row[0]:
                return []
            attrs = row[0]
            if isinstance(attrs, str):
                attrs = json.loads(attrs)
            colors = (attrs or {}).get("colors", {}).get("colors", [])
            return [
                (c["name"], float(c.get("fraction", 0)))
                for c in colors
                if isinstance(c, dict) and "name" in c
            ]
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to get observation colors for {observation_id}: {exc}"
            )
            return []

    def _find_better_observation_for_ref(
        self,
        scene_id: int,
        ref: Dict[str, Any],
        required_colors: set[str],
        current_observation_id: int,
    ) -> Optional[Tuple[int, int, Tuple[int, int, int, int]]]:
        """Find an observation in the same scene whose colors match the caption better.

        Returns (observation_id, object_id, bbox) or None.
        """
        try:
            class_id = ref.get("class_id")
            x_min = ref.get("x_min", 0)
            y_min = ref.get("y_min", 0)
            x_max = ref.get("x_max", 0)
            y_max = ref.get("y_max", 0)
            ref_area = max(1, (x_max - x_min) * (y_max - y_min))

            rows = self._db_fetchall(
                """
                SELECT id, object_id, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max, attributes_json
                FROM object_observations
                WHERE scene_id = %s AND class_id = %s
                ORDER BY created_at DESC
                """,
                (scene_id, class_id),
            )
            best_match = None
            best_score = -1.0
            for row in rows:
                obs_id, obj_id, ox1, oy1, ox2, oy2, attrs = row
                if obs_id == current_observation_id:
                    continue
                if not attrs:
                    continue
                if isinstance(attrs, str):
                    attrs = json.loads(attrs)
                colors = (attrs or {}).get("colors", {}).get("colors", [])
                color_names = {c["name"] for c in colors if isinstance(c, dict) and "name" in c}

                # Compute overlap with ref bbox.
                inter_x1 = max(x_min, ox1 or 0)
                inter_y1 = max(y_min, oy1 or 0)
                inter_x2 = min(x_max, ox2 or 0)
                inter_y2 = min(y_max, oy2 or 0)
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                overlap_ratio = inter_area / ref_area

                matched_colors = required_colors & color_names
                if not matched_colors:
                    continue
                if overlap_ratio < 0.15:
                    continue

                # Score: prefer strong overlap and multiple color matches.
                score = overlap_ratio + len(matched_colors) * 0.5
                if score > best_score:
                    best_score = score
                    best_match = (obs_id, obj_id, (ox1, oy1, ox2, oy2))

            # Require reasonable overlap (same region of the image) and at least one color hit.
            return best_match if best_match is not None else None
        except Exception as exc:
            self.get_logger().warning(f"Failed to find better observation for ref: {exc}")
            return None

    def _correct_interaction_refs_by_caption(
        self,
        record: PendingInteractionRecord,
        scene_id: int,
        observation_ids_by_role: Dict[str, int],
    ) -> bool:
        """Verify caption color mentions match linked observations; swap refs if needed."""
        caption_colors = self._extract_color_words(record.caption)
        if not caption_colors:
            return False

        corrected = False
        for ref in record.refs:
            role = str(ref.get("role") or "").strip().lower()
            if not role:
                continue
            observation_id = observation_ids_by_role.get(role)
            if observation_id is None:
                continue

            obs_colors = self._get_observation_colors(observation_id)
            obs_color_names = {name for name, _ in obs_colors[:6]}
            missing_colors = caption_colors - obs_color_names
            if not missing_colors:
                continue

            self.get_logger().info(
                f"Interaction caption '{record.caption}' color check for {role}: "
                f"observation {observation_id} colors {obs_color_names} do not include "
                f"{missing_colors}; searching for better match"
            )

            better = self._find_better_observation_for_ref(
                scene_id=scene_id,
                ref=ref,
                required_colors=missing_colors,
                current_observation_id=observation_id,
            )
            if better is None:
                continue

            new_obs_id, new_obj_id, new_bbox = better
            self.get_logger().info(
                f"Corrected interaction {role} from observation {observation_id} "
                f"to {new_obs_id} based on caption colors"
            )
            ref["x_min"] = int(new_bbox[0])
            ref["y_min"] = int(new_bbox[1])
            ref["x_max"] = int(new_bbox[2])
            ref["y_max"] = int(new_bbox[3])
            # Clear track_id so the next lookup uses the corrected bbox.
            ref["track_id"] = ""
            corrected = True

        return corrected

    def _interaction_cache_key(self, record: PendingInteractionRecord) -> Tuple[Any, ...]:
        normalized_refs = tuple(
            sorted(
                (
                    str(ref.get("role") or ""),
                    str(ref.get("track_id") or ""),
                    int(ref.get("class_id") or -1),
                )
                for ref in record.refs
            )
        )
        return (
            record.stamp_key,
            record.action.strip().lower(),
            record.caption.strip().lower(),
            normalized_refs,
        )

    def _interaction_dedup_signature(self, record: PendingInteractionRecord) -> Tuple[Any, ...]:
        """Track/class-based signature for temporal dedup (stable across consecutive frames).

        Uses subject/target track_id + class_id and the normalized action. Excludes the
        per-frame stamp and caption (those vary every frame, which is what the existing
        exact-dup key catches)."""
        subject_track = ""
        subject_class = -1
        target_track = ""
        target_class = -1
        for ref in record.refs:
            role = str(ref.get("role") or "").strip().lower()
            if role == "subject":
                subject_track = str(ref.get("track_id") or "")
                subject_class = int(ref.get("class_id") or -1)
            elif role in ("target", "object"):
                target_track = str(ref.get("track_id") or "")
                target_class = int(ref.get("class_id") or -1)
        return (
            subject_track,
            subject_class,
            target_track,
            target_class,
            record.action.strip().lower(),
        )

    def _queue_interaction_record(self, record: PendingInteractionRecord) -> None:
        cache_key = self._interaction_cache_key(record)
        if cache_key in self._processed_interaction_keys:
            return
        # Temporal dedup: skip if the same (subject,object,action) was persisted recently.
        window = self._interaction_dedup_window_sec
        if window and window > 0:
            now = time.time()
            signature = self._interaction_dedup_signature(record)
            last_seen = self._recent_interaction_signatures.get(signature)
            if last_seen is not None and (now - last_seen) < window:
                self.get_logger().debug(
                    f"Temporal dedup: skipping duplicate interaction '{record.action}' "
                    f"(seen {now - last_seen:.2f}s ago, window {window:.1f}s)"
                )
                return
            self._recent_interaction_signatures[signature] = now
            # Prune stale signatures to bound memory.
            stale = [s for s, ts in self._recent_interaction_signatures.items() if (now - ts) > window]
            for s in stale:
                self._recent_interaction_signatures.pop(s, None)
        self._pending_interactions.append(record)
        self._processed_interaction_keys.add(cache_key)

    def _flush_pending_interactions(self) -> None:
        if not self._pending_interactions:
            return

        remaining: List[PendingInteractionRecord] = []
        for record in self._pending_interactions:
            if not self._try_persist_interaction(record):
                remaining.append(record)
        self._pending_interactions = remaining

    def _has_pending_observations_for_stamp(
        self, stamp_key: Tuple[int, int]
    ) -> bool:
        return any(
            pending.stamp_key == stamp_key for pending in self._pending_observations
        )

    def _try_persist_interaction(self, record: PendingInteractionRecord) -> bool:
        scene_id = self._lookup_scene_id_for_stamp(record.stamp_key)
        if scene_id is None:
            if self._has_pending_observations_for_stamp(record.stamp_key):
                return False
            record.attempts += 1
            if record.attempts < self._interaction_max_retry_attempts:
                return False
            self.get_logger().warning(
                f"Dropping interaction '{record.action}' for stamp {record.stamp_key}: "
                f"no scene found after {record.attempts} attempts"
            )
            return True  # drop: no scene and no pending observations

        observation_ids = []
        observation_ids_by_role: Dict[str, int] = {}
        unresolved_refs = []
        for ref in record.refs:
            observation_ref = self._lookup_observation_id_for_ref(scene_id, ref)
            if observation_ref is None:
                unresolved_refs.append(ref)
                continue
            observation_id, _object_id = observation_ref
            observation_ids.append(observation_id)
            role = str(ref.get("role") or "").strip().lower()
            if role:
                observation_ids_by_role[role] = observation_id

        if unresolved_refs:
            if self._has_pending_observations_for_stamp(record.stamp_key):
                return False
            record.attempts += 1
            if record.attempts < self._interaction_max_retry_attempts:
                return False
            self.get_logger().warning(
                f"Dropping interaction '{record.action}' for stamp {record.stamp_key} in scene "
                f"{scene_id}: {len(unresolved_refs)} unresolved ref(s) after {record.attempts} "
                f"attempts: {unresolved_refs}"
            )
            return True  # drop: unresolved refs and no pending observations

        # Caption consistency correction: if the caption mentions colors/clothing that do not
        # match the linked observation, try to swap to a better matching observation in the
        # same scene (e.g. foreground vs background person when IDs overlap).
        if self._correct_interaction_refs_by_caption(record, scene_id, observation_ids_by_role):
            observation_ids = []
            observation_ids_by_role = {}
            for ref in record.refs:
                observation_ref = self._lookup_observation_id_for_ref(scene_id, ref)
                if observation_ref is None:
                    continue
                observation_id, _object_id = observation_ref
                observation_ids.append(observation_id)
                role = str(ref.get("role") or "").strip().lower()
                if role:
                    observation_ids_by_role[role] = observation_id

        return self._persist_interaction(
            record,
            scene_id,
            observation_ids,
            subject_id=observation_ids_by_role.get("subject"),
            object_id=observation_ids_by_role.get("target") or observation_ids_by_role.get("object"),
        )

    def _persist_interaction(
        self,
        record: PendingInteractionRecord,
        scene_id: Optional[int],
        observation_ids: Optional[List[int]] = None,
        subject_id: Optional[int] = None,
        object_id: Optional[int] = None,
    ) -> bool:
        caption = (record.caption or "").strip()
        if not caption:
            caption = record.action or "interaction"
        subject_bbox = self._interaction_bbox_for_role(record.refs, "subject")
        object_bbox = self._interaction_bbox_for_role(record.refs, "target")

        interaction_id = self._insert_interaction(
            record.action,
            caption,
            model_source=record.model_source,
            subject_bbox=subject_bbox,
            object_bbox=object_bbox,
            subject_id=subject_id,
            object_id=object_id,
            scene_id=scene_id,
            map_id=record.map_id,
            confidence=record.confidence,
        )

        if interaction_id is None:
            return True
        self.get_logger().info(
            f"Saved interaction {interaction_id} for scene {scene_id} with "
            f"{len(observation_ids or [])} matched observations; "
            f"raw_response='{record.raw_response[:200]}...'"
        )
        _int_ts_sec = self._stamp_key_to_seconds(record.stamp_key)
        _int_age_ms = (time.time() - _int_ts_sec) * 1000.0 if _int_ts_sec > 0 else -1.0
        self.get_logger().info(
            f"[latency] interaction: capture_to_interaction_saved={_int_age_ms:.0f}ms "
            f"interaction_id={interaction_id} scene_id={scene_id}"
        )
        return True
    
    def _init_db(self):
        # Tolerate the database being unavailable at startup (e.g. Postgres
        # still booting or being restarted): retry briefly, then continue
        # unconnected. _ensure_db_connection() lazily reconnects on the first
        # query once the database is reachable again.
        deadline = time.monotonic() + 30.0
        attempt = 0
        while True:
            attempt += 1
            try:
                self._connect_db()
                return
            except (psycopg2.InterfaceError, psycopg2.OperationalError) as exc:
                self.db_conn = None
                self.db_cursor = None
                if time.monotonic() >= deadline:
                    self.get_logger().error(
                        f"Database unavailable at startup after {attempt} attempts "
                        f"({exc}); continuing unconnected and retrying lazily"
                    )
                    return
                self.get_logger().warn(
                    f"Database connect failed at startup (attempt {attempt}): {exc}. Retrying..."
                )
                time.sleep(min(0.5 * attempt, 3.0))


    def _insert_object_observation(
        self,
        object_id,
        cropped_image,
        mask_image,
        original_cropped_image,
        embedding,
        x=None,
        y=None,
        z=None,
        scene_id=None,
        yolo_track_id=None,
        class_id=None,
        bbox_x_min=None,
        bbox_y_min=None,
        bbox_x_max=None,
        bbox_y_max=None,
        robot_x=None,
        robot_y=None,
        robot_z=None,
        rel_x=None,
        rel_y=None,
        rel_z=None,
        position_source=None,
        map_id=None,
        confidence=None,
        attributes_json=None,
        quality_score=None,
        detection_backend=None,
        embedding_backend=None,
    ):
        try:
            if class_id is None:
                row = self._db_execute(
                    "SELECT class_id FROM objects WHERE id = %s",
                    (object_id,),
                    fetchone=True,
                )
                if row is not None:
                    class_id = row[0]
            slam_pose = self._get_slam_robot_pose()
            if slam_pose is not None:
                srx, sry, srz, syaw = slam_pose
                robot_x = srx
                robot_y = sry
                robot_z = srz
                if x is not None and y is not None and z is not None and position_source is not None and position_source in ('dynosam', 'depth'):
                    dx = float(x) - srx
                    dy = float(y) - sry
                    dz = float(z) - srz
                    cos_yaw = math.cos(syaw)
                    sin_yaw = math.sin(syaw)
                    rel_x = dx * cos_yaw + dy * sin_yaw
                    rel_y = -dx * sin_yaw + dy * cos_yaw
                    rel_z = dz

            sql = """
            INSERT INTO object_observations (
                object_id,
                scene_id,
                cropped_image,
                mask_image,
                original_cropped_image,
                x,
                y,
                z,
                embedding,
                yolo_track_id,
                class_id,
                robot_x,
                robot_y,
                robot_z,
                rel_x,
                rel_y,
                rel_z,
                position_source,
                bbox_x_min,
                bbox_y_min,
                bbox_x_max,
                bbox_y_max,
                map_id,
                confidence,
                attributes_json,
                quality_score,
                detection_backend,
                embedding_backend
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s,
                %s, %s, %s, %s,
                %s, %s, %s::jsonb, %s,
                %s, %s
            )
            RETURNING id
            """
            self.get_logger().info(f"Inserting object observation for object_id={object_id}, scene_id={scene_id}")
            row = self._db_execute(
                sql,
                (
                    object_id,
                    scene_id,
                    psycopg2.Binary(cropped_image),
                    psycopg2.Binary(mask_image) if mask_image else None,
                    psycopg2.Binary(original_cropped_image) if original_cropped_image else None,
                    x,
                    y,
                    z,
                    self._to_vector_literal(embedding),
                    yolo_track_id,
                    class_id,
                    robot_x,
                    robot_y,
                    robot_z,
                    rel_x,
                    rel_y,
                    rel_z,
                    position_source,
                    bbox_x_min,
                    bbox_y_min,
                    bbox_x_max,
                    bbox_y_max,
                    map_id,
                    confidence,
                    json.dumps(attributes_json) if attributes_json is not None else None,
                    quality_score,
                    detection_backend,
                    embedding_backend,
                ),
                fetchone=True,
            )
            self._observation_insert_count += 1
            self._maybe_run_consolidation()
            return int(row[0]) if row is not None else None
        except Exception as exc:
            self.get_logger().error(f"Failed to insert object observation for object_id={object_id}, scene_id={scene_id}: {exc}")
            raise

    def _enqueue_pending_observation(self, record: PendingObservationRecord) -> None:
        with self._pending_observations_lock:
            self._pending_observations.append(record)

    def _flush_pending_observations(self) -> None:
        # Atomically take ownership of the current queue: appends arriving
        # while we process (slow DB work, possibly seconds under backlog) go
        # to a fresh list and are merged back afterwards instead of being lost.
        with self._pending_observations_lock:
            if not self._pending_observations:
                return
            records = self._pending_observations
            self._pending_observations = []

        remaining: List[PendingObservationRecord] = []
        now_sec = self.get_clock().now().nanoseconds / 1e9
        for record in records:
            dynosam_position = self._lookup_object_position_for_stamp(
                record.dynosam_instance_id,
                record.stamp_key,
            )
            position, position_source, position_frame = self._select_observation_position(
                record.estimated_position,
                dynosam_position,
                record.estimated_position_frame,
            )
            if position is None:
                record.attempts += 1
                if (now_sec - record.created_wall_time_sec) < self._pending_observation_timeout_sec:
                    remaining.append(record)
                    continue
                if record.robot_position is not None:
                    self.get_logger().warning(
                        f"No DynoSAM pose found for object {record.dynosam_instance_id} at "
                        f"{record.stamp_key[0]}.{record.stamp_key[1]} before timeout; "
                        "falling back to robot position"
                    )
                    position = (
                        record.robot_position[0],
                        record.robot_position[1],
                        record.robot_position[2],
                    )
                    position_source = 'yolo'
                else:
                    self.get_logger().warning(
                        f"No DynoSAM pose found for object {record.dynosam_instance_id} at "
                        f"{record.stamp_key[0]}.{record.stamp_key[1]} before timeout; "
                        "saving observation without coordinates"
                    )
                    position = (None, None, None)
                    position_source = 'yolo'

            if position is not None and position[0] is not None and position_frame != 'map':
                position = self._transform_object_position_to_map_frame(position, record.stamp_key)

            try:
                # The merge remap covers records queued at merge time; this catches
                # any record that raced in afterwards (object deleted meanwhile).
                self._ensure_object_exists(record.object_id, record.class_id)
            except Exception as exc:
                self.get_logger().error(
                    f"Failed to ensure object {record.object_id} exists before delayed insert: {exc}"
                )
            try:
                observation_id = self._insert_object_observation(
                    record.object_id,
                    record.cropped_image,
                    record.mask_image,
                    record.original_cropped_image,
                    record.embedding,
                    position[0],
                    position[1],
                    position[2],
                    record.scene_id,
                    record.yolo_track_id,
                    record.class_id,
                    record.bbox_x_min,
                    record.bbox_y_min,
                    record.bbox_x_max,
                    record.bbox_y_max,
                    record.robot_position[0] if record.robot_position is not None else None,
                    record.robot_position[1] if record.robot_position is not None else None,
                    record.robot_position[2] if record.robot_position is not None else None,
                    None,
                    None,
                    None,
                    position_source,
                    record.map_id,
                    record.confidence,
                    record.attributes_json,
                    record.quality_score,
                    record.detection_backend,
                    record.embedding_backend,
                )
                if observation_id is not None:
                    self._insert_pending_part_observations(
                        observation_id=observation_id,
                        object_id=record.object_id,
                        pending={
                            "part_images": record.part_images,
                            "part_embeddings": record.part_embeddings,
                            "part_preprocessing": record.part_preprocessing,
                            "attributes_json": record.attributes_json or {},
                        },
                    )
                if position is not None and position[0] is not None:
                    self._update_recent_detection_cache(
                        int(record.object_id), position,
                        record.stamp_key[0] + record.stamp_key[1] / 1e9,
                        record.class_id,
                    )
            except Exception as exc:
                self.get_logger().error(
                    f"Skipping failed delayed observation insert for object {record.object_id}: {exc}"
                )

        with self._pending_observations_lock:
            self._pending_observations = remaining + self._pending_observations

    def _maybe_run_consolidation(self) -> None:
        if self.consolidation_every_n_observations <= 0:
            return
        if self._observation_insert_count < self.consolidation_every_n_observations:
            return
        self._observation_insert_count = 0
        self._consolidate_clusters_by_similarity()

    def _consolidate_person_clusters_by_face_identity(self) -> None:
        """Merge person body-clusters that share the same face identity.

        Faces (ArcFace) are far more discriminative than body embeddings. If two
        person object clusters are both linked to face observations with the same
        person_id, they are the same physical person even if their body centroids
        are below the similarity threshold. Merge them (keep the larger/older one).
        """
        try:
            rows = self._db_fetchall(
                """
                SELECT
                    fo.person_id,
                    fo.object_id,
                    COUNT(oo.id) AS n_obs,
                    MIN(oo.created_at) AS first_seen
                FROM face_observations fo
                JOIN objects o ON o.id = fo.object_id
                LEFT JOIN object_observations oo ON oo.object_id = fo.object_id
                WHERE fo.object_id IS NOT NULL
                  AND fo.person_id IS NOT NULL
                  AND o.class_id = %s
                GROUP BY fo.person_id, fo.object_id
                ORDER BY fo.person_id, n_obs DESC, first_seen ASC
                """,
                (int(self.person_class_id),),
            )
        except Exception as exc:
            self.get_logger().warning(f"Face-based consolidation lookup failed: {exc}")
            return

        # Group object clusters by face person_id.
        by_person: Dict[str, list] = {}
        for person_id, object_id, n_obs, first_seen in rows or []:
            by_person.setdefault(str(person_id), []).append(
                (str(object_id), int(n_obs or 0), first_seen)
            )

        max_obs = int(getattr(self, "face_correction_max_cluster_observations", 0) or 0)
        for person_id, clusters in by_person.items():
            if len(clusters) < 2:
                continue
            # Keep the cluster with the most observations (ties: oldest first_seen).
            clusters.sort(key=lambda c: (-c[1], c[2] if c[2] is not None else 0))
            keep_id = clusters[0][0]
            # Guard: never merge into an over-merged sink. If the would-be keep
            # cluster already holds a huge number of observations, it is a
            # multi-person sink (its face person_id is polluted); absorbing more
            # clusters only reinforces the over-merge. Skip this person entirely.
            if max_obs > 0 and clusters[0][1] >= max_obs:
                self.get_logger().info(
                    f"Face-based consolidation: skipped person {person_id}; keep cluster {keep_id} "
                    f"too large ({clusters[0][1]} obs >= {max_obs}), likely a sink"
                )
                continue
            for drop_id, _n, _fs in clusters[1:]:
                if drop_id == keep_id:
                    continue
                try:
                    moved = self._db_fetchall(
                        "UPDATE object_observations SET object_id = %s WHERE object_id = %s RETURNING id",
                        (keep_id, drop_id),
                    )
                    # Re-link face observations from the dropped cluster to the kept one.
                    self._db_execute(
                        "UPDATE face_observations SET object_id = %s WHERE object_id = %s",
                        (keep_id, drop_id),
                    )
                    self._db_execute("DELETE FROM objects WHERE id = %s", (drop_id,))
                    for track_id, mapped_object_id in list(self.yolo_id_to_object_id.items()):
                        if mapped_object_id == drop_id:
                            self.yolo_id_to_object_id[track_id] = keep_id
                    self._remap_object_references_after_merge(keep_id, drop_id)
                    self.get_logger().info(
                        f"Face-based consolidation: merged person clusters keep={keep_id}, "
                        f"drop={drop_id}, moved={len(moved or [])}, shared person_id={person_id}"
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"Face-based consolidation failed for person {person_id} (keep={keep_id}, drop={drop_id}): {exc}"
                    )

    def _split_mixed_person_clusters_by_face(self) -> None:
        """Periodic face recluster (Step A): split person clusters that mix people.

        BoTSORT tracks drift across people sitting close together, so a single body
        cluster can accumulate face observations of several DISTINCT person_ids (a
        multi-person sink in the making). OSNet body embeddings cannot separate the
        cast (cross-person sim ~0.95) but ArcFace can (cross-person <=~0.17). For
        every person cluster with a clear DOMINANT face person plus one or more
        significant MINORITY face persons, move the minority persons' observations
        out into their own cluster so each body cluster stays single-identity.

        Conservative by design:
          * only clusters with >= face_recluster_min_cluster_obs observations;
          * only face person_ids with >= face_recluster_min_person_faces faces count
            as a real sub-identity (stray bystander faces are ignored);
          * only split when the dominant person has >= face_recluster_dominant_ratio
            x the runner-up's faces (genuinely ambiguous 50/50 clusters are left alone);
          * an observation is moved only when it is confidently the minority person:
            it has a direct minority face, OR it shares (yolo_track_id, scene_id)
            with a minority-faced observation. Everything else stays with dominant.
        """
        if not getattr(self, "face_recluster_split_enabled", False):
            return
        min_faces = int(getattr(self, "face_recluster_min_person_faces", 4) or 4)
        dom_ratio = float(getattr(self, "face_recluster_dominant_ratio", 1.5) or 1.5)
        min_obs = int(getattr(self, "face_recluster_min_cluster_obs", 12) or 12)

        # Face-person distribution per person cluster.
        try:
            rows = self._db_fetchall(
                """
                SELECT fo.object_id, fo.person_id, COUNT(*) AS n_faces
                FROM face_observations fo
                JOIN objects o ON o.id = fo.object_id
                WHERE fo.object_id IS NOT NULL
                  AND fo.person_id IS NOT NULL
                  AND o.class_id = %s
                GROUP BY fo.object_id, fo.person_id
                ORDER BY fo.object_id, n_faces DESC
                """,
                (int(self.person_class_id),),
            ) or []
        except Exception as exc:
            self.get_logger().warning(f"Face recluster: distribution lookup failed: {exc}")
            return

        by_cluster: Dict[str, list] = {}
        for object_id, person_id, n_faces in rows:
            by_cluster.setdefault(str(object_id), []).append((str(person_id), int(n_faces)))

        for object_id, person_counts in by_cluster.items():
            # Significant sub-identities only.
            significant = [(p, n) for p, n in person_counts if n >= min_faces]
            if len(significant) < 2:
                continue  # single-identity (or only noise minorities) -> clean
            significant.sort(key=lambda kv: -kv[1])
            dominant_person, dominant_faces = significant[0]
            runner_faces = significant[1][1]
            if dominant_faces < dom_ratio * runner_faces:
                continue  # genuinely ambiguous; do not shred
            # Cluster must be big enough to bother.
            try:
                row = self._db_execute(
                    "SELECT COUNT(*) FROM object_observations WHERE object_id = %s",
                    (int(object_id),), fetchone=True,
                )
                cluster_obs = int(row[0]) if row else 0
            except Exception:
                cluster_obs = 0
            if cluster_obs < min_obs:
                continue

            # Split each significant minority person out of this cluster.
            for minority_person, minority_faces in significant[1:]:
                if minority_person == dominant_person:
                    continue
                try:
                    self._split_one_person_out_of_cluster(
                        cluster_id=object_id,
                        minority_person=minority_person,
                        dominant_person=dominant_person,
                        minority_faces=minority_faces,
                    )
                except Exception as exc:
                    self.get_logger().error(
                        f"Face recluster: split of person {minority_person} from cluster "
                        f"{object_id} failed: {exc}"
                    )

    def _split_one_person_out_of_cluster(
        self,
        cluster_id: str,
        minority_person: str,
        dominant_person: str,
        minority_faces: int,
    ) -> None:
        """Move `minority_person`'s observations out of `cluster_id`.

        Target cluster: the minority person's existing primary person cluster (the
        non-`cluster_id` cluster holding most of that person's faced observations),
        else a freshly created person cluster. Observations moved:
          * those with a direct face of minority_person; plus
          * those sharing (yolo_track_id, scene_id) with such a faced observation.
        """
        # 1. Find the minority person's primary cluster elsewhere (if any). We ONLY
        # split into an existing cluster that already holds this person's faces — we
        # never create a fresh cluster here. Creating fresh clusters caused a
        # pathological oscillation: the same drifting-track observations were moved
        # into a brand-new cluster, then re-split out of it next cycle, leaving a
        # trail of hundreds of empty clusters. If the person has no established
        # cluster elsewhere, leave their observations put (non-destructive); a later
        # face correction or the face-based merge pass will consolidate them.
        row = self._db_execute(
            """
            SELECT fo.object_id, COUNT(*) AS n
            FROM face_observations fo
            JOIN objects o ON o.id = fo.object_id
            WHERE fo.person_id = %s
              AND fo.object_id IS NOT NULL
              AND fo.object_id <> %s
              AND o.class_id = %s
            GROUP BY fo.object_id
            ORDER BY n DESC
            LIMIT 1
            """,
            (int(minority_person), int(cluster_id), int(self.person_class_id)),
            fetchone=True,
        )
        if row is None:
            return  # no established identity cluster to split into; skip
        target_id = int(row[0])

        # Safety: only split into a target where this person is ALREADY the dominant
        # face identity. Moving them into a cluster dominated by someone else would
        # just re-mix them (and ping-pong on the next cycle). The lookup above picks
        # the cluster with the most of this person's faces, so this is normally true.
        dom_row = self._db_execute(
            """
            SELECT fo.person_id, COUNT(*) AS n
            FROM face_observations fo
            WHERE fo.object_id = %s AND fo.person_id IS NOT NULL
            GROUP BY fo.person_id
            ORDER BY n DESC
            LIMIT 1
            """,
            (target_id,), fetchone=True,
        )
        if dom_row is not None and int(dom_row[0]) != int(minority_person):
            return  # target is dominated by a different person; skip to avoid ping-pong

        # 2. Observations to move: direct minority faces + same (track, scene).
        move_rows = self._db_fetchall(
            """
            WITH faced AS (
                SELECT oo.id AS obs_id, oo.yolo_track_id, oo.scene_id
                FROM object_observations oo
                JOIN face_observations fo ON fo.observation_id = oo.id
                WHERE oo.object_id = %s AND fo.person_id = %s
            ),
            companions AS (
                SELECT oo2.id AS obs_id
                FROM object_observations oo2
                JOIN faced f
                  ON oo2.yolo_track_id IS NOT NULL
                 AND oo2.yolo_track_id = f.yolo_track_id
                 AND oo2.scene_id IS NOT NULL
                 AND oo2.scene_id = f.scene_id
                WHERE oo2.object_id = %s
            )
            SELECT DISTINCT obs_id FROM (
                SELECT obs_id FROM faced
                UNION
                SELECT obs_id FROM companions
            ) u
            """,
            (int(cluster_id), int(minority_person), int(cluster_id)),
        ) or []
        obs_ids = [int(r[0]) for r in move_rows]
        if not obs_ids:
            return

        self._db_execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (target_id, obs_ids),
        )
        self._db_execute(
            "UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)",
            (target_id, obs_ids),
        )
        # Re-link the moved face observations to the target cluster.
        self._db_execute(
            "UPDATE face_observations SET object_id = %s WHERE observation_id = ANY(%s)",
            (target_id, obs_ids),
        )
        self.get_logger().info(
            f"Face recluster: split person {minority_person} out of cluster {cluster_id} "
            f"-> {target_id} (moved {len(obs_ids)} obs, {minority_faces} faces; "
            f"dominant person {dominant_person} kept)"
        )

    def _remap_object_references_after_merge(self, keep_id: int, drop_id: int) -> None:
        """Point in-memory references at the surviving cluster after a merge.

        Consolidation moves DB observations to keep_id and deletes drop_id, but
        queued observation records and the recent-detection cache still reference
        drop_id; without remapping, their later inserts violate the
        object_observations_object_id_fkey and are silently lost.
        """
        try:
            keep_key = int(keep_id)
            drop_key = int(drop_id)
        except (TypeError, ValueError):
            return
        with self._pending_observations_lock:
            for record in self._pending_observations:
                try:
                    if int(record.object_id) == drop_key:
                        record.object_id = keep_key
                except (TypeError, ValueError):
                    continue
        entry = self._recent_detection_cache.pop(drop_key, None)
        if entry is not None:
            self._recent_detection_cache.setdefault(keep_key, entry)

    def _consolidate_clusters_by_similarity(self) -> None:
        # Step A: periodically split person clusters that mix multiple real people
        # (face-guided), before the merge passes below.
        try:
            self._split_mixed_person_clusters_by_face()
        except Exception as exc:
            self.get_logger().error(f"Face recluster split failed: {exc}")
        self._consolidate_person_clusters_by_face_identity()
        threshold = self.consolidation_similarity_threshold
        threshold_objects = self.consolidation_similarity_threshold_objects
        if threshold <= 0.0 and threshold_objects <= 0.0:
            return

        try:
            pairs_sql = """
            WITH centroids AS (
                SELECT
                    o.id AS object_id,
                    o.class_id,
                    AVG(oo.embedding) AS centroid,
                    COUNT(*) AS n_obs,
                    MIN(oo.created_at) AS first_seen
                FROM objects o
                JOIN object_observations oo ON oo.object_id = o.id
                WHERE COALESCE(oo.quality_score, 1.0) >= %s
                GROUP BY o.id, o.class_id
                HAVING COUNT(*) >= %s
            )
            SELECT
                c1.object_id AS object_a,
                c2.object_id AS object_b,
                c1.n_obs AS n_a,
                c2.n_obs AS n_b,
                c1.first_seen AS first_a,
                c2.first_seen AS first_b,
                c1.class_id AS class_a,
                c2.class_id AS class_b,
                (1.0 - (c1.centroid <=> c2.centroid)) AS similarity
            FROM centroids c1
            JOIN centroids c2
              ON c1.object_id < c2.object_id
             AND c1.class_id = c2.class_id
            WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= CASE WHEN c1.class_id = %s THEN %s ELSE %s END
            ORDER BY similarity DESC
            LIMIT %s
            """

            candidates = self._db_fetchall(
                pairs_sql,
                (
                    self.consolidation_min_observation_quality,
                    self.consolidation_min_observations_per_cluster,
                    self.person_class_id,
                    threshold,
                    threshold_objects,
                    self.consolidation_max_pairs_per_run,
                ),
            )

            self.get_logger().info(
                f"Cluster consolidation found {len(candidates or [])} candidate pairs above threshold "
                f"{threshold} (persons) / {threshold_objects} (objects)."
            )

            merged = 0
            for row in candidates or []:
                object_a = str(row[0])
                object_b = str(row[1])
                n_a = int(row[2])
                n_b = int(row[3])
                first_a = row[4]
                first_b = row[5]
                class_a = row[6]
                class_b = row[7]
                similarity = float(row[8])
                self.get_logger().info(
                    f"Consolidation candidate: {object_a} (class={class_a}) <-> {object_b} (class={class_b}) "
                    f"similarity={similarity:.4f}"
                )

                # Keep the larger cluster; break ties by older first_seen.
                if n_a > n_b:
                    keep_id, drop_id = object_a, object_b
                elif n_b > n_a:
                    keep_id, drop_id = object_b, object_a
                elif first_a <= first_b:
                    keep_id, drop_id = object_a, object_b
                else:
                    keep_id, drop_id = object_b, object_a

                # Color-compatibility guard (non-person objects): do not merge two
                # clusters whose dominant colors clearly conflict (e.g. a pink laptop
                # cluster and a black laptop cluster), even if their embedding
                # centroids are similar. Keeps color-split clusters from being
                # re-merged by centroid similarity alone.
                if not self._clusters_color_compatible(keep_id, drop_id, class_a):
                    self.get_logger().info(
                        f"Consolidation skipped: {keep_id} <-> {drop_id} (class={class_a}) "
                        f"have conflicting dominant colors despite similarity={similarity:.3f}"
                    )
                    continue

                # Face-identity guard (persons): do NOT centroid-merge two person
                # clusters whose face observations prove they are DIFFERENT people.
                # OSNet body centroids of distinct people can exceed the merge
                # threshold (offline: cross-person centroids reach ~0.97), so
                # without this guard the centroid pass re-merges clean per-character
                # clusters back into a sink. Faces (ArcFace) are authoritative.
                if self._person_clusters_face_conflict(keep_id, drop_id, class_a):
                    self.get_logger().info(
                        f"Consolidation skipped: {keep_id} <-> {drop_id} (class={class_a}) "
                        f"have conflicting FACE identities despite body-centroid similarity={similarity:.3f}"
                    )
                    continue

                # Stale-candidate guard: the candidate list was computed once at the
                # start of this run, but earlier merges in this same loop DELETE objects
                # (drop_id). A later pair may reference an already-deleted object as its
                # keep_id/drop_id, which would violate object_observations_object_id_fkey
                # and abort the whole run. Skip any pair whose endpoints no longer exist.
                if not self._object_exists(keep_id) or not self._object_exists(drop_id):
                    self.get_logger().info(
                        f"Consolidation skipped: {keep_id} <-> {drop_id} (class={class_a}) "
                        f"references an object already merged/deleted earlier in this run"
                    )
                    continue

                moved = self._db_fetchall(
                    "UPDATE object_observations SET object_id = %s WHERE object_id = %s RETURNING id",
                    (keep_id, drop_id),
                )
                if not moved:
                    continue

                self._db_execute("DELETE FROM objects WHERE id = %s", (drop_id,))

                # Update active track mappings so short-term continuity stays stable.
                for track_id, mapped_object_id in list(self.yolo_id_to_object_id.items()):
                    if mapped_object_id == drop_id:
                        self.yolo_id_to_object_id[track_id] = keep_id
                self._remap_object_references_after_merge(keep_id, drop_id)

                merged += 1
                self.get_logger().info(
                    f"Consolidated clusters: keep={keep_id}, drop={drop_id}, moved={len(moved)}, similarity={similarity:.3f}"
                )

            if merged > 0:
                self.get_logger().info(f"Cluster consolidation merged {merged} cluster pairs.")
        except Exception as exc:
            self.get_logger().error(f"Cluster consolidation failed: {exc}")

        # Second pass: merge same-class clusters whose MAX pairwise observation
        # similarity is high, even when pose/viewpoint-diluted centroids fall below
        # the centroid threshold (catches same-person fragments).
        try:
            self._consolidate_clusters_by_best_observation()
        except Exception as exc:
            self.get_logger().error(f"Best-observation consolidation failed: {exc}")

    def _consolidate_clusters_by_best_observation(self) -> None:
        person_threshold = float(getattr(self, "consolidation_best_obs_threshold", 0.90) or 0.0)
        object_threshold = float(getattr(self, "consolidation_best_obs_threshold_objects", 0.70) or 0.0)
        if person_threshold <= 0.0 and object_threshold <= 0.0:
            return
        prefilter = float(getattr(self, "consolidation_best_obs_centroid_prefilter", 0.60) or 0.0)
        max_pairs = int(getattr(self, "consolidation_max_pairs_per_run", 50) or 50)
        max_obs = int(getattr(self, "reid_max_match_cluster_observations", 0) or 0)
        small_max = int(getattr(self, "consolidation_best_obs_max_small_observations", 3) or 3)
        min_strong = int(getattr(self, "consolidation_best_obs_min_strong_pairs", 2) or 2)

        # Small-fragment rescue pass. Centroid merging misses tiny clusters (1-3 obs):
        # a 1-obs cluster is excluded by consolidation_min_observations_per_cluster and
        # a 2-3-obs centroid is too noisy to clear the (high) person centroid threshold.
        # To avoid the over-merging a global best-observation rule causes (OSNet person
        # embeddings sit in the 0.85-0.95 range across DIFFERENT people), this pass is
        # restricted to pairs where EXACTLY ONE side is a tiny fragment (<= small_max
        # obs) and the other is an established, non-sink cluster.
        #
        # A pair merges only when at least `min_strong` DISTINCT (non-identical)
        # observation pairs clear the best-observation threshold. Requiring
        # non-identical matches (sim < 0.999) avoids false merges from the
        # track-embedding cache (which reuses bit-identical vectors across a track).
        # Sink clusters (>= reid_max_match_cluster_observations) are excluded so the
        # merge cannot feed an over-merged cluster.
        sink_filter = ""
        sink_params: tuple = ()
        if max_obs > 0:
            sink_filter = "AND c1.n_obs < %s AND c2.n_obs < %s"
            sink_params = (max_obs, max_obs)
        # Per-class enable: a threshold <= 0 disables that class. The CASE-based
        # threshold would treat a 0 threshold as "merge everything", so disabled
        # classes must be excluded from the pairs CTE entirely.
        #   persons disabled -> exclude person_class_id
        #   objects disabled -> keep only person_class_id
        class_filter = ""
        class_params: tuple = ()
        if person_threshold <= 0.0 and object_threshold > 0.0:
            class_filter = "AND c1.class_id <> %s"
            class_params = (int(self.person_class_id),)
        elif object_threshold <= 0.0 and person_threshold > 0.0:
            class_filter = "AND c1.class_id = %s"
            class_params = (int(self.person_class_id),)
        sql = f"""
        WITH centroids AS (
            SELECT
                o.id AS object_id,
                o.class_id,
                AVG(oo.embedding) AS centroid,
                COUNT(*) AS n_obs,
                MIN(oo.created_at) AS first_seen
            FROM objects o
            JOIN object_observations oo ON oo.object_id = o.id
            WHERE COALESCE(oo.quality_score, 1.0) >= %s
            GROUP BY o.id, o.class_id
            HAVING COUNT(*) >= 1
        ),
        pairs AS (
            SELECT
                c1.object_id AS object_a,
                c2.object_id AS object_b,
                c1.n_obs AS n_a,
                c2.n_obs AS n_b,
                c1.first_seen AS first_a,
                c2.first_seen AS first_b,
                c1.class_id AS class_a
            FROM centroids c1
            JOIN centroids c2
              ON c1.object_id < c2.object_id
             AND c1.class_id = c2.class_id
            WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= %s
              -- exactly one side is a tiny fragment, the other is established
              AND ((c1.n_obs <= %s AND c2.n_obs > %s) OR (c2.n_obs <= %s AND c1.n_obs > %s))
              {class_filter}
              {sink_filter}
        )
        SELECT
            p.object_a,
            p.object_b,
            p.n_a,
            p.n_b,
            p.first_a,
            p.first_b,
            p.class_a,
            MAX(1.0 - (a.embedding <=> b.embedding)) AS best_obs_sim,
            COUNT(*) FILTER (
                WHERE (1.0 - (a.embedding <=> b.embedding))
                      >= CASE WHEN p.class_a = %s THEN %s ELSE %s END
                  AND (1.0 - (a.embedding <=> b.embedding)) < 0.999
            ) AS strong_nonident_pairs
        FROM pairs p
        JOIN object_observations a ON a.object_id = p.object_a
        JOIN object_observations b ON b.object_id = p.object_b
        WHERE COALESCE(a.quality_score, 1.0) >= %s
          AND COALESCE(b.quality_score, 1.0) >= %s
        GROUP BY p.object_a, p.object_b, p.n_a, p.n_b, p.first_a, p.first_b, p.class_a
        HAVING COUNT(*) FILTER (
                WHERE (1.0 - (a.embedding <=> b.embedding))
                      >= CASE WHEN p.class_a = %s THEN %s ELSE %s END
                  AND (1.0 - (a.embedding <=> b.embedding)) < 0.999
            ) >= %s
        ORDER BY best_obs_sim DESC
        LIMIT %s
        """
        candidates = self._db_fetchall(
            sql,
            (
                self.consolidation_min_observation_quality,
                prefilter,
                small_max,
                small_max,
                small_max,
                small_max,
            )
            + class_params
            + sink_params
            + (
                self.person_class_id,
                person_threshold,
                object_threshold,
                self.consolidation_min_observation_quality,
                self.consolidation_min_observation_quality,
                self.person_class_id,
                person_threshold,
                object_threshold,
                min_strong,
                max_pairs,
            ),
        ) or []

        if candidates:
            self.get_logger().info(
                f"Best-observation consolidation found {len(candidates)} candidate pairs "
                f"(person>={person_threshold}, object>={object_threshold})."
            )

        merged = 0
        for row in candidates:
            object_a = str(row[0])
            object_b = str(row[1])
            n_a = int(row[2])
            n_b = int(row[3])
            first_a = row[4]
            first_b = row[5]
            class_a = row[6]
            best_obs_sim = float(row[7])

            # Keep the larger cluster; break ties by older first_seen.
            if n_a > n_b:
                keep_id, drop_id = object_a, object_b
            elif n_b > n_a:
                keep_id, drop_id = object_b, object_a
            elif first_a <= first_b:
                keep_id, drop_id = object_a, object_b
            else:
                keep_id, drop_id = object_b, object_a

            # Color-compatibility guard (non-person objects).
            if not self._clusters_color_compatible(keep_id, drop_id, class_a):
                self.get_logger().info(
                    f"Best-obs consolidation skipped: {keep_id} <-> {drop_id} (class={class_a}) "
                    f"conflicting dominant colors despite best_obs_sim={best_obs_sim:.3f}"
                )
                continue

            # Stale-candidate guard: candidates were computed before this loop ran, but
            # earlier merges DELETE objects. Skip pairs referencing an already-deleted
            # object to avoid an object_observations_object_id_fkey violation.
            if not self._object_exists(keep_id) or not self._object_exists(drop_id):
                self.get_logger().info(
                    f"Best-obs consolidation skipped: {keep_id} <-> {drop_id} (class={class_a}) "
                    f"references an object already merged/deleted earlier in this run"
                )
                continue

            moved = self._db_fetchall(
                "UPDATE object_observations SET object_id = %s WHERE object_id = %s RETURNING id",
                (keep_id, drop_id),
            )
            if not moved:
                continue
            self._db_execute("DELETE FROM objects WHERE id = %s", (drop_id,))
            for track_id, mapped_object_id in list(self.yolo_id_to_object_id.items()):
                if mapped_object_id == drop_id:
                    self.yolo_id_to_object_id[track_id] = keep_id
            self._remap_object_references_after_merge(keep_id, drop_id)
            merged += 1
            self.get_logger().info(
                f"Best-obs consolidated clusters: keep={keep_id}, drop={drop_id}, "
                f"moved={len(moved)}, best_obs_sim={best_obs_sim:.3f}"
            )

        if merged > 0:
            self.get_logger().info(f"Best-observation consolidation merged {merged} cluster pairs.")

    def __init__(self) -> None:
        super().__init__("database_node")

        # The node runs in a multi-threaded executor; guard shared cursor access.
        self._db_lock = threading.RLock()

        self.declare_parameter("db_host", "db")
        self.declare_parameter("db_port", 5432)
        self.declare_parameter("db_name", "bordsupr")
        self.declare_parameter("db_user", "postgres")
        self.declare_parameter("db_password", "postgres")
        self.declare_parameter("person_class_id", 0)

        self.db_host = str(self.get_parameter("db_host").value)
        self.db_port = int(self.get_parameter("db_port").value)
        self.db_name = str(self.get_parameter("db_name").value)
        self.db_user = str(self.get_parameter("db_user").value)
        self.db_password = str(self.get_parameter("db_password").value)
        self.person_class_id = int(self.get_parameter("person_class_id").value)

        self._db_embedding_dim: Optional[int] = None
        self._embedding_dim_warning_emitted = False
        self._db_face_embedding_dim: Optional[int] = None
        self._face_embedding_dim_warning_emitted = False

        self._init_db()

        # Persistent mapping from YOLO ID (track_id or detection_id) to object_id in DB
        self.yolo_id_to_object_id: Dict[str, str] = {}

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("vlm_topic", "/dynosam/vlm_result")
        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("stitched_rgb_topic", "/spot/camera/frontmiddle_virtual/image")
        self.declare_parameter("depth_topic", "/spot/depth_registered/frontleft/image_rotated")
        self.declare_parameter("camera_info_topic", "/spot/camera/frontleft/camera_info_rotated")
        self.declare_parameter("image_rotation", "cw")
        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("detector_backend", "yolo")
        self.declare_parameter("face_output_topic", "/dynosam/face_output")
        self.declare_parameter("interaction_topic", "/dynosam/interaction_output")
        self.declare_parameter("object_odometry_topic", "/dynosam/frontend/object_odometry")
        self.declare_parameter("tf_topic", "/tf")
        self.declare_parameter("odometry_topic", "/spot/odometry")
        self.declare_parameter("embedding_service", "/get_convnext_embedding")
        self.declare_parameter("person_embedding_service", "/get_osnet_embedding")
        self.declare_parameter("embedding_timeout_sec", 10.0)
        self.declare_parameter("debug_log_services", False)
        self.declare_parameter("embedding_reid_similarity_threshold", 0.85)
        # Separate (lower) acceptance threshold for non-person classes: object
        # embeddings (ConvNeXt) are less viewpoint-robust than the fine-tuned
        # person OSNet, so the person threshold over-fragments objects.
        self.declare_parameter("embedding_reid_similarity_threshold_objects", 0.50)
        self.declare_parameter("face_reid_similarity_threshold", 0.68)
        self.declare_parameter("face_reid_min_score_margin", 0.03)
        self.declare_parameter("face_reid_track_ttl_sec", 20.0)
        self.declare_parameter("face_track_db_lookup_window_sec", 180.0)
        # Window (seconds) during which a confident face identification can reassign
        # this track's recently-clustered body observations to the correct person cluster.
        self.declare_parameter("face_correction_window_sec", 30.0)
        # When true, a confident face identification reassigns ALL of this track's
        # person observations to the face-identified cluster (not just the recent
        # window). Faces (ArcFace) are far more discriminative than the OSNet body
        # embedding that mis-assigned them, so the whole track should follow the face.
        self.declare_parameter("face_correction_whole_track", True)
        # Anti-sink face gate (persons): a body-ONLY match (no confirming face) may not
        # feed a person cluster that already holds this many observations. OSNet body
        # embeddings are too weak to be trusted for large merges (offline: cross-person
        # best-match up to ~0.95), so an unguarded body merge snowballs a multi-person
        # sink. 0 disables the gate.
        self.declare_parameter("person_body_merge_face_required_obs", 50)
        # Step B: a body-ONLY person match into a cluster that already has a confirmed
        # face identity must reach this body similarity to merge (OSNet cross-person
        # sim reaches ~0.95-0.97, so the default 0.85 lets different people merge).
        # 0 disables the gate.
        self.declare_parameter("person_body_merge_face_identified_sim", 0.97)
        # When true, the centroid-consolidation pass will NOT merge two person clusters
        # whose face observations prove they are different people (dominant face
        # person_id differs). Prevents the body-centroid merge from re-merging clean
        # per-person clusters into a sink (OSNet cross-person centroids reach ~0.97).
        self.declare_parameter("face_identity_consolidation_guard_enabled", True)
        # Periodic face recluster (Step A): split person body-clusters whose face
        # observations prove they mix multiple real people. BoTSORT tracks drift
        # across people when they sit close, so a body cluster can accumulate faces
        # of several distinct person_ids; OSNet body embeddings cannot separate them
        # (cross-person sim ~0.95) but ArcFace can (cross-person <=~0.17). This pass
        # periodically moves the minority-person observations out of a mixed cluster
        # so clusters stay single-identity. Runs inside _maybe_run_consolidation.
        self.declare_parameter("face_recluster_split_enabled", True)
        # A face person_id must hold at least this many faces in a cluster to count
        # as a real sub-identity (below this it is noise / a stray bystander face).
        self.declare_parameter("face_recluster_min_person_faces", 4)
        # Only split when the dominant face person is clearly dominant: its face
        # count >= this ratio x the runner-up. Avoids shredding genuinely ambiguous
        # clusters where two people are evenly mixed.
        self.declare_parameter("face_recluster_dominant_ratio", 1.5)
        # Never split a cluster smaller than this (tiny fragments are not worth it).
        self.declare_parameter("face_recluster_min_cluster_obs", 12)
        # Max seconds between a face and the nearest body observation of the same
        # track for them to be linked into the same cluster. Bounds face->body
        # linking against BoTSORT track-id reuse (a later reuse of the same track id
        # is a different person). Also bounds the post-hoc face-link sweep.
        self.declare_parameter("face_body_link_window_sec", 5.0)
        # Minimum IoU between a face's person crop bbox and a candidate observation's
        # bbox for them to be linked. The face detector reports the exact person crop
        # the face was detected on; matching by track+scene is unreliable (tracks drift
        # across people sitting close, and the face's scene_id can point at an adjacent
        # frame whose same-track observation is a DIFFERENT person). The correct crop
        # matches at IoU ~0.85-1.0 while a wrong crop is ~0.4-0.5, so 0.5 separates them.
        # When a face has a person crop and no observation matches, the face is dropped
        # (its true observation was not persisted) rather than mis-linked. 0 disables.
        self.declare_parameter("face_body_link_min_iou", 0.50)
        self.declare_parameter("face_pending_max_age_sec", 120.0)
        # Safety guard: never let face-correction dump a track's body observations
        # into an already-huge cluster. A single real person cannot have thousands
        # of body observations, so a target cluster this large is an over-merged
        # sink; correcting into it only feeds the sink. 0 disables the guard.
        self.declare_parameter("face_correction_max_cluster_observations", 200)
        self.declare_parameter("reclassify_furniture_with_faces", True)
        self.declare_parameter("reid_knn_neighbors", 12)
        self.declare_parameter("reid_average_weight", 0.65)
        self.declare_parameter("reid_min_score_margin", 0.03)
        self.declare_parameter("reid_min_neighbor_hits", 1)
        # Exclude clusters already holding this many observations from being body
        # embedding match targets (over-merged sinks self-reinforce). 0 disables.
        self.declare_parameter("reid_max_match_cluster_observations", 300)
        # Color-compatibility gate for non-person object re-id: reject a match when
        # the detection's dominant color conflicts with the cluster's dominant color
        # (e.g. pink vs black laptop). Prevents over-merging distinctly-colored
        # objects. 1 enables, 0 disables.
        self.declare_parameter("object_color_gate_enabled", True)
        self.declare_parameter("reid_min_observation_quality", 0.05)
        self.declare_parameter("reid_max_position_distance_m", 2.0)
        self.declare_parameter("depth_estimation_min_valid_pixels", 5)
        self.declare_parameter("depth_estimation_max_relative_std", 0.5)
        self.declare_parameter("depth_estimation_person_min_valid_pixels", 10)
        self.declare_parameter("depth_estimation_person_max_relative_std", 0.60)
        self.declare_parameter("depth_estimation_center_crop_classes", ["person"])
        self.declare_parameter("depth_estimation_center_crop_ratio", 0.50)
        self.declare_parameter("track_id_ttl_sec", 60.0)
        self.declare_parameter("stream_queue_depth", 1000)
        self.declare_parameter("rgb_match_window_sec", 1.25)
        self.declare_parameter("rgb_cache_ttl_sec", 120.0)
        self.declare_parameter("rgb_cache_max_entries", 4096)
        self.declare_parameter("vlm_cache_ttl_sec", 300.0)
        self.declare_parameter("vlm_cache_max_entries", 4096)
        self.declare_parameter("scene_id_cache_max_entries", 4096)
        self.declare_parameter("object_pose_match_window_sec", 0.35)
        self.declare_parameter("object_pose_cache_ttl_sec", 120.0)
        self.declare_parameter("observation_position_source_preference", "")
        self.declare_parameter("prefer_depth_over_dynosam", True)
        self.declare_parameter("pending_observation_timeout_sec", 30.0)
        self.declare_parameter("db_retry_timeout_sec", 15.0)
        self.declare_parameter("interaction_scene_match_window_sec", 0.5)
        self.declare_parameter("interaction_max_retry_attempts", 6)
        self.declare_parameter("interaction_dedup_window_sec", 5.0)
        self.declare_parameter("consolidation_every_n_observations", 120)
        self.declare_parameter("consolidation_similarity_threshold", 0.80)
        self.declare_parameter("consolidation_similarity_threshold_objects", 0.50)
        self.declare_parameter("consolidation_best_obs_threshold", 0.90)
        self.declare_parameter("consolidation_best_obs_threshold_objects", 0.70)
        self.declare_parameter("consolidation_best_obs_centroid_prefilter", 0.60)
        self.declare_parameter("consolidation_best_obs_max_small_observations", 3)
        self.declare_parameter("consolidation_best_obs_min_strong_pairs", 2)
        self.declare_parameter("consolidation_min_observations_per_cluster", 2)
        self.declare_parameter("consolidation_max_pairs_per_run", 6)
        self.declare_parameter("static_object_max_position_distance_m", 0.8)
        self.declare_parameter("static_object_class_ids", [])
        from rcl_interfaces.msg import ParameterDescriptor
        self.declare_parameter(
            "portable_object_class_ids",
            [],
            ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("consolidation_min_observation_quality", 0.05)
        self.declare_parameter("attributes_enabled", True)
        self.declare_parameter("attribute_quality_enabled", True)
        self.declare_parameter("attribute_color_enabled", True)
        self.declare_parameter("attribute_person_parts_enabled", True)
        self.declare_parameter("attribute_part_embeddings_enabled", False)
        self.declare_parameter("attribute_super_resolution_enabled", False)
        self.declare_parameter("attribute_super_resolution_method", "lanczos")
        self.declare_parameter("attribute_super_resolution_min_side", 96)
        self.declare_parameter("attribute_segmentation_mode", "mask")
        self.declare_parameter("scene_dedup_enabled", True)
        self.declare_parameter("scene_save_mode", "change_or_interval")
        self.declare_parameter("scene_min_mean_abs_diff", 5.0)
        self.declare_parameter("scene_min_changed_pixel_ratio", 0.03)
        self.declare_parameter("scene_pixel_change_threshold", 12)
        self.declare_parameter("scene_force_save_interval_sec", 30.0)
        # Scene image storage size knobs: downscale the longest side to
        # scene_image_max_side px (0 = keep full resolution) and encode JPEG at
        # scene_image_jpeg_quality (1-100). Lower values shrink the DB footprint.
        self.declare_parameter("scene_image_max_side", 640)
        self.declare_parameter("scene_image_jpeg_quality", 80)

        # Per-track content-change (scene-cut / person-swap) detection. When a
        # tracked detection's crop changes sharply vs the previous frame of the SAME
        # track, the tracker likely reused the track id across a cut to a different
        # object/person. We then drop the track->object mapping and bypass the cached
        # embedding so the detection is re-resolved (and re-embedded) fresh.
        self.declare_parameter("track_content_change_enabled", True)
        self.declare_parameter("track_content_change_mean_abs_diff", 22.0)
        self.declare_parameter("track_content_change_changed_pixel_ratio", 0.45)
        self.declare_parameter("track_content_change_pixel_threshold", 25)

        self.vlm_topic = str(self.get_parameter("vlm_topic").value)
        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.stitched_rgb_topic = str(self.get_parameter("stitched_rgb_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self._image_rotation = str(self.get_parameter("image_rotation").value).strip().lower()
        self.yolo_output_topic = str(self.get_parameter("yolo_output_topic").value)
        self.detector_backend = str(self.get_parameter("detector_backend").value)
        self.face_output_topic = str(self.get_parameter("face_output_topic").value)
        self.interaction_topic = str(self.get_parameter("interaction_topic").value)
        self.object_odometry_topic = str(self.get_parameter("object_odometry_topic").value)
        self.tf_topic = str(self.get_parameter("tf_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.object_embedding_service_name = str(self.get_parameter("embedding_service").value)
        self.person_embedding_service_name = str(self.get_parameter("person_embedding_service").value)
        self.embedding_service_name = self.object_embedding_service_name
        self.embedding_timeout_sec = float(self.get_parameter("embedding_timeout_sec").value)
        self.debug_log_services = bool(self.get_parameter("debug_log_services").value)
        self.embedding_reid_similarity_threshold = float(
            self.get_parameter("embedding_reid_similarity_threshold").value
        )
        self.embedding_reid_similarity_threshold_objects = float(
            self.get_parameter("embedding_reid_similarity_threshold_objects").value
        )
        self.face_reid_similarity_threshold = float(
            self.get_parameter("face_reid_similarity_threshold").value
        )
        self.face_reid_min_score_margin = float(
            self.get_parameter("face_reid_min_score_margin").value
        )
        self.face_track_id_ttl_sec = float(
            self.get_parameter("face_reid_track_ttl_sec").value
        )
        self.face_track_db_lookup_window_sec = float(
            self.get_parameter("face_track_db_lookup_window_sec").value
        )
        self.face_correction_window_sec = float(
            self.get_parameter("face_correction_window_sec").value
        )
        self.face_correction_whole_track = bool(
            self.get_parameter("face_correction_whole_track").value
        )
        self.person_body_merge_face_required_obs = int(
            self.get_parameter("person_body_merge_face_required_obs").value
        )
        self.person_body_merge_face_identified_sim = float(
            self.get_parameter("person_body_merge_face_identified_sim").value
        )
        self.face_identity_consolidation_guard_enabled = bool(
            self.get_parameter("face_identity_consolidation_guard_enabled").value
        )
        self.face_recluster_split_enabled = bool(
            self.get_parameter("face_recluster_split_enabled").value
        )
        self.face_recluster_min_person_faces = int(
            self.get_parameter("face_recluster_min_person_faces").value
        )
        self.face_recluster_dominant_ratio = float(
            self.get_parameter("face_recluster_dominant_ratio").value
        )
        self.face_recluster_min_cluster_obs = int(
            self.get_parameter("face_recluster_min_cluster_obs").value
        )
        self.face_body_link_window_sec = float(
            self.get_parameter("face_body_link_window_sec").value
        )
        self.face_body_link_min_iou = float(
            self.get_parameter("face_body_link_min_iou").value
        )
        self.face_pending_max_age_sec = float(
            self.get_parameter("face_pending_max_age_sec").value
        )
        self.face_correction_max_cluster_observations = int(
            self.get_parameter("face_correction_max_cluster_observations").value
        )
        self.reclassify_furniture_with_faces = bool(
            self.get_parameter("reclassify_furniture_with_faces").value
        )
        self.person_class_id = int(self.get_parameter("person_class_id").value)
        self.reid_knn_neighbors = int(self.get_parameter("reid_knn_neighbors").value)
        self.reid_average_weight = float(self.get_parameter("reid_average_weight").value)
        self.reid_min_score_margin = float(self.get_parameter("reid_min_score_margin").value)
        self.reid_min_neighbor_hits = int(self.get_parameter("reid_min_neighbor_hits").value)
        self.reid_max_match_cluster_observations = int(
            self.get_parameter("reid_max_match_cluster_observations").value
        )
        self.object_color_gate_enabled = bool(
            self.get_parameter("object_color_gate_enabled").value
        )
        self.reid_min_observation_quality = float(self.get_parameter("reid_min_observation_quality").value)
        self.track_id_ttl_sec = float(self.get_parameter("track_id_ttl_sec").value)
        self.reid_max_position_distance_m = float(
            self.get_parameter("reid_max_position_distance_m").value
        )
        self.depth_estimation_min_valid_pixels = int(
            self.get_parameter("depth_estimation_min_valid_pixels").value
        )
        self.depth_estimation_max_relative_std = float(
            self.get_parameter("depth_estimation_max_relative_std").value
        )
        self.depth_estimation_person_min_valid_pixels = int(
            self.get_parameter("depth_estimation_person_min_valid_pixels").value
        )
        self.depth_estimation_person_max_relative_std = float(
            self.get_parameter("depth_estimation_person_max_relative_std").value
        )
        self.depth_estimation_center_crop_classes = list(
            self.get_parameter("depth_estimation_center_crop_classes").value or []
        )
        self.depth_estimation_center_crop_ratio = float(
            self.get_parameter("depth_estimation_center_crop_ratio").value
        )
        # Throttle dict for depth-estimation failure logs: key -> last_log_time
        self._depth_failure_log_throttle: Dict[str, float] = {}
        self.stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)
        self._rgb_match_window_sec = float(self.get_parameter("rgb_match_window_sec").value)
        self._rgb_cache_ttl_sec = float(self.get_parameter("rgb_cache_ttl_sec").value)
        self._rgb_cache_max_entries = int(self.get_parameter("rgb_cache_max_entries").value)
        self._vlm_cache_ttl_sec = float(self.get_parameter("vlm_cache_ttl_sec").value)
        self._vlm_cache_max_entries = int(self.get_parameter("vlm_cache_max_entries").value)
        self._scene_id_cache_max_entries = int(
            self.get_parameter("scene_id_cache_max_entries").value
        )
        self._depth_scale = 0.001
        self._object_pose_match_window_sec = float(
            self.get_parameter("object_pose_match_window_sec").value
        )
        self._object_pose_cache_ttl_sec = float(
            self.get_parameter("object_pose_cache_ttl_sec").value
        )
        legacy_prefer_depth_over_dynosam = bool(
            self.get_parameter("prefer_depth_over_dynosam").value
        )
        position_source_preference = str(
            self.get_parameter("observation_position_source_preference").value
        ).strip().lower()
        if not position_source_preference:
            position_source_preference = 'depth' if legacy_prefer_depth_over_dynosam else 'dynosam'
        if position_source_preference not in {'depth', 'dynosam'}:
            self.get_logger().warning(
                "Unsupported observation_position_source_preference "
                f"'{position_source_preference}', falling back to legacy preference"
            )
            position_source_preference = 'depth' if legacy_prefer_depth_over_dynosam else 'dynosam'
        self._position_source_preference = position_source_preference
        self._prefer_depth_over_dynosam = self._position_source_preference == 'depth'
        self._pending_observation_timeout_sec = float(
            self.get_parameter("pending_observation_timeout_sec").value
        )
        self._db_retry_timeout_sec = float(
            self.get_parameter("db_retry_timeout_sec").value
        )
        self._robot_pose_match_window_sec = self._object_pose_match_window_sec
        self._robot_pose_cache_ttl_sec = self._object_pose_cache_ttl_sec
        self._interaction_scene_match_window_sec = float(
            self.get_parameter("interaction_scene_match_window_sec").value
        )
        self._interaction_max_retry_attempts = int(
            self.get_parameter("interaction_max_retry_attempts").value
        )
        self._interaction_dedup_window_sec = float(
            self.get_parameter("interaction_dedup_window_sec").value
        )
        self.consolidation_every_n_observations = int(
            self.get_parameter("consolidation_every_n_observations").value
        )
        self.consolidation_similarity_threshold = float(
            self.get_parameter("consolidation_similarity_threshold").value
        )
        self.consolidation_similarity_threshold_objects = float(
            self.get_parameter("consolidation_similarity_threshold_objects").value
        )
        self.consolidation_best_obs_threshold = float(
            self.get_parameter("consolidation_best_obs_threshold").value
        )
        self.consolidation_best_obs_threshold_objects = float(
            self.get_parameter("consolidation_best_obs_threshold_objects").value
        )
        self.consolidation_best_obs_centroid_prefilter = float(
            self.get_parameter("consolidation_best_obs_centroid_prefilter").value
        )
        self.consolidation_best_obs_max_small_observations = int(
            self.get_parameter("consolidation_best_obs_max_small_observations").value
        )
        self.consolidation_best_obs_min_strong_pairs = int(
            self.get_parameter("consolidation_best_obs_min_strong_pairs").value
        )
        self.consolidation_min_observations_per_cluster = int(
            self.get_parameter("consolidation_min_observations_per_cluster").value
        )
        self.consolidation_max_pairs_per_run = int(
            self.get_parameter("consolidation_max_pairs_per_run").value
        )
        self.static_object_max_position_distance_m = float(
            self.get_parameter("static_object_max_position_distance_m").value
        )
        raw_static_class_ids = self.get_parameter_or(
            "static_object_class_ids", Parameter("static_object_class_ids", value=None)
        ).value
        if raw_static_class_ids is None or len(raw_static_class_ids) == 0:
            self.static_object_class_ids: Optional[set[int]] = None
        else:
            self.static_object_class_ids = {int(v) for v in raw_static_class_ids}
        raw_portable_class_ids = self.get_parameter_or(
            "portable_object_class_ids", Parameter("portable_object_class_ids", value=None)
        ).value
        if raw_portable_class_ids is None or len(raw_portable_class_ids) == 0:
            self.portable_object_class_ids: Optional[set[int]] = None
        else:
            self.portable_object_class_ids = {int(v) for v in raw_portable_class_ids}
        if self.static_object_max_position_distance_m < 0.0:
            self.static_object_max_position_distance_m = 0.0
        self.consolidation_min_observation_quality = float(
            self.get_parameter("consolidation_min_observation_quality").value
        )
        self.attribute_config = AttributeConfig(
            enabled=bool(self.get_parameter("attributes_enabled").value),
            quality_enabled=bool(self.get_parameter("attribute_quality_enabled").value),
            color_enabled=bool(self.get_parameter("attribute_color_enabled").value),
            person_parts_enabled=bool(self.get_parameter("attribute_person_parts_enabled").value),
            part_embeddings_enabled=bool(self.get_parameter("attribute_part_embeddings_enabled").value),
            super_resolution_enabled=bool(self.get_parameter("attribute_super_resolution_enabled").value),
            super_resolution_method=str(self.get_parameter("attribute_super_resolution_method").value),
            super_resolution_min_side=int(self.get_parameter("attribute_super_resolution_min_side").value),
            segmentation_mode=str(self.get_parameter("attribute_segmentation_mode").value),
            person_class_id=self.person_class_id,
        )
        self.scene_dedup_enabled = bool(self.get_parameter("scene_dedup_enabled").value)
        self.scene_save_mode = str(self.get_parameter("scene_save_mode").value).strip().lower()
        self.scene_min_mean_abs_diff = float(
            self.get_parameter("scene_min_mean_abs_diff").value
        )
        self.scene_min_changed_pixel_ratio = float(
            self.get_parameter("scene_min_changed_pixel_ratio").value
        )
        self.scene_pixel_change_threshold = int(
            self.get_parameter("scene_pixel_change_threshold").value
        )
        self.scene_force_save_interval_sec = float(
            self.get_parameter("scene_force_save_interval_sec").value
        )
        self.scene_image_max_side = int(self.get_parameter("scene_image_max_side").value)
        self.scene_image_jpeg_quality = int(self.get_parameter("scene_image_jpeg_quality").value)
        self.track_content_change_enabled = bool(
            self.get_parameter("track_content_change_enabled").value
        )
        self.track_content_change_mean_abs_diff = float(
            self.get_parameter("track_content_change_mean_abs_diff").value
        )
        self.track_content_change_changed_pixel_ratio = float(
            self.get_parameter("track_content_change_changed_pixel_ratio").value
        )
        self.track_content_change_pixel_threshold = int(
            self.get_parameter("track_content_change_pixel_threshold").value
        )
        if self.scene_save_mode not in {"change_or_interval", "interval"}:
            self.get_logger().warning(
                f"Unsupported scene_save_mode '{self.scene_save_mode}', falling back to 'change_or_interval'"
            )
            self.scene_save_mode = "change_or_interval"

        if self.reid_knn_neighbors < 2:
            self.reid_knn_neighbors = 2
        self.reid_average_weight = min(1.0, max(0.0, self.reid_average_weight))
        if self.reid_min_score_margin < 0.0:
            self.reid_min_score_margin = 0.0
        if self.reid_min_neighbor_hits < 1:
            self.reid_min_neighbor_hits = 1
        if self.face_reid_min_score_margin < 0.0:
            self.face_reid_min_score_margin = 0.0
        if self.face_track_id_ttl_sec < 1.0:
            self.face_track_id_ttl_sec = 1.0
        if self.face_track_db_lookup_window_sec < 1.0:
            self.face_track_db_lookup_window_sec = 1.0
        if self.stream_queue_depth < 1:
            self.stream_queue_depth = 1
        if self._rgb_cache_max_entries < 1:
            self._rgb_cache_max_entries = 1
        if self._vlm_cache_max_entries < 1:
            self._vlm_cache_max_entries = 1
        if self._scene_id_cache_max_entries < 1:
            self._scene_id_cache_max_entries = 1
        if self._image_rotation not in {"none", "cw", "ccw", "180"}:
            self.get_logger().warning(
                f"Unsupported image_rotation '{self._image_rotation}', falling back to 'none'"
            )
            self._image_rotation = "none"

        self.get_logger().info(f"VLM topic: {self.vlm_topic}")
        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"Stitched RGB topic: {self.stitched_rgb_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
        self.get_logger().info(f"Image rotation compensation: {self._image_rotation}")
        self.get_logger().info(f"YOLO output topic: {self.yolo_output_topic}")
        self.get_logger().info(f"Face output topic: {self.face_output_topic}")
        self.get_logger().info(f"Interaction topic: {self.interaction_topic}")
        self.get_logger().info(f"Object odometry topic: {self.object_odometry_topic}")
        self.get_logger().info(f"TF topic: {self.tf_topic}")
        self.get_logger().info(f"Odometry topic: {self.odometry_topic}")
        self.get_logger().info(f"Object embedding service: {self.object_embedding_service_name}")
        self.get_logger().info(f"Person embedding service: {self.person_embedding_service_name}")
        self.get_logger().info(
            f"Database target: postgresql://{self.db_user}:***@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        self.get_logger().info(f"Embedding timeout sec: {self.embedding_timeout_sec}")
        self.get_logger().info(
            f"Embedding re-id similarity threshold: {self.embedding_reid_similarity_threshold} "
            f"(objects: {self.embedding_reid_similarity_threshold_objects})"
        )
        self.get_logger().info(f"Re-id KNN neighbors: {self.reid_knn_neighbors}")
        self.get_logger().info(f"Re-id average weight: {self.reid_average_weight}")
        self.get_logger().info(f"Re-id min score margin: {self.reid_min_score_margin}")
        self.get_logger().info(f"Re-id min neighbor hits: {self.reid_min_neighbor_hits}")
        self.get_logger().info(f"Track ID TTL sec: {self.track_id_ttl_sec}")
        self.get_logger().info(f"Stream queue depth: {self.stream_queue_depth}")
        self.get_logger().info(f"RGB match window sec: {self._rgb_match_window_sec}")
        self.get_logger().info(f"RGB cache ttl sec: {self._rgb_cache_ttl_sec}")
        self.get_logger().info(f"RGB cache max entries: {self._rgb_cache_max_entries}")
        self.get_logger().info(f"VLM cache ttl sec: {self._vlm_cache_ttl_sec}")
        self.get_logger().info(f"VLM cache max entries: {self._vlm_cache_max_entries}")
        self.get_logger().info(f"Scene id cache max entries: {self._scene_id_cache_max_entries}")
        self.get_logger().info(
            f"Face re-id similarity threshold: {self.face_reid_similarity_threshold}"
        )
        self.get_logger().info(
            f"Face re-id min score margin: {self.face_reid_min_score_margin}"
        )
        self.get_logger().info(f"Face track ID TTL sec: {self.face_track_id_ttl_sec}")
        self.get_logger().info(
            f"Face track DB lookup window sec: {self.face_track_db_lookup_window_sec}"
        )
        self.get_logger().info(f"Person class id: {self.person_class_id}")
        self.get_logger().info(
            f"Object pose match window sec: {self._object_pose_match_window_sec}"
        )
        self.get_logger().info(
            f"Object pose cache TTL sec: {self._object_pose_cache_ttl_sec}"
        )
        self.get_logger().info(
            f"Observation position source preference: {self._position_source_preference}"
        )
        self.get_logger().info(
            f"Prefer depth over DynoSAM (legacy): {self._prefer_depth_over_dynosam}"
        )
        self.get_logger().info(
            f"Interaction scene match window sec: {self._interaction_scene_match_window_sec}"
        )
        self.get_logger().info(
            f"Consolidation every n observations: {self.consolidation_every_n_observations}"
        )
        self.get_logger().info(
            f"Consolidation similarity threshold: {self.consolidation_similarity_threshold} "
            f"(objects: {self.consolidation_similarity_threshold_objects})"
        )
        self.get_logger().info(
            f"Consolidation min observations per cluster: {self.consolidation_min_observations_per_cluster}"
        )
        self.get_logger().info(f"Visual attribute extraction enabled: {self.attribute_config.enabled}")
        self.get_logger().info(f"Part embeddings enabled: {self.attribute_config.part_embeddings_enabled}")
        self.get_logger().info(f"Scene dedup enabled: {self.scene_dedup_enabled}")
        self.get_logger().info(f"Scene save mode: {self.scene_save_mode}")
        self.get_logger().info(
            f"Scene min mean abs diff: {self.scene_min_mean_abs_diff}"
        )
        self.get_logger().info(
            f"Scene min changed pixel ratio: {self.scene_min_changed_pixel_ratio}"
        )
        self.get_logger().info(
            f"Scene pixel change threshold: {self.scene_pixel_change_threshold}"
        )
        self.get_logger().info(
            f"Scene force save interval sec: {self.scene_force_save_interval_sec}"
        )
        self.get_logger().info(
            f"Static object max position distance: {self.static_object_max_position_distance_m}m"
        )
        if self.static_object_class_ids is None:
            self.get_logger().info("Static object class ids: all non-person")
        else:
            self.get_logger().info(
                f"Static object class ids: {sorted(self.static_object_class_ids)}"
            )
        if self.portable_object_class_ids is None:
            self.get_logger().info("Portable object class ids: none (distance check applies to all)")
        else:
            self.get_logger().info(
                f"Portable object class ids: {sorted(self.portable_object_class_ids)}"
            )

        # Run one pass at startup so existing near-duplicate clusters are consolidated
        # without waiting for more observations.
        self._consolidate_clusters_by_similarity()

        # -----------------------------
        # Callback groups
        # -----------------------------
        # Separate groups so a subscription callback is not blocked waiting on the client.
        self.sub_group = MutuallyExclusiveCallbackGroup()
        self.rgb_group = ReentrantCallbackGroup()
        self.tf_group = ReentrantCallbackGroup()
        self.client_group = ReentrantCallbackGroup()
        self.timer_group = ReentrantCallbackGroup()
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=self.stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # -----------------------------
        # Service client
        # -----------------------------
        self.object_embedding_client = self.create_client(
            GetImageEmbedding,
            self.object_embedding_service_name,
            callback_group=self.client_group,
        )
        self.person_embedding_client = self.create_client(
            GetImageEmbedding,
            self.person_embedding_service_name,
            callback_group=self.client_group,
        )
        self.embedding_client = self.object_embedding_client

        # -----------------------------
        # Example subscriptions
        # -----------------------------
        # Keep / adapt these to your actual pipeline.
        self.vlm_sub = self.create_subscription(
            StampedString,
            self.vlm_topic,
            self.vlm_callback,
            stream_qos,
            callback_group=self.sub_group,
        )
        self.rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic,
            self._cache_rgb_image,
            stream_qos,
            callback_group=self.rgb_group,
        )
        self.corrected_rgb_sub = self.create_subscription(
            Image,
            self.rgb_topic + "_corrected",
            self._cache_corrected_rgb_image,
            stream_qos,
            callback_group=self.rgb_group,
        )
        self.stitched_rgb_sub = self.create_subscription(
            Image,
            self.stitched_rgb_topic,
            self._cache_stitched_rgb_image,
            stream_qos,
            callback_group=self.rgb_group,
        )
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self._cache_depth_image,
            qos_profile_sensor_data,
            callback_group=self.rgb_group,
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._cache_camera_info,
            qos_profile_sensor_data,
            callback_group=self.rgb_group,
        )
        # Subscribe to YOLO output topic
        
        self.yolo_output_sub = self.create_subscription(
            YoloOutput,
            self.yolo_output_topic,
            self.yolo_output_callback,
            stream_qos,
            callback_group=self.sub_group,
        )
        self.face_output_sub = self.create_subscription(
            FaceOutput,
            self.face_output_topic,
            self.face_output_callback,
            stream_qos,
            callback_group=self.sub_group,
        )
        self.object_odometry_sub = self.create_subscription(
            ObjectOdometry,
            self.object_odometry_topic,
            self._cache_object_odometry,
            100,
            callback_group=self.tf_group,
        )
        self.tf_sub = self.create_subscription(
            TFMessage,
            self.tf_topic,
            self._cache_object_transform,
            stream_qos,
            callback_group=self.sub_group,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            self.odometry_topic,
            self._cache_robot_pose,
            50,
            callback_group=self.tf_group,
        )
        self.depth_object_odometry_pub = self.create_publisher(
            ObjectOdometry, "/bordsupr/depth_object_odometry", 100
        )
        self.interaction_sub = self.create_subscription(
            InteractionOutput,
            self.interaction_topic,
            self.interaction_output_callback,
            stream_qos,
            callback_group=self.sub_group,
        )


        # -----------------------------
        # State
        # -----------------------------
        self._next_request_id = 1
        self.cv_bridge = CvBridge()
        self._pending_requests: Dict[int, PendingEmbeddingRequest] = {}
        self._latest_rgb_by_stamp: "OrderedDict[Tuple[int, int], Image]" = OrderedDict()
        self._latest_corrected_rgb_by_stamp: "OrderedDict[Tuple[int, int], Image]" = OrderedDict()
        self._latest_stitched_rgb_by_stamp: "OrderedDict[Tuple[int, int], Image]" = OrderedDict()
        self._latest_vlm_by_stamp: "OrderedDict[Tuple[int, int], str]" = OrderedDict()
        self._scene_id_by_stamp: "OrderedDict[Tuple[int, int], int]" = OrderedDict()
        self._latest_depth_by_stamp: Dict[Tuple[int, int], Image] = {}
        self._latest_camera_info_by_stamp: Dict[Tuple[int, int], CameraInfo] = {}
        self._vlm_match_window_sec = 2.0
        self._caption_backfill_window_sec = 2.0
        self._pending_interactions: List[PendingInteractionRecord] = []
        self._pending_observations: List[PendingObservationRecord] = []
        # Guards _pending_observations: the flush timer (timer_group, reentrant)
        # iterates and rebinds the list while embedding done-callbacks
        # (client_group, reentrant) append to it from other executor threads.
        # Without a lock, an append landing mid-flush is silently orphaned when
        # the flush rebinds the attribute (record lost with no log/error).
        self._pending_observations_lock = threading.Lock()
        self._pending_faces: List[dict] = []
        self._recent_detection_cache: Dict[int, Tuple[float, float, float, float, int]] = {}
        self._recent_detection_cache_ttl_sec = 30.0
        self._processed_interaction_keys = set()
        # Temporal dedup: signature -> last-seen epoch sec. Suppresses re-inserting the
        # same (subject,object,action) on consecutive frames within the dedup window.
        self._recent_interaction_signatures: Dict[Tuple[Any, ...], float] = {}
        self.track_id_last_seen_sec: Dict[str, float] = {}
        self._embedding_by_track_id: Dict[str, list[float]] = {}
        self._face_embedding_by_track: Dict[str, Tuple[list[float], float]] = {}
        # Per-track previous-frame crop thumbnail (grayscale int16) for scene-cut /
        # person-swap detection. Keyed by yolo track id.
        self._crop_thumb_by_track_id: Dict[str, np.ndarray] = {}
        self.face_track_to_person_id: Dict[str, str] = {}
        self.face_track_last_seen_sec: Dict[str, float] = {}
        self._object_pose_history: Dict[int, List[Tuple[float, Tuple[int, int], Tuple[float, float, float], bool]]] = {}
        self._robot_pose_history: List[Tuple[float, Tuple[int, int], Tuple[float, float, float, str, float]]] = []
        self._latest_robot_pose: Optional[Dict[str, float | str]] = None
        self._observation_insert_count = 0
        self._last_saved_scene_thumb: Optional[np.ndarray] = None
        self._last_saved_scene_id: Optional[int] = None
        self._last_scene_save_time: float = 0.0
        self._scene_dedup_skip_count: int = 0
        self.tf_buffer = Buffer(cache_time=Duration(seconds=120.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=False)
        self._cached_camera_transforms: Dict[Tuple[int, int], Any] = {}

        # -----------------------------
        # Timers
        # -----------------------------
        self.service_check_timer = self.create_timer(
            1.0,
            self._check_embedding_service,
            callback_group=self.timer_group,
        )

        self.timeout_check_timer = self.create_timer(
            0.5,
            self._check_pending_timeouts,
            callback_group=self.timer_group,
        )
        self.interaction_retry_timer = self.create_timer(
            1.0,
            self._flush_pending_interactions,
            callback_group=self.timer_group,
        )
        self.face_retry_timer = self.create_timer(
            2.0,
            self._flush_pending_faces,
            callback_group=self.timer_group,
        )
        self.pending_observation_retry_timer = self.create_timer(
            0.25,
            self._flush_pending_observations,
            callback_group=self.timer_group,
        )

    def _handle_detection_embedding_response(self, future, pending):
        self.get_logger().info(
            f"Handling embedding response for track {pending.get('yolo_track_id')} in scene {pending['scene_id']}"
        )
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"Embedding service future failed for track {pending.get('yolo_track_id')}: {exc}"
            )
            return
        if result is None or not result.success:
            self.get_logger().error(
                f"Embedding service {pending.get('embedding_service_name', self.embedding_service_name)} "
                f"failed for track {pending.get('yolo_track_id')}: {getattr(result, 'message', 'no result')}"
            )
            return
        embedding = self._coerce_embedding_dim(list(result.embedding))
        detection_colors = (
            ((pending.get('attributes_json') or {}).get('colors') or {}).get('histogram')
        )
        object_id = self._resolve_object_id_for_detection(
            embedding=embedding,
            mapped_object_id=pending.get('mapped_object_id'),
            class_id=pending.get('class_id'),
            yolo_track_id=pending.get('yolo_track_id'),
            frame_sec=pending.get('frame_sec'),
            estimated_position=pending.get('estimated_position'),
            detection_colors=detection_colors,
        )
        self.get_logger().info(
            f"Received embedding for object {object_id} with dim {result.embedding_dim}"
        )
        if pending.get('yolo_track_id') is not None:
            self._embedding_by_track_id[pending['yolo_track_id']] = list(embedding)
        embedding_dim = int(result.embedding_dim)
        # Record which embedding backend produced this vector
        svc = pending.get('embedding_service_name', self.embedding_service_name)
        if 'osnet' in svc.lower():
            pending['embedding_backend'] = 'osnet'
        elif 'convnext' in svc.lower():
            pending['embedding_backend'] = 'convnext'
        else:
            pending['embedding_backend'] = svc
        try:
            self._persist_or_queue_observation(
                object_id=object_id,
                embedding=embedding,
                pending=pending,
            )
        except Exception as exc:
            self.get_logger().error(
                f"Skipping failed observation insert for object {object_id}: {exc}"
            )
            return
        if self.debug_log_services:
            self.get_logger().info(
                f"Requested service: {pending.get('embedding_service_name', self.embedding_service_name)}"
            )
            for name, types in self.get_service_names_and_types():
                self.get_logger().info(f"Visible service: {name} -> {types}")

    # ------------------------------------------------------------------
    # Example callbacks
    # ------------------------------------------------------------------
    def vlm_callback(self, msg: StampedString) -> None:
        key = self._stamp_key(msg)
        self._latest_vlm_by_stamp[key] = msg.data
        self._latest_vlm_by_stamp.move_to_end(key)
        self._prune_old_vlm_entries(current_stamp=key)
        self._backfill_scene_caption_for_stamp(key, msg.data)
        self.get_logger().info(
            f"Stored VLM text for timestamp {key[0]}.{key[1]}: {msg.data}"
        )

    def _backfill_scene_caption_for_stamp(self, stamp_key: Tuple[int, int], caption: str) -> None:
        if caption is None or not str(caption).strip():
            return

        scene_ts_sec = self._stamp_key_to_seconds(stamp_key)

        # Write the caption ONLY to the scene whose timestamp exactly equals this caption's
        # own source-frame stamp. The caption was generated from that exact frame; grafting
        # it onto a *neighboring* scene (the old "nearest empty-caption scene within 2s"
        # behaviour) produced captions that did not match the scene's image. If no scene
        # exists at this exact stamp (e.g. the frame was dedup-skipped), discard the caption.
        try:
            sql = """
            UPDATE scenes s
            SET caption = %s
            WHERE s.timestamp = to_timestamp(%s)
              AND (s.caption IS NULL OR btrim(s.caption) = '')
            RETURNING s.id
            """
            updated = self._db_execute(
                sql,
                (
                    caption,
                    scene_ts_sec,
                ),
                fetchone=True,
            )
            if updated is not None:
                _cap_age_ms = (time.time() - scene_ts_sec) * 1000.0 if scene_ts_sec > 0 else -1.0
                self.get_logger().info(
                    f"Backfilled caption for scene id {updated[0]} at its exact VLM timestamp "
                    f"{stamp_key[0]}.{stamp_key[1]}"
                )
                self.get_logger().info(
                    f"[latency] caption: capture_to_caption_saved={_cap_age_ms:.0f}ms scene_id={updated[0]}"
                )
            else:
                self.get_logger().debug(
                    f"Discarded VLM caption for {stamp_key[0]}.{stamp_key[1]}: no scene at that "
                    f"exact timestamp needing a caption (frame likely dedup-skipped)."
                )
        except Exception as exc:
            self.get_logger().error(
                f"Failed to backfill scene caption for {stamp_key[0]}.{stamp_key[1]}: {exc}"
            )

    def _stamp_key_to_seconds(self, stamp_key: Tuple[int, int]) -> float:
        return float(stamp_key[0]) + (float(stamp_key[1]) / 1e9)

    def _prune_old_vlm_entries(self, current_stamp: Tuple[int, int]) -> None:
        now_sec = self._stamp_key_to_seconds(current_stamp)
        stale_keys = [
            key for key in self._latest_vlm_by_stamp
            if (now_sec - self._stamp_key_to_seconds(key)) > self._vlm_cache_ttl_sec
        ]
        for key in stale_keys:
            self._latest_vlm_by_stamp.pop(key, None)
        while len(self._latest_vlm_by_stamp) > self._vlm_cache_max_entries:
            self._latest_vlm_by_stamp.popitem(last=False)

    def _remember_scene_id_for_stamp(self, stamp_key: Tuple[int, int], scene_id: int) -> None:
        self._scene_id_by_stamp[stamp_key] = int(scene_id)
        self._scene_id_by_stamp.move_to_end(stamp_key)
        while len(self._scene_id_by_stamp) > self._scene_id_cache_max_entries:
            self._scene_id_by_stamp.popitem(last=False)

    def _lookup_vlm_text_for_stamp(self, stamp_key: Tuple[int, int]) -> Optional[str]:
        # Exact match ONLY. A VLM caption is tagged (by scene_description_node) with the
        # stamp of the exact frame it described. Substituting a *different* frame's caption
        # (e.g. the most recent one within a time window) writes a caption that does not
        # match this frame's image — the caption/image desync bug. If this frame's caption
        # has not arrived yet, return None and leave the scene caption empty; the exact
        # backfill (_backfill_scene_caption_for_stamp) will fill it when the VLM responds.
        return self._latest_vlm_by_stamp.get(stamp_key)

    def image_callback(self, image_msg: Image) -> None:
        """
        Replace this with your real detection/image pipeline entry point.

        This callback demonstrates the important pattern:
        - do not block here
        - send async service request
        - continue processing in the future callback
        """
        embedding_client = self._embedding_client_for_class(class_id)
        embedding_service_name = self._embedding_service_name_for_class(class_id)
        if not embedding_client.service_is_ready():
            self.get_logger().warn(
                f"Embedding service {embedding_service_name} not ready yet, skipping image "
                f"{image_msg.header.stamp.sec}.{image_msg.header.stamp.nanosec}"
            )
            return

        self.request_embedding_async(image_msg, source="image_callback")

    # ------------------------------------------------------------------
    # Async embedding request flow
    # ------------------------------------------------------------------
    def _new_request_id(self) -> int:
        request_id = self._next_request_id
        self._next_request_id += 1
        return request_id

    def request_embedding_async(self, image_msg: Image, source: str = "", class_id: int = 0) -> None:
        stamp = self._stamp_key(image_msg)
        request_id = self._new_request_id()

        self.get_logger().info(
            f"Requesting embedding for image with timestamp "
            f"{stamp[0]}.{stamp[1]}"
        )

        req = GetImageEmbedding.Request()
        req.image = image_msg

        pending = PendingEmbeddingRequest(
            request_id=request_id,
            image_msg=image_msg,
            stamp_key=stamp,
            source=source,
            class_id=class_id,
        )
        self._pending_requests[request_id] = pending

        future = self._embedding_client_for_class(class_id).call_async(req)
        future.add_done_callback(
            lambda fut, rid=request_id: self._handle_embedding_response(rid, fut)
        )

    def _handle_embedding_response(self, request_id: int, future) -> None:
        pending = self._pending_requests.pop(request_id, None)
        if pending is None:
            self.get_logger().warn(
                f"Received embedding response for unknown request_id={request_id}"
            )
            return

        stamp = pending.stamp_key

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(
                f"Embedding service future failed for "
                f"{stamp[0]}.{stamp[1]}: {exc}"
            )
            return

        if result is None:
            self.get_logger().error(
                f"Embedding service returned no result for "
                f"{stamp[0]}.{stamp[1]}"
            )
            return

        if not result.success:
            self.get_logger().error(
                f"Embedding service failed for "
                f"{stamp[0]}.{stamp[1]}: {result.message}"
            )
            return

        embedding = self._coerce_embedding_dim(list(result.embedding))
        embedding_dim = int(result.embedding_dim)

        self.get_logger().info(
            f"Received embedding for timestamp {stamp[0]}.{stamp[1]} "
            f"with dim {embedding_dim}"
        )

        vlm_text = self._latest_vlm_by_stamp.get(stamp)

        self.process_embedding_result(
            image_msg=pending.image_msg,
            embedding=embedding,
            embedding_dim=embedding_dim,
            vlm_text=vlm_text,
            source=pending.source,
            class_id=pending.class_id,
        )

    # ------------------------------------------------------------------
    # Timeout handling
    # ------------------------------------------------------------------
    def _check_pending_timeouts(self) -> None:
        """
        Optional timeout guard for bookkeeping.
        Since call_async futures do not carry your own timestamp metadata,
        we track age using ROS time now versus image header time.
        """
        now = self.get_clock().now().nanoseconds / 1e9
        timed_out_ids = []

        for request_id, pending in self._pending_requests.items():
            stamp_sec = pending.image_msg.header.stamp.sec + (
                pending.image_msg.header.stamp.nanosec / 1e9
            )
            age = now - stamp_sec

            if age > self.embedding_timeout_sec:
                timed_out_ids.append(request_id)

        for request_id in timed_out_ids:
            pending = self._pending_requests.pop(request_id, None)
            if pending is None:
                continue

            stamp = pending.stamp_key
            self.get_logger().error(
                f"Embedding service call timed out for "
                f"{stamp[0]}.{stamp[1]}"
            )

    # ------------------------------------------------------------------
    # Downstream processing
    # ------------------------------------------------------------------
    def process_embedding_result(
        self,
        image_msg: Image,
        embedding: list[float],
        embedding_dim: int,
        vlm_text: Optional[str],
        source: str = "",
        yolo_id: Optional[str] = None,
        class_id: int = 0,
    ) -> None:
        """
        Save embedding result to pgvector database, mapping YOLO IDs to object_ids.
        """
        stamp = self._stamp_key(image_msg)

        self.get_logger().info(
            f"Processing embedding result for {stamp[0]}.{stamp[1]} "
            f"(dim={embedding_dim}, source={source})"
        )

        if vlm_text is not None:
            self.get_logger().info(f"Matched VLM text: {vlm_text}")
        else:
            self.get_logger().info("No VLM text matched for this timestamp")

        # --- YOLO ID to object_id mapping logic ---
        # yolo_id should be passed from detection pipeline (e.g., track_id)
        if yolo_id is None:
            yolo_id = "unknown"

        cropped_image = b''    # Replace with actual cropped image bytes
        x = y = z = None      # Replace with actual coordinates if available

        # Check if YOLO ID is already mapped to an object_id
        if yolo_id in self.yolo_id_to_object_id:
            object_id = self.yolo_id_to_object_id[yolo_id]
        else:
            # Use YOLO ID as object_id for DB, or generate a new one if needed
            object_id = str(yolo_id)
            try:
                self._ensure_object_exists(object_id, class_id)
                self.yolo_id_to_object_id[yolo_id] = object_id
                self.get_logger().info(f"Mapped YOLO ID {yolo_id} to object_id {object_id}")
            except Exception as exc:
                self.get_logger().error(f"Failed to ensure object exists: {exc}")

        # Insert scene if VLM text is available
        scene_id = None
        active_map_id = self._get_active_map_id()
        if vlm_text:
            try:
                # Use image timestamp for scene timestamp
                scene_timestamp = self._timestamp_from_header(image_msg.header)
                scene_id = self._insert_scene(
                    vlm_text,
                    x,
                    y,
                    scene_timestamp,
                    str(getattr(image_msg.header, "frame_id", "") or ""),
                    map_id=active_map_id,
                )
                self._remember_scene_id_for_stamp(stamp, scene_id)
                self.get_logger().info(f"Saved scene with id {scene_id} and caption '{vlm_text}'")
            except Exception as exc:
                self.get_logger().error(f"Failed to save scene: {exc}")

        try:
            self._insert_object_observation(
                object_id=object_id,
                cropped_image=cropped_image,
                mask_image=b"",
                original_cropped_image=b"",
                embedding=embedding,
                x=x,
                y=y,
                z=z,
                scene_id=scene_id,
                yolo_track_id=yolo_id,
                class_id=class_id,
                confidence=None,
                position_source='yolo',
                map_id=active_map_id,
            )
            self.get_logger().info("Saved object observation to database.")
        except Exception as exc:
            self.get_logger().error(f"Failed to save object observation to database: {exc}")

    # ------------------------------------------------------------------
    # Legacy-compatible helper
    # ------------------------------------------------------------------
    def request_embedding(self, image_msg: Image):
        """
        Deprecated synchronous-style wrapper.

        Kept only so existing call sites do not break immediately.
        It now dispatches async and returns None.
        Migrate callers to `request_embedding_async(...)`.
        """
        self.get_logger().warn(
            "request_embedding() was called synchronously. "
            "This node now uses async service requests. "
            "Use request_embedding_async(...) instead."
        )
        self.request_embedding_async(image_msg, source="legacy_request_embedding")
        return None


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = DatabaseNode()

    # MultiThreadedExecutor is recommended when mixing subscriptions,
    # service clients, timers, and other potentially blocking work.
    executor = rclpy.executors.MultiThreadedExecutor()

    try:
        executor.add_node(node)
        # Keep the node alive across database outages: a psycopg2 error that
        # escapes a callback (e.g. Postgres killed longer than the retry
        # window) is logged and the executor keeps spinning. _db_execute
        # reconnects automatically once the database is back.
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=1.0)
            except psycopg2.Error as exc:
                node.get_logger().error(
                    f"Database error escaped a callback; node kept alive "
                    f"(will reconnect automatically): {exc}"
                )
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
