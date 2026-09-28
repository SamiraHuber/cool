#!/usr/bin/env python3
"""
ROS 2 video recorder node for the stitched front image.

Subscribes to a sensor_msgs/Image topic and writes incoming frames
to an MP4 video file. Start/stop is controlled via a shared JSON
request file so the web backend can drive it without a ROS connection.

Usage:
  ros2 run bordsupr video_recorder_node --ros-args \
    -p image_topic:=/spot/camera/frontmiddle_virtual/image \
    -p output_dir:=/shared/videos \
    -p fps:=20

Environment:
  VIDEO_RECORD_REQUEST_PATH  default /shared/video_record_request.json
  VIDEO_RECORD_STATUS_PATH   default /shared/video_record_status.json
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


VIDEO_RECORD_REQUEST_PATH = Path(
    os.getenv("VIDEO_RECORD_REQUEST_PATH", "/shared/video_record_request.json")
)
VIDEO_RECORD_STATUS_PATH = Path(
    os.getenv("VIDEO_RECORD_STATUS_PATH", "/shared/video_record_status.json")
)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


class VideoRecorderNode(Node):
    def __init__(self):
        super().__init__("video_recorder_node")

        self.declare_parameter("image_topic", "/spot/camera/frontmiddle_virtual/image")
        self.declare_parameter("output_dir", "/shared/videos")
        self.declare_parameter("fps", 20.0)
        self.declare_parameter("codec", "mp4v")
        self.declare_parameter("poll_interval_sec", 0.5)

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.output_dir = Path(
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self.fps = self.get_parameter("fps").get_parameter_value().double_value
        self.codec = (
            self.get_parameter("codec").get_parameter_value().string_value
        )
        self.poll_interval_sec = (
            self.get_parameter("poll_interval_sec").get_parameter_value().double_value
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.writer: cv2.VideoWriter | None = None
        self.current_filename: str | None = None
        self.frames_recorded = 0
        self.recording_start_time: float | None = None
        self.last_request_id: str | None = None
        self.last_request_time: float = 0.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.subscription = self.create_subscription(
            Image, self.image_topic, self.image_callback, qos
        )

        self.poll_timer = self.create_timer(
            self.poll_interval_sec, self._poll_request
        )
        self.status_timer = self.create_timer(2.0, self._publish_status)

        self._write_status("idle", "Video recorder ready.")
        self.get_logger().info(
            f"Video recorder ready. Topic: {self.image_topic}, "
            f"Output: {self.output_dir}, FPS: {self.fps}"
        )

    def _write_status(self, state: str, message: str, **extra) -> None:
        payload = {
            "state": state,
            "message": message,
            "updated_at": time.time(),
        }
        if self.current_filename:
            payload["filename"] = self.current_filename
        if self.recording_start_time is not None:
            payload["started_at"] = self.recording_start_time
        if state == "recording" or self.frames_recorded:
            payload["frames_recorded"] = self.frames_recorded
        payload.update(extra)
        atomic_write_json(VIDEO_RECORD_STATUS_PATH, payload)

    def _publish_status(self) -> None:
        if self.current_filename is None:
            return
        if self.writer is not None:
            self._write_status(
                "recording",
                f"Recording: {self.current_filename} ({self.frames_recorded} frames)",
            )
        else:
            self._write_status("idle", "Video recorder ready.")

    def _poll_request(self) -> None:
        request = load_json_file(VIDEO_RECORD_REQUEST_PATH)
        if not request:
            return

        request_id = request.get("request_id")
        if not request_id or request_id == self.last_request_id:
            return

        action = request.get("action")
        self.last_request_id = request_id
        self.last_request_time = time.time()

        if action == "start":
            self._start_recording(request)
        elif action == "stop":
            self._stop_recording()
        else:
            self.get_logger().warning(f"Unknown video record action: {action}")

    def _start_recording(self, request: dict) -> None:
        if self.writer is not None:
            self._stop_recording()

        requested_name = request.get("filename")
        if requested_name:
            filename = Path(requested_name).name
            if not filename.endswith(".mp4"):
                filename += ".mp4"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stitched_{timestamp}.mp4"

        self.current_filename = filename
        self.frames_recorded = 0
        self.recording_start_time = time.time()

        # Defer writer creation until first frame so we know the resolution
        self.writer = None

        self._write_status(
            "recording",
            f"Recording started: {filename}",
            filename=filename,
        )
        self.get_logger().info(f"Started recording: {self.output_dir / filename}")

    def _stop_recording(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        filename = self.current_filename
        duration = (
            time.time() - self.recording_start_time
            if self.recording_start_time else 0.0
        )

        self._write_status(
            "stopped",
            f"Recording stopped: {filename} ({self.frames_recorded} frames, {duration:.1f}s)",
            filename=filename,
            frames_recorded=self.frames_recorded,
            duration_sec=round(duration, 1),
        )
        self.get_logger().info(
            f"Stopped recording: {filename} ({self.frames_recorded} frames, {duration:.1f}s)"
        )

        self.current_filename = None
        self.frames_recorded = 0
        self.recording_start_time = None

    def image_callback(self, msg: Image) -> None:
        if self.writer is None and self.current_filename is None:
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"cv_bridge conversion failed: {exc}")
            return

        if bgr is None or bgr.size == 0:
            return

        # Initialise writer on first frame so we know resolution
        if self.writer is None:
            h, w = bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            out_path = str(self.output_dir / self.current_filename)
            self.writer = cv2.VideoWriter(out_path, fourcc, self.fps, (w, h))
            if not self.writer.isOpened():
                self.get_logger().error(f"Failed to open VideoWriter for {out_path}")
                self.writer = None
                self.current_filename = None
                self._write_status("error", f"Failed to open VideoWriter for {out_path}")
                return

        self.writer.write(bgr)
        self.frames_recorded += 1

    def destroy_node(self) -> None:
        self._stop_recording()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
