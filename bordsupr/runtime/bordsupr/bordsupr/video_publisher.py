import os
from pathlib import Path
from typing import List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2


DEFAULT_HOST_RUNTIME_ROOT = os.getenv("BORDSUPR_HOST_RUNTIME_ROOT", str(Path(__file__).resolve().parents[4] / "bordsupr/runtime"))
DEFAULT_CONTAINER_RUNTIME_ROOT = os.getenv("BORDSUPR_CONTAINER_RUNTIME_ROOT", "/workspace/src")
EXTRA_RUNTIME_MOUNTS = os.getenv("BORDSUPR_EXTRA_RUNTIME_MOUNTS", "")


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("image_folder", "/workspace/src/bordsupr/resource/videos/frames_output/many_people_stairs_1920_1080_30fps")
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("image_stride", 1)
        self.declare_parameter("recursive", False)
        self.declare_parameter("loop", True)
        self.declare_parameter("max_images", 0)
        self.declare_parameter("frame_id_root", "")
        self.declare_parameter("stream_queue_depth", 1000)
        rgb_topic = str(self.get_parameter("rgb_topic").value)
        image_folder = str(self.get_parameter("image_folder").value)
        publish_hz = float(self.get_parameter("publish_hz").value)
        image_stride = int(self.get_parameter("image_stride").value)
        recursive = bool(self.get_parameter("recursive").value)
        self.loop = bool(self.get_parameter("loop").value)
        max_images = int(self.get_parameter("max_images").value)
        frame_id_root = str(self.get_parameter("frame_id_root").value).strip()
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)

        if publish_hz <= 0.0:
            publish_hz = 1.0
        if image_stride < 1:
            image_stride = 1
        if stream_queue_depth < 1:
            stream_queue_depth = 1

        self.image_folder = self._resolve_image_folder(Path(image_folder))
        self.frame_id_root = Path(frame_id_root).resolve() if frame_id_root else self.image_folder

        self.publisher_ = self.create_publisher(
            Image,
            rgb_topic,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=stream_queue_depth,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.bridge = CvBridge()

        self.image_files = self._collect_image_files(self.image_folder, recursive=recursive, image_stride=image_stride)
        if max_images > 0:
            self.image_files = self.image_files[:max_images]

        if not self.image_files:
            self.get_logger().error(f'No images found in {self.image_folder}')
        else:
            self.get_logger().info(
                f"Prepared {len(self.image_files)} image(s) from {self.image_folder} "
                f"(recursive={recursive}, loop={self.loop}, publish_hz={publish_hz}, image_stride={image_stride}, "
                f"stream_queue_depth={stream_queue_depth})"
            )

        self.current_index = 0
        self.published_count = 0
        self.done = False

        timer_period = 1.0 / publish_hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def _finish(self) -> None:
        if self.done:
            return
        self.done = True
        if self.timer is not None:
            self.timer.cancel()
        self.get_logger().info(
            f'Published {self.published_count} image(s); stopping publisher.'
        )

    @staticmethod
    def _resolve_image_folder(folder: Path) -> Path:
        mappings = [(DEFAULT_HOST_RUNTIME_ROOT, DEFAULT_CONTAINER_RUNTIME_ROOT)]
        for mount in EXTRA_RUNTIME_MOUNTS.split(";"):
            mount = mount.strip()
            if not mount or "=" not in mount:
                continue
            host_root, container_root = mount.split("=", 1)
            mappings.append((host_root.strip(), container_root.strip()))

        for host_root_value, container_root_value in mappings:
            if not host_root_value or not container_root_value:
                continue
            try:
                relative = folder.relative_to(Path(host_root_value).expanduser())
                translated = Path(container_root_value) / relative
                if translated.exists():
                    return translated.resolve()
            except Exception:
                pass
        return folder.resolve()

    @staticmethod
    def _collect_image_files(folder: Path, recursive: bool, image_stride: int) -> List[Path]:
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        files: List[Path] = []
        for pattern in patterns:
            if recursive:
                files.extend(folder.rglob(pattern))
            else:
                files.extend(folder.glob(pattern))
        return sorted({p.resolve() for p in files})[::max(1, image_stride)]

    def _frame_id_for_path(self, image_path: Path) -> str:
        try:
            relative = image_path.resolve().relative_to(self.frame_id_root)
            return relative.as_posix()
        except Exception:
            return image_path.name

    def timer_callback(self):
        if not self.image_files:
            return

        img_path = self.image_files[self.current_index]
        img = cv2.imread(str(img_path))
        if img is None:
            self.get_logger().error(f'Failed to read image: {img_path}')
            self.current_index += 1
            if self.current_index >= len(self.image_files):
                if self.loop:
                    self.current_index = 0
                    return
                self._finish()
            return

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id_for_path(img_path)
        self.publisher_.publish(msg)

        self.published_count += 1
        self.get_logger().info(
            f'Publishing image {self.current_index + 1}/{len(self.image_files)}: {msg.header.frame_id}'
        )

        self.current_index += 1
        if self.current_index >= len(self.image_files):
            if self.loop:
                self.current_index = 0
                return

            self._finish()


def main():
    rclpy.init()
    node = ImagePublisher()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

# python3 video_publisher.py
