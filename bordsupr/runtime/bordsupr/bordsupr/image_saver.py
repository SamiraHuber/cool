#!/usr/bin/env python3
"""
ROS 2 image saver node.

Subscribes to a ROS 2 sensor_msgs/Image topic and saves incoming images
to a folder as PNG or JPG files.

Usage:
  ros2 run <your_package> ros2_image_saver --ros-args \
    -p image_topic:=/camera/image_raw \
    -p output_dir:=/tmp/ros2_images \
    -p image_format:=png \
    -p save_every_n:=1

Direct Python usage:
  python3 ros2_image_saver.py --ros-args \
    -p image_topic:=/camera/image_raw \
    -p output_dir:=./images
"""

from pathlib import Path
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageSaver(Node):
    def __init__(self):
        super().__init__("image_saver")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_dir", "./saved_images")
        self.declare_parameter("image_format", "png")
        self.declare_parameter("save_every_n", 1)
        self.declare_parameter("max_age_sec", 5.0)

        self.image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self.output_dir = Path(
            self.get_parameter("output_dir").get_parameter_value().string_value
        )
        self.image_format = (
            self.get_parameter("image_format").get_parameter_value().string_value
            .lower()
            .strip(".")
        )
        self.save_every_n = (
            self.get_parameter("save_every_n").get_parameter_value().integer_value
        )
        self.max_age_sec = (
            self.get_parameter("max_age_sec").get_parameter_value().double_value
        )

        if self.image_format not in {"png", "jpg", "jpeg"}:
            raise ValueError("image_format must be one of: png, jpg, jpeg")

        if self.save_every_n < 1:
            raise ValueError("save_every_n must be >= 1")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.image_count = 0
        self.saved_count = 0

        qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription = self.create_subscription(Image, self.image_topic, self.image_callback, qos)

        self.get_logger().info(f"Listening to image topic: {self.image_topic}")
        self.get_logger().info(f"Saving images to: {self.output_dir.resolve()}")
        self.get_logger().info(f"Image format: {self.image_format}")
        self.get_logger().info(f"Saving every {self.save_every_n} frame(s)")

    def image_callback(self, msg: Image):
        self.image_count += 1

        if self.image_count % self.save_every_n != 0:
            return

        if self.max_age_sec > 0:
            stamp = msg.header.stamp
            if stamp.sec > 0:
                import time as _time
                age = _time.time() - (stamp.sec + stamp.nanosec * 1e-9)
                if age > self.max_age_sec:
                    self.get_logger().warn(
                        f"Skipping stale message (age={age:.1f}s > max_age_sec={self.max_age_sec}s)",
                        throttle_duration_sec=5.0,
                    )
                    return

        try:
            # "bgr8" works for normal OpenCV color images.
            # For mono images, cv_bridge will convert as needed.
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Failed to convert image: {exc}")
            return

        stamp = msg.header.stamp
        if stamp.sec != 0 or stamp.nanosec != 0:
            timestamp = f"{stamp.sec}_{stamp.nanosec:09d}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        filename = self.output_dir / f"image_{self.saved_count:06d}_{timestamp}.{self.image_format}"

        success = cv2.imwrite(str(filename), cv_image)
        if not success:
            self.get_logger().error(f"Failed to save image: {filename}")
            return

        self.saved_count += 1
        self.get_logger().info(f"Saved image: {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, Exception):
        node.get_logger().info("Stopped by user")
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
