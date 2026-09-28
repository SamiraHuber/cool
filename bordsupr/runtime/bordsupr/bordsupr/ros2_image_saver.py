#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class ImageSaver(Node):
    def __init__(self):
        super().__init__("image_saver")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("output_dir", "./saved_images")
        self.declare_parameter("image_format", "png")
        self.declare_parameter("save_every_n", 1)

        self.image_topic = self.get_parameter("image_topic").value
        self.output_dir = Path(self.get_parameter("output_dir").value)
        self.image_format = self.get_parameter("image_format").value.lower().strip(".")
        self.save_every_n = self.get_parameter("save_every_n").value

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.image_count = 0
        self.saved_count = 0

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            10,
        )

        self.get_logger().info(f"Listening to: {self.image_topic}")
        self.get_logger().info(f"Saving to: {self.output_dir.resolve()}")

    def image_callback(self, msg):
        self.image_count += 1

        if self.image_count % self.save_every_n != 0:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Image conversion failed: {exc}")
            return

        stamp = msg.header.stamp
        timestamp = f"{stamp.sec}_{stamp.nanosec:09d}"
        filename = self.output_dir / f"image_{self.saved_count:06d}_{timestamp}.{self.image_format}"

        if cv2.imwrite(str(filename), cv_image):
            self.get_logger().info(f"Saved {filename}")
            self.saved_count += 1
        else:
            self.get_logger().error(f"Failed to save {filename}")


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
