#!/usr/bin/env python3
"""Local optical-flow node.

DEPRECATION NOTE: Despite the historical name ``dynosam_optical_flow_node`` and the
``/dynosam/flow`` topic it publishes to, this node does NOT use the DynoSAM
dynamic-SLAM backend. DynoSAM is no longer part of the active pipeline. This node
computes optical flow locally from the RGB camera stream. The topic/node names are
kept only for compatibility. (This node is disabled by default; see
``BORDSUPR_ENABLE_OPTICAL_FLOW`` in the launch file.)
"""

from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class DynoSAMOpticalFlowNode(Node):
    def __init__(self) -> None:
        super().__init__("dynosam_optical_flow_node")

        self.bridge = CvBridge()

        self.prev_gray = None
        self.prev_header = None
        self.frame_count = 0

        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("flow_topic", "/dynosam/flow")

        rgb_topic = self.get_parameter("rgb_topic").value
        flow_topic = self.get_parameter("flow_topic").value

        self.sub = self.create_subscription(Image, rgb_topic, self.callback, 10)
        self.pub = self.create_publisher(Image, flow_topic, 10)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"Flow topic: {flow_topic}")

    def callback(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed: {exc}")
            return

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # First frame → no flow yet
        if self.prev_gray is None:
            self.prev_gray = gray
            # Keeping the header makes sure we have a valid timestamp for the first flow message (even if it's all zeros)
            self.prev_header = msg.header
            return

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ).astype(np.float32)

        out = self.bridge.cv2_to_imgmsg(flow, encoding="32FC2")
        out.header = msg.header

        self.pub.publish(out)

        self.prev_gray = gray
        self.prev_header = msg.header

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Published {self.frame_count} optical flow frames"
            )


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = DynoSAMOpticalFlowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

# python3 dynosam_optical_flow_node.py --ros-args --params-file flow.yaml