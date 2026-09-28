import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import numpy as np
import cv2


class ImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')
        self.declare_parameter("rgb_topic", "/camera/frontleft/image_rotated")
        rgb_topic = str(self.get_parameter("rgb_topic").value)

        self.publisher_ = self.create_publisher(Image, rgb_topic, 10)
        self.bridge = CvBridge()

        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        # Create a dummy image (e.g., 640x480 blue image)
        img = cv2.imread('/workspace/src/bordsupr/resource/test_image_2.jpg')

        msg = self.bridge.cv2_to_imgmsg(img, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(msg)

        self.get_logger().info('Publishing test image')


def main():
    rclpy.init()
    node = ImagePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

# python3 image_publisher.py