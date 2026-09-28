#!/usr/bin/env python3

import copy

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


class RotateImagesAndCameraInfo(Node):
    def __init__(self):
        super().__init__("rotate_images_and_camera_info")

        self.declare_parameter("origin_image_topic", "/spot/camera/frontleft/image")
        self.declare_parameter("origin_depth_topic", "/spot/depth_registered/frontleft/image")
        self.declare_parameter("origin_camera_info_topic", "/spot/camera/frontleft/camera_info")
        self.declare_parameter("rotation", "cw")  # cw, ccw, 180
        self.declare_parameter("sync_queue_size", 20)
        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("publish_every_n_frames", 60)

        self.declare_parameter("image_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("depth_topic", "/spot/depth_registered/frontleft/image_rotated")
        self.declare_parameter("camera_info_topic", "/spot/camera/frontleft/camera_info_rotated")

        self.image_topic = self.get_parameter("origin_image_topic").get_parameter_value().string_value
        self.depth_topic = self.get_parameter("origin_depth_topic").get_parameter_value().string_value
        self.camera_info_topic = self.get_parameter("origin_camera_info_topic").get_parameter_value().string_value
        self.rotation = self.get_parameter("rotation").get_parameter_value().string_value.lower()
        self.sync_queue_size = self.get_parameter("sync_queue_size").get_parameter_value().integer_value
        self.sync_slop_sec = self.get_parameter("sync_slop_sec").get_parameter_value().double_value
        self.publish_every_n_frames = max(
            1, self.get_parameter("publish_every_n_frames").get_parameter_value().integer_value
        )

        self.out_image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.out_depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.out_camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        self.frame_counter = 0
        self.latest_camera_info = None
        self._last_pub_sec = -1
        self._last_pub_nsec = -1

        if self.rotation not in ("cw", "ccw", "180"):
            raise ValueError("rotation must be one of: 'cw', 'ccw', '180'")

        self.bridge = CvBridge()

        self.image_pub = self.create_publisher(Image, self.out_image_topic, 10)
        self.depth_pub = self.create_publisher(Image, self.out_depth_topic, 10)
        camera_info_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.camera_info_pub = self.create_publisher(CameraInfo, self.out_camera_info_topic, camera_info_qos)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.rgb_sub = Subscriber(self, Image, self.image_topic, qos_profile=sensor_qos)
        self.depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=sensor_qos)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, sensor_qos
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=self.sync_queue_size,
            slop=self.sync_slop_sec,
        )
        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            f"Started with rotation='{self.rotation}', "
            f"image_topic='{self.image_topic}', depth_topic='{self.depth_topic}', "
            f"camera_info_topic='{self.camera_info_topic}', "
            f"sync_queue_size={self.sync_queue_size}, sync_slop_sec={self.sync_slop_sec}, "
            f"publish_every_n_frames={self.publish_every_n_frames}"
        )

    def rotate_cv_image(self, img: np.ndarray) -> np.ndarray:
        if self.rotation == "cw":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation == "ccw":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return cv2.rotate(img, cv2.ROTATE_180)

    def rotated_camera_info(self, msg: CameraInfo) -> CameraInfo:
        out = copy.deepcopy(msg)

        W = msg.width
        H = msg.height

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        R = np.array(msg.r, dtype=np.float64).reshape(3, 3)
        P = np.array(msg.p, dtype=np.float64).reshape(3, 4)

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        Tx = P[0, 3]
        Ty = P[1, 3]

        if self.rotation == "cw":
            new_W = H
            new_H = W

            fx_new = fy
            fy_new = fx
            cx_new = H - 1 - cy
            cy_new = cx

            Tx_new = Ty
            Ty_new = Tx

        elif self.rotation == "ccw":
            new_W = H
            new_H = W

            fx_new = fy
            fy_new = fx
            cx_new = cy
            cy_new = W - 1 - cx

            Tx_new = Ty
            Ty_new = Tx

        else:  # 180
            new_W = W
            new_H = H

            fx_new = fx
            fy_new = fy
            cx_new = W - 1 - cx
            cy_new = H - 1 - cy

            Tx_new = Tx
            Ty_new = Ty

        K_new = np.array(
            [
                [fx_new, 0.0, cx_new],
                [0.0, fy_new, cy_new],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        # Keep rectification as identity if it was identity, otherwise rotate it in image space.
        # For many monocular/RGBD use cases this is sufficient. If your pipeline depends heavily
        # on calibrated stereo rectification, verify R/P for your specific setup.
        R_new = R.copy()

        P_new = np.array(
            [
                [fx_new, 0.0, cx_new, Tx_new],
                [0.0, fy_new, cy_new, Ty_new],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        out.width = int(new_W)
        out.height = int(new_H)
        out.k = K_new.reshape(-1).tolist()
        out.r = R_new.reshape(-1).tolist()
        out.p = P_new.reshape(-1).tolist()

        return out

    def camera_info_callback(self, camera_info_msg: CameraInfo) -> None:
        self.latest_camera_info = camera_info_msg
        self.camera_info_pub.publish(self.rotated_camera_info(camera_info_msg))

    def synced_callback(self, rgb_msg: Image, depth_msg: Image) -> None:
        self.frame_counter += 1
        if self.frame_counter % self.publish_every_n_frames != 0:
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="passthrough")
            rotated_rgb = self.rotate_cv_image(cv_img)
            out_rgb_msg = self.bridge.cv2_to_imgmsg(rotated_rgb, encoding=rgb_msg.encoding)

            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            rotated_depth = self.rotate_cv_image(cv_depth)
            out_depth_msg = self.bridge.cv2_to_imgmsg(rotated_depth, encoding=depth_msg.encoding)

            synced_header = rgb_msg.header

            # Enforce strictly monotonic timestamps to prevent DynoSAM backend
            # crash (CHECK_GT(timestamp_k, last_nav_state_time_)).
            current_ns = synced_header.stamp.sec * 1_000_000_000 + synced_header.stamp.nanosec
            last_ns = self._last_pub_sec * 1_000_000_000 + self._last_pub_nsec
            if current_ns <= last_ns:
                # Add 1 us (1000 ns) to guarantee the difference survives
                # the float64 conversion DynoSAM uses (static_cast<double>(ns)/1e9).
                new_ns = last_ns + 1_000
                synced_header.stamp.sec = new_ns // 1_000_000_000
                synced_header.stamp.nanosec = new_ns % 1_000_000_000
                self.get_logger().warning(
                    f"Adjusted non-monotonic timestamp by +1 us: "
                    f"{self._last_pub_sec}.{self._last_pub_nsec:09d} -> "
                    f"{synced_header.stamp.sec}.{synced_header.stamp.nanosec:09d}"
                )

            self._last_pub_sec = synced_header.stamp.sec
            self._last_pub_nsec = synced_header.stamp.nanosec

            self.get_logger().info(
                f"Publishing frame {self.frame_counter} with stamp "
                f"{synced_header.stamp.sec}.{synced_header.stamp.nanosec:09d}"
            )

            out_rgb_msg.header = synced_header
            out_depth_msg.header = synced_header

            self.image_pub.publish(out_rgb_msg)
            self.depth_pub.publish(out_depth_msg)
            if self.latest_camera_info is not None:
                cam_info_out = self.rotated_camera_info(self.latest_camera_info)
                cam_info_out.header = synced_header
                self.camera_info_pub.publish(cam_info_out)
        except Exception as e:
            self.get_logger().error(f"Failed to process synchronized image pair: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RotateImagesAndCameraInfo()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
