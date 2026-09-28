#!/usr/bin/env python3
"""Lightweight scene captioner for the stitched front-middle camera.

Subscribes to /spot/camera/frontmiddle_virtual/image, samples at a low
rate (default 5 s), calls the same VLM, and publishes the caption on
/dynosam/stitched_vlm_result.  The caption is also written to
/shared/stitched_caption.json so the web backend can read it without
needing a ROS subscription.

This node is intentionally separate from scene_description_node so that
switching the stitched image does not break timestamp matching in the
database node (which still pairs frontleft images with YOLO outputs).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
from openai import OpenAI

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from bordsupr_interfaces.msg import StampedString

logger = logging.getLogger(__name__)
SHARED_CAPTION_PATH = Path("/shared/stitched_caption.json")


class StitchedSceneDescriptionNode(Node):
    def __init__(self) -> None:
        super().__init__("stitched_scene_description_node")

        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/spot/camera/frontmiddle_virtual/image")
        self.declare_parameter("vlm_topic", "/dynosam/stitched_vlm_result")
        self.declare_parameter("vlm_api_url", "http://vlm_server:8000/v1")
        self.declare_parameter("model_name", "Qwen/Qwen3-VL-4B-Instruct")
        self.declare_parameter("vlm_disable_thinking", False)
        self.declare_parameter("use_kimi", False)
        self.declare_parameter("kimi_api_url", "https://api.moonshot.ai/v1")
        self.declare_parameter("kimi_api_key", "")
        self.declare_parameter("kimi_model", "kimi-k2.6")
        self.declare_parameter("use_gemini", False)
        self.declare_parameter("gemini_api_key", "")
        self.declare_parameter("gemini_model", "gemini-2.5-pro")
        self.declare_parameter("caption_interval_sec", 5.0)
        self.declare_parameter("captions_enabled", True)
        self.declare_parameter("prompt_text", "Describe what is visible in this image in 2 sentences.")

        rgb_topic = self.get_parameter("rgb_topic").value
        vlm_topic = self.get_parameter("vlm_topic").value
        vlm_api_url = self.get_parameter("vlm_api_url").value
        model_name = self.get_parameter("model_name").value
        use_kimi = self.get_parameter("use_kimi").value
        kimi_api_url = self.get_parameter("kimi_api_url").value
        kimi_api_key = self.get_parameter("kimi_api_key").value
        kimi_model = self.get_parameter("kimi_model").value
        use_gemini = self.get_parameter("use_gemini").value
        gemini_api_key = self.get_parameter("gemini_api_key").value
        gemini_model = self.get_parameter("gemini_model").value
        self.caption_interval_sec = float(self.get_parameter("caption_interval_sec").value)
        self.captions_enabled = bool(self.get_parameter("captions_enabled").value)
        self.vlm_disable_thinking = bool(self.get_parameter("vlm_disable_thinking").value)
        self.prompt_text = self.get_parameter("prompt_text").value

        if use_gemini:
            if not gemini_api_key:
                self.get_logger().error("use_gemini is true but gemini_api_key is empty")
            self.model_name = gemini_model
            self.client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=gemini_api_key,
            )
            self.get_logger().info("Using Gemini API for stitched scene description")
        elif use_kimi:
            if not kimi_api_key:
                self.get_logger().error("use_kimi is true but kimi_api_key is empty")
            self.model_name = kimi_model
            self.client = OpenAI(base_url=kimi_api_url, api_key=kimi_api_key)
            self.get_logger().info("Using Kimi API for stitched scene description")
        else:
            self.model_name = model_name
            self.client = OpenAI(base_url=vlm_api_url, api_key="lm-studio")

        self._latest_image: Optional[Image] = None
        self._worker_busy = False
        self._last_caption_time = 0.0

        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub = self.create_subscription(Image, rgb_topic, self._image_cb, stream_qos)
        self.pub = self.create_publisher(StampedString, vlm_topic, stream_qos)
        self.timer = self.create_timer(self.caption_interval_sec, self._timer_cb)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"VLM topic: {vlm_topic}")
        self.get_logger().info(f"Caption interval: {self.caption_interval_sec}s")

    def _image_cb(self, msg: Image) -> None:
        self._latest_image = msg

    def _timer_cb(self) -> None:
        if not self.captions_enabled or self._worker_busy or self._latest_image is None:
            return
        self._worker_busy = True
        msg = self._latest_image
        self._latest_image = None
        try:
            self._caption_image(msg)
        finally:
            self._worker_busy = False

    def _caption_image(self, msg: Image) -> None:
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed: {exc}")
            return

        try:
            # Rotate 90° clockwise to match frontleft orientation
            bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
            ok, buffer = cv2.imencode(".jpg", bgr)
            if not ok:
                self.get_logger().error("Failed to encode image to JPEG")
                return

            image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

            extra_body = (
                {"chat_template_kwargs": {"enable_thinking": False}}
                if getattr(self, "vlm_disable_thinking", False)
                else None
            )
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                            },
                        ],
                    }
                ],
                extra_body=extra_body,
            )

            response_text = completion.choices[0].message.content

            out = StampedString()
            out.data = response_text
            out.header = msg.header
            self.pub.publish(out)

            # Write to shared file for web backend
            try:
                SHARED_CAPTION_PATH.write_text(
                    json.dumps(
                        {
                            "caption": response_text,
                            "timestamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                            "frame_id": msg.header.frame_id,
                        }
                    )
                )
            except Exception as exc:
                self.get_logger().warning(f"Failed to write shared caption file: {exc}")

            self.get_logger().info(f"Published stitched VLM text: {response_text}")

        except Exception as exc:
            self.get_logger().error(f"VLM request failed: {exc}")


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = StitchedSceneDescriptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
