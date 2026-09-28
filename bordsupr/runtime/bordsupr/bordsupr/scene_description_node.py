#!/usr/bin/env python3

import base64
from collections import deque
from typing import Optional

import cv2
from openai import OpenAI

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
from bordsupr_interfaces.msg import StampedString

# pip install ultralytics
from ultralytics import YOLO


class SceneDescriptionNode(Node):
    def __init__(self) -> None:
        super().__init__("scene_description_node")

        self.bridge = CvBridge()

        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("vlm_topic", "/dynosam/vlm_result")
        self.declare_parameter("vlm_api_url", "http://42b9e761e7e5:8000/v1")
        self.declare_parameter("lm_studio_url", "")
        self.declare_parameter("model_name", "Qwen/Qwen3-VL-4B-Instruct")
        self.declare_parameter("vlm_disable_thinking", False)
        self.declare_parameter("use_kimi", False)
        self.declare_parameter("kimi_api_url", "https://api.moonshot.ai/v1")
        self.declare_parameter("kimi_api_key", "")
        self.declare_parameter("kimi_model", "kimi-k2.6")
        self.declare_parameter("use_gemini", False)
        self.declare_parameter("gemini_api_key", "")
        self.declare_parameter("gemini_model", "gemini-2.5-pro")
        self.declare_parameter("captions_enabled", True)
        self.declare_parameter("prompt_text", "Describe what is visible in this image in 2 sentences.")
        self.declare_parameter("processing_queue_warn_threshold", 20)
        self.declare_parameter("stream_queue_depth", 1000)

        rgb_topic = self.get_parameter("rgb_topic").value
        vlm_topic = self.get_parameter("vlm_topic").value
        vlm_api_url = self.get_parameter("vlm_api_url").value
        lm_studio_url = self.get_parameter("lm_studio_url").value
        model_name = self.get_parameter("model_name").value
        use_kimi = self.get_parameter("use_kimi").value
        kimi_api_url = self.get_parameter("kimi_api_url").value
        kimi_api_key = self.get_parameter("kimi_api_key").value
        kimi_model = self.get_parameter("kimi_model").value
        use_gemini = self.get_parameter("use_gemini").value
        gemini_api_key = self.get_parameter("gemini_api_key").value
        gemini_model = self.get_parameter("gemini_model").value
        self.captions_enabled = bool(self.get_parameter("captions_enabled").value)
        self.prompt_text = str(self.get_parameter("prompt_text").value)
        self.vlm_disable_thinking = bool(self.get_parameter("vlm_disable_thinking").value)
        self.processing_queue_warn_threshold = int(
            self.get_parameter("processing_queue_warn_threshold").value
        )
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)
        if stream_queue_depth < 1:
            stream_queue_depth = 1

        if lm_studio_url:
            self.get_logger().warn(
                "Parameter 'lm_studio_url' is deprecated, use 'vlm_api_url' instead."
            )
            vlm_api_url = lm_studio_url

        if use_gemini:
            if not gemini_api_key:
                self.get_logger().error("use_gemini is true but gemini_api_key is empty")
            self.model_name = gemini_model
            self.client = OpenAI(base_url="https://generativelanguage.googleapis.com/v1beta/openai/", api_key=gemini_api_key)
            self.get_logger().info("Using Gemini API for scene description")
        elif use_kimi:
            if not kimi_api_key:
                self.get_logger().error("use_kimi is true but kimi_api_key is empty")
            self.model_name = kimi_model
            self.client = OpenAI(base_url=kimi_api_url, api_key=kimi_api_key)
            self.get_logger().info("Using Kimi API for scene description")
        else:
            self.model_name = model_name
            self.client = OpenAI(base_url=vlm_api_url, api_key="lm-studio")

        self._pending_images = deque()
        self._worker_busy = False
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub = self.create_subscription(Image, rgb_topic, self.callback, stream_qos)
        self.pub = self.create_publisher(StampedString, vlm_topic, stream_qos)
        self.worker_timer = self.create_timer(0.01, self._process_next_image)
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"VLM topic: {vlm_topic}")
        self.get_logger().info(f"VLM API URL: {vlm_api_url}")
        self.get_logger().info(f"Model: {model_name}")
        self.get_logger().info(f"Captions enabled: {self.captions_enabled}")
        self.get_logger().info(f"Stream queue depth: {stream_queue_depth}")

    def _on_parameter_update(self, params) -> SetParametersResult:
        for param in params:
            if param.name == "captions_enabled":
                self.captions_enabled = bool(param.value)
                if not self.captions_enabled:
                    self._pending_images.clear()
                self.get_logger().info(f"Captions enabled: {self.captions_enabled}")
        return SetParametersResult(successful=True)

    def callback(self, msg: Image) -> None:
        if not self.captions_enabled:
            return
        self._pending_images.append(msg)
        pending_count = len(self._pending_images)
        if (
            self.processing_queue_warn_threshold > 0
            and pending_count == self.processing_queue_warn_threshold
        ):
            self.get_logger().warning(
                f"Scene description queue has grown to {pending_count} frame(s); "
                "processing continues in publish order."
            )

    def _process_next_image(self) -> None:
        if not self.captions_enabled:
            self._pending_images.clear()
            return
        if self._worker_busy or not self._pending_images:
            return

        msg = self._pending_images.popleft()
        self._worker_busy = True
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed: {exc}")
            self._worker_busy = False
            return

        try:
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
                            {
                                "type": "text",
                                "text": self.prompt_text,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}"
                                },
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

            self.get_logger().info(f"Published VLM text: {response_text}")

        except Exception as exc:
            self.get_logger().error(f"VLM request failed: {exc}")
        finally:
            self._worker_busy = False

def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = SceneDescriptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
