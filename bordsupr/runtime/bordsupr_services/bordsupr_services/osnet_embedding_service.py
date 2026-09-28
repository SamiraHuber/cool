#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from bordsupr_interfaces.srv import GetImageEmbedding

from .osnet_model import osnet_x1_0


class OSNetEmbeddingService(Node):
    def __init__(self) -> None:
        super().__init__("osnet_embedding_service")

        self.declare_parameter("service_name", "/get_osnet_embedding")
        self.declare_parameter("checkpoint_path", "/shared/cluster_testsets/osnet_finetuned_persons_big_2_improved.pth")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("white_balance", True)
        self.declare_parameter("auto_brightness", True)
        self.declare_parameter("normalize", True)

        service_name = str(self.get_parameter("service_name").value)
        checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value))
        device_name = str(self.get_parameter("device").value)
        self.white_balance = bool(self.get_parameter("white_balance").value)
        self.auto_brightness = bool(self.get_parameter("auto_brightness").value)
        self.normalize = bool(self.get_parameter("normalize").value)

        if device_name == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested but unavailable. Falling back to CPU.")
            device_name = "cpu"
        self.device = torch.device(device_name)
        self.bridge = CvBridge()
        self.model = self._load_model(checkpoint_path)
        self.model.eval()
        self.model.to(self.device)

        self.srv = self.create_service(GetImageEmbedding, service_name, self.handle_get_embedding)
        self.get_logger().info(
            "OSNet fine-tuned improved embedding service ready on "
            f"{service_name} using {checkpoint_path}"
        )

    def _load_model(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"OSNet checkpoint not found: {checkpoint_path}")

        model = osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
        model_state = model.state_dict()
        compatible_state = {}
        for key, value in state_dict.items():
            normalized_key = key[7:] if key.startswith("module.") else key
            if normalized_key in model_state and model_state[normalized_key].shape == value.shape:
                compatible_state[normalized_key] = value
        model.load_state_dict(compatible_state, strict=False)
        return model

    def ros_image_to_bgr(self, image_msg: Image) -> np.ndarray:
        cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        if len(cv_img.shape) == 2:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
        elif len(cv_img.shape) == 3:
            enc = image_msg.encoding.lower()
            if enc in ("rgb8", "rgb16"):
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
            elif enc in ("rgba8", "rgba16"):
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGR)
            elif enc in ("bgra8", "bgra16"):
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
        if cv_img.dtype != np.uint8:
            cv_img = self.to_uint8(cv_img)
        return cv_img

    @staticmethod
    def to_uint8(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        if max_v <= min_v:
            return np.zeros_like(img, dtype=np.uint8)
        return ((img - min_v) / (max_v - min_v) * 255.0).clip(0, 255).astype(np.uint8)

    @staticmethod
    def _white_balance_rgb(rgb: np.ndarray) -> np.ndarray:
        arr = rgb.astype(np.float32)
        mean_r = float(arr[:, :, 0].mean())
        mean_g = float(arr[:, :, 1].mean())
        mean_b = float(arr[:, :, 2].mean())
        if mean_g > 0 and mean_r > 0 and mean_b > 0:
            arr[:, :, 0] = np.clip(arr[:, :, 0] * (mean_g / mean_r), 0, 255)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * (mean_g / mean_b), 0, 255)
        return arr.astype(np.uint8)

    @staticmethod
    def _auto_brightness_rgb(rgb: np.ndarray) -> np.ndarray:
        mean_brightness = float(rgb.mean())
        if 0.0 < mean_brightness < 80.0:
            scale = min(2.0, 128.0 / mean_brightness)
            return np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)
        return rgb

    @torch.inference_mode()
    def embed_bgr_image(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.white_balance:
            rgb = self._white_balance_rgb(rgb)
        if self.auto_brightness:
            rgb = self._auto_brightness_rgb(rgb)
        rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        emb = self.model(tensor)[0].detach().float().cpu().numpy().astype(np.float32)
        if self.normalize:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        return emb.astype(np.float32)

    def handle_get_embedding(self, request, response):
        try:
            bgr = self.ros_image_to_bgr(request.image)
            if bgr.size == 0:
                response.success = False
                response.message = "Empty image"
                response.embedding = []
                response.embedding_dim = 0
                return response

            embedding = self.embed_bgr_image(bgr)
            response.embedding = embedding.tolist()
            response.embedding_dim = int(embedding.shape[0])
            response.success = True
            response.message = "ok"
            return response
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.embedding = []
            response.embedding_dim = 0
            self.get_logger().error(f"OSNet embedding request failed: {exc}")
            return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = OSNetEmbeddingService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
