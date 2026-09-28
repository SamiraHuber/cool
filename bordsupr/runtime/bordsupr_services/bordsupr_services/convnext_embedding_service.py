#!/usr/bin/env python3

from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from bordsupr_interfaces.srv import GetImageEmbedding


class ConvNeXtEmbeddingService(Node):
    def __init__(self) -> None:
        super().__init__("convnext_embedding_service")

        self.declare_parameter("service_name", "/get_convnext_embedding")
        self.declare_parameter("model_name", "convnext_tiny.fb_in1k")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("normalize", True)
        self.declare_parameter("use_multiview", False)
        self.declare_parameter("center_crop_ratio", 0.82)
        self.declare_parameter("border_suppression_ratio", 0.12)
        self.declare_parameter("min_side_for_extra_views_px", 72)
        self.declare_parameter("white_balance", True)
        self.declare_parameter("auto_brightness", True)

        service_name = str(self.get_parameter("service_name").value)
        model_name = str(self.get_parameter("model_name").value)
        device_name = str(self.get_parameter("device").value)
        self.normalize = bool(self.get_parameter("normalize").value)
        self.use_multiview = bool(self.get_parameter("use_multiview").value)
        self.center_crop_ratio = float(self.get_parameter("center_crop_ratio").value)
        self.border_suppression_ratio = float(self.get_parameter("border_suppression_ratio").value)
        self.min_side_for_extra_views_px = int(self.get_parameter("min_side_for_extra_views_px").value)
        self.white_balance = bool(self.get_parameter("white_balance").value)
        self.auto_brightness = bool(self.get_parameter("auto_brightness").value)

        if self.center_crop_ratio <= 0.0 or self.center_crop_ratio > 1.0:
            self.get_logger().warn("center_crop_ratio must be in (0, 1]; using default 0.82")
            self.center_crop_ratio = 0.82
        if self.border_suppression_ratio < 0.0 or self.border_suppression_ratio >= 0.45:
            self.get_logger().warn("border_suppression_ratio must be in [0, 0.45); using default 0.12")
            self.border_suppression_ratio = 0.12
        if self.min_side_for_extra_views_px < 16:
            self.min_side_for_extra_views_px = 16

        if device_name == "cuda" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested but unavailable. Falling back to CPU.")
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.bridge = CvBridge()

        self.model = self._load_model(model_name)
        self.get_logger().info(f"Embedding model loaded: {model_name}")
        self.model.eval()
        self.model.to(self.device)

        self.srv = self.create_service(
            GetImageEmbedding,
            service_name,
            self.handle_get_embedding,
        )

        self.get_logger().info(f"Service ready on {service_name}")

    def _load_model(self, model_name: str):
        self.get_logger().info(f"Loading ConvNeXt model: {model_name}")
        import timm
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
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

    def to_uint8(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        if max_v <= min_v:
            return np.zeros_like(img, dtype=np.uint8)
        img = (img - min_v) / (max_v - min_v)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def _build_views(self, rgb: np.ndarray) -> list[np.ndarray]:
        views: list[np.ndarray] = [rgb]
        if not self.use_multiview:
            return views

        height, width = rgb.shape[:2]
        if min(height, width) < self.min_side_for_extra_views_px:
            return views

        crop_h = int(height * self.center_crop_ratio)
        crop_w = int(width * self.center_crop_ratio)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        y1 = min(height, y0 + crop_h)
        x1 = min(width, x0 + crop_w)
        center_crop = rgb[y0:y1, x0:x1]
        if center_crop.size > 0:
            views.append(center_crop)

        border_h = int(round(height * self.border_suppression_ratio))
        border_w = int(round(width * self.border_suppression_ratio))
        if border_h > 0 or border_w > 0:
            blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=6.0, sigmaY=6.0)
            focused = rgb.copy()
            if border_h > 0:
                focused[:border_h, :, :] = blurred[:border_h, :, :]
                focused[height - border_h:, :, :] = blurred[height - border_h:, :, :]
            if border_w > 0:
                focused[:, :border_w, :] = blurred[:, :border_w, :]
                focused[:, width - border_w:, :] = blurred[:, width - border_w:, :]
            views.append(focused)

        return views

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        """Resize to 224x224 and apply ImageNet normalization."""
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        arr = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)

    @torch.inference_mode()
    def _embed_rgb_image(self, rgb: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(rgb)
        tensor = tensor.to(self.device)
        features = self.model.forward_features(tensor)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            features = features[:, 0]
        emb = features[0].detach().float().cpu().numpy().astype(np.float32)
        return emb

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

    def embed_bgr_image(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.white_balance:
            rgb = self._white_balance_rgb(rgb)
        if self.auto_brightness:
            rgb = self._auto_brightness_rgb(rgb)
        views = self._build_views(rgb)
        vectors = [self._embed_rgb_image(view) for view in views]
        emb = np.mean(np.stack(vectors, axis=0), axis=0)

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
            self.get_logger().info(f"Successfully generated embedding of dim {response.embedding_dim}")
            return response

        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.embedding = []
            response.embedding_dim = 0
            self.get_logger().error(f"Embedding request failed: {exc}")
            return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = ConvNeXtEmbeddingService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
