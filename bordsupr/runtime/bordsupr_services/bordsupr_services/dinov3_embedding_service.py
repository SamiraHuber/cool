#!/usr/bin/env python3

from typing import Optional

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from transformers import AutoImageProcessor, AutoModel

from bordsupr_interfaces.srv import GetImageEmbedding


class DinoV3EmbeddingService(Node):
    def __init__(self) -> None:
        super().__init__("dinov3_embedding_service")

        self.declare_parameter("service_name", "/get_dinov3_embedding")
        self.declare_parameter("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
        self.declare_parameter("fallback_model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("normalize", True)
        self.declare_parameter("use_multiview", True)
        self.declare_parameter("center_crop_ratio", 0.82)
        self.declare_parameter("border_suppression_ratio", 0.12)
        self.declare_parameter("min_side_for_extra_views_px", 72)

        service_name = str(self.get_parameter("service_name").value)
        model_name = str(self.get_parameter("model_name").value)
        fallback_model_name = str(self.get_parameter("fallback_model_name").value)
        device_name = str(self.get_parameter("device").value)
        self.normalize = bool(self.get_parameter("normalize").value)
        self.use_multiview = bool(self.get_parameter("use_multiview").value)
        self.center_crop_ratio = float(self.get_parameter("center_crop_ratio").value)
        self.border_suppression_ratio = float(self.get_parameter("border_suppression_ratio").value)
        self.min_side_for_extra_views_px = int(self.get_parameter("min_side_for_extra_views_px").value)

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

        self.processor, self.model, loaded_model_name = self._load_model(model_name, fallback_model_name)
        self.get_logger().info(f"Embedding model loaded: {loaded_model_name}")
        self.model.eval()
        self.model.to(self.device)

        self.srv = self.create_service(
            GetImageEmbedding,
            service_name,
            self.handle_get_embedding,
        )

        self.get_logger().info(f"Service ready on {service_name}")

    def _load_model(self, model_name: str, fallback_model_name: str):
        self.get_logger().info(f"Loading DINOv3 model: {model_name}")
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return processor, model, model_name
        except Exception as exc:
            if fallback_model_name == model_name:
                raise
            self.get_logger().warn(
                f"Failed to load model '{model_name}' ({exc}). Falling back to '{fallback_model_name}'."
            )
            processor = AutoImageProcessor.from_pretrained(fallback_model_name)
            model = AutoModel.from_pretrained(fallback_model_name)
            return processor, model, fallback_model_name

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

    @torch.inference_mode()
    def _embed_rgb_image(self, rgb: np.ndarray) -> np.ndarray:
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        else:
            emb = outputs.last_hidden_state[:, 0, :][0]

        return emb.detach().float().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_bgr_image(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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
    node = DinoV3EmbeddingService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()