#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from bordsupr_interfaces.srv import GetImageEmbedding


class FaceEmbeddingService(Node):
    def __init__(self) -> None:
        super().__init__("face_embedding_service")

        self.declare_parameter("service_name", "/get_face_embedding")
        self.declare_parameter("backend", "insightface")
        self.declare_parameter("insightface_model", "buffalo_l")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("normalize", True)
        self.declare_parameter("embedding_dim", 512)
        self.declare_parameter("fallback_on_failure", True)

        service_name = str(self.get_parameter("service_name").value)
        self.backend = str(self.get_parameter("backend").value).strip().lower()
        self.insightface_model = str(self.get_parameter("insightface_model").value)
        self.device = str(self.get_parameter("device").value).strip().lower()
        self.normalize = bool(self.get_parameter("normalize").value)
        self.embedding_dim = int(self.get_parameter("embedding_dim").value)
        self.fallback_on_failure = bool(self.get_parameter("fallback_on_failure").value)

        if self.embedding_dim <= 0:
            self.embedding_dim = 512

        self.bridge = CvBridge()
        self._face_app = None
        self._backend_active = "fallback"

        self._initialize_backend()

        self.srv = self.create_service(
            GetImageEmbedding,
            service_name,
            self.handle_get_embedding,
        )

        self.get_logger().info(f"Face embedding backend: {self._backend_active}")
        self.get_logger().info(f"Service ready on {service_name}")

    def _initialize_backend(self) -> None:
        if self.backend != "insightface":
            self._backend_active = "fallback"
            return

        try:
            from insightface.app import FaceAnalysis

            providers = ["CPUExecutionProvider"]
            ctx_id = -1
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0

            self._face_app = FaceAnalysis(name=self.insightface_model, providers=providers)
            self._face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._backend_active = "insightface"
        except Exception as exc:
            self._face_app = None
            self._backend_active = "fallback"
            self.get_logger().warning(
                f"Could not initialize insightface backend ({exc}). Falling back to deterministic embedding."
            )

    def _ros_image_to_bgr(self, image_msg: Image) -> np.ndarray:
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
        if image is None or image.size == 0:
            raise ValueError("Empty image")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3:
            enc = image_msg.encoding.lower()
            if enc in ("rgb8", "rgb16"):
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif enc in ("rgba8", "rgba16"):
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif enc in ("bgra8", "bgra16"):
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.dtype != np.uint8:
            image = self._to_uint8(image)

        return image

    @staticmethod
    def _to_uint8(img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        min_v = float(np.min(img))
        max_v = float(np.max(img))
        if max_v <= min_v:
            return np.zeros_like(img, dtype=np.uint8)
        img = (img - min_v) / (max_v - min_v)
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return img

    def _embed_with_insightface(self, bgr: np.ndarray) -> np.ndarray:
        if self._face_app is None:
            raise RuntimeError("insightface backend is not initialized")

        faces = self._face_app.get(bgr)
        if not faces:
            raise ValueError("No face found in image")

        best = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
        )
        emb = np.asarray(best.embedding, dtype=np.float32)
        return emb

    def _embed_fallback(self, bgr: np.ndarray) -> np.ndarray:
        # Lightweight deterministic fallback so the pipeline keeps functioning
        # when ArcFace runtime dependencies are unavailable.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 16), interpolation=cv2.INTER_AREA)
        emb = resized.astype(np.float32).reshape(-1)
        emb -= float(np.mean(emb))
        return emb

    def _coerce_dim(self, emb: np.ndarray) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        current = int(emb.shape[0])
        if current == self.embedding_dim:
            out = emb
        elif current > self.embedding_dim:
            out = emb[: self.embedding_dim]
        else:
            out = np.concatenate(
                [emb, np.zeros((self.embedding_dim - current,), dtype=np.float32)],
                axis=0,
            )

        if self.normalize:
            norm = float(np.linalg.norm(out))
            if norm > 0.0:
                out = out / norm

        return out.astype(np.float32)

    def handle_get_embedding(self, request, response):
        try:
            bgr = self._ros_image_to_bgr(request.image)

            if self._backend_active == "insightface":
                try:
                    emb = self._embed_with_insightface(bgr)
                except Exception as exc:
                    if not self.fallback_on_failure:
                        raise
                    self.get_logger().warning(
                        f"insightface inference failed ({exc}); using fallback embedding"
                    )
                    emb = self._embed_fallback(bgr)
            else:
                emb = self._embed_fallback(bgr)

            emb = self._coerce_dim(emb)
            response.embedding = emb.tolist()
            response.embedding_dim = int(emb.shape[0])
            response.success = True
            response.message = "ok"
            return response

        except Exception as exc:
            response.embedding = []
            response.embedding_dim = 0
            response.success = False
            response.message = str(exc)
            self.get_logger().error(f"Face embedding request failed: {exc}")
            return response


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = FaceEmbeddingService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
