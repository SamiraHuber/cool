#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from bordsupr_interfaces.msg import FaceDetection, FaceOutput, YoloOutput


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return vec
    return vec / norm


class FaceDetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("face_detector_node")

        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("face_output_topic", "/dynosam/face_output")
        self.declare_parameter("person_class_id", 0)
        self.declare_parameter("extra_face_candidate_class_ids", [56, 57, 58, 59, 60])
        self.declare_parameter("face_candidate_min_aspect_ratio", 0.5)
        self.declare_parameter("face_candidate_max_aspect_ratio", 2.5)
        self.declare_parameter("min_person_confidence", 0.25)
        self.declare_parameter("max_persons_per_frame", 3)
        self.declare_parameter("min_face_confidence", 0.45)
        self.declare_parameter("min_face_size_px", 24)
        self.declare_parameter("max_faces_per_person", 1)
        self.declare_parameter("face_margin_ratio", 0.18)
        self.declare_parameter("insightface_model", "buffalo_l")
        self.declare_parameter("insightface_device", "cuda")
        self.declare_parameter("insightface_det_size", 640)
        self.declare_parameter("processing_queue_warn_threshold", 20)
        self.declare_parameter("stream_queue_depth", 1000)

        self.yolo_output_topic = str(self.get_parameter("yolo_output_topic").value)
        self.face_output_topic = str(self.get_parameter("face_output_topic").value)
        self.person_class_id = int(self.get_parameter("person_class_id").value)
        self.extra_face_candidate_class_ids = set(
            int(v)
            for v in (self.get_parameter("extra_face_candidate_class_ids").value or [])
        )
        self.face_candidate_min_aspect_ratio = float(
            self.get_parameter("face_candidate_min_aspect_ratio").value
        )
        self.face_candidate_max_aspect_ratio = float(
            self.get_parameter("face_candidate_max_aspect_ratio").value
        )
        self.min_person_confidence = float(self.get_parameter("min_person_confidence").value)
        self.max_persons_per_frame = int(self.get_parameter("max_persons_per_frame").value)
        self.min_face_confidence = float(self.get_parameter("min_face_confidence").value)
        self.min_face_size_px = int(self.get_parameter("min_face_size_px").value)
        self.max_faces_per_person = int(self.get_parameter("max_faces_per_person").value)
        self.face_margin_ratio = float(self.get_parameter("face_margin_ratio").value)
        self.insightface_model = str(self.get_parameter("insightface_model").value)
        self.insightface_requested_device = (
            str(self.get_parameter("insightface_device").value).strip().lower()
        )
        self.insightface_device = self.insightface_requested_device
        self.insightface_det_size = int(self.get_parameter("insightface_det_size").value)
        self.processing_queue_warn_threshold = int(
            self.get_parameter("processing_queue_warn_threshold").value
        )
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)

        if self.max_faces_per_person < 1:
            self.max_faces_per_person = 1
        if self.max_persons_per_frame < 1:
            self.max_persons_per_frame = 1
        if self.min_face_confidence < 0.0:
            self.min_face_confidence = 0.0
        if self.min_face_confidence > 1.0:
            self.min_face_confidence = 1.0
        if self.min_face_size_px < 8:
            self.min_face_size_px = 8
        if self.face_margin_ratio < 0.0:
            self.face_margin_ratio = 0.0
        if self.insightface_det_size < 160:
            self.insightface_det_size = 160
        if stream_queue_depth < 1:
            stream_queue_depth = 1

        self.bridge = CvBridge()
        self._face_aligner = None
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._face_app = self._build_insightface_app()
        self._face_aligner = self._load_face_aligner()

        self.get_logger().info("Face detector backend: insightface")
        self.get_logger().info(f"InsightFace model: {self.insightface_model}")
        self.get_logger().info(f"InsightFace requested device: {self.insightface_requested_device}")
        self.get_logger().info(f"InsightFace device: {self.insightface_device}")
        self.get_logger().info(f"InsightFace det size: {self.insightface_det_size}")
        self.get_logger().info(f"Min face confidence: {self.min_face_confidence}")

        self._pending_yolo_outputs = deque()
        self._worker_busy = False

        self.face_pub = self.create_publisher(FaceOutput, self.face_output_topic, stream_qos)
        self.yolo_sub = self.create_subscription(
            YoloOutput,
            self.yolo_output_topic,
            self.yolo_output_callback,
            stream_qos,
        )
        self.worker_timer = self.create_timer(0.01, self._process_next_yolo_output)

        self.get_logger().info(f"YOLO input topic: {self.yolo_output_topic}")
        self.get_logger().info(f"Face output topic: {self.face_output_topic}")
        self.get_logger().info(f"Max persons per frame: {self.max_persons_per_frame}")
        self.get_logger().info(f"Stream queue depth: {stream_queue_depth}")

    def _build_insightface_app(self):
        try:
            from insightface.app import FaceAnalysis
        except Exception as exc:
            raise RuntimeError(
                f"Failed to import insightface. Install insightface + onnxruntime first ({exc})"
            ) from exc

        providers, ctx_id, runtime_device = self._resolve_insightface_runtime()
        self.insightface_device = runtime_device

        face_app = FaceAnalysis(name=self.insightface_model, providers=providers)
        face_app.prepare(ctx_id=ctx_id, det_size=(self.insightface_det_size, self.insightface_det_size))
        return face_app

    def _resolve_insightface_runtime(self) -> tuple[list[str], int, str]:
        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        runtime_device = "cpu"

        if self.insightface_requested_device != "cuda":
            return providers, ctx_id, runtime_device

        try:
            import onnxruntime as ort

            available_providers = set(ort.get_available_providers())
        except Exception as exc:
            self.get_logger().warning(
                f"Failed to query ONNX Runtime providers ({exc}); using CPUExecutionProvider"
            )
            return providers, ctx_id, runtime_device

        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
            runtime_device = "cuda"
            return providers, ctx_id, runtime_device

        self.get_logger().warning(
            "InsightFace CUDA was requested but CUDAExecutionProvider is unavailable; "
            "falling back to CPUExecutionProvider"
        )
        return providers, ctx_id, runtime_device

    def _load_face_aligner(self):
        try:
            from insightface.utils import face_align
            return face_align
        except Exception:
            return None

    def _extract_bgr(self, ros_image) -> Optional[np.ndarray]:
        try:
            bgr = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")
            if bgr is None or bgr.size == 0:
                return None
            return bgr
        except Exception as exc:
            self.get_logger().warning(f"Failed to decode cropped person image: {exc}")
            return None

    def _detect_faces(self, person_bgr: np.ndarray) -> list[dict]:
        h, w = person_bgr.shape[:2]
        try:
            faces = self._face_app.get(person_bgr)
        except Exception as exc:
            self.get_logger().warning(f"InsightFace inference failed for frame: {exc}")
            return []

        detections: list[dict] = []
        for face in faces:
            bbox = np.asarray(getattr(face, "bbox", []), dtype=np.float32).reshape(-1)
            emb = np.asarray(getattr(face, "embedding", []), dtype=np.float32).reshape(-1)
            if bbox.shape[0] < 4 or emb.size == 0:
                continue

            x1, y1, x2, y2 = self._clip_box(
                int(round(float(bbox[0]))),
                int(round(float(bbox[1]))),
                int(round(float(bbox[2]))),
                int(round(float(bbox[3]))),
                w,
                h,
            )
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            if bw < self.min_face_size_px or bh < self.min_face_size_px:
                continue

            score = float(getattr(face, "det_score", 1.0))
            if score < self.min_face_confidence:
                continue

            kps = np.asarray(getattr(face, "kps", []), dtype=np.float32)
            if kps.size == 0:
                kps = None

            detections.append(
                {
                    "box": (x1, y1, x2, y2),
                    "score": score,
                    "embedding": _l2_normalize(emb.astype(np.float32)),
                    "kps": kps,
                    "area": float(bw * bh),
                }
            )

        detections.sort(
            key=lambda item: (float(item["score"]), float(item["area"])),
            reverse=True,
        )
        return detections

    @staticmethod
    def _clip_box(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))
        return x1, y1, x2, y2

    def _extract_aligned_face(self, person_bgr: np.ndarray, face_det: dict) -> Optional[np.ndarray]:
        h, w = person_bgr.shape[:2]
        x1, y1, x2, y2 = face_det["box"]

        # Keypoint-based alignment yields more stable identity embeddings.
        kps = face_det.get("kps")
        if self._face_aligner is not None and kps is not None:
            try:
                aligned = self._face_aligner.norm_crop(
                    person_bgr,
                    landmark=np.asarray(kps, dtype=np.float32),
                )
                if aligned is not None and aligned.size > 0:
                    return aligned
            except Exception:
                pass

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0
        side = max(bw, bh) * (1.0 + (2.0 * self.face_margin_ratio))
        half = side / 2.0

        sx1 = int(round(cx - half))
        sy1 = int(round(cy - half))
        sx2 = int(round(cx + half))
        sy2 = int(round(cy + half))

        sx1, sy1, sx2, sy2 = self._clip_box(sx1, sy1, sx2, sy2, w, h)

        crop = person_bgr[sy1:sy2, sx1:sx2]
        if crop is None or crop.size == 0:
            return None

        # A fixed-size aligned crop simplifies downstream embedding model expectations.
        aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
        return aligned

    def yolo_output_callback(self, msg: YoloOutput) -> None:
        self._pending_yolo_outputs.append(msg)
        pending_count = len(self._pending_yolo_outputs)
        if (
            self.processing_queue_warn_threshold > 0
            and pending_count == self.processing_queue_warn_threshold
        ):
            self.get_logger().warning(
                f"Face detection queue has grown to {pending_count} frame(s); "
                "processing continues in publish order."
            )

    def _process_next_yolo_output(self) -> None:
        if self._worker_busy or not self._pending_yolo_outputs:
            return

        msg = self._pending_yolo_outputs.popleft()
        self._worker_busy = True
        if not msg.objects:
            self._worker_busy = False
            return

        out = FaceOutput()
        out.header = msg.header

        candidate_people = []
        for det in msg.objects:
            try:
                class_id = int(det.class_id)
                score = float(det.score)
            except Exception:
                continue

            width = max(1, int(det.x_max) - int(det.x_min))
            height = max(1, int(det.y_max) - int(det.y_min))
            aspect_ratio = height / max(1, width)
            is_person_class = class_id == self.person_class_id
            is_likely_person_furniture = (
                class_id in self.extra_face_candidate_class_ids
                and self.face_candidate_min_aspect_ratio <= aspect_ratio <= self.face_candidate_max_aspect_ratio
            )
            if not (is_person_class or is_likely_person_furniture):
                continue
            if score < self.min_person_confidence:
                continue

            area = width * height
            candidate_people.append((area, score, det))

        if not candidate_people:
            self._worker_busy = False
            return

        candidate_people.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_people = [item[2] for item in candidate_people[: self.max_persons_per_frame]]

        for det in selected_people:

            person_bgr = self._extract_bgr(det.cropped_image)
            if person_bgr is None:
                continue

            faces = self._detect_faces(person_bgr)
            if not faces:
                continue

            if self.max_faces_per_person > 0:
                faces = faces[: self.max_faces_per_person]

            for idx, face_det in enumerate(faces):
                fx1, fy1, fx2, fy2 = face_det["box"]
                face_score = float(face_det["score"])
                embedding = np.asarray(face_det["embedding"], dtype=np.float32).reshape(-1)
                if embedding.size == 0:
                    continue

                aligned_face = self._extract_aligned_face(person_bgr, face_det)
                if aligned_face is None:
                    continue

                face_msg = FaceDetection()
                face_msg.scene_id = -1
                face_msg.object_id = ""
                face_msg.track_id = int(getattr(det, "track_id", -1))

                face_msg.person_x_min = int(det.x_min)
                face_msg.person_y_min = int(det.y_min)
                face_msg.person_x_max = int(det.x_max)
                face_msg.person_y_max = int(det.y_max)

                face_msg.face_x_min = int(det.x_min) + int(fx1)
                face_msg.face_y_min = int(det.y_min) + int(fy1)
                face_msg.face_x_max = int(det.x_min) + int(fx2)
                face_msg.face_y_max = int(det.y_min) + int(fy2)

                face_msg.face_score = float(face_score)
                face_msg.embedding_id = (
                    f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}:"
                    f"{face_msg.track_id}:{idx}"
                )
                face_msg.embedding_dim = int(embedding.shape[0])
                face_msg.embedding = embedding.astype(np.float32).tolist()
                face_msg.aligned_face_image = self.bridge.cv2_to_imgmsg(aligned_face, encoding="bgr8")
                face_msg.aligned_face_image.header = msg.header

                out.faces.append(face_msg)

        if out.faces:
            self.face_pub.publish(out)
        self._worker_busy = False


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
