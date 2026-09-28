#!/usr/bin/env python3

from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from bordsupr_interfaces.msg import YoloDetection, YoloOutput

from .detection_engine import build_detection_engine, DetectionResult


class YoloSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_segmentation_node")

        self.bridge = CvBridge()
        self.previous_track_ids = set()
        self.frame_index = 0

        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("corrected_rgb_topic", "/spot/camera/frontleft/image_rotated_corrected")
        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("model_path", "yolo11s.pt")
        self.declare_parameter("detector_backend", "yolo")
        self.declare_parameter("conf_threshold", 0.25)
        self.declare_parameter("tracker_config", "botsort.yaml")
        self.declare_parameter("stream_queue_depth", 1000)
        self.declare_parameter("processing_queue_warn_threshold", 20)
        self.declare_parameter("brightness_correction", "clahe")
        self.declare_parameter("brightness_target", 100.0)
        self.declare_parameter("owlv2_model_path", "google/owlv2-base-patch16-ensemble")
        self.declare_parameter("owlv2_classes", [])
        self.declare_parameter("owlv2_device", "cuda")
        self.declare_parameter("owlv2_imgsz", 640)
        self.declare_parameter("two_stage_person", False)

        rgb_topic = str(self.get_parameter("rgb_topic").value)
        corrected_rgb_topic = str(self.get_parameter("corrected_rgb_topic").value)
        yolo_output_topic = str(self.get_parameter("yolo_output_topic").value)
        model_path = str(self.get_parameter("model_path").value)
        detector_backend = str(self.get_parameter("detector_backend").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.tracker_config = str(self.get_parameter("tracker_config").value)
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)
        self.processing_queue_warn_threshold = int(
            self.get_parameter("processing_queue_warn_threshold").value
        )
        self.brightness_correction = str(self.get_parameter("brightness_correction").value).strip().lower()
        self.brightness_target = float(self.get_parameter("brightness_target").value)
        owlv2_model_path = str(self.get_parameter("owlv2_model_path").value)
        owlv2_classes_raw = self.get_parameter("owlv2_classes").value
        owlv2_device = str(self.get_parameter("owlv2_device").value)
        owlv2_imgsz = int(self.get_parameter("owlv2_imgsz").value)
        two_stage_person = bool(self.get_parameter("two_stage_person").value)

        if stream_queue_depth < 1:
            stream_queue_depth = 1

        # Parse owlv2_classes: ROS parameter may return a plain string or a list
        owlv2_classes: Optional[List[str]] = None
        if owlv2_classes_raw:
            if isinstance(owlv2_classes_raw, str):
                owlv2_classes = [c.strip() for c in owlv2_classes_raw.split(",") if c.strip()]
            elif isinstance(owlv2_classes_raw, (list, tuple)):
                owlv2_classes = [str(c).strip() for c in owlv2_classes_raw if str(c).strip()]

        self.engine = build_detection_engine(
            backend=detector_backend,
            model_path=model_path,
            conf_threshold=self.conf_threshold,
            tracker_config=self.tracker_config,
            imgsz=None,
            owlv2_model_path=owlv2_model_path,
            owlv2_classes=owlv2_classes if owlv2_classes else None,
            owlv2_device=owlv2_device,
            owlv2_imgsz=owlv2_imgsz,
            two_stage_person=two_stage_person,
        )

        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub = self.create_subscription(Image, rgb_topic, self.callback, stream_qos)
        self.corrected_rgb_pub = self.create_publisher(Image, corrected_rgb_topic, stream_qos)
        self.yolo_pub = self.create_publisher(YoloOutput, yolo_output_topic, stream_qos)
        self.worker_timer = self.create_timer(0.01, self._process_next_image)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"Corrected RGB topic: {corrected_rgb_topic}")
        self.get_logger().info(f"YOLO output topic: {yolo_output_topic}")
        self.get_logger().info(f"Detector backend: {detector_backend}")
        self.get_logger().info(f"Model path: {model_path}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"Tracker config: {self.tracker_config}")
        self.get_logger().info(f"Brightness correction: {self.brightness_correction}, target: {self.brightness_target}")
        self.get_logger().info(f"Two-stage person detection: {two_stage_person}")

        self._pending_images = deque()
        self._worker_busy = False

    def _class_name_for_id(self, class_id: int) -> str:
        return self.engine.class_name_for_id(class_id)

    def _correct_brightness(self, bgr: np.ndarray) -> np.ndarray:
        if self.brightness_correction == "none":
            return bgr
        if self.brightness_correction == "clahe":
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = float(np.mean(l))
            if mean_l < self.brightness_target:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if self.brightness_correction == "gamma":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mean_val = float(np.mean(gray))
            if mean_val >= self.brightness_target:
                return bgr
            gamma = np.log(mean_val / 255.0) / np.log(self.brightness_target / 255.0)
            if gamma <= 0 or not np.isfinite(gamma):
                return bgr
            inv_gamma = 1.0 / gamma
            lookup = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            return cv2.LUT(bgr, lookup)
        return bgr

    def _extract_detection_mask(
        self,
        masks: Optional[np.ndarray],
        index: int,
        image_shape: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        if masks is None or index >= len(masks):
            return None

        h, w = image_shape
        x1, y1, x2, y2 = bbox
        mask = masks[index]

        if mask.ndim != 2:
            return None

        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        mask_bin = mask > 0.5
        if np.any(mask_bin):
            return mask_bin

        bbox_mask = np.zeros((h, w), dtype=bool)
        bbox_mask[y1:y2, x1:x2] = True
        return bbox_mask

    def callback(self, msg: Image) -> None:
        self._pending_images.append(msg)
        pending_count = len(self._pending_images)
        if (
            self.processing_queue_warn_threshold > 0
            and pending_count == self.processing_queue_warn_threshold
        ):
            self.get_logger().warning(
                f"Segmentation queue has grown to {pending_count} frame(s); "
                "processing continues in publish order."
            )

    def _process_next_image(self) -> None:
        if self._worker_busy or not self._pending_images:
            return

        # Drop stale frames to stay near real-time (critical for slow backends
        # like OWLv2 and for depth timestamp synchronization).
        while len(self._pending_images) > 1:
            self._pending_images.popleft()

        msg = self._pending_images.popleft()
        self._worker_busy = True
        self.frame_index += 1

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed: {exc}")
            self._worker_busy = False
            return

        bgr_original = bgr
        bgr = self._correct_brightness(bgr)
        if self.brightness_correction != "none" and not np.array_equal(bgr, bgr_original):
            mean_before = float(np.mean(cv2.cvtColor(bgr_original, cv2.COLOR_BGR2GRAY)))
            mean_after = float(np.mean(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)))
            self.get_logger().info(
                f"Brightness correction ({self.brightness_correction}): "
                f"mean_before={mean_before:.1f}, mean_after={mean_after:.1f}"
            )

        # Publish brightness-corrected full frame for scene rendering
        corrected_msg = self.bridge.cv2_to_imgmsg(bgr, encoding="bgr8")
        corrected_msg.header = msg.header
        self.corrected_rgb_pub.publish(corrected_msg)

        try:
            det_result = self.engine.detect_and_track(bgr)
        except Exception as exc:
            self.get_logger().error(f"Detection/tracking failed: {exc}")
            self._worker_busy = False
            return

        h, w = bgr.shape[:2]

        yolo_output = YoloOutput()
        yolo_output.header = msg.header

        current_track_ids = set()

        if det_result is not None and det_result.num_detections > 0:
            self.get_logger().info(
                f"Detections: boxes={det_result.num_detections}, masks={det_result.masks is not None}"
            )

            xyxy = det_result.xyxy.astype(int)
            scores = det_result.scores
            class_ids = det_result.class_ids
            track_ids = det_result.track_ids
            masks = det_result.masks

            detection_indices = sorted(range(len(xyxy)), key=lambda idx: float(scores[idx]))

            for i in detection_indices:
                if scores[i] < self.conf_threshold:
                    continue

                x1, y1, x2, y2 = xyxy[i]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                if x2 <= x1 or y2 <= y1:
                    continue

                track_id = int(track_ids[i])
                class_id = int(class_ids[i])
                class_name = self._class_name_for_id(class_id)
                detected_in_previous_frame = track_id >= 0 and track_id in self.previous_track_ids
                instance_id = track_id if track_id >= 0 else (i + 1)

                if track_id >= 0:
                    current_track_ids.add(track_id)

                crop = bgr[y1:y2, x1:x2].copy()
                mask_bin = self._extract_detection_mask(masks, i, (h, w), (x1, y1, x2, y2))
                if mask_bin is None:
                    mask_bin = np.zeros((h, w), dtype=bool)
                    mask_bin[y1:y2, x1:x2] = True

                crop_mask = (mask_bin[y1:y2, x1:x2].astype(np.uint8) * 255)

                det = YoloDetection()
                det.header = msg.header
                det.instance_id = instance_id
                det.track_id = track_id
                det.class_id = class_id
                det.class_name = class_name
                det.score = float(scores[i])
                det.detected_in_previous_frame = detected_in_previous_frame
                det.x_min = int(x1)
                det.y_min = int(y1)
                det.x_max = int(x2)
                det.y_max = int(y2)

                det.cropped_image = self.bridge.cv2_to_imgmsg(crop, encoding="bgr8")
                det.cropped_image.header = msg.header

                # Extract original crop from uncorrected frame for debugging
                if bgr_original is not None:
                    orig_crop = bgr_original[y1:y2, x1:x2]
                    det.original_cropped_image = self.bridge.cv2_to_imgmsg(orig_crop, encoding="bgr8")
                    det.original_cropped_image.header = msg.header

                det.mask_image = self.bridge.cv2_to_imgmsg(crop_mask, encoding="mono8")
                det.mask_image.header = msg.header

                yolo_output.objects.append(det)
        else:
            self.get_logger().info("No detections this frame")

        yolo_output.header = msg.header
        self.yolo_pub.publish(yolo_output)

        self.previous_track_ids = current_track_ids
        self._worker_busy = False


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = YoloSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
