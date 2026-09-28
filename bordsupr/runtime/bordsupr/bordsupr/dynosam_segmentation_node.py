#!/usr/bin/env python3
"""Local object-detection node (YOLO / OWLv2).

DEPRECATION NOTE: Despite the historical name ``dynosam_segmentation_node`` and the
``/dynosam/*`` topic names it publishes to, this node does NOT use the DynoSAM
dynamic-SLAM backend. DynoSAM is no longer part of the active pipeline. This is a
plain local detection node that runs YOLO or OWLv2 (via ``detection_engine``) on the
RGB camera stream and publishes detections to ``/dynosam/yolo_output`` (and a motion
mask to ``/dynosam/segmentation``). The topic/node names are kept only for
compatibility.
"""

import heapq
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from bordsupr_interfaces.msg import YoloDetection, YoloOutput

from .detection_engine import build_detection_engine, DetectionResult


class DynoSAMSegmentationNode(Node):
    def __init__(self) -> None:
        super().__init__("dynosam_segmentation_node")

        self.bridge = CvBridge()
        self.previous_track_ids = set()
        self.track_id_to_instance_id: Dict[int, int] = {}
        self.track_last_seen_frame: Dict[int, int] = {}
        self.frame_index = 0

        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("seg_topic", "/dynosam/segmentation")
        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("model_path", "yolo11s.pt")
        self.declare_parameter("detector_backend", "yolo")
        self.declare_parameter("conf_threshold", 0.4)
        self.declare_parameter("tracker_config", "botsort.yaml")
        self.declare_parameter("track_lost_ttl_frames", 5)
        self.declare_parameter("max_instance_id", 255)
        self.declare_parameter("processing_queue_warn_threshold", 20)
        self.declare_parameter("stream_queue_depth", 1000)
        self.declare_parameter("yolo_imgsz", 320)
        self.declare_parameter("max_queue_drop_older_than", 3)
        self.declare_parameter("owlv2_model_path", "google/owlv2-base-patch16-ensemble")
        self.declare_parameter("owlv2_classes", "")
        self.declare_parameter("owlv2_device", "cuda")
        self.declare_parameter("owlv2_imgsz", 640)
        self.declare_parameter("two_stage_person", False)

        rgb_topic = str(self.get_parameter("rgb_topic").value)
        seg_topic = str(self.get_parameter("seg_topic").value)
        yolo_output_topic = str(self.get_parameter("yolo_output_topic").value)
        model_path = str(self.get_parameter("model_path").value)
        detector_backend = str(self.get_parameter("detector_backend").value)
        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.tracker_config = str(self.get_parameter("tracker_config").value)
        self.track_lost_ttl_frames = int(self.get_parameter("track_lost_ttl_frames").value)
        configured_max_instance_id = int(self.get_parameter("max_instance_id").value)
        self.processing_queue_warn_threshold = int(
            self.get_parameter("processing_queue_warn_threshold").value
        )
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)
        self.yolo_imgsz = int(self.get_parameter("yolo_imgsz").value)
        self.max_queue_drop_older_than = int(self.get_parameter("max_queue_drop_older_than").value)
        owlv2_model_path = str(self.get_parameter("owlv2_model_path").value)
        owlv2_classes_raw = self.get_parameter("owlv2_classes").value
        owlv2_device = str(self.get_parameter("owlv2_device").value)
        owlv2_imgsz = int(self.get_parameter("owlv2_imgsz").value)
        two_stage_person = bool(self.get_parameter("two_stage_person").value)

        self.max_instance_id = min(255, max(1, configured_max_instance_id))
        if stream_queue_depth < 1:
            stream_queue_depth = 1
        self.available_instance_ids = list(range(1, self.max_instance_id + 1))
        heapq.heapify(self.available_instance_ids)
        self.allocated_instance_ids = set()
        self._pending_images = deque()
        self._worker_busy = False
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

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
            imgsz=self.yolo_imgsz,
            owlv2_model_path=owlv2_model_path,
            owlv2_classes=owlv2_classes if owlv2_classes else None,
            owlv2_device=owlv2_device,
            owlv2_imgsz=owlv2_imgsz,
            two_stage_person=two_stage_person,
        )

        self.sub = self.create_subscription(Image, rgb_topic, self.callback, stream_qos)
        self.seg_pub = self.create_publisher(Image, seg_topic, stream_qos)
        self.yolo_pub = self.create_publisher(YoloOutput, yolo_output_topic, stream_qos)
        self.worker_timer = self.create_timer(0.01, self._process_next_image)

        self.get_logger().info(f"RGB topic: {rgb_topic}")
        self.get_logger().info(f"Segmentation topic: {seg_topic}")
        self.get_logger().info(f"YOLO output topic: {yolo_output_topic}")
        self.get_logger().info(f"Detector backend: {detector_backend}")
        self.get_logger().info(f"Model path: {model_path}")
        self.get_logger().info(f"Tracker config: {self.tracker_config}")
        self.get_logger().info(f"Confidence threshold: {self.conf_threshold}")
        self.get_logger().info(f"Track lost TTL frames: {self.track_lost_ttl_frames}")
        self.get_logger().info(f"Max DynoSAM instance ID: {self.max_instance_id}")
        self.get_logger().info(f"Stream queue depth: {stream_queue_depth}")
        self.get_logger().info(f"Two-stage person detection: {two_stage_person}")
        if configured_max_instance_id != self.max_instance_id:
            self.get_logger().warning(
                f"Configured max_instance_id={configured_max_instance_id} adjusted to {self.max_instance_id} "
                "to satisfy DynoSAM's 8-bit object ID limit"
            )

    def _allocate_instance_id(self) -> Optional[int]:
        if not self.available_instance_ids:
            return None
        instance_id = heapq.heappop(self.available_instance_ids)
        self.allocated_instance_ids.add(instance_id)
        return instance_id

    def _release_instance_id(self, instance_id: int) -> None:
        if instance_id not in self.allocated_instance_ids:
            return
        self.allocated_instance_ids.remove(instance_id)
        heapq.heappush(self.available_instance_ids, instance_id)

    def _class_name_for_id(self, class_id: int) -> str:
        return self.engine.class_name_for_id(class_id)

    def _get_instance_id(self, track_id: int) -> Optional[int]:
        if track_id >= 0:
            instance_id = self.track_id_to_instance_id.get(track_id)
            if instance_id is None:
                instance_id = self._allocate_instance_id()
                if instance_id is None:
                    return None
                self.track_id_to_instance_id[track_id] = instance_id
            self.track_last_seen_frame[track_id] = self.frame_index
            return instance_id

        return self._allocate_instance_id()

    def _prune_stale_tracks(self) -> None:
        stale_track_ids = [
            track_id
            for track_id, last_seen_frame in self.track_last_seen_frame.items()
            if (self.frame_index - last_seen_frame) > self.track_lost_ttl_frames
        ]
        for track_id in stale_track_ids:
            self.track_last_seen_frame.pop(track_id, None)
            instance_id = self.track_id_to_instance_id.pop(track_id, None)
            if instance_id is not None:
                self._release_instance_id(instance_id)

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
        # Continuously drop oldest frames so the queue never exceeds the threshold.
        # This keeps latency bounded when inference is slower than the frame rate.
        if self.max_queue_drop_older_than > 0:
            while len(self._pending_images) > self.max_queue_drop_older_than:
                self._pending_images.popleft()
        if (
            self.processing_queue_warn_threshold > 0
            and pending_count == self.processing_queue_warn_threshold
        ):
            self.get_logger().warning(
                f"Segmentation queue has grown to {pending_count} frame(s); "
                "processing continues with latest frame(s)."
            )

    def _process_next_image(self) -> None:
        if self._worker_busy or not self._pending_images:
            return

        # Always process the most recent frame to minimize latency.
        while len(self._pending_images) > 1:
            self._pending_images.popleft()
        msg = self._pending_images.pop()
        self._worker_busy = True
        self.frame_index += 1
        self._prune_stale_tracks()

        # Latency instrumentation: frame age at pickup (capture -> YOLO start).
        _capture_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        _t_start = time.perf_counter()
        _frame_age_at_pickup = time.time() - _capture_ts if _capture_ts > 0 else -1.0

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed: {exc}")
            self._worker_busy = False
            return

        try:
            _t_inf = time.perf_counter()
            det_result = self.engine.detect_and_track(bgr)
            _inference_ms = (time.perf_counter() - _t_inf) * 1000.0
        except Exception as exc:
            self.get_logger().error(f"Detection/tracking failed: {exc}")
            self._worker_busy = False
            return

        h, w = bgr.shape[:2]
        labels = np.zeros((h, w), dtype=np.int32)
        yolo_output = YoloOutput()
        yolo_output.header = msg.header

        current_track_ids = set()
        temporary_instance_ids = set()

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
                instance_id = self._get_instance_id(track_id)
                if instance_id is None:
                    self.get_logger().warning(
                        f"Skipping detection track_id={track_id} class={class_name}: "
                        f"no free DynoSAM instance IDs remain in 1..{self.max_instance_id}"
                    )
                    continue

                if track_id >= 0:
                    current_track_ids.add(track_id)
                else:
                    temporary_instance_ids.add(instance_id)

                crop = bgr[y1:y2, x1:x2].copy()
                mask_bin = self._extract_detection_mask(masks, i, (h, w), (x1, y1, x2, y2))
                if mask_bin is None:
                    mask_bin = np.zeros((h, w), dtype=bool)
                    mask_bin[y1:y2, x1:x2] = True

                labels[mask_bin] = instance_id
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

                det.mask_image = self.bridge.cv2_to_imgmsg(crop_mask, encoding="mono8")
                det.mask_image.header = msg.header

                yolo_output.objects.append(det)
        else:
            self.get_logger().info("No detections this frame")

        for instance_id in temporary_instance_ids:
            self._release_instance_id(instance_id)

        unique_vals = np.unique(labels)
        self.get_logger().info(
            f"Label values present: {unique_vals[:10]} (total unique={len(unique_vals)})"
        )

        seg_msg = self.bridge.cv2_to_imgmsg(labels, encoding="32SC1")
        seg_msg.header = msg.header
        yolo_output.header = msg.header

        self.seg_pub.publish(seg_msg)
        self.yolo_pub.publish(yolo_output)

        # Latency instrumentation: total capture -> YOLO-output-published.
        _total_ms = (time.perf_counter() - _t_start) * 1000.0
        _e2e_age_ms = (time.time() - _capture_ts) * 1000.0 if _capture_ts > 0 else -1.0
        self.get_logger().info(
            f"[latency] yolo: inference={_inference_ms:.0f}ms "
            f"proc_total={_total_ms:.0f}ms frame_age_at_pickup={_frame_age_at_pickup*1000.0:.0f}ms "
            f"capture_to_pub={_e2e_age_ms:.0f}ms objects={len(yolo_output.objects)}"
        )

        self.previous_track_ids = current_track_ids
        self._worker_busy = False


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = DynoSAMSegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
