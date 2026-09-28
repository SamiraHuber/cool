#!/usr/bin/env python3

from __future__ import annotations

import base64
from collections import OrderedDict, deque
import json
import re
from typing import Dict, Optional, Tuple

import cv2
from cv_bridge import CvBridge
import numpy as np
from openai import OpenAI

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from bordsupr_interfaces.msg import (
    InteractionEvent,
    InteractionObjectRef,
    InteractionOutput,
    YoloDetection,
    YoloOutput,
)


class InteractionDescriptionNode(Node):
    def __init__(self) -> None:
        super().__init__("interaction_description_node")

        self.bridge = CvBridge()
        self._image_cache: "OrderedDict[Tuple[int, int], Image]" = OrderedDict()

        self.declare_parameter("rgb_topic", "/spot/camera/frontleft/image_rotated")
        self.declare_parameter("yolo_output_topic", "/dynosam/yolo_output")
        self.declare_parameter("interaction_topic", "/dynosam/interaction_output")
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
        self.declare_parameter("interaction_cache_ttl_sec", 120.0)
        self.declare_parameter("interaction_cache_max_images", 4096)
        self.declare_parameter("interaction_min_confidence", 0.4)
        self.declare_parameter("interaction_pregate_enabled", True)
        self.declare_parameter("interaction_pregate_max_center_dist_ratio", 0.6)
        self.declare_parameter("interaction_pregate_min_iou", 0.01)
        self.declare_parameter("processing_queue_warn_threshold", 20)
        self.declare_parameter("stream_queue_depth", 1000)

        self.rgb_topic = str(self.get_parameter("rgb_topic").value)
        self.yolo_output_topic = str(self.get_parameter("yolo_output_topic").value)
        self.interaction_topic = str(self.get_parameter("interaction_topic").value)
        vlm_api_url = str(self.get_parameter("vlm_api_url").value)
        lm_studio_url = str(self.get_parameter("lm_studio_url").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.vlm_disable_thinking = bool(self.get_parameter("vlm_disable_thinking").value)
        use_kimi = self.get_parameter("use_kimi").value
        kimi_api_url = str(self.get_parameter("kimi_api_url").value)
        kimi_api_key = str(self.get_parameter("kimi_api_key").value)
        kimi_model = str(self.get_parameter("kimi_model").value)
        use_gemini = self.get_parameter("use_gemini").value
        gemini_api_key = str(self.get_parameter("gemini_api_key").value)
        gemini_model = str(self.get_parameter("gemini_model").value)
        self.cache_ttl_sec = float(self.get_parameter("interaction_cache_ttl_sec").value)
        self.cache_max_images = int(self.get_parameter("interaction_cache_max_images").value)
        self.min_confidence = float(self.get_parameter("interaction_min_confidence").value)
        self.pregate_enabled = bool(self.get_parameter("interaction_pregate_enabled").value)
        self.pregate_max_center_dist_ratio = float(
            self.get_parameter("interaction_pregate_max_center_dist_ratio").value
        )
        self.pregate_min_iou = float(self.get_parameter("interaction_pregate_min_iou").value)
        self.processing_queue_warn_threshold = int(
            self.get_parameter("processing_queue_warn_threshold").value
        )
        stream_queue_depth = int(self.get_parameter("stream_queue_depth").value)
        if self.cache_max_images < 1:
            self.cache_max_images = 1
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
            self.get_logger().info("Using Gemini API for interaction description")
        elif use_kimi:
            if not kimi_api_key:
                self.get_logger().error("use_kimi is true but kimi_api_key is empty")
            self.model_name = kimi_model
            self.client = OpenAI(base_url=kimi_api_url, api_key=kimi_api_key)
            self.get_logger().info("Using Kimi API for interaction description")
        else:
            self.client = OpenAI(base_url=vlm_api_url, api_key="lm-studio")

        self._pending_yolo_outputs = deque()
        self._worker_busy = False
        stream_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=stream_queue_depth,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.image_sub = self.create_subscription(Image, self.rgb_topic, self.image_callback, stream_qos)
        self.yolo_sub = self.create_subscription(YoloOutput, self.yolo_output_topic, self.yolo_callback, stream_qos)
        self.pub = self.create_publisher(InteractionOutput, self.interaction_topic, stream_qos)
        self.worker_timer = self.create_timer(0.01, self._process_next_yolo_output)

        self.get_logger().info(f"RGB topic: {self.rgb_topic}")
        self.get_logger().info(f"YOLO output topic: {self.yolo_output_topic}")
        self.get_logger().info(f"Interaction topic: {self.interaction_topic}")
        self.get_logger().info(f"VLM API URL: {vlm_api_url}")
        self.get_logger().info(f"Model: {self.model_name}")
        self.get_logger().info(f"Interaction cache ttl sec: {self.cache_ttl_sec}")
        self.get_logger().info(f"Interaction cache max images: {self.cache_max_images}")
        self.get_logger().info(f"Stream queue depth: {stream_queue_depth}")

    @staticmethod
    def _stamp_key(msg) -> Tuple[int, int]:
        stamp = msg.header.stamp
        return int(stamp.sec), int(stamp.nanosec)

    @staticmethod
    def _stamp_to_sec(stamp_key: Tuple[int, int]) -> float:
        return float(stamp_key[0]) + (float(stamp_key[1]) / 1e9)

    @staticmethod
    def _normalize_label_id(det: YoloDetection) -> str:
        track_id = int(det.track_id)
        if track_id >= 0:
            return str(track_id)
        return str(int(det.instance_id))

    @staticmethod
    def _normalize_response_id(raw_id: str) -> str:
        value = str(raw_id or "").strip()
        if not value:
            return ""
        match = re.search(r"(\d+)", value)
        if match:
            return match.group(1)
        return value

    @staticmethod
    def _parse_confidence(raw) -> float:
        """Parse VLM self-reported confidence, clamped to [0,1]; 0.0 on any failure."""
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if value != value:  # NaN
            return 0.0
        return max(0.0, min(1.0, value))

    def image_callback(self, msg: Image) -> None:
        stamp_key = self._stamp_key(msg)
        self._image_cache[stamp_key] = msg
        self._image_cache.move_to_end(stamp_key)
        self._prune_image_cache(stamp_key)

    def _prune_image_cache(self, current_stamp: Tuple[int, int]) -> None:
        now_sec = self._stamp_to_sec(current_stamp)
        stale_keys = [
            stamp_key
            for stamp_key in self._image_cache
            if (now_sec - self._stamp_to_sec(stamp_key)) > self.cache_ttl_sec
        ]
        for stamp_key in stale_keys:
            self._image_cache.pop(stamp_key, None)
        while len(self._image_cache) > self.cache_max_images:
            self._image_cache.popitem(last=False)

    def _find_matching_image(self, msg: YoloOutput) -> Optional[Image]:
        # Exact timestamp match ONLY. The YoloOutput's detections/crops come from the camera
        # frame with this exact stamp. If we fall back to a *neighboring* frame (the old
        # 0.35s nearest-frame fallback), we would draw frame N's bounding boxes on frame M's
        # pixels and send that to the VLM — producing a caption about frame M that is then
        # stamped/stored as frame N (the interaction caption/image desync bug). At 5Hz a
        # 0.35s window spans 1-2 frames, so this happened frequently under fast input.
        # Returning None here causes _process_next_yolo_output to skip this frame cleanly.
        target_key = self._stamp_key(msg)
        exact = self._image_cache.get(target_key)
        if exact is not None:
            return exact

        self.get_logger().debug(
            f"No exact RGB frame in cache for interaction at "
            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}; skipping (refusing to "
            f"caption a neighboring frame)."
        )
        return None

    def yolo_callback(self, msg: YoloOutput) -> None:
        self._pending_yolo_outputs.append(msg)
        pending_count = len(self._pending_yolo_outputs)
        if (
            self.processing_queue_warn_threshold > 0
            and pending_count == self.processing_queue_warn_threshold
        ):
            self.get_logger().warning(
                f"Interaction queue has grown to {pending_count} frame(s); "
                "processing continues in publish order."
            )

    @staticmethod
    def _is_bbox_in_image(det: YoloDetection, frame_shape: tuple) -> bool:
        h, w = frame_shape[:2]
        return (
            0 <= int(det.x_min) < int(det.x_max) <= w
            and 0 <= int(det.y_min) < int(det.y_max) <= h
        )

    @staticmethod
    def _bbox_iou(a: YoloDetection, b: YoloDetection) -> float:
        ix1 = max(int(a.x_min), int(b.x_min))
        iy1 = max(int(a.y_min), int(b.y_min))
        ix2 = min(int(a.x_max), int(b.x_max))
        iy2 = min(int(a.y_max), int(b.y_max))
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (int(a.x_max) - int(a.x_min)) * (int(a.y_max) - int(a.y_min)))
        area_b = max(1, (int(b.x_max) - int(b.x_min)) * (int(b.y_max) - int(b.y_min)))
        return inter / float(area_a + area_b - inter)

    @staticmethod
    def _center_distance_ratio(person: YoloDetection, obj: YoloDetection) -> float:
        """Center-to-center distance normalized by the person bbox diagonal."""
        pcx = (int(person.x_min) + int(person.x_max)) / 2.0
        pcy = (int(person.y_min) + int(person.y_max)) / 2.0
        ocx = (int(obj.x_min) + int(obj.x_max)) / 2.0
        ocy = (int(obj.y_min) + int(obj.y_max)) / 2.0
        dist = ((pcx - ocx) ** 2 + (pcy - ocy) ** 2) ** 0.5
        pw = max(1, int(person.x_max) - int(person.x_min))
        ph = max(1, int(person.y_max) - int(person.y_min))
        diag = (pw ** 2 + ph ** 2) ** 0.5
        return dist / diag

    def _passes_geometric_pregate(self, detections: list[YoloDetection]) -> bool:
        """Loose gate: only proceed to the VLM if some person is plausibly near an object.

        Person-person frames always pass (talking/interaction between people is never
        gated out). Returns True if the VLM call should proceed, False to skip.
        """
        if not self.pregate_enabled:
            return True
        persons = [d for d in detections if int(d.class_id) == 0]
        objects = [d for d in detections if int(d.class_id) != 0]
        if len(persons) >= 2:
            return True  # person-person interaction possible; never gate
        if not persons or not objects:
            return False  # single person, nothing else to interact with
        for person in persons:
            for obj in objects:
                if self._bbox_iou(person, obj) >= self.pregate_min_iou:
                    return True
                if self._center_distance_ratio(person, obj) <= self.pregate_max_center_dist_ratio:
                    return True
        return False

    def _validate_interaction_refs(
        self,
        interaction: dict,
        detection_map: Dict[str, YoloDetection],
        frame_shape: tuple,
    ) -> bool:
        subject_id = self._normalize_response_id(interaction.get("subject_id", ""))
        target_id = self._normalize_response_id(interaction.get("target_id", ""))
        action = str(interaction.get("action", "")).strip()

        if not subject_id or not action:
            return False

        subject_det = detection_map.get(subject_id)
        if subject_det is None or int(subject_det.class_id) != 0:
            self.get_logger().warn(
                f"Interaction subject_id {subject_id} missing or not a person; rejecting"
            )
            return False
        if not self._is_bbox_in_image(subject_det, frame_shape):
            self.get_logger().warn(
                f"Interaction subject_id {subject_id} bbox out of image bounds; rejecting"
            )
            return False

        if target_id:
            target_det = detection_map.get(target_id)
            if target_det is None:
                self.get_logger().warn(
                    f"Interaction target_id {target_id} not in detection map; rejecting"
                )
                return False
            if not self._is_bbox_in_image(target_det, frame_shape):
                self.get_logger().warn(
                    f"Interaction target_id {target_id} bbox out of image bounds; rejecting"
                )
                return False

        return True

    def _process_next_yolo_output(self) -> None:
        if self._worker_busy or not self._pending_yolo_outputs:
            return

        msg = self._pending_yolo_outputs.popleft()
        self._worker_busy = True
        person_detections = [det for det in msg.objects if int(det.class_id) == 0]
        if not person_detections:
            self.get_logger().info(
                f"Skipping interaction inference for {msg.header.stamp.sec}.{msg.header.stamp.nanosec}: no person detections"
            )
            self._worker_busy = False
            return

        if not self._passes_geometric_pregate(msg.objects):
            self.get_logger().debug(
                f"Skipping interaction inference for {msg.header.stamp.sec}.{msg.header.stamp.nanosec}: "
                "geometric pre-gate found no person near any object"
            )
            self._worker_busy = False
            return

        image_msg = self._find_matching_image(msg)
        if image_msg is None:
            self.get_logger().warn(
                "No matching RGB frame found for interaction inference at "
                f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
            )
            self._worker_busy = False
            return

        try:
            bgr = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"RGB conversion failed for interaction node: {exc}")
            self._worker_busy = False
            return

        detection_map = {self._normalize_label_id(det): det for det in msg.objects}
        annotated = self._annotate_frame(bgr, msg.objects)
        prompt = self._build_prompt(msg.objects)

        try:
            response_text = self._query_vlm(prompt, annotated)
        except Exception as exc:
            self.get_logger().error(f"Interaction VLM request failed: {exc}")
            self._worker_busy = False
            return

        interaction_output = InteractionOutput()
        interaction_output.header = msg.header
        interaction_output.source_stamp_sec = int(msg.header.stamp.sec)
        interaction_output.source_stamp_nanosec = int(msg.header.stamp.nanosec)
        interaction_output.raw_response = response_text

        parsed = self._parse_response(response_text)
        if not parsed:
            self.get_logger().info(
                "Interaction model returned no parsable interactions for "
                f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}: {response_text[:200]}"
            )
            self.get_logger().warn(f"Raw interaction model response: {response_text}")
        for interaction in parsed:
            if not self._validate_interaction_refs(interaction, detection_map, bgr.shape):
                continue

            subject_id = self._normalize_response_id(interaction.get("subject_id", ""))
            target_id = self._normalize_response_id(interaction.get("target_id", ""))
            action = str(interaction.get("action", "")).strip()
            caption = str(interaction.get("caption", "")).strip()
            confidence = self._parse_confidence(interaction.get("confidence"))
            if confidence < self.min_confidence:
                self.get_logger().debug(
                    f"Dropping interaction '{action}' (subject {subject_id}, target {target_id}) "
                    f"with confidence {confidence:.2f} < {self.min_confidence:.2f}"
                )
                continue

            event = InteractionEvent()
            event.action = action
            event.caption = caption or self._default_caption(subject_id, action, target_id, detection_map)
            event.confidence = confidence

            subject_det = detection_map[subject_id]
            event.objects.append(self._build_object_ref("subject", subject_id, subject_det))

            target_det = detection_map.get(target_id) if target_id else None
            if target_det is not None:
                event.objects.append(self._build_object_ref("target", target_id, target_det))

            interaction_output.interactions.append(event)

        if not interaction_output.interactions:
            self.get_logger().warn(
                "Interaction response parsed but produced no valid interactions. "
                f"Raw response: {response_text}"
            )
            self.get_logger().info(
                f"No validated interactions remained for {msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
            )
            self._worker_busy = False
            return

        self.pub.publish(interaction_output)
        self.get_logger().info(
            f"Published {len(interaction_output.interactions)} interactions for "
            f"{msg.header.stamp.sec}.{msg.header.stamp.nanosec}"
        )
        self._worker_busy = False

    def _build_prompt(self, detections: list[YoloDetection]) -> str:
        lines = []
        for det in detections:
            label_id = self._normalize_label_id(det)
            lines.append(
                f"ID {label_id}: {det.class_name} at "
                f"bbox [{int(det.x_min)}, {int(det.y_min)}, {int(det.x_max)}, {int(det.y_max)}]"
            )

        return (
            "You are looking at a robot camera image that already has object IDs drawn on it. "
            "Focus on visible human interactions with objects or other people. "
            "Return JSON only with the schema "
            "{\"interactions\": [{\"subject_id\": \"<person id>\", \"target_id\": \"<object id or empty>\", "
            "\"action\": \"<short verb>\", \"caption\": \"<short sentence>\", "
            "\"confidence\": <float 0.0-1.0>}]}. "
            "confidence is your confidence that the interaction is clearly visible and the IDs are correct. "
            "Only include interactions that are clearly visible. "
            "CRITICAL: when multiple bounding boxes overlap, prefer the LARGER / FOREGROUND box. "
            "The smaller box behind it usually belongs to a background object or person. "
            "Verify that the ID you select matches the description: if the caption says "
            "'person in green shirt', the selected box must actually show green. "
            "If the described colors or object do not match any visible ID, return {\"interactions\": []}. "
            "Known detections: "
            + " ; ".join(lines)
        )

    def _query_vlm(self, prompt: str, bgr_image) -> str:
        ok, buffer = cv2.imencode(".jpg", bgr_image)
        if not ok:
            raise RuntimeError("Failed to encode annotated frame to JPEG")

        image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
        # Reasoning-style models (e.g. Qwen3.5) emit chain-of-thought before answering,
        # which breaks strict JSON parsing and wastes tokens. When vlm_disable_thinking is
        # set, ask the server (vLLM/Qwen3 chat template) to skip the thinking block.
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
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            extra_body=extra_body,
        )
        return str(completion.choices[0].message.content or "").strip()

    def _parse_response(self, response_text: str) -> list[dict]:
        if not response_text:
            return []

        json_text = response_text.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", json_text, re.DOTALL)
        if fence_match:
            json_text = fence_match.group(1)

        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            brace_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            if brace_match is None:
                self.get_logger().warn(f"Could not parse interaction JSON: {response_text}")
                return []
            try:
                payload = json.loads(brace_match.group(1))
            except json.JSONDecodeError:
                self.get_logger().warn(f"Could not parse interaction JSON: {response_text}")
                return []

        interactions = payload.get("interactions", [])
        if not isinstance(interactions, list):
            return []
        return [item for item in interactions if isinstance(item, dict)]

    def _annotate_frame(self, frame, detections: list[YoloDetection]):
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Sort by area ascending so larger (likely foreground) boxes are drawn last,
        # making their labels and borders visible on top of smaller background boxes.
        sorted_dets = sorted(
            detections,
            key=lambda d: (d.x_max - d.x_min) * (d.y_max - d.y_min),
        )

        for idx, det in enumerate(sorted_dets):
            x1, y1, x2, y2 = int(det.x_min), int(det.y_min), int(det.x_max), int(det.y_max)
            label_id = self._normalize_label_id(det)
            label = f"ID {label_id} | {det.class_name}"

            # Distinct color per detection (HSV -> BGR) so overlapping IDs are easier to tell apart.
            hue = int((idx * 35) % 180)
            color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0][0])

            # Semi-transparent fill to show overlap regions.
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)

            # Border.
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)

            # Label background sized to text.
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            label_x = max(0, min(x1, w - text_w - 12))
            label_y = max(text_h + 8, y1 - 6)

            cv2.rectangle(
                annotated,
                (label_x, label_y - text_h - 6),
                (label_x + text_w + 10, label_y + 2),
                color_bgr,
                -1,
            )

            # Contrast text color based on label background brightness.
            brightness = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
            text_color = (0, 0, 0) if brightness > 135 else (255, 255, 255)
            cv2.putText(
                annotated,
                label,
                (label_x + 5, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                text_color,
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _build_object_ref(self, role: str, label_id: str, det: YoloDetection) -> InteractionObjectRef:
        ref = InteractionObjectRef()
        ref.role = role
        ref.track_id = label_id
        ref.detection_id = int(det.instance_id)
        ref.class_id = int(det.class_id)
        ref.class_name = str(det.class_name)
        ref.x_min = int(det.x_min)
        ref.y_min = int(det.y_min)
        ref.x_max = int(det.x_max)
        ref.y_max = int(det.y_max)
        return ref

    def _default_caption(
        self,
        subject_id: str,
        action: str,
        target_id: str,
        detection_map: Dict[str, YoloDetection],
    ) -> str:
        subject_det = detection_map.get(subject_id)
        subject_name = subject_det.class_name if subject_det is not None else "person"
        if not target_id:
            return f"{subject_name} {subject_id} is {action}."
        target_det = detection_map.get(target_id)
        target_name = target_det.class_name if target_det is not None else "object"
        return f"{subject_name} {subject_id} is {action} {target_name} {target_id}."


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = InteractionDescriptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
