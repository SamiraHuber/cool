#!/usr/bin/env python3
"""Detection engine abstraction supporting YOLO and OWLv2 backends with BoTSort tracking."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# Standard COCO 80-class mapping used by YOLO models
COCO_CLASSES: Dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}

COCO_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in COCO_CLASSES.items()}

# OWLv2 query text → COCO class ID (for queries that differ from COCO names)
OWLV2_TO_COCO_ID: Dict[str, int] = {
    "ball": 32,  # OWLv2 "ball" maps to COCO "sports ball"
}

DEFAULT_OWLV2_CLASSES: List[str] = [COCO_CLASSES[i] for i in range(80)]


class SimpleBoxes:
    """Minimal Boxes-like wrapper compatible with ultralytics BOTSort."""

    def __init__(
        self,
        xyxy: np.ndarray,
        conf: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32)
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self._update_derived()

    def _update_derived(self) -> None:
        if len(self.xyxy) > 0:
            xywh = self.xyxy.copy()
            xywh[:, 2] = self.xyxy[:, 2] - self.xyxy[:, 0]
            xywh[:, 3] = self.xyxy[:, 3] - self.xyxy[:, 1]
            xywh[:, 0] = self.xyxy[:, 0] + xywh[:, 2] / 2.0
            xywh[:, 1] = self.xyxy[:, 1] + xywh[:, 3] / 2.0
            self.xywh = xywh
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.xyxy)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SimpleBoxes(self.xyxy[idx], self.conf[idx], self.cls[idx])
        idx_arr = np.asarray(idx)
        if idx_arr.size == 0:
            return SimpleBoxes(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
        if idx_arr.dtype == bool:
            return SimpleBoxes(
                self.xyxy[idx],
                self.conf[idx],
                self.cls[idx],
            )
        return SimpleBoxes(
            self.xyxy[idx],
            self.conf[idx],
            self.cls[idx],
        )

    @property
    def xywhr(self) -> np.ndarray:
        return self.xywh


class DetectionResult:
    """Normalized output from any detection engine."""

    def __init__(
        self,
        xyxy: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        track_ids: np.ndarray,
        class_names: List[str],
        masks: Optional[np.ndarray] = None,
    ) -> None:
        self.xyxy = xyxy
        self.scores = scores
        self.class_ids = class_ids
        self.track_ids = track_ids
        self.class_names = class_names
        self.masks = masks

    @property
    def num_detections(self) -> int:
        return len(self.xyxy)


def _init_tracker(tracker_config: str):
    """Create a standalone ultralytics BoTSORT tracker."""
    from ultralytics.trackers import BOTSORT
    import ultralytics
    import yaml
    from pathlib import Path

    tracker_path = Path(tracker_config)
    if not tracker_path.is_absolute():
        builtin = Path(ultralytics.__file__).parent / "cfg" / "trackers" / tracker_config
        if builtin.exists():
            tracker_path = builtin
        else:
            tracker_path = Path.cwd() / tracker_config
    with open(tracker_path, "r") as f:
        cfg = yaml.safe_load(f)
    args = SimpleNamespace(**cfg)
    return BOTSORT(args)


def _nms(
    xyxy: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class NMS to deduplicate overlapping detections."""
    if len(xyxy) == 0:
        return xyxy, scores, class_ids

    keep_indices: List[int] = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_indices = np.where(cls_mask)[0]
        cls_xyxy = xyxy[cls_mask]
        cls_scores = scores[cls_mask]

        order = np.argsort(-cls_scores)
        while len(order) > 0:
            i = order[0]
            keep_indices.append(cls_indices[i])
            if len(order) == 1:
                break
            ious = _compute_iou(cls_xyxy[i : i + 1], cls_xyxy[order[1:]])
            order = order[1:][ious[0] <= iou_thresh]

    keep = np.array(keep_indices, dtype=int)
    return xyxy[keep], scores[keep], class_ids[keep]


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and N boxes."""
    x1 = np.maximum(box_a[:, 0:1], box_b[:, 0])
    y1 = np.maximum(box_a[:, 1:2], box_b[:, 1])
    x2 = np.minimum(box_a[:, 2:3], box_b[:, 2])
    y2 = np.minimum(box_a[:, 3:4], box_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    union = area_a[:, None] + area_b - inter
    return inter / np.maximum(union, 1e-6)


def _two_stage_detect(
    bgr: np.ndarray,
    detect_raw_fn,
    person_class_id: int = 0,
    crop_padding: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Stage-1 detect all, then stage-2 detect on each person crop and merge."""
    xyxy, scores, class_ids, masks = detect_raw_fn(bgr)

    if len(xyxy) == 0:
        return xyxy, scores, class_ids, masks

    person_mask = class_ids == person_class_id
    if not np.any(person_mask):
        return xyxy, scores, class_ids, masks

    h, w = bgr.shape[:2]
    all_xyxy: List[np.ndarray] = [xyxy]
    all_scores: List[np.ndarray] = [scores]
    all_class_ids: List[np.ndarray] = [class_ids]
    # Masks from crops are not aggregated (complex geometry); keep stage-1 only.
    combined_masks: Optional[np.ndarray] = masks

    for idx in np.where(person_mask)[0]:
        x1, y1, x2, y2 = xyxy[idx].astype(int)
        pad_x = max(1, int((x2 - x1) * crop_padding))
        pad_y = max(1, int((y2 - y1) * crop_padding))
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue

        c_xyxy, c_scores, c_class_ids, _ = detect_raw_fn(crop)
        if len(c_xyxy) == 0:
            continue

        # Map crop coordinates back to original image
        c_xyxy[:, [0, 2]] += cx1
        c_xyxy[:, [1, 3]] += cy1

        # Exclude person re-detections (already in stage 1)
        non_person = c_class_ids != person_class_id
        if not np.any(non_person):
            continue

        all_xyxy.append(c_xyxy[non_person])
        all_scores.append(c_scores[non_person])
        all_class_ids.append(c_class_ids[non_person])

    if len(all_xyxy) == 1:
        return all_xyxy[0], all_scores[0], all_class_ids[0], combined_masks

    merged_xyxy = np.vstack(all_xyxy)
    merged_scores = np.concatenate(all_scores)
    merged_class_ids = np.concatenate(all_class_ids)

    # Deduplicate overlaps between stage-1 and stage-2 detections
    merged_xyxy, merged_scores, merged_class_ids = _nms(
        merged_xyxy, merged_scores, merged_class_ids, iou_thresh=0.5
    )

    return merged_xyxy, merged_scores, merged_class_ids, combined_masks


def _tracker_to_result(
    tracked,
    class_name_fn,
    masks: Optional[np.ndarray] = None,
) -> Optional[DetectionResult]:
    """Convert tracker output array to DetectionResult."""
    tracked = np.asarray(tracked, dtype=np.float32)
    if tracked.ndim != 2 or tracked.shape[1] < 7:
        return None

    xyxy = tracked[:, :4]
    track_ids = tracked[:, 4].astype(int)
    out_scores = tracked[:, 5]
    out_cls = tracked[:, 6].astype(int)
    class_names = [class_name_fn(int(cid)) for cid in out_cls]

    return DetectionResult(
        xyxy=xyxy,
        scores=out_scores,
        class_ids=out_cls,
        track_ids=track_ids,
        class_names=class_names,
        masks=masks,
    )


class BaseDetectionEngine:
    def detect_and_track(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        raise NotImplementedError

    def class_name_for_id(self, class_id: int) -> str:
        raise NotImplementedError


class YoloDetectionEngine(BaseDetectionEngine):
    """YOLO + BoTSort tracking (built-in or standalone for two-stage)."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        tracker_config: str,
        imgsz: Optional[int] = None,
        two_stage_person: bool = False,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.imgsz = imgsz
        self._names = getattr(self.model, "names", {})
        self.two_stage_person = two_stage_person

        if self.two_stage_person:
            self._tracker = _init_tracker(tracker_config)

    def class_name_for_id(self, class_id: int) -> str:
        if isinstance(self._names, dict):
            return str(self._names.get(class_id, class_id))
        if isinstance(self._names, (list, tuple)) and 0 <= class_id < len(self._names):
            return str(self._names[class_id])
        return str(class_id)

    def _detect_raw(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run YOLO predict (no tracking) and return raw detections."""
        kwargs: Dict[str, Any] = dict(
            source=bgr,
            conf=self.conf_threshold,
            verbose=False,
        )
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        results = self.model.predict(**kwargs)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, empty.copy()[:, 0], np.zeros(0, dtype=np.int32), None

        result = results[0]
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()

        return xyxy, scores, class_ids, masks

    def detect_and_track(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        if self.two_stage_person:
            return self._detect_and_track_two_stage(bgr)
        return self._detect_and_track_builtin(bgr)

    def _detect_and_track_builtin(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        """Original single-stage path using ultralytics built-in tracking."""
        kwargs: Dict[str, Any] = dict(
            source=bgr,
            conf=self.conf_threshold,
            persist=True,
            tracker=self.tracker_config,
            verbose=False,
        )
        if self.imgsz is not None:
            kwargs["imgsz"] = self.imgsz

        results = self.model.track(**kwargs)
        if not results:
            return None

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return None

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = -np.ones(len(xyxy), dtype=int)

        masks = None
        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.cpu().numpy()

        class_names = [self.class_name_for_id(int(cid)) for cid in class_ids]

        return DetectionResult(
            xyxy=xyxy,
            scores=scores,
            class_ids=class_ids,
            track_ids=track_ids,
            class_names=class_names,
            masks=masks,
        )

    def _detect_and_track_two_stage(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        """Two-stage: detect all, then re-detect on each person crop."""
        xyxy, scores, class_ids, masks = _two_stage_detect(
            bgr, self._detect_raw, person_class_id=0, crop_padding=0.1
        )

        if len(xyxy) == 0:
            empty = SimpleBoxes(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
            _ = self._tracker.update(empty, img=bgr)
            return None

        dets = SimpleBoxes(xyxy, scores, class_ids)
        tracked = self._tracker.update(dets, img=bgr)

        if tracked is None or len(tracked) == 0:
            return None

        return _tracker_to_result(tracked, self.class_name_for_id, masks=masks)


class Owlv2DetectionEngine(BaseDetectionEngine):
    """OWLv2 zero-shot detection + standalone BoTSort tracking."""

    _DEVICE_FALLBACK: Dict[str, str] = {}

    def __init__(
        self,
        model_path: str,
        conf_threshold: float,
        tracker_config: str,
        classes: Optional[List[str]] = None,
        device: str = "cuda",
        imgsz: int = 640,
        two_stage_person: bool = False,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.tracker_config = tracker_config
        self.classes = classes if classes is not None else DEFAULT_OWLV2_CLASSES[:]
        self.device = device
        self.imgsz = imgsz
        self._cache_key = f"owlv2:{model_path}"
        self.two_stage_person = two_stage_person

        # Build mapping from OWLv2 class index → COCO/custom class ID
        self._class_id_map: List[int] = []
        self._custom_id_to_name: Dict[int, str] = {}
        next_custom_id = 80
        for name in self.classes:
            if name in COCO_NAME_TO_ID:
                self._class_id_map.append(COCO_NAME_TO_ID[name])
            elif name in OWLV2_TO_COCO_ID:
                self._class_id_map.append(OWLV2_TO_COCO_ID[name])
            else:
                self._class_id_map.append(next_custom_id)
                self._custom_id_to_name[next_custom_id] = name
                next_custom_id += 1

        self.person_class_id = self._class_id_map[self.classes.index("person")] if "person" in self.classes else 0

        self._load_model()
        self._tracker = _init_tracker(tracker_config)

    def _resolve_device(self) -> str:
        import torch

        fallback = self._DEVICE_FALLBACK.get(self._cache_key)
        if fallback is not None:
            return fallback
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return self.device

    def _load_model(self) -> None:
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        self.processor = Owlv2Processor.from_pretrained(self.model_path)
        self.model = Owlv2ForObjectDetection.from_pretrained(self.model_path)
        device = self._resolve_device()
        self.model = self.model.to(device)
        self.model.eval()

    def class_name_for_id(self, class_id: int) -> str:
        if class_id in COCO_CLASSES:
            return COCO_CLASSES[class_id]
        if class_id in self._custom_id_to_name:
            return self._custom_id_to_name[class_id]
        return str(class_id)

    def _preprocess(self, bgr: np.ndarray) -> Tuple[Any, Tuple[int, int]]:
        from PIL import Image

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) <= self.imgsz:
            return Image.fromarray(rgb), (h, w)
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(resized), (new_h, new_w)

    def _detect_raw(
        self, bgr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run OWLv2 inference (no tracking) and return raw detections."""
        import torch

        image_pil, target_size = self._preprocess(bgr)
        texts = [self.classes]
        inputs = self.processor(text=texts, images=image_pil, return_tensors="pt")

        device = self._resolve_device()
        try:
            self.model = self.model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
        except RuntimeError as exc:
            if self._is_oom(exc) and device != "cpu":
                self._DEVICE_FALLBACK[self._cache_key] = "cpu"
                device = "cpu"
                self.model = self.model.to(device)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
            else:
                raise

        target_sizes = torch.Tensor([target_size]).to(device)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            threshold=self.conf_threshold,
            target_sizes=target_sizes,
        )[0]

        if len(results["scores"]) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, empty.copy()[:, 0], np.zeros(0, dtype=np.int32), None

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results.get("labels")
        if labels is not None:
            raw_class_ids = labels.cpu().numpy().astype(int)
            class_ids = np.array([self._class_id_map[cid] for cid in raw_class_ids], dtype=np.int32)
        else:
            class_ids = np.zeros(len(boxes), dtype=np.int32)

        return boxes, scores, class_ids, None

    def detect_and_track(self, bgr: np.ndarray) -> Optional[DetectionResult]:
        if self.two_stage_person:
            xyxy, scores, class_ids, _ = _two_stage_detect(
                bgr,
                self._detect_raw,
                person_class_id=self.person_class_id,
                crop_padding=0.1,
            )
        else:
            xyxy, scores, class_ids, _ = self._detect_raw(bgr)

        if len(xyxy) == 0:
            empty = SimpleBoxes(
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.zeros(0, dtype=np.float32),
            )
            _ = self._tracker.update(empty, img=bgr)
            return None

        dets = SimpleBoxes(xyxy, scores, class_ids)
        tracked = self._tracker.update(dets, img=bgr)

        if tracked is None or len(tracked) == 0:
            return None

        return _tracker_to_result(tracked, self.class_name_for_id, masks=None)

    @staticmethod
    def _is_oom(exc: RuntimeError) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "cuda out of memory" in msg
            or "outofmemory" in msg
        )


def build_detection_engine(
    backend: str,
    model_path: str,
    conf_threshold: float,
    tracker_config: str,
    imgsz: Optional[int] = None,
    owlv2_model_path: Optional[str] = None,
    owlv2_classes: Optional[List[str]] = None,
    owlv2_device: str = "cuda",
    owlv2_imgsz: int = 640,
    two_stage_person: bool = False,
) -> BaseDetectionEngine:
    backend = backend.strip().lower()
    if backend == "owlv2":
        path = owlv2_model_path or model_path
        return Owlv2DetectionEngine(
            model_path=path,
            conf_threshold=conf_threshold,
            tracker_config=tracker_config,
            classes=owlv2_classes,
            device=owlv2_device,
            imgsz=owlv2_imgsz,
            two_stage_person=two_stage_person,
        )
    if backend == "yolo":
        return YoloDetectionEngine(
            model_path=model_path,
            conf_threshold=conf_threshold,
            tracker_config=tracker_config,
            imgsz=imgsz,
            two_stage_person=two_stage_person,
        )
    raise ValueError(f"Unknown detector backend: {backend!r}")
