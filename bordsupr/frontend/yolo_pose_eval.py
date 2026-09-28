"""YOLO pose / detection evaluation engine for the web frontend.

This module is imported by app.py; heavy dependencies (ultralytics, transformers)
are loaded lazily inside functions so that FastAPI startup stays fast.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = "person"
    score: float = 1.0
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BBox) -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        union = self.area + other.area - inter_area
        return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

@dataclass
class ImageRecord:
    image_path: Path
    boxes: list[BBox]


# Standard COCO class names used as fallback for ONNX YOLO models when
# model.names triggers a GPU OOM during predictor setup.
_COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]
_COCO_NAMES_DICT: dict[int, str] = {i: name for i, name in enumerate(_COCO_NAMES)}


def _normalise_label(label: str) -> str:
    label = str(label).strip().lower()
    if label in ("person", "0", "human"):
        return "person"
    return label


def _normalise_classes(classes: set[str] | None) -> set[str] | None:
    """Normalize class filter entries the same way as detection labels."""
    if classes is None:
        return None
    return {_normalise_label(c) for c in classes}


def load_ground_truth(path: Path, fmt: str = "xyxy") -> list[ImageRecord]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("images", data)
    records: list[ImageRecord] = []
    for item in data:
        raw_boxes = item.get("boxes", item.get("annotations", []))
        boxes: list[BBox] = []
        for b in raw_boxes:
            label = _normalise_label(b.get("label", b.get("class", "person")))
            if fmt == "xyxy":
                x1 = float(b["x1"])
                y1 = float(b["y1"])
                x2 = float(b["x2"])
                y2 = float(b["y2"])
            elif fmt == "xywh":
                x1 = float(b["x"])
                y1 = float(b["y"])
                x2 = x1 + float(b["w"])
                y2 = y1 + float(b["h"])
            else:
                raise ValueError(f"Unsupported gt-format: {fmt}")
            boxes.append(BBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label))
        records.append(ImageRecord(image_path=Path(item["image_path"]), boxes=boxes))
    return records


# ---------------------------------------------------------------------------
# Matching & AP computation
# ---------------------------------------------------------------------------

def match_image(
    gts: list[BBox],
    dets: list[BBox],
    iou_thresh: float,
) -> tuple[int, int, int]:
    """Return (tp, fp, fn) for a single image at a given IoU threshold."""
    gt_matched = [False] * len(gts)
    tp = 0
    fp = 0
    dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
    for det in dets_sorted:
        best_iou = 0.0
        best_gt = -1
        for i, gt in enumerate(gts):
            if gt_matched[i]:
                continue
            if gt.label != det.label:
                continue
            iou = det.iou(gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = i
        if best_iou >= iou_thresh and best_gt >= 0:
            gt_matched[best_gt] = True
            tp += 1
        else:
            fp += 1
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn


def match_image_detailed(
    gts: list[BBox],
    dets: list[BBox],
    iou_thresh: float,
) -> tuple[int, int, int, list[str], list[int], list[bool]]:
    """Return (tp, fp, fn, det_status, det_matched_gt, gt_matched).

    det_status[i] = 'tp' or 'fp' for dets[i] (original order preserved).
    det_matched_gt[i] = index of matched GT or -1.
    gt_matched[j] = True if GT j was matched.
    """
    gt_matched = [False] * len(gts)
    det_status = ["fp"] * len(dets)
    det_matched_gt = [-1] * len(dets)

    sorted_indices = sorted(range(len(dets)), key=lambda i: dets[i].score, reverse=True)
    for det_idx in sorted_indices:
        det = dets[det_idx]
        best_iou = 0.0
        best_gt = -1
        for i, gt in enumerate(gts):
            if gt_matched[i]:
                continue
            if gt.label != det.label:
                continue
            iou = det.iou(gt)
            if iou > best_iou:
                best_iou, best_gt = iou, i
        if best_iou >= iou_thresh and best_gt >= 0:
            gt_matched[best_gt] = True
            det_status[det_idx] = "tp"
            det_matched_gt[det_idx] = best_gt

    tp = sum(1 for s in det_status if s == "tp")
    fp = sum(1 for s in det_status if s == "fp")
    fn = sum(1 for m in gt_matched if not m)
    return tp, fp, fn, det_status, det_matched_gt, gt_matched


def compute_ap(
    gts_all: list[list[BBox]],
    dets_all: list[list[BBox]],
    iou_thresh: float,
    label: str | None = None,
) -> float:
    """Compute Average Precision for a single IoU threshold (COCO 101-point)."""
    aps = compute_ap_multi(gts_all, dets_all, [iou_thresh], label=label)
    return aps[iou_thresh]


def compute_ap_multi(
    gts_all: list[list[BBox]],
    dets_all: list[list[BBox]],
    iou_thresholds: list[float],
    label: str | None = None,
) -> dict[float, float]:
    """Compute AP for multiple IoU thresholds in a single pass (COCO 101-point each)."""
    all_scores_by_thresh: dict[float, list[tuple[float, bool]]] = {t: [] for t in iou_thresholds}
    n_gt = 0
    for gts_img, dets_img in zip(gts_all, dets_all):
        gts = [g for g in gts_img if label is None or g.label == label]
        dets = [d for d in dets_img if label is None or d.label == label]
        if not gts and not dets:
            continue
        n_gt += len(gts)
        dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
        for thresh in iou_thresholds:
            gt_matched = [False] * len(gts)
            for det in dets_sorted:
                best_iou = 0.0
                best_gt = -1
                for i, gt in enumerate(gts):
                    if gt_matched[i]:
                        continue
                    iou = det.iou(gt)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = i
                if best_iou >= thresh and best_gt >= 0:
                    gt_matched[best_gt] = True
                    all_scores_by_thresh[thresh].append((det.score, True))
                else:
                    all_scores_by_thresh[thresh].append((det.score, False))

    result: dict[float, float] = {}
    for thresh in iou_thresholds:
        all_scores = all_scores_by_thresh[thresh]
        if n_gt == 0:
            result[thresh] = float("nan")
            continue
        all_scores.sort(key=lambda x: x[0], reverse=True)
        tp_cumsum = np.cumsum([s[1] for s in all_scores])
        fp_cumsum = np.cumsum([not s[1] for s in all_scores])
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            if np.sum(recalls >= t) == 0:
                p = 0.0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 101.0
        result[thresh] = float(ap)
    return result


# ---------------------------------------------------------------------------
# Model config & result
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_path: str
    model_type: str = "yolo"          # "yolo" | "owlv2" | "grounding_dino"
    conf: float = 0.25
    imgsz: int = 640
    device: str = "0"
    iou: float = 0.7
    text_prompt: str = "person"
    ablation_brightness: bool = False
    ablation_crop_redetect: bool = False
    tracker: str = "none"             # "none" | "bytetrack" | "botsort" | "strongsort" | "ef-strongsort"


@dataclass
class EvalResult:
    config: ModelConfig
    per_image: list[dict[str, Any]] = field(default_factory=list)
    total_tp: int = 0
    total_fp: int = 0
    total_fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    ap50: float = 0.0
    ap5095: float = 0.0
    elapsed_sec: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Model cache (lazy, module-level)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}
# Tracks device fallback for zero-shot models (GPU OOM -> CPU)
_DEVICE_FALLBACK: dict[str, str] = {}


def _torch_device(device_str: str) -> str:
    if device_str == "cpu":
        return "cpu"
    if device_str.startswith("cuda"):
        return device_str
    return f"cuda:{device_str}"


def _load_yolo_model(model_path: str):
    from ultralytics import YOLO

    key = f"yolo:{model_path}"
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(model_path)
    return _MODEL_CACHE[key]


def _load_owlv2_model(model_path: str, device: str = "cpu"):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    import torch

    key = f"owlv2:{model_path}"
    if key not in _MODEL_CACHE:
        processor = Owlv2Processor.from_pretrained(model_path)
        model = Owlv2ForObjectDetection.from_pretrained(model_path)
        # Proactively skip GPU if almost no memory is free to avoid OOM churn
        if str(device) != "cpu" and _gpu_free_mb() < 500:
            device = "cpu"
            _DEVICE_FALLBACK[key] = "cpu"
        try:
            model = model.to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            print(f"[WARN] OWLv2 model.to({device}) failed: {exc}. Falling back to CPU.", file=sys.stderr)
            try:
                model = model.to("cpu")
            except RuntimeError:
                pass
            _DEVICE_FALLBACK[key] = "cpu"
        _MODEL_CACHE[key] = (processor, model)
    return _MODEL_CACHE[key]


def _load_grounding_dino_model(model_path: str, device: str = "cpu"):
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    import torch

    key = f"gdino:{model_path}"
    if key not in _MODEL_CACHE:
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path)
        # Proactively skip GPU if almost no memory is free to avoid OOM churn
        if str(device) != "cpu" and _gpu_free_mb() < 500:
            device = "cpu"
            _DEVICE_FALLBACK[key] = "cpu"
        try:
            model = model.to(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            print(f"[WARN] GroundingDINO model.to({device}) failed: {exc}. Falling back to CPU.", file=sys.stderr)
            try:
                model = model.to("cpu")
            except RuntimeError:
                pass
            _DEVICE_FALLBACK[key] = "cpu"
        _MODEL_CACHE[key] = (processor, model)
    return _MODEL_CACHE[key]


def _ensure_model_on_device(model, device: str):
    """Move model to target device only if not already there."""
    import torch

    target = torch.device(device)
    if next(model.parameters()).device == target:
        return model
    try:
        model = model.to(target)
    except (torch.cuda.OutOfMemoryError, RuntimeError):
        if str(target) == "cpu":
            raise
        model = model.to("cpu")
    return model


# ---------------------------------------------------------------------------
# Pre-processing ablations
# ---------------------------------------------------------------------------

def _apply_brightness_correction(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE brightness correction to a BGR image."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_crop_redetect(
    dets: list[BBox],
    img: np.ndarray,
    infer_fn,
    config: ModelConfig,
    classes: set[str] | None,
) -> list[BBox]:
    """For each person detection, crop around it and run a second detection."""
    if not dets:
        return dets

    refined: list[BBox] = []
    h, w = img.shape[:2]

    for det in dets:
        if det.label != "person":
            refined.append(det)
            continue

        # 20% padding around the box
        pad_x = det.width * 0.2
        pad_y = det.height * 0.2
        cx1 = max(0, int(det.x1 - pad_x))
        cy1 = max(0, int(det.y1 - pad_y))
        cx2 = min(w, int(det.x2 + pad_x))
        cy2 = min(h, int(det.y2 + pad_y))

        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            refined.append(det)
            continue

        # Run inference on crop WITHOUT re-triggering crop-redetect (prevent recursion)
        crop_config = ModelConfig(
            model_path=config.model_path,
            model_type=config.model_type,
            conf=config.conf,
            imgsz=config.imgsz,
            device=config.device,
            iou=config.iou,
            text_prompt=config.text_prompt,
            ablation_brightness=False,
            ablation_crop_redetect=False,
        )
        try:
            crop_dets = infer_fn(crop, crop_config, classes)
        except RuntimeError as exc:
            if _is_k_out_of_range_error(exc):
                print(
                    f"[WARN] Crop-redetect failed with 'selected index k out of range' for "
                    f"{crop_config.model_type} model={crop_config.model_path}. Keeping original detection.",
                    file=sys.stderr,
                )
                refined.append(det)
                continue
            raise

        # Map back and find best overlapping detection
        best_iou = 0.0
        best_det = det
        for cd in crop_dets:
            mapped = BBox(
                x1=cd.x1 + cx1,
                y1=cd.y1 + cy1,
                x2=cd.x2 + cx1,
                y2=cd.y2 + cy1,
                label=cd.label,
                score=cd.score,
            )
            if mapped.label == det.label:
                iou = det.iou(mapped)
                if iou > best_iou:
                    best_iou = iou
                    best_det = mapped

        refined.append(best_det)

    return refined


# ---------------------------------------------------------------------------
# Inference dispatchers
# ---------------------------------------------------------------------------

def _run_yolo_inference(
    img: np.ndarray,
    config: ModelConfig,
    classes: set[str] | None,
) -> list[BBox]:
    model = _load_yolo_model(config.model_path)
    try:
        model_names = getattr(model, "names", {})
    except RuntimeError as exc:
        # ONNX models trigger predictor setup on first access to .names, which
        # can OOM on a full GPU even though we force CPU inference below.
        if _is_oom_error(exc):
            print(
                f"[WARN] model.names OOM for {config.model_path}, falling back to COCO names.",
                file=sys.stderr,
            )
            model_names = _COCO_NAMES_DICT
        else:
            raise

    # ONNX Runtime models frequently fail on GPU due to missing data-transfer
    # providers; force CPU for .onnx files to avoid the "selected index k out
    # of range" style binding errors.
    device = config.device
    if str(config.model_path).lower().endswith(".onnx"):
        device = "cpu"

    res = model.predict(
        source=img,
        conf=config.conf,
        imgsz=config.imgsz,
        device=device,
        iou=config.iou,
        verbose=False,
    )[0]

    dets: list[BBox] = []
    if res.boxes is not None:
        for box in res.boxes:
            xyxy = box.xyxy.cpu().numpy().flatten()
            score = float(box.conf.cpu().item())
            cid = int(box.cls.cpu().item())
            label = _normalise_label(_class_name(model_names, cid))
            if classes is not None and label not in classes:
                continue
            dets.append(
                BBox(
                    x1=float(xyxy[0]),
                    y1=float(xyxy[1]),
                    x2=float(xyxy[2]),
                    y2=float(xyxy[3]),
                    label=label,
                    score=score,
                )
            )
    return dets


def _preprocess_for_zero_shot(img: np.ndarray, imgsz: int) -> tuple[Image.Image, tuple[int, int]]:
    """Resize image so longest side <= imgsz, preserving aspect ratio.
    Returns (PIL Image, (original_h, original_w)) for processor input.
    The processor handles its own internal resizing; we only cap the
    image size to keep memory reasonable.  Boxes returned by
    post_process_grounded_object_detection are rescaled to the
    original image coordinate system using target_sizes, so we must
    pass the original dimensions, not the resized ones."""
    from PIL import Image
    h, w = img.shape[:2]
    if max(h, w) <= imgsz:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return pil, (h, w)
    scale = imgsz / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    return pil, (h, w)


def _run_owlv2_inference(
    img: np.ndarray,
    config: ModelConfig,
    classes: set[str] | None,
) -> list[BBox]:
    import torch

    processor, model = _load_owlv2_model(config.model_path, _torch_device(config.device))
    cache_key = f"owlv2:{config.model_path}"
    device = _DEVICE_FALLBACK.get(cache_key, _torch_device(config.device))

    image_pil, target_size = _preprocess_for_zero_shot(img, config.imgsz)
    # Support comma-separated class queries (e.g. "person, chair, ball, bottle")
    queries = [q.strip() for q in config.text_prompt.split(",")]
    texts = [queries]
    inputs = processor(text=texts, images=image_pil, return_tensors="pt")

    try:
        model = _ensure_model_on_device(model, device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
    except RuntimeError as exc:
        if device != "cpu":
            print(
                f"[WARN] OWLv2 inference failed on {device}: {exc}. Falling back to CPU for {config.model_path}",
                file=sys.stderr,
            )
            _DEVICE_FALLBACK[cache_key] = "cpu"
            device = "cpu"
            model = _ensure_model_on_device(model, device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
        else:
            raise

    output_device = next((t.device for t in outputs.values() if isinstance(t, torch.Tensor)), torch.device("cpu"))
    target_sizes = torch.Tensor([target_size]).to(output_device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=config.conf,
        target_sizes=target_sizes,
    )[0]

    dets: list[BBox] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels_tensor = results.get("labels")

    for i in range(len(boxes)):
        if labels_tensor is not None and i < len(labels_tensor):
            cls_idx = int(labels_tensor[i])
            label = _normalise_label(queries[cls_idx] if cls_idx < len(queries) else config.text_prompt)
        else:
            label = _normalise_label(queries[0] if queries else config.text_prompt)
        if classes is not None and label not in classes:
            continue
        dets.append(
            BBox(
                x1=float(boxes[i][0]),
                y1=float(boxes[i][1]),
                x2=float(boxes[i][2]),
                y2=float(boxes[i][3]),
                label=label,
                score=float(scores[i]),
            )
        )
    if not dets:
        print(
            f"[DEBUG] OWLv2 returned 0 detections for {config.model_path} "
            f"conf={config.conf} text_prompt={config.text_prompt} image_size={target_size}",
            file=sys.stderr,
        )
    return dets


def _run_grounding_dino_inference(
    img: np.ndarray,
    config: ModelConfig,
    classes: set[str] | None,
) -> list[BBox]:
    import torch

    processor, model = _load_grounding_dino_model(config.model_path, _torch_device(config.device))
    cache_key = f"gdino:{config.model_path}"
    device = _DEVICE_FALLBACK.get(cache_key, _torch_device(config.device))

    image_pil, target_size = _preprocess_for_zero_shot(img, config.imgsz)
    # Grounding DINO works best with period-separated caption format
    queries = [q.strip() for q in config.text_prompt.split(",")]
    text = " . ".join(queries)
    inputs = processor(images=image_pil, text=text, return_tensors="pt")

    try:
        model = _ensure_model_on_device(model, device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
    except RuntimeError as exc:
        if device != "cpu":
            print(
                f"[WARN] Grounding DINO inference failed on {device}: {exc}. Falling back to CPU for {config.model_path}",
                file=sys.stderr,
            )
            _DEVICE_FALLBACK[cache_key] = "cpu"
            device = "cpu"
            model = _ensure_model_on_device(model, device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
        else:
            raise

    output_device = next((t.device for t in outputs.values() if isinstance(t, torch.Tensor)), torch.device("cpu"))
    target_sizes = torch.Tensor([target_size]).to(output_device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids=inputs["input_ids"],
        threshold=config.conf,
        text_threshold=config.conf,
        target_sizes=target_sizes,
    )[0]

    dets: list[BBox] = []
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    text_labels = results.get("text_labels", [])

    for i in range(len(boxes)):
        raw_label = text_labels[i] if i < len(text_labels) else config.text_prompt
        # Map the matched phrase back to the closest individual query
        label = _normalise_label(raw_label)
        lowered = label.lower()
        matched = None
        for q in queries:
            if q.lower() in lowered:
                matched = q
                break
        if matched:
            label = _normalise_label(matched)
        if classes is not None and label not in classes:
            continue
        dets.append(
            BBox(
                x1=float(boxes[i][0]),
                y1=float(boxes[i][1]),
                x2=float(boxes[i][2]),
                y2=float(boxes[i][3]),
                label=label,
                score=float(scores[i]),
            )
        )
    if not dets:
        print(
            f"[DEBUG] Grounding DINO returned 0 detections for {config.model_path} "
            f"conf={config.conf} text_prompt={config.text_prompt} image_size={target_size}",
            file=sys.stderr,
        )
    return dets


# ---------------------------------------------------------------------------
# Batched zero-shot inference (much faster for eval loops)
# ---------------------------------------------------------------------------

_ZERO_SHOT_BATCH_SIZE = 8


def _run_owlv2_inference_batched(
    images: list[np.ndarray],
    config: ModelConfig,
    classes: set[str] | None,
) -> list[list[BBox]]:
    """Run OWLv2 on a batch of images and return per-image detections."""
    import torch
    import gc

    try:
        processor, model = _load_owlv2_model(config.model_path, _torch_device(config.device))
        cache_key = f"owlv2:{config.model_path}"

        queries = [q.strip() for q in config.text_prompt.split(",")]

        # Preprocess all images
        preprocessed: list[tuple[Image.Image, tuple[int, int]]] = []
        for img in images:
            preprocessed.append(_preprocess_for_zero_shot(img, config.imgsz))

        all_dets: list[list[BBox]] = []
        for batch_start in range(0, len(preprocessed), _ZERO_SHOT_BATCH_SIZE):
            batch_end = min(batch_start + _ZERO_SHOT_BATCH_SIZE, len(preprocessed))
            batch_pils = [preprocessed[i][0] for i in range(batch_start, batch_end)]
            batch_targets = [preprocessed[i][1] for i in range(batch_start, batch_end)]

            inputs = processor(text=[queries] * len(batch_pils), images=batch_pils, return_tensors="pt")

            # Explicit device handling: avoid silent CPU fallback that causes device-mismatch crashes
            device = _DEVICE_FALLBACK.get(cache_key, _torch_device(config.device))
            if str(device) != "cpu" and _gpu_free_mb() < 300:
                device = "cpu"
            try:
                model = model.to(device)
            except RuntimeError as exc:
                if str(device) != "cpu":
                    print(f"[WARN] model.to({device}) failed for {config.model_type}: {exc}. Falling back to CPU.", file=sys.stderr)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        model = model.to(device)
                    except RuntimeError:
                        device = "cpu"
                        model = model.to(device)
                else:
                    raise
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = None
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except RuntimeError as exc:
                if str(device) != "cpu":
                    print(f"[WARN] Inference failed on {device} for {config.model_type}: {exc}. Falling back to CPU.", file=sys.stderr)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        model = model.to(device)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                    except RuntimeError:
                        device = "cpu"
                        model = model.to(device)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        try:
                            with torch.no_grad():
                                outputs = model(**inputs)
                        except RuntimeError as exc2:
                            print(
                                f"[WARN] CPU inference also failed for {config.model_type} model={config.model_path}: {exc2}. "
                                f"Returning empty detections for this batch.",
                                file=sys.stderr,
                            )
                else:
                    raise
            if outputs is None:
                all_dets.extend([[] for _ in batch_pils])
                continue

            output_device = next((t.device for t in outputs.values() if isinstance(t, torch.Tensor)), torch.device("cpu"))
            target_sizes = torch.Tensor(batch_targets).to(output_device)
            results = processor.post_process_grounded_object_detection(
                outputs, threshold=config.conf, target_sizes=target_sizes
            )

            for batch_idx, res in enumerate(results):
                dets: list[BBox] = []
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                labels_tensor = res.get("labels")
                for i in range(len(boxes)):
                    if labels_tensor is not None and i < len(labels_tensor):
                        cls_idx = int(labels_tensor[i])
                        label = _normalise_label(queries[cls_idx] if cls_idx < len(queries) else config.text_prompt)
                    else:
                        label = _normalise_label(queries[0] if queries else config.text_prompt)
                    if classes is not None and label not in classes:
                        continue
                    dets.append(
                        BBox(
                            x1=float(boxes[i][0]),
                            y1=float(boxes[i][1]),
                            x2=float(boxes[i][2]),
                            y2=float(boxes[i][3]),
                            label=label,
                            score=float(scores[i]),
                        )
                    )
                all_dets.append(dets)

            # Keep GPU memory clean between batches
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return all_dets
    except Exception as exc:
        print(f"[WARN] OWLv2 batched inference crashed: {exc}. Returning empty detections.", file=sys.stderr)
        return [[] for _ in images]


def _run_grounding_dino_inference_batched(
    images: list[np.ndarray],
    config: ModelConfig,
    classes: set[str] | None,
) -> list[list[BBox]]:
    """Run Grounding DINO on a batch of images and return per-image detections."""
    import torch
    import gc

    try:
        processor, model = _load_grounding_dino_model(config.model_path, _torch_device(config.device))
        cache_key = f"gdino:{config.model_path}"

        queries = [q.strip() for q in config.text_prompt.split(",")]
        text = " . ".join(queries)

        preprocessed: list[tuple[Image.Image, tuple[int, int]]] = []
        for img in images:
            preprocessed.append(_preprocess_for_zero_shot(img, config.imgsz))

        all_dets: list[list[BBox]] = []
        for batch_start in range(0, len(preprocessed), _ZERO_SHOT_BATCH_SIZE):
            batch_end = min(batch_start + _ZERO_SHOT_BATCH_SIZE, len(preprocessed))
            batch_pils = [preprocessed[i][0] for i in range(batch_start, batch_end)]
            batch_targets = [preprocessed[i][1] for i in range(batch_start, batch_end)]

            inputs = processor(images=batch_pils, text=[text] * len(batch_pils), return_tensors="pt")

            # Work around a GroundingDinoProcessor bug: when all text prompts are identical,
            # it may deduplicate them and return text tensors with batch size 1 instead of N.
            n_images = len(batch_pils)
            for key in ("input_ids", "token_type_ids", "attention_mask"):
                if key in inputs and inputs[key].shape[0] == 1 and n_images > 1:
                    inputs[key] = inputs[key].expand(n_images, -1).contiguous()

            # Explicit device handling: avoid silent CPU fallback that causes device-mismatch crashes
            device = _DEVICE_FALLBACK.get(cache_key, _torch_device(config.device))
            if str(device) != "cpu" and _gpu_free_mb() < 300:
                device = "cpu"
            try:
                model = model.to(device)
            except RuntimeError as exc:
                if str(device) != "cpu":
                    print(f"[WARN] model.to({device}) failed for {config.model_type}: {exc}. Falling back to CPU.", file=sys.stderr)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        model = model.to(device)
                    except RuntimeError:
                        device = "cpu"
                        model = model.to(device)
                else:
                    raise
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = None
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except RuntimeError as exc:
                if str(device) != "cpu":
                    print(f"[WARN] Inference failed on {device} for {config.model_type}: {exc}. Falling back to CPU.", file=sys.stderr)
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                    gc.collect()
                    try:
                        model = model.to(device)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        with torch.no_grad():
                            outputs = model(**inputs)
                    except RuntimeError:
                        device = "cpu"
                        model = model.to(device)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        try:
                            with torch.no_grad():
                                outputs = model(**inputs)
                        except RuntimeError as exc2:
                            print(
                                f"[WARN] CPU inference also failed for {config.model_type} model={config.model_path}: {exc2}. "
                                f"Returning empty detections for this batch.",
                                file=sys.stderr,
                            )
                else:
                    raise
            if outputs is None:
                all_dets.extend([[] for _ in batch_pils])
                continue

            output_device = next((t.device for t in outputs.values() if isinstance(t, torch.Tensor)), torch.device("cpu"))
            target_sizes = torch.Tensor(batch_targets).to(output_device)
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                threshold=config.conf,
                text_threshold=config.conf,
                target_sizes=target_sizes,
            )

            for batch_idx, res in enumerate(results):
                dets: list[BBox] = []
                boxes = res["boxes"].cpu().numpy()
                scores = res["scores"].cpu().numpy()
                text_labels = res.get("text_labels", [])
                for i in range(len(boxes)):
                    raw_label = text_labels[i] if i < len(text_labels) else config.text_prompt
                    label = _normalise_label(raw_label)
                    lowered = label.lower()
                    matched = None
                    for q in queries:
                        if q.lower() in lowered:
                            matched = q
                            break
                    if matched:
                        label = _normalise_label(matched)
                    if classes is not None and label not in classes:
                        continue
                    dets.append(
                        BBox(
                            x1=float(boxes[i][0]),
                            y1=float(boxes[i][1]),
                            x2=float(boxes[i][2]),
                            y2=float(boxes[i][3]),
                            label=label,
                            score=float(scores[i]),
                        )
                    )
                all_dets.append(dets)

            # Keep GPU memory clean between batches
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        return all_dets
    except Exception as exc:
        print(f"[WARN] Grounding DINO batched inference crashed: {exc}. Returning empty detections.", file=sys.stderr)
        return [[] for _ in images]


def _is_k_out_of_range_error(exc: BaseException) -> bool:
    """Check whether an exception is the PyTorch topk 'selected index k out of range' error."""
    return isinstance(exc, RuntimeError) and "selected index k out of range" in str(exc)


def _is_oom_error(exc: BaseException) -> bool:
    """Check whether an exception is a CUDA / MPS out-of-memory error."""
    msg = str(exc).lower()
    return isinstance(exc, RuntimeError) and ("out of memory" in msg or "outofmemory" in msg)


def _gpu_free_mb() -> float:
    """Return free GPU memory in MiB, or 0.0 if unavailable / error."""
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 * 1024)
    except Exception:
        return 0.0


def _run_inference(
    img: np.ndarray,
    config: ModelConfig,
    classes: set[str] | None,
) -> list[BBox]:
    """Dispatch to the correct model type and apply ablations."""
    classes = _normalise_classes(classes)

    # Pre-processing ablation
    input_img = _apply_brightness_correction(img) if config.ablation_brightness else img

    # Base inference
    try:
        if config.model_type == "yolo":
            dets = _run_yolo_inference(input_img, config, classes)
        elif config.model_type == "owlv2":
            dets = _run_owlv2_inference(input_img, config, classes)
        elif config.model_type == "grounding_dino":
            dets = _run_grounding_dino_inference(input_img, config, classes)
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")
    except RuntimeError as exc:
        if _is_k_out_of_range_error(exc):
            print(f"[WARN] Inference failed with 'selected index k out of range' for {config.model_type} "
                  f"model={config.model_path} imgsz={config.imgsz}. Returning empty detections.", file=sys.stderr)
            return []
        if _is_oom_error(exc):
            print(f"[WARN] Inference OOM for {config.model_type} model={config.model_path} imgsz={config.imgsz}. "
                  f"Returning empty detections.", file=sys.stderr)
            return []
        raise

    # Post-processing ablation
    if config.ablation_crop_redetect:
        dets = _apply_crop_redetect(dets, input_img, _run_inference, config, classes)

    return dets


def _class_name(model_names: dict[int, str] | list[str], cid: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(cid, str(cid)))
    if isinstance(model_names, (list, tuple)) and 0 <= cid < len(model_names):
        return str(model_names[cid])
    return str(cid)


# ---------------------------------------------------------------------------
# Tracking helpers
# ---------------------------------------------------------------------------

def _apply_tracking(
    records: list[ImageRecord],
    all_detections: list[list[BBox]],
    tracker_name: str,
    device: str,
    min_conf: float = 0.001,
    images: list[np.ndarray | None] | None = None,
) -> list[list[BBox]]:
    """Apply a multi-object tracker across the image sequence.

    Converts detections to the boxmot Nx6 format, runs the tracker frame-by-frame,
    and converts outputs back to BBox objects (with optional track_id).
    If ``images`` is provided, it is used directly instead of re-reading from disk.
    """
    if tracker_name == "none":
        return all_detections

    def _ensure_reid_weights(device_name: str) -> Path:
        candidate_paths = [
            Path("/usr/local/lib/python3.11/site-packages/models/osnet_x0_25_msmt17.pt"),
            Path("/usr/local/lib/python3.10/dist-packages/models/osnet_x0_25_msmt17.pt"),
            Path("/usr/local/lib/python3.10/site-packages/models/osnet_x0_25_msmt17.pt"),
        ]
        for path in candidate_paths:
            if path.exists():
                return path

        from boxmot.reid.core.reid import ReID  # type: ignore[import-not-found]

        ReID(weights="osnet_x0_25_msmt17.pt", device=_torch_device(device_name))
        for path in candidate_paths:
            if path.exists():
                return path
        return candidate_paths[0]

    # Lazy tracker import (multiple paths for compatibility)
    tracker = None
    if tracker_name == "bytetrack":
        try:
            from boxmot import ByteTrack as _Tracker  # type: ignore[attr-defined]
        except Exception:
            from boxmot.trackers.bytetrack.bytetrack import ByteTrack as _Tracker  # type: ignore[import-not-found]
        tracker = _Tracker(min_conf=min_conf, track_thresh=min_conf)
    elif tracker_name == "botsort":
        try:
            from boxmot import BotSort as _Tracker  # type: ignore[attr-defined]
        except Exception:
            from boxmot.trackers.botsort.botsort import BotSort as _Tracker  # type: ignore[import-not-found]
        import torch as _torch
        tracker = _Tracker(
            reid_weights=_ensure_reid_weights(device),
            device=_torch.device(_torch_device(device)),
            half=False,
            track_high_thresh=max(min_conf, 0.5),
            track_low_thresh=min(min_conf, 0.1),
            new_track_thresh=max(min_conf, 0.6),
            track_buffer=30,
            match_thresh=0.8,
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            cmc_method="sof",
            with_reid=True,
        )
    elif tracker_name in ("strongsort", "ef-strongsort"):
        try:
            from boxmot import StrongSort as _Tracker  # type: ignore[attr-defined]
        except Exception:
            from boxmot.trackers.strongsort.strongsort import StrongSort as _Tracker  # type: ignore[import-not-found]
        import torch as _torch
        _ss_kwargs: dict[str, Any] = dict(
            reid_weights=_ensure_reid_weights(device),
            device=_torch.device(_torch_device(device)),
            half=False,
            min_conf=min_conf,
        )
        if tracker_name == "ef-strongsort":
            # Embedding-free: disable appearance gating so dummy zero embeddings
            # never reject a match.
            _ss_kwargs["max_cos_dist"] = 1.0
        tracker = _Tracker(**_ss_kwargs)
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

    # Build label <-> class_id mapping from all detections
    label_to_id: dict[str, int] = {}
    id_to_label: dict[int, str] = {}
    next_cid = 0
    for dets in all_detections:
        for d in dets:
            if d.label not in label_to_id:
                label_to_id[d.label] = next_cid
                id_to_label[next_cid] = d.label
                next_cid += 1

    tracked: list[list[BBox]] = []
    image_iter = images if images is not None else [None] * len(records)
    for rec, dets, img in zip(records, all_detections, image_iter):
        if img is None:
            img = cv2.imread(str(rec.image_path))
        if img is None:
            img = np.zeros((10, 10, 3), dtype=np.uint8)

        if not dets:
            dets_array = np.empty((0, 6), dtype=np.float32)
        else:
            dets_array = np.array(
                [
                    [d.x1, d.y1, d.x2, d.y2, d.score, label_to_id.get(d.label, 0)]
                    for d in dets
                ],
                dtype=np.float32,
            )

        try:
            if tracker_name == "ef-strongsort":
                # Pass dummy uniform embeddings to skip the expensive ReID forward pass.
                # All identical unit vectors → cosine distance = 0, so appearance never rejects.
                _dummy_embs = np.ones((len(dets_array), 512), dtype=np.float32) / np.sqrt(512)
                outputs = tracker.update(dets_array, img, embs=_dummy_embs)
            else:
                outputs = tracker.update(dets_array, img)
        except RuntimeError as exc:
            if _is_k_out_of_range_error(exc):
                print(
                    f"[WARN] Tracker update failed with 'selected index k out of range' for tracker={tracker_name}. "
                    f"Returning empty detections for this frame.",
                    file=sys.stderr,
                )
                tracked.append([])
                continue
            raise

        frame_dets: list[BBox] = []
        if hasattr(outputs, "xyxy") and len(outputs) > 0:
            # TrackResults (boxmot >= 10)
            for i in range(len(outputs)):
                cls_val = int(outputs.cls[i]) if hasattr(outputs, "cls") else 0
                conf_val = float(outputs.conf[i]) if hasattr(outputs, "conf") else 1.0
                tid_val = int(outputs.id[i]) if hasattr(outputs, "id") else None
                frame_dets.append(
                    BBox(
                        x1=float(outputs.xyxy[i][0]),
                        y1=float(outputs.xyxy[i][1]),
                        x2=float(outputs.xyxy[i][2]),
                        y2=float(outputs.xyxy[i][3]),
                        label=id_to_label.get(cls_val, "person"),
                        score=conf_val,
                        track_id=tid_val,
                    )
                )
        elif isinstance(outputs, np.ndarray) and outputs.size > 0:
            # Older boxmot returns ndarray
            for row in outputs:
                cls_val = int(row[6]) if len(row) > 6 else 0
                conf_val = float(row[5]) if len(row) > 5 else 1.0
                tid_val = int(row[4]) if len(row) > 4 else None
                frame_dets.append(
                    BBox(
                        x1=float(row[0]),
                        y1=float(row[1]),
                        x2=float(row[2]),
                        y2=float(row[3]),
                        label=id_to_label.get(cls_val, "person"),
                        score=conf_val,
                        track_id=tid_val,
                    )
                )

        tracked.append(frame_dets)

    return tracked


# ---------------------------------------------------------------------------
# Main evaluation loop (generic)
# ---------------------------------------------------------------------------

def _natural_sort_key(path: Path) -> list:
    """Natural-sort key so img_2.jpg comes before img_10.jpg."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.name)]


def run_yolo_on_dataset(
    records: list[ImageRecord],
    config: ModelConfig,
    classes: set[str] | None,
    progress_callback: callable | None = None,
) -> EvalResult:
    classes = _normalise_classes(classes)
    result = EvalResult(config=config)
    t0 = time.perf_counter()

    # Sort records by filename so trackers receive frames in ascending order
    records = sorted(records, key=lambda r: _natural_sort_key(r.image_path))

    # Phase 1: load images once and run inference
    loaded_images: list[np.ndarray | None] = []
    all_raw_dets: list[list[BBox]] = []

    if config.model_type in ("owlv2", "grounding_dino"):
        # Batched zero-shot inference: load all images first, then run in batches
        for rec in records:
            img = cv2.imread(str(rec.image_path))
            loaded_images.append(img)

        for batch_start in range(0, len(records), _ZERO_SHOT_BATCH_SIZE):
            batch_end = min(batch_start + _ZERO_SHOT_BATCH_SIZE, len(records))
            batch_imgs: list[np.ndarray] = []
            batch_indices: list[int] = []
            for i in range(batch_start, batch_end):
                img = loaded_images[i]
                if img is None:
                    all_raw_dets.append((i, []))
                else:
                    batch_imgs.append(img)
                    batch_indices.append(i)

            if batch_imgs:
                # Apply brightness ablation if requested (on the batch)
                if config.ablation_brightness:
                    batch_imgs = [_apply_brightness_correction(img) for img in batch_imgs]

                if config.model_type == "owlv2":
                    batch_dets = _run_owlv2_inference_batched(batch_imgs, config, classes)
                else:
                    batch_dets = _run_grounding_dino_inference_batched(batch_imgs, config, classes)

                for bi, det_list in zip(batch_indices, batch_dets):
                    all_raw_dets.append((bi, det_list))

            if progress_callback:
                progress_callback(batch_end, len(records))

        # Rebuild as a dense list ordered by image index.
        dets_by_idx: dict[int, list[BBox]] = dict(all_raw_dets)
        all_raw_dets = [dets_by_idx.get(i, []) for i in range(len(records))]

        # Apply crop-redetect ablation if requested (must run per-image)
        # Skip for zero-shot models: each crop would trigger a full model forward pass,
        # which is prohibitively slow (especially on CPU fallback).
        if config.ablation_crop_redetect and config.model_type in ("owlv2", "grounding_dino"):
            print(
                f"[WARN] Crop-redetect ablation is skipped for {config.model_type} "
                f"because it would require a full forward pass per detection crop.",
                file=sys.stderr,
            )
        elif config.ablation_crop_redetect:
            for i, rec in enumerate(records):
                img = loaded_images[i]
                if img is not None and all_raw_dets[i]:
                    all_raw_dets[i] = _apply_crop_redetect(all_raw_dets[i], img, _run_inference, config, classes)
    else:
        # YOLO / other models: per-image inference ( Ultralytics handles its own batching)
        for idx, rec in enumerate(records):
            img = cv2.imread(str(rec.image_path))
            loaded_images.append(img)
            if img is None:
                all_raw_dets.append([])
            else:
                dets = _run_inference(img, config, classes)
                all_raw_dets.append(dets)
            if progress_callback:
                progress_callback(idx + 1, len(records))

    # Phase 2: optional tracking across the sequence
    all_det_boxes = _apply_tracking(
        records, all_raw_dets, config.tracker, config.device, min_conf=config.conf, images=loaded_images
    )

    # Phase 3: match detections to GT and aggregate metrics
    all_gt_boxes: list[list[BBox]] = []
    total_tp = total_fp = total_fn = 0
    for rec, dets in zip(records, all_det_boxes):
        tp, fp, fn, det_status, det_matched_gt, gt_matched = match_image_detailed(rec.boxes, dets, iou_thresh=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_gt_boxes.append(rec.boxes)
        result.per_image.append(
            {
                "image": str(rec.image_path),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detections": len(dets),
                "gt_boxes": [
                    {"label": b.label, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "matched": bool(gt_matched[i])}
                    for i, b in enumerate(rec.boxes)
                ],
                "det_boxes": [
                    {"label": b.label, "x1": b.x1, "y1": b.y1, "x2": b.x2, "y2": b.y2, "score": b.score, "track_id": b.track_id, "status": det_status[i], "matched_gt": det_matched_gt[i]}
                    for i, b in enumerate(dets)
                ],
            }
        )

    elapsed = time.perf_counter() - t0
    result.elapsed_sec = elapsed
    result.total_tp = total_tp
    result.total_fp = total_fp
    result.total_fn = total_fn
    result.precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    result.recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    if result.precision + result.recall > 0:
        result.f1 = 2 * result.precision * result.recall / (result.precision + result.recall)

    result.ap50 = compute_ap(all_gt_boxes, all_det_boxes, iou_thresh=0.5)
    aps = [
        compute_ap(all_gt_boxes, all_det_boxes, iou_thresh=t)
        for t in np.arange(0.50, 1.00, 0.05)
    ]
    valid_aps = [a for a in aps if not math.isnan(a)]
    result.ap5095 = float(np.mean(valid_aps)) if valid_aps else float("nan")

    return result


# ---------------------------------------------------------------------------
# Auto-labelling helper (generic)
# ---------------------------------------------------------------------------

def auto_label_images(
    image_paths: list[Path],
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    classes: set[str] | None = None,
    model_type: str = "yolo",
    text_prompt: str = "person",
    ablation_brightness: bool = False,
    ablation_crop_redetect: bool = False,
) -> list[dict[str, Any]]:
    config = ModelConfig(
        model_path=model_path,
        model_type=model_type,
        conf=conf,
        imgsz=imgsz,
        device=device,
        text_prompt=text_prompt,
        ablation_brightness=ablation_brightness,
        ablation_crop_redetect=ablation_crop_redetect,
    )
    classes = _normalise_classes(classes)
    records: list[dict[str, Any]] = []
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            records.append({"image_path": str(img_path), "boxes": []})
            continue
        dets = _run_inference(img, config, classes)
        boxes: list[dict] = []
        for d in dets:
            boxes.append(
                {
                    "label": d.label,
                    "x1": round(float(d.x1), 2),
                    "y1": round(float(d.y1), 2),
                    "x2": round(float(d.x2), 2),
                    "y2": round(float(d.y2), 2),
                    "confidence": round(float(d.score), 4),
                }
            )
        records.append({"image_path": str(img_path), "boxes": boxes})
    return records


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def draw_eval_visualisations(
    per_image: list[dict[str, Any]],
    output_dir: Path,
    max_images: int = 12,
) -> list[str]:
    """Draw GT (green) and detection (red) boxes on sample images.
    Returns list of relative filenames that were written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for item in per_image[:max_images]:
        img_path = Path(item["image"])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # GT boxes in green
        for b in item.get("gt_boxes", []):
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{b['label']}", (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Detection boxes in red
        for b in item.get("det_boxes", []):
            x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            score_text = f"{b['label']}:{b.get('score', 0):.2f}"
            tid = b.get('track_id')
            if tid is not None:
                label_text = f"T{tid} {score_text}"
            else:
                label_text = score_text
            cv2.putText(img, label_text, (x1, max(y1 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out_name = img_path.name
        out_path = output_dir / out_name
        # Avoid collisions by appending index if needed
        counter = 1
        original_out_path = out_path
        while out_path.exists():
            stem = original_out_path.stem
            suffix = original_out_path.suffix
            out_path = output_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        cv2.imwrite(str(out_path), img)
        written.append(out_path.name)

    return written


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _config_display_name(config: ModelConfig) -> str:
    parts = [Path(config.model_path).name]
    if config.model_type != "yolo":
        parts.append(config.model_type)
    if config.ablation_brightness:
        parts.append("brightness")
    if config.ablation_crop_redetect:
        parts.append("crop")
    if config.tracker != "none":
        parts.append(config.tracker)
    return "-".join(parts)


def results_to_csv(results: list[EvalResult]) -> str:
    lines = [
        "model,model_type,conf,imgsz,device,brightness,crop,tracker,tp,fp,fn,precision,recall,f1,ap50,ap50_95,elapsed_sec"
    ]
    for r in results:
        cfg = r.config
        lines.append(
            f"{Path(cfg.model_path).name},"
            f"{cfg.model_type},"
            f"{cfg.conf},"
            f"{cfg.imgsz},"
            f"{cfg.device},"
            f"{int(cfg.ablation_brightness)},"
            f"{int(cfg.ablation_crop_redetect)},"
            f"{cfg.tracker},"
            f"{r.total_tp},"
            f"{r.total_fp},"
            f"{r.total_fn},"
            f"{r.precision:.6f},"
            f"{r.recall:.6f},"
            f"{r.f1:.6f},"
            f"{r.ap50:.6f},"
            f"{r.ap5095:.6f},"
            f"{r.elapsed_sec:.3f}"
        )
    return "\n".join(lines) + "\n"


def results_to_json(results: list[EvalResult]) -> list[dict[str, Any]]:
    payload = []
    for r in results:
        cfg = r.config
        payload.append(
            {
                "model": cfg.model_path,
                "model_type": cfg.model_type,
                "conf": cfg.conf,
                "imgsz": cfg.imgsz,
                "device": cfg.device,
                "ablation_brightness": cfg.ablation_brightness,
                "ablation_crop_redetect": cfg.ablation_crop_redetect,
                "tracker": cfg.tracker,
                "text_prompt": cfg.text_prompt,
                "tp": r.total_tp,
                "fp": r.total_fp,
                "fn": r.total_fn,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "ap50": r.ap50,
                "ap50_95": r.ap5095,
                "elapsed_sec": r.elapsed_sec,
                "error": r.error,
                "per_image": r.per_image,
            }
        )
    return payload
