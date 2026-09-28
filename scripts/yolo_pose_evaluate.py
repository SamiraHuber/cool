#!/usr/bin/env python3
"""Evaluate YOLO pose (and detection) models against a labeled bounding-box dataset.

Usage example::

    python scripts/yolo_pose_evaluate.py \
        --ground-truth data/pose_ground_truth.json \
        --models yolov8n-pose.pt yolov8s-pose.pt yolo11n-pose.pt \
        --conf-thresholds 0.25 0.5 \
        --imgsz 640 1280 \
        --output-dir reports/yolo_pose_eval \
        --device cuda

Ground-truth JSON format
------------------------
A JSON list of image records.  Coordinates may be ``xyxy`` (default) or
``xywh`` (set ``--gt-format xywh``).

.. code-block:: json

    [
      {
        "image_path": "images/frame_001.jpg",
        "boxes": [
          {"label": "person", "x1": 100, "y1": 200, "x2": 150, "y2": 300},
          {"label": "person", "x1": 300, "y1": 100, "x2": 380, "y2": 250}
        ]
      }
    ]

The label ``"person"`` is normalised to ``0`` for COCO-class evaluation.
Any other label is kept as-is but will only match predictions of the same
class name / class id.

Metrics produced
----------------
* Per-image TP / FP / FN
* Precision, Recall, F1
* AP@50 and AP@50:95 (COCO-style 101-point interpolation)
* mAP across all classes
* A CSV with one row per (model, conf, imgsz) configuration
* A JSON with full per-image breakdowns
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

# ultralytics is imported lazily inside the evaluator so that --help works
# even when the package is not installed in the current environment.


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
        records.append(
            ImageRecord(image_path=Path(item["image_path"]), boxes=boxes)
        )
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
    # Sort detections by score descending
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


def compute_ap(
    gts_all: list[list[BBox]],
    dets_all: list[list[BBox]],
    iou_thresh: float,
    label: str | None = None,
) -> float:
    """Compute Average Precision for a single IoU threshold (COCO 101-point)."""
    # Collect all detections across images
    all_scores: list[tuple[float, bool]] = []
    n_gt = 0
    for gts_img, dets_img in zip(gts_all, dets_all):
        gts = [g for g in gts_img if label is None or g.label == label]
        dets = [d for d in dets_img if label is None or d.label == label]
        if not gts and not dets:
            continue
        n_gt += len(gts)
        gt_matched = [False] * len(gts)
        dets_sorted = sorted(dets, key=lambda d: d.score, reverse=True)
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
            if best_iou >= iou_thresh and best_gt >= 0:
                gt_matched[best_gt] = True
                all_scores.append((det.score, True))
            else:
                all_scores.append((det.score, False))

    if n_gt == 0:
        return float("nan")

    all_scores.sort(key=lambda x: x[0], reverse=True)
    tp_cumsum = np.cumsum([s[1] for s in all_scores])
    fp_cumsum = np.cumsum([not s[1] for s in all_scores])
    recalls = tp_cumsum / n_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)

    # 101-point interpolation (COCO standard)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        if np.sum(recalls >= t) == 0:
            p = 0.0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 101.0
    return float(ap)


# ---------------------------------------------------------------------------
# YOLO inference
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    model_path: str
    conf: float
    imgsz: int
    device: str
    iou: float  # NMS IoU


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


def _class_name(model_names: dict[int, str] | list[str], cid: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(cid, str(cid)))
    if isinstance(model_names, (list, tuple)) and 0 <= cid < len(model_names):
        return str(model_names[cid])
    return str(cid)


def run_yolo_on_dataset(
    records: list[ImageRecord],
    config: ModelConfig,
    classes: set[str] | None,
) -> EvalResult:
    classes = _normalise_classes(classes)
    from ultralytics import YOLO

    model = YOLO(config.model_path)
    model_names = getattr(model, "names", {})

    all_gt_boxes: list[list[BBox]] = []
    all_det_boxes: list[list[BBox]] = []
    result = EvalResult(config=config)
    total_tp = total_fp = total_fn = 0

    t0 = time.perf_counter()
    for rec in records:
        img = cv2.imread(str(rec.image_path))
        if img is None:
            print(f"[WARN] Could not load {rec.image_path}", file=sys.stderr)
            all_gt_boxes.append(rec.boxes)
            all_det_boxes.append([])
            result.per_image.append(
                {
                    "image": str(rec.image_path),
                    "tp": 0,
                    "fp": 0,
                    "fn": len(rec.boxes),
                    "detections": 0,
                }
            )
            continue

        # ONNX Runtime models frequently fail on GPU due to missing data-transfer
        # providers; force CPU for .onnx files.
        device = config.device
        if str(config.model_path).lower().endswith(".onnx"):
            device = "cpu"

        try:
            res = model.predict(
                source=img,
                conf=config.conf,
                imgsz=config.imgsz,
                device=device,
                iou=config.iou,
                verbose=False,
            )[0]
        except RuntimeError as exc:
            if "selected index k out of range" in str(exc):
                print(
                    f"[WARN] Inference failed with 'selected index k out of range' for "
                    f"{config.model_path} on {rec.image_path}. Returning empty detections.",
                    file=sys.stderr,
                )
                all_gt_boxes.append(rec.boxes)
                all_det_boxes.append([])
                result.per_image.append(
                    {
                        "image": str(rec.image_path),
                        "tp": 0,
                        "fp": 0,
                        "fn": len(rec.boxes),
                        "detections": 0,
                    }
                )
                continue
            raise

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

        tp, fp, fn = match_image(rec.boxes, dets, iou_thresh=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_gt_boxes.append(rec.boxes)
        all_det_boxes.append(dets)
        result.per_image.append(
            {
                "image": str(rec.image_path),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "detections": len(dets),
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

    # AP@50
    result.ap50 = compute_ap(all_gt_boxes, all_det_boxes, iou_thresh=0.5)
    # AP@50:95
    aps = [
        compute_ap(all_gt_boxes, all_det_boxes, iou_thresh=t)
        for t in np.arange(0.50, 1.00, 0.05)
    ]
    valid_aps = [a for a in aps if not math.isnan(a)]
    result.ap5095 = float(np.mean(valid_aps)) if valid_aps else float("nan")

    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results: list[EvalResult]) -> None:
    print("\n" + "=" * 100)
    print(
        f"{'Model':<30} {'Conf':>6} {'Imgsz':>6} {'TP':>5} {'FP':>5} {'FN':>5} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'AP50':>7} {'AP50:95':>8} {'Time':>8}"
    )
    print("-" * 100)
    for r in results:
        name = Path(r.config.model_path).name
        print(
            f"{name:<30} {r.config.conf:>6.2f} {r.config.imgsz:>6} "
            f"{r.total_tp:>5} {r.total_fp:>5} {r.total_fn:>5} "
            f"{r.precision:>6.3f} {r.recall:>6.3f} {r.f1:>6.3f} "
            f"{r.ap50:>7.3f} {r.ap5095:>8.3f} {r.elapsed_sec:>7.1f}s"
        )
    print("=" * 100)


def save_csv(results: list[EvalResult], path: Path) -> None:
    lines = [
        "model,conf,imgsz,device,tp,fp,fn,precision,recall,f1,ap50,ap50_95,elapsed_sec"
    ]
    for r in results:
        lines.append(
            f"{Path(r.config.model_path).name},"
            f"{r.config.conf},"
            f"{r.config.imgsz},"
            f"{r.config.device},"
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
    path.write_text("\n".join(lines) + "\n")
    print(f"[INFO] CSV saved to {path}")


def save_json(results: list[EvalResult], path: Path) -> None:
    payload = []
    for r in results:
        payload.append(
            {
                "model": r.config.model_path,
                "conf": r.config.conf,
                "imgsz": r.config.imgsz,
                "device": r.config.device,
                "tp": r.total_tp,
                "fp": r.total_fp,
                "fn": r.total_fn,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "ap50": r.ap50,
                "ap50_95": r.ap5095,
                "elapsed_sec": r.elapsed_sec,
                "per_image": r.per_image,
            }
        )
    path.write_text(json.dumps(payload, indent=2))
    print(f"[INFO] JSON saved to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate YOLO pose / detection models against labeled ground-truth boxes."
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to ground-truth JSON (see script docstring for format).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more YOLO model paths / names (e.g. yolov8n-pose.pt yolo11n-pose.pt).",
    )
    parser.add_argument(
        "--conf-thresholds",
        nargs="+",
        type=float,
        default=[0.25],
        help="Confidence thresholds to sweep (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640],
        help="Inference sizes to sweep (default: 640).",
    )
    parser.add_argument(
        "--device",
        default="0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu",
        help="Torch device string (default: '0' if CUDA available else 'cpu').",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold passed to YOLO (default: 0.7).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Filter to these class names only (default: all).",
    )
    parser.add_argument(
        "--gt-format",
        choices=["xyxy", "xywh"],
        default="xyxy",
        help="Ground-truth box coordinate format (default: xyxy).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/yolo_pose_eval"),
        help="Directory for results (default: reports/yolo_pose_eval).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Draw GT (green) and predictions (red) and save to output-dir/visualisations/.",
    )
    return parser.parse_args(argv)


def _draw_visualisations(
    records: list[ImageRecord],
    results: list[EvalResult],
    out_dir: Path,
) -> None:
    """Save side-by-side annotated images for each configuration."""
    vis_dir = out_dir / "visualisations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Group results by model name for quick lookup
    result_map: dict[str, EvalResult] = {}
    for r in results:
        key = f"{Path(r.config.model_path).name}_c{r.config.conf}_s{r.config.imgsz}"
        result_map[key] = r

    # We need to re-run inference to get actual boxes because we didn't store
    # them in EvalResult.  For simplicity, skip storing all boxes and just
    # re-run once per unique config.  This is acceptable for small datasets.
    from ultralytics import YOLO

    for r in results:
        model = YOLO(r.config.model_path)
        model_names = getattr(model, "names", {})
        key = f"{Path(r.config.model_path).name}_c{r.config.conf}_s{r.config.imgsz}"
        sub = vis_dir / key
        sub.mkdir(exist_ok=True)

        for rec in records:
            img = cv2.imread(str(rec.image_path))
            if img is None:
                continue
            out = img.copy()
            # GT in green
            for gt in rec.boxes:
                cv2.rectangle(
                    out,
                    (int(gt.x1), int(gt.y1)),
                    (int(gt.x2), int(gt.y2)),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    out,
                    f"GT:{gt.label}",
                    (int(gt.x1), int(gt.y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            # ONNX Runtime models frequently fail on GPU due to missing data-transfer
            # providers; force CPU for .onnx files.
            device = r.config.device
            if str(r.config.model_path).lower().endswith(".onnx"):
                device = "cpu"

            res = model.predict(
                source=img,
                conf=r.config.conf,
                imgsz=r.config.imgsz,
                device=device,
                iou=r.config.iou,
                verbose=False,
            )[0]
            if res.boxes is not None:
                for box in res.boxes:
                    xyxy = box.xyxy.cpu().numpy().flatten()
                    score = float(box.conf.cpu().item())
                    cid = int(box.cls.cpu().item())
                    label = _normalise_label(_class_name(model_names, cid))
                    cv2.rectangle(
                        out,
                        (int(xyxy[0]), int(xyxy[1])),
                        (int(xyxy[2]), int(xyxy[3])),
                        (0, 0, 255),
                        2,
                    )
                    cv2.putText(
                        out,
                        f"{label}:{score:.2f}",
                        (int(xyxy[0]), int(xyxy[1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )

            out_path = sub / rec.image_path.name
            cv2.imwrite(str(out_path), out)

    print(f"[INFO] Visualisations saved to {vis_dir}")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.ground_truth.exists():
        print(f"[ERROR] Ground-truth file not found: {args.ground_truth}", file=sys.stderr)
        return 1

    records = load_ground_truth(args.ground_truth, fmt=args.gt_format)
    if not records:
        print("[ERROR] No images loaded from ground truth.", file=sys.stderr)
        return 1

    print(f"[INFO] Loaded {len(records)} images with ground-truth boxes.")
    total_gt_boxes = sum(len(r.boxes) for r in records)
    print(f"[INFO] Total GT boxes: {total_gt_boxes}")

    classes: set[str] | None = None
    if args.classes:
        classes = {_normalise_label(c) for c in args.classes}
        print(f"[INFO] Filtering to classes: {classes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[EvalResult] = []
    combinations = [
        ModelConfig(
            model_path=m,
            conf=c,
            imgsz=s,
            device=args.device,
            iou=args.nms_iou,
        )
        for m in args.models
        for c in args.conf_thresholds
        for s in args.imgsz
    ]
    print(f"[INFO] Running {len(combinations)} configuration(s)...\n")

    for cfg in combinations:
        print(
            f"[RUN] model={Path(cfg.model_path).name} conf={cfg.conf} imgsz={cfg.imgsz} device={cfg.device}"
        )
        res = run_yolo_on_dataset(records, cfg, classes)
        print(
            f"      -> TP={res.total_tp} FP={res.total_fp} FN={res.total_fn} "
            f"P={res.precision:.3f} R={res.recall:.3f} F1={res.f1:.3f} "
            f"AP50={res.ap50:.3f} AP50:95={res.ap5095:.3f} ({res.elapsed_sec:.1f}s)"
        )
        results.append(res)

    print_summary(results)
    save_csv(results, args.output_dir / "results.csv")
    save_json(results, args.output_dir / "results.json")

    if args.visualize:
        _draw_visualisations(records, results, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
