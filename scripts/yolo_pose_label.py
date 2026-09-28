#!/usr/bin/env python3
"""Generate a ground-truth JSON for yolo_pose_evaluate.py from images.

Can use a YOLO model to pre-fill bounding boxes, or emit empty templates
for manual labelling.

Usage examples::

    # Auto-label with a pose model (boxes only, no keypoints in GT)
    python scripts/yolo_pose_label.py \
        --images-dir data/my_dataset/images \
        --model yolov8n-pose.pt \
        --output data/my_dataset/ground_truth.json \
        --classes person

    # Create empty templates for manual labelling
    python scripts/yolo_pose_label.py \
        --images-dir data/my_dataset/images \
        --output data/my_dataset/ground_truth.json \
        --empty
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(directory: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = sorted(directory.glob(pattern))
    return [f for f in files if f.suffix.lower() in SUPPORTED_EXTS]


def _normalise_label(label: str) -> str:
    label = str(label).strip().lower()
    if label in ("person", "0", "human"):
        return "person"
    return label


def _class_name(model_names: dict[int, str] | list[str], cid: int) -> str:
    if isinstance(model_names, dict):
        return str(model_names.get(cid, str(cid)))
    if isinstance(model_names, (list, tuple)) and 0 <= cid < len(model_names):
        return str(model_names[cid])
    return str(cid)


def generate_empty(images: list[Path]) -> list[dict]:
    return [
        {
            "image_path": str(img),
            "boxes": [],
        }
        for img in images
    ]


def generate_from_model(
    images: list[Path],
    model_path: str,
    conf: float,
    imgsz: int,
    device: str,
    classes: set[str] | None,
) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            f"[ERROR] ultralytics is required for auto-labelling: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    model = YOLO(model_path)
    model_names = getattr(model, "names", {})

    records: list[dict] = []
    for img_path in images:
        res = model.predict(
            source=str(img_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]

        boxes: list[dict] = []
        if res.boxes is not None:
            for box in res.boxes:
                xyxy = box.xyxy.cpu().numpy().flatten()
                score = float(box.conf.cpu().item())
                cid = int(box.cls.cpu().item())
                label = _normalise_label(_class_name(model_names, cid))
                if classes is not None and label not in classes:
                    continue
                boxes.append(
                    {
                        "label": label,
                        "x1": round(float(xyxy[0]), 2),
                        "y1": round(float(xyxy[1]), 2),
                        "x2": round(float(xyxy[2]), 2),
                        "y2": round(float(xyxy[3]), 2),
                        "confidence": round(score, 4),
                    }
                )

        records.append(
            {
                "image_path": str(img_path),
                "boxes": boxes,
            }
        )
        print(f"  {img_path.name}: {len(boxes)} box(es)")

    return records


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ground-truth JSON for yolo_pose_evaluate.py."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan images recursively.",
    )
    parser.add_argument(
        "--empty",
        action="store_true",
        help="Emit empty box lists instead of running a model.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="YOLO model to use for auto-labelling (default: yolov8n-pose.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for auto-labelling (default: 0.25).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size for auto-labelling (default: 640).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Torch device (default: 0).",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Keep only these class names (default: all).",
    )
    parser.add_argument(
        "--relative-to",
        type=Path,
        default=None,
        help="Store image paths relative to this directory (default: absolute).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    if not args.images_dir.is_dir():
        print(f"[ERROR] Not a directory: {args.images_dir}", file=sys.stderr)
        return 1

    images = find_images(args.images_dir, recursive=args.recursive)
    if not images:
        print("[ERROR] No images found.", file=sys.stderr)
        return 1

    print(f"[INFO] Found {len(images)} image(s).")

    classes: set[str] | None = None
    if args.classes:
        classes = {_normalise_label(c) for c in args.classes}

    if args.empty:
        records = generate_empty(images)
    else:
        print(f"[INFO] Running {args.model} for auto-labelling...")
        records = generate_from_model(
            images,
            model_path=args.model,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            classes=classes,
        )

    if args.relative_to is not None:
        for rec in records:
            p = Path(rec["image_path"])
            try:
                rec["image_path"] = str(p.relative_to(args.relative_to))
            except ValueError:
                pass  # keep absolute if not under relative_to

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2))
    print(f"[INFO] Ground truth saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
