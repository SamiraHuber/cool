from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import cv2
import numpy as np


PERSON_PARTS = ("upper_body", "lower_body", "feet")


@dataclass(frozen=True)
class AttributeConfig:
    enabled: bool = True
    quality_enabled: bool = True
    color_enabled: bool = True
    person_parts_enabled: bool = True
    part_embeddings_enabled: bool = False
    super_resolution_enabled: bool = False
    super_resolution_method: str = "lanczos"
    super_resolution_min_side: int = 96
    segmentation_mode: str = "mask"
    person_class_id: int = 0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _valid_image(image: Optional[np.ndarray]) -> bool:
    return image is not None and image.ndim == 3 and image.shape[0] > 0 and image.shape[1] > 0


def resize_for_embedding(
    image_bgr: np.ndarray,
    *,
    enabled: bool,
    method: str = "lanczos",
    min_side: int = 96,
) -> tuple[np.ndarray, Dict[str, Any]]:
    if not _valid_image(image_bgr) or not enabled:
        return image_bgr, {"enabled": False, "method": "none", "scale": 1.0}

    height, width = image_bgr.shape[:2]
    shortest = min(height, width)
    if shortest >= max(1, int(min_side)):
        return image_bgr, {"enabled": False, "method": "none", "scale": 1.0}

    scale = float(min_side) / float(max(1, shortest))
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    normalized_method = str(method or "lanczos").strip().lower()

    # Real-ESRGAN is intentionally a hook here. The runtime can enable it later
    # without changing stored metadata or evaluation scripts.
    interpolation = cv2.INTER_CUBIC if normalized_method == "real_esrgan" else cv2.INTER_LANCZOS4
    resized = cv2.resize(image_bgr, new_size, interpolation=interpolation)
    stored_method = "real_esrgan_fallback_cubic" if normalized_method == "real_esrgan" else "lanczos"
    return resized, {"enabled": True, "method": stored_method, "scale": scale}


def compute_quality_score(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    detection_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    if not _valid_image(image_bgr):
        return {
            "quality_score": 0.0,
            "blur_score": 0.0,
            "brightness_score": 0.0,
            "size_score": 0.0,
            "mask_coverage": 0.0,
            "detection_confidence": _safe_float(detection_confidence),
        }

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    valid_mask = None
    if mask is not None and mask.shape[:2] == image_bgr.shape[:2]:
        valid_mask = mask > 0
        if np.count_nonzero(valid_mask) < 8:
            valid_mask = None

    pixels_gray = gray[valid_mask] if valid_mask is not None else gray.reshape(-1)
    brightness = float(np.mean(pixels_gray)) if pixels_gray.size else 0.0
    brightness_score = 1.0 - abs(brightness - 128.0) / 128.0
    brightness_score = _clamp01(brightness_score)

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = _clamp01(blur_var / 500.0)
    size_score = _clamp01((width * height) / float(160 * 320))
    mask_coverage = 1.0
    if valid_mask is not None:
        mask_coverage = _clamp01(float(np.count_nonzero(valid_mask)) / float(width * height))

    confidence = _safe_float(detection_confidence, 0.65)
    quality = (
        0.30 * blur_score
        + 0.25 * brightness_score
        + 0.25 * size_score
        + 0.10 * mask_coverage
        + 0.10 * _clamp01(confidence)
    )
    return {
        "quality_score": _clamp01(quality),
        "blur_score": blur_score,
        "blur_laplacian_var": blur_var,
        "brightness_score": brightness_score,
        "mean_brightness": brightness,
        "size_score": size_score,
        "width": int(width),
        "height": int(height),
        "mask_coverage": mask_coverage,
        "detection_confidence": confidence,
    }


def _coarse_color_name(hue: float, sat: float, value: float) -> str:
    if value < 45:
        return "black"
    if value > 215 and sat < 35:
        return "white"
    if sat < 30:
        return "gray"
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 22:
        return "orange"
    if hue < 34:
        return "yellow"
    if hue < 82:
        return "green"
    if hue < 100:
        return "cyan"
    if hue < 128:
        return "blue"
    if hue < 150:
        return "purple"
    return "pink"


def extract_dominant_colors(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    max_colors: int = 4,
) -> Dict[str, Any]:
    if not _valid_image(image_bgr):
        return {"colors": [], "histogram": {}, "pixel_count": 0}

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    valid = np.ones(hsv.shape[:2], dtype=bool)
    if mask is not None and mask.shape[:2] == image_bgr.shape[:2]:
        valid &= mask > 0

    # Drop extreme shadows/highlights where color names become unstable.
    valid &= hsv[:, :, 2] > 25
    valid &= hsv[:, :, 2] < 245
    pixels = hsv[valid]
    if pixels.size == 0:
        return {"colors": [], "histogram": {}, "pixel_count": 0}

    names: list[str] = []
    for hue, sat, value in pixels:
        names.append(_coarse_color_name(float(hue), float(sat), float(value)))

    counts: Dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    total = float(sum(counts.values()))
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    colors = [
        {"name": name, "fraction": round(count / total, 4)}
        for name, count in ranked[: max(1, int(max_colors))]
    ]
    return {
        "colors": colors,
        "histogram": {name: round(count / total, 4) for name, count in ranked},
        "pixel_count": int(total),
    }


def split_person_parts(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    if not _valid_image(image_bgr):
        return {}

    height, width = image_bgr.shape[:2]
    splits = {
        "upper_body": (0, int(round(height * 0.48))),
        "lower_body": (int(round(height * 0.40)), int(round(height * 0.86))),
        "feet": (int(round(height * 0.78)), height),
    }
    parts: Dict[str, Dict[str, Any]] = {}
    for name, (y1, y2) in splits.items():
        y1 = max(0, min(height, y1))
        y2 = max(y1 + 1, min(height, y2))
        crop = image_bgr[y1:y2, 0:width]
        part_mask = mask[y1:y2, 0:width] if mask is not None and mask.shape[:2] == image_bgr.shape[:2] else None
        parts[name] = {
            "image": crop,
            "mask": part_mask,
            "bbox": [0, y1, width, y2],
        }
    return parts


def extract_visual_attributes(
    image_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    *,
    class_id: Optional[int],
    detection_confidence: Optional[float],
    config: AttributeConfig,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    if not config.enabled:
        return {"enabled": False}, {}

    attributes: Dict[str, Any] = {
        "enabled": True,
        "class_id": class_id,
        "extractors": {
            "quality": bool(config.quality_enabled),
            "color": bool(config.color_enabled),
            "person_parts": bool(config.person_parts_enabled),
            "part_embeddings": bool(config.part_embeddings_enabled),
            "super_resolution": bool(config.super_resolution_enabled),
            "segmentation_mode": config.segmentation_mode,
        },
    }

    if config.quality_enabled:
        attributes["quality"] = compute_quality_score(
            image_bgr,
            mask,
            detection_confidence=detection_confidence,
        )
        attributes["quality_score"] = attributes["quality"]["quality_score"]

    if config.color_enabled:
        attributes["colors"] = extract_dominant_colors(image_bgr, mask)

    part_images: Dict[str, Dict[str, Any]] = {}
    if config.person_parts_enabled and class_id is not None and int(class_id) == int(config.person_class_id):
        part_images = split_person_parts(image_bgr, mask)
        part_attrs: Dict[str, Any] = {}
        for name, part in part_images.items():
            part_entry: Dict[str, Any] = {"bbox": part["bbox"]}
            if config.quality_enabled:
                part_entry["quality"] = compute_quality_score(
                    part["image"],
                    part.get("mask"),
                    detection_confidence=detection_confidence,
                )
                part_entry["quality_score"] = part_entry["quality"]["quality_score"]
            if config.color_enabled:
                part_entry["colors"] = extract_dominant_colors(part["image"], part.get("mask"))
            part_attrs[name] = part_entry
        attributes["parts"] = part_attrs

    return attributes, part_images
