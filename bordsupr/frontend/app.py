
import io
import logging
logger = logging.getLogger(__name__)
from fastapi import FastAPI, HTTPException, Query, Request

# --- Ensure FastAPI app is defined before any usage ---
app = FastAPI()

from fastapi.responses import Response, HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
import asyncio
from pydantic import BaseModel, Field
import psycopg2
import os
import json
import shlex
import pathlib
import socket
import subprocess
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import heapq
import math
import random
import re
import tempfile
import threading
import time
import uuid
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from PIL import ImageFilter, ImageStat
import base64
from typing import Callable, List

def pil_image_to_base64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@app.post("/api/yolo-probe/crops")
def yolo_probe_crops(
    path: str = Query(...),
    detect_mode: str = Query("hand_crop"),
    preprocess_clahe: bool = Query(False),
    preprocess_gamma: float = Query(1.0),
    preprocess_sharpen: bool = Query(False),
    preprocess_denoise: bool = Query(False),
    preprocess_auto_brighten: bool = Query(False),
    person_model_path: str = Query(""),
    hand_model_path: str = Query(""),
    person_confidence: float = Query(0.35),
    hand_confidence: float = Query(0.25),
    iou_threshold: float = Query(0.5),
    person_pad_px: int = Query(30),
    hand_pad_px: int = Query(20),
    hand_class_id: int = Query(0),
):
    """
    Returns all person and hand crops for a given image as base64 PNGs.
    """
    resolved = str(Path(path).resolve())
    if not any(resolved.startswith(folder) for folder in _YOLO_PROBE_ALLOWED_FOLDERS):
        raise HTTPException(status_code=403, detail="Image path is not in a scanned folder")
    p = Path(resolved)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    image = PILImage.open(str(p)).convert("RGB")
    if preprocess_clahe or abs(preprocess_gamma - 1.0) > 0.01 or preprocess_sharpen or preprocess_denoise or preprocess_auto_brighten:
        image = _yolo_probe_preprocess(
            image,
            clahe=preprocess_clahe,
            gamma=preprocess_gamma,
            sharpen=preprocess_sharpen,
            denoise=preprocess_denoise,
            auto_brighten=preprocess_auto_brighten,
        )
    # Load models
    model_path = ""
    session = None
    person_session = None
    hand_session = None
    # Use the same logic as detect endpoint
    if detect_mode == "hand_crop":
        from .app import _load_yolo_session
        model_path = person_model_path or hand_model_path or ""
        # fallback to main model if not provided
        if not model_path:
            raise HTTPException(status_code=400, detail="Model path required for crops")
        person_session = _load_yolo_session(person_model_path or model_path)
        hand_session = _load_yolo_session(hand_model_path or model_path)
        obj_session = _load_yolo_session(model_path)
        person_dets, hand_dets, obj_dets = _run_yolo_hand_crop_objects(
            image,
            person_session=person_session,
            hand_session=hand_session,
            obj_session=obj_session,
            person_confidence=person_confidence,
            hand_confidence=hand_confidence,
            obj_confidence=0.25,
            iou_threshold=iou_threshold,
            max_obj_detections=10,
            obj_classes=None,
            person_pad_px=person_pad_px,
            hand_pad_px=hand_pad_px,
            hand_class_id=hand_class_id,
        )
        crops = []
        # Person crops
        for det in person_dets:
            x1, y1, x2, y2 = map(int, det["bbox"])
            px1 = max(0, x1 - person_pad_px)
            py1 = max(0, y1 - person_pad_px)
            px2 = min(image.width, x2 + person_pad_px)
            py2 = min(image.height, y2 + person_pad_px)
            crop = image.crop((px1, py1, px2, py2))
            crops.append({"type": "person", "bbox": [px1, py1, px2, py2], "img_b64": pil_image_to_base64(crop)})
        # Hand crops
        for det in hand_dets:
            x1, y1, x2, y2 = map(int, det["bbox"])
            hx1 = max(0, x1 - hand_pad_px)
            hy1 = max(0, y1 - hand_pad_px)
            hx2 = min(image.width, x2 + hand_pad_px)
            hy2 = min(image.height, y2 + hand_pad_px)
            crop = image.crop((hx1, hy1, hx2, hy2))
            crops.append({"type": "hand", "bbox": [hx1, hy1, hx2, hy2], "img_b64": pil_image_to_base64(crop)})
        return JSONResponse(content={"crops": crops})
    else:
        raise HTTPException(status_code=400, detail="Only hand_crop mode supported for crop gallery.")
import io
import base64

try:
    PIL_RESAMPLE_BILINEAR = PILImage.Resampling.BILINEAR
    PIL_RESAMPLE_LANCZOS = PILImage.Resampling.LANCZOS
    PIL_RESAMPLE_NEAREST = PILImage.Resampling.NEAREST
except AttributeError:
    PIL_RESAMPLE_BILINEAR = PILImage.BILINEAR
    PIL_RESAMPLE_LANCZOS = PILImage.LANCZOS
    PIL_RESAMPLE_NEAREST = PILImage.NEAREST

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")
LIDAR_OCCUPANCY_MAP_PATH = os.getenv("LIDAR_OCCUPANCY_MAP_PATH", "/shared/lidar_occupancy_map.json")
TOOLBOX_MAP_PATH = os.getenv("TOOLBOX_MAP_PATH", "/shared/toolbox_map_snapshot.json")
NAV2_GLOBAL_COSTMAP_PATH = os.getenv("NAV2_GLOBAL_COSTMAP_PATH", "/shared/nav2_global_costmap_snapshot.json")
MAP_SNAPSHOT_PATH = os.getenv("MAP_SNAPSHOT_PATH", "/shared/map_snapshot.json")
VELODYNE_SCAN_PATH = os.getenv("VELODYNE_SCAN_PATH", "/shared/velodyne_scan_snapshot.json")
VLM_API_URL = os.getenv("VLM_API_URL", "http://42b9e761e7e5:8000/v1")
VLM_MODEL = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
USE_KIMI = os.getenv("USE_KIMI", "").lower() in ("1", "true", "yes")
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k2.6")
USE_GEMINI = os.getenv("USE_GEMINI", "").lower() in ("1", "true", "yes")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
WORLD_FRAME_ALIASES = {"odom", "spot/odom", "spot/vision"}
VIDEO_PUBLISHER_CONTAINER = os.getenv("VIDEO_PUBLISHER_CONTAINER", "bordsupr")
SLAM_TAB_DIR = Path(os.getenv("SLAM_TAB_DIR", "/shared/slam_tab"))
SLAM_TAB_REQUEST_PATH = SLAM_TAB_DIR / "request.json"
SLAM_TAB_STATUS_PATH = SLAM_TAB_DIR / "status.json"
SLAM_TAB_MAP_PATH = SLAM_TAB_DIR / "map.json"
SLAM_TAB_LOCAL_COSTMAP_PATH = SLAM_TAB_DIR / "local_costmap.json"
SLAM_TAB_ROBOT_POSE_PATH = SLAM_TAB_DIR / "robot_pose.json"
SLAM_TAB_CLAIM_PATH = SLAM_TAB_DIR / "claim.json"
ROBOT_PATH_FILE = Path(os.getenv("ROBOT_PATH_FILE", "/shared/robot_path.jsonl"))
SLAM_TAB_SAVE_DIR = SLAM_TAB_DIR / "saved"
VIDEO_PUBLISHER_SCRIPT_PATH = os.getenv(
    "VIDEO_PUBLISHER_SCRIPT_PATH",
    "/workspace/src/bordsupr/bordsupr/video_publisher.py",
)
VIDEO_PUBLISHER_RGB_TOPIC = os.getenv("VIDEO_PUBLISHER_RGB_TOPIC", "/spot/camera/frontleft/image_rotated")
VIDEO_PUBLISHER_PID_FILE = os.getenv("VIDEO_PUBLISHER_PID_FILE", "/tmp/bordsupr_video_publisher.pid")
VIDEO_PUBLISHER_LOG_FILE = os.getenv("VIDEO_PUBLISHER_LOG_FILE", "/tmp/bordsupr_video_publisher.log")
FACE_DETECTOR_PID_FILE = os.getenv("FACE_DETECTOR_PID_FILE", "/tmp/bordsupr_face_detector.pid")
VIDEO_PUBLISHER_DEFAULT_FRAME_DIR = os.getenv(
    "VIDEO_PUBLISHER_DEFAULT_FRAME_DIR",
    str(Path.cwd() / "bordsupr/runtime/bordsupr/resource/backup_image_queue_20260209_121144/images"),
)
VIDEO_PUBLISHER_HOST_RUNTIME_ROOT = os.getenv(
    "VIDEO_PUBLISHER_HOST_RUNTIME_ROOT",
    str(Path.cwd() / "bordsupr/runtime"),
)
VIDEO_PUBLISHER_CONTAINER_RUNTIME_ROOT = os.getenv("VIDEO_PUBLISHER_CONTAINER_RUNTIME_ROOT", "/workspace/src")
VIDEO_PUBLISHER_EXTRA_RUNTIME_MOUNTS = os.getenv("VIDEO_PUBLISHER_EXTRA_RUNTIME_MOUNTS", "")
VALID_HOME_TABS = {"scenes", "objects", "persons", "interactions", "map", "robot_view"}
DEFAULT_DATASET_PATH = os.getenv(
    "DEFAULT_DATASET_PATH",
    "exploration/office_time_log_2026-05-12.json",
)
DB_TABLES = [
    "maps",
    "rooms",
    "scenes",
    "objects",
    "object_observations",
    "face_observations",
    "interactions",
]
CLUSTER_TESTSET_STORE_DIR = pathlib.Path(
    os.getenv("CLUSTER_TESTSET_STORE_DIR", "/shared/cluster_testsets")
)
CLUSTER_TESTSET_DEFAULT_DATASET_ROOT = os.getenv(
    "CLUSTER_TESTSET_DEFAULT_DATASET_ROOT",
    "data/Market-1501-v15.09.15",
)
CLUSTER_TESTSET_YOLO_MODEL = os.getenv("CLUSTER_TESTSET_YOLO_MODEL", "")

_video_publisher_last_request: dict = {}
_yolo_compare_models: dict = {}

NAVIGATION_REQUEST_PATH = pathlib.Path(
    os.getenv("NAVIGATION_REQUEST_PATH", "/shared/navigation_request.json")
)
NAVIGATION_STATUS_PATH = pathlib.Path(
    os.getenv("NAVIGATION_STATUS_PATH", "/shared/navigation_status.json")
)
NAVIGATION_CANCEL_REQUEST_PATH = pathlib.Path(
    os.getenv("NAVIGATION_CANCEL_REQUEST_PATH", "/shared/navigation_cancel_request.json")
)
CMD_VEL_STATUS_PATH = pathlib.Path(
    os.getenv("CMD_VEL_STATUS_PATH", "/shared/cmd_vel_status.json")
)
ROBOT_ODOM_STALE_SEC = float(os.getenv("ROBOT_ODOM_STALE_SEC", "3.0"))
NAVIGATION_QUEUE_STALE_SEC = float(
    os.getenv("NAVIGATION_QUEUE_STALE_SEC", "10.0")
)
MAP_RESET_REQUEST_PATH = pathlib.Path(
    os.getenv("MAP_RESET_REQUEST_PATH", "/shared/map_reset_request.json")
)
MAP_RESET_STATUS_PATH = pathlib.Path(
    os.getenv("MAP_RESET_STATUS_PATH", "/shared/map_reset_state.json")
)
TOOLBOX_MAP_REQUEST_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_REQUEST_PATH", "/shared/toolbox_map_request.json")
)
TOOLBOX_MAP_STATUS_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_STATUS_PATH", "/shared/toolbox_map_status.json")
)
TOOLBOX_MAP_SAVE_DIR = pathlib.Path(
    os.getenv("TOOLBOX_MAP_SAVE_DIR", "/shared/maps/toolbox_saved")
)
TOOLBOX_MAP_AUTOLOAD_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_AUTOLOAD_PATH", str(TOOLBOX_MAP_SAVE_DIR / "autoload.json"))
)
TOOLBOX_MAP_ACTIVE_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_ACTIVE_PATH", str(TOOLBOX_MAP_SAVE_DIR / "active.json"))
)
TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH", str(TOOLBOX_MAP_SAVE_DIR / "active_snapshot.json"))
)
RESET_STATE_PATH = pathlib.Path(
    os.getenv("MAP_RESET_STATE_PATH", "/shared/map_reset_state.json")
)
TOOLBOX_MAP_METADATA_PATH = pathlib.Path(
    os.getenv("TOOLBOX_MAP_METADATA_PATH", str(TOOLBOX_MAP_SAVE_DIR / "metadata.json"))
)
TOOLBOX_MAP_DEFAULT_NAME = os.getenv("TOOLBOX_MAP_DEFAULT_NAME", "default_map")
TOOLBOX_MAP_SEARCH_DIRS = tuple(
    dict.fromkeys((TOOLBOX_MAP_SAVE_DIR, SLAM_TAB_SAVE_DIR)).keys()
)
NAV2_PLAN_REQUEST_PATH = pathlib.Path(
    os.getenv("NAV2_PLAN_REQUEST_PATH", "/shared/nav2_plan_request.json")
)
NAV2_PLAN_RESPONSE_PATH = pathlib.Path(
    os.getenv("NAV2_PLAN_RESPONSE_PATH", "/shared/nav2_plan_response.json")
)
NAV2_PLAN_TIMEOUT_SEC = float(os.getenv("NAV2_PLAN_TIMEOUT_SEC", "30.0"))
NAV2_PLAN_POLL_SEC = float(os.getenv("NAV2_PLAN_POLL_SEC", "0.1"))
MAP_RESET_TIMEOUT_SEC = float(os.getenv("MAP_RESET_TIMEOUT_SEC", "12.0"))
MAP_RESET_POLL_SEC = float(os.getenv("MAP_RESET_POLL_SEC", "0.2"))
NAV_FREE_THRESHOLD = int(os.getenv("NAV_FREE_THRESHOLD", "35"))
NAV_OCCUPIED_THRESHOLD = int(os.getenv("NAV_OCCUPIED_THRESHOLD", "68"))
NAV_OBSTACLE_INFLATION_RADIUS_M = float(os.getenv("NAV_OBSTACLE_INFLATION_RADIUS_M", "0.30"))
NAV_UNKNOWN_INFLATION_RADIUS_M = float(os.getenv("NAV_UNKNOWN_INFLATION_RADIUS_M", "0.20"))
NAV_MAX_SNAP_DISTANCE_M = float(os.getenv("NAV_MAX_SNAP_DISTANCE_M", "0.75"))
NAV_WALL_CLOSING_RADIUS_M = float(os.getenv("NAV_WALL_CLOSING_RADIUS_M", "0.08"))
SLAM_SET_POSE_PENDING_STALE_BUFFER_SEC = float(os.getenv("SLAM_SET_POSE_PENDING_STALE_BUFFER_SEC", "2.0"))

# Scene-change dataset directories
# Repo checkout (bordsupr/frontend/app.py) or the web container (/app/app.py with ./data mounted at /app/data)
SCENE_CHANGE_DATASETS_ROOT = next(
    (
        candidate
        for candidate in (
            Path(__file__).resolve().parent.parent.parent / "data" / "curiosity" / "datasets",
            Path("/app") / "data" / "curiosity" / "datasets",
        )
        if candidate.exists()
    ),
    Path(__file__).resolve().parent.parent.parent / "data" / "curiosity" / "datasets",
)
SCENE_CHANGE_DATA_DIR = SCENE_CHANGE_DATASETS_ROOT / "default"
DATASET_BACKUPS = {
    "default": SCENE_CHANGE_DATASETS_ROOT / "default",
    "adversarial": SCENE_CHANGE_DATASETS_ROOT / "adversarial",
    "regime": SCENE_CHANGE_DATASETS_ROOT / "regime",
}

YOLO_CLASS_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake", 56: "chair",
    57: "couch", 58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet",
    62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
    80: "office table",
}


def class_name_from_id(class_id):
    if class_id is None:
        return None
    try:
        return YOLO_CLASS_NAMES.get(int(class_id))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# YOLO Model Comparison helpers
# ---------------------------------------------------------------------------

YOLO_COMPARE_MAX_IMAGES = 50
YOLO_COMPARE_PALETTE = [
    (220, 64, 64),    # red
    (64, 220, 64),    # green
    (64, 120, 220),   # blue
    (220, 180, 64),   # yellow/orange
    (180, 64, 220),   # purple
    (64, 220, 200),   # cyan
    (220, 120, 180),  # pink
    (120, 120, 120),  # gray
]


def _get_yolo_compare_model(model_path: str):
    """Load and cache a YOLO model for the compare tab."""
    global _yolo_compare_models
    if model_path in _yolo_compare_models:
        return _yolo_compare_models[model_path]
    from ultralytics import YOLO
    model = YOLO(model_path)
    _yolo_compare_models[model_path] = model
    return model


def _list_image_paths(image_dir: Path, image_stride: int, image_count: int):
    if not image_dir.exists() or not image_dir.is_dir():
        return []
    all_paths = sorted([
        *image_dir.glob("*.jpg"),
        *image_dir.glob("*.jpeg"),
        *image_dir.glob("*.png"),
        *image_dir.glob("*.JPG"),
        *image_dir.glob("*.JPEG"),
        *image_dir.glob("*.PNG"),
    ])
    return all_paths[::image_stride][:image_count]


def _compute_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _greedy_overlap_match(dets_a, dets_b, iou_threshold):
    """Greedy bipartite matching between two detection lists. Returns match count."""
    if not dets_a or not dets_b:
        return 0
    pairs = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou = _compute_iou(da["bbox"], db["bbox"])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_a = set()
    used_b = set()
    count = 0
    for iou, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            count += 1
    return count


def _greedy_overlap_pairs(dets_a, dets_b, iou_threshold=0.10):
    """Greedy bipartite matching between two detection lists. Returns list of matched pairs with IoU."""
    if not dets_a or not dets_b:
        return []
    pairs = []
    for i, da in enumerate(dets_a):
        for j, db in enumerate(dets_b):
            iou = _compute_iou(da["bbox"], db["bbox"])
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_a = set()
    used_b = set()
    matches = []
    for iou, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            matches.append({"det_a": dets_a[i], "det_b": dets_b[j], "iou": round(float(iou), 3)})
    return matches


def _detections_for_image(image_path: Path, model, conf_threshold: float):
    """Run YOLO predict and return list of detection dicts."""
    result = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)[0]
    names = getattr(model, "names", {})
    detections = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy().astype(float)
        confs = result.boxes.conf.cpu().numpy().astype(float)
        cls_ids = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            class_name = str(names.get(int(cls_id), cls_id)) if isinstance(names, dict) else str(names[int(cls_id)] if 0 <= int(cls_id) < len(names) else cls_id)
            detections.append({
                "class_id": int(cls_id),
                "class_name": class_name,
                "score": float(conf),
                "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            })
    return detections


def _annotate_image(image_path: Path, model_results: dict, model_colors: dict):
    """
    Draw detections from multiple models on one image.
    model_results: {model_name: [detections]}
    model_colors: {model_name: (R, G, B)}
    Returns base64-encoded JPEG string.
    """
    img = PILImage.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Try to load a font; fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    for model_name, dets in model_results.items():
        color = model_colors.get(model_name, (200, 200, 200))
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            label = f"{det['class_name']} {det['score']:.2f}"
            # text background
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            label_y1 = max(0, y1 - text_h - 4)
            label_y2 = label_y1 + text_h + 4
            draw.rectangle([x1, label_y1, x1 + text_w + 6, label_y2], fill=color)
            draw.text((x1 + 3, label_y1 + 1), label, fill=(255, 255, 255), font=font)

    # Draw legend
    legend_x = 8
    legend_y = 8
    legend_h = 18
    for model_name in model_results.keys():
        color = model_colors.get(model_name, (200, 200, 200))
        draw.rectangle([legend_x, legend_y, legend_x + 14, legend_y + 14], fill=color)
        draw.text((legend_x + 18, legend_y), model_name, fill=(255, 255, 255), font=font)
        legend_y += legend_h

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class YoloCompareRunRequest(BaseModel):
    frame_dir: str
    model_paths: list[str]
    conf_threshold: float = 0.25
    image_stride: int = 5
    max_images: int = 20
    iou_threshold: float = 0.5


class YoloCompareDetection(BaseModel):
    class_id: int
    class_name: str
    score: float
    bbox: list[float]


class YoloCompareImageResult(BaseModel):
    image_name: str
    image_b64: str
    model_results: dict
    overlap_pairs: list[dict] = []


class YoloCompareRunResponse(BaseModel):
    frame_dir: str
    processed_images: int
    model_names: list[str]
    per_model_counts: dict[str, int]
    overlap_matrix: dict[str, dict[str, int]]
    unique_detections: dict[str, int]
    image_results: list[YoloCompareImageResult]


def _resolve_map_id(building_name: str | None) -> int | None:
    if not building_name:
        return None
    normalized = str(building_name).strip()
    if not normalized:
        return None
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM maps WHERE name = %s", (normalized,))
            row = cur.fetchone()
    return int(row[0]) if row else None


def normalize_frame_id(frame_id):
    if frame_id in WORLD_FRAME_ALIASES:
        return "world"
    return frame_id


def normalize_world_payload(payload):
    if not isinstance(payload, dict):
        return payload

    map_data = payload.get("map")
    if isinstance(map_data, dict):
        map_data["frame_id"] = normalize_frame_id(map_data.get("frame_id"))

    for key in ("robot", "goal"):
        entry = payload.get(key)
        if isinstance(entry, dict):
            entry["frame_id"] = normalize_frame_id(entry.get("frame_id"))

    path = payload.get("path")
    if isinstance(path, list):
        for point in path:
            if isinstance(point, dict) and "frame_id" in point:
                point["frame_id"] = normalize_frame_id(point.get("frame_id"))

    if "frame_id" in payload:
        payload["frame_id"] = normalize_frame_id(payload.get("frame_id"))
    if "sensor_frame_id" in payload:
        payload["sensor_frame_id"] = normalize_frame_id(payload.get("sensor_frame_id"))

    return payload


def _load_map_payload_raw(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload = normalize_world_payload(payload)
    payload["available"] = True
    payload["source"] = os.path.basename(path)
    return payload


def load_lidar_occupancy_map_payload():
    return load_map_payload_from_path(LIDAR_OCCUPANCY_MAP_PATH)


def load_map_payload_from_path(path):
    payload = _load_map_payload_raw(path)
    if payload is None:
        return None
    if _map_is_suppressed(path, payload):
        return None
    return payload


def atomic_write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def load_json_file(path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_slam_pose_pending_state() -> dict:
    status_payload = load_json_file(SLAM_TAB_STATUS_PATH)
    if not isinstance(status_payload, dict):
        return {"pending": False, "stale": False, "detail": "", "timeout_sec": None}
    if not bool(status_payload.get("localization_pending")):
        return {"pending": False, "stale": False, "detail": "", "timeout_sec": None}

    message = str(status_payload.get("message") or "").strip()
    detail = (
        message
        or "The frontend pose was sent to slam_toolbox, but localization has not confirmed it in live TF yet."
    )

    timeout_sec = None
    try:
        timeout_sec = float(status_payload.get("localization_pending_timeout_sec") or 12.0)
    except (TypeError, ValueError):
        timeout_sec = 12.0

    pending_since = status_payload.get("localization_pending_since")
    if pending_since is None:
        pending_since = status_payload.get("updated_at")

    stale = False
    try:
        if pending_since is not None and timeout_sec is not None:
            stale = (time.time() - float(pending_since)) > (timeout_sec + SLAM_SET_POSE_PENDING_STALE_BUFFER_SEC)
    except (TypeError, ValueError):
        stale = False

    return {
        "pending": True,
        "stale": stale,
        "detail": detail,
        "timeout_sec": timeout_sec,
    }


def assert_slam_pose_not_pending_for_nav2(*, allow_stale_for_planning: bool = False) -> bool:
    pending_state = _get_slam_pose_pending_state()
    if not pending_state["pending"]:
        return False

    if allow_stale_for_planning and pending_state["stale"]:
        return True

    detail = pending_state["detail"]
    if pending_state["stale"]:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{detail} Localization did not confirm the set pose within "
                f"{pending_state['timeout_sec']:.0f}s, so navigation remains blocked. "
                "Re-localize or reset the pose before executing a route."
            ),
        )

    raise HTTPException(
        status_code=409,
        detail=f"{detail} Wait for set-pose confirmation before planning or navigating.",
    )


def _map_payload_signature(payload):
    if not isinstance(payload, dict):
        return None

    map_data = payload.get("map")
    if not isinstance(map_data, dict):
        return None

    origin = map_data.get("origin") or {}
    data = map_data.get("data") or []
    try:
        normalized_map = {
            "frame_id": normalize_frame_id(map_data.get("frame_id")),
            "resolution": float(map_data.get("resolution") or 0.0),
            "width": int(map_data.get("width") or 0),
            "height": int(map_data.get("height") or 0),
            "origin": {
                "x": float(origin.get("x") or 0.0),
                "y": float(origin.get("y") or 0.0),
            },
            "data": [int(value) for value in data],
        }
    except (TypeError, ValueError):
        return None

    digest = hashlib.sha256(
        json.dumps(normalized_map, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _load_reset_state():
    payload = load_json_file(RESET_STATE_PATH)
    return payload if isinstance(payload, dict) else {}


def _get_suppressed_map_entry(path):
    state = _load_reset_state()
    suppressed_maps = state.get("suppressed_maps")
    if not isinstance(suppressed_maps, dict):
        return None
    entry = suppressed_maps.get(os.path.basename(path))
    return entry if isinstance(entry, dict) else None


def _map_is_suppressed(path, payload):
    entry = _get_suppressed_map_entry(path)
    if entry is None:
        return False
    try:
        generated_at = float(payload.get("generated_at"))
        captured_at = float(entry.get("captured_at"))
        if generated_at > captured_at:
            return False
    except (TypeError, ValueError):
        pass
    signature = _map_payload_signature(payload)
    return signature is not None and signature == entry.get("signature")


def _map_unavailable_reason(path, default_reason):
    if _get_suppressed_map_entry(path) is not None:
        return "map_reset"
    return default_reason


def _build_suppressed_map_state():
    suppressed_maps = {}
    for path in (LIDAR_OCCUPANCY_MAP_PATH, TOOLBOX_MAP_PATH, MAP_SNAPSHOT_PATH):
        payload = _load_map_payload_raw(path)
        signature = _map_payload_signature(payload)
        if signature is None:
            continue
        suppressed_maps[os.path.basename(path)] = {
            "signature": signature,
            "captured_at": time.time(),
        }
    return suppressed_maps


def delete_file_if_exists(path):
    try:
        os.remove(path)
        return True
    except FileNotFoundError:
        return False


def sanitize_toolbox_map_name(name):
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(name or "").strip())
    text = text.strip("._-")
    return text or TOOLBOX_MAP_DEFAULT_NAME


def normalize_toolbox_building_name(value, fallback_name=None):
    text = str(value or "").strip()
    if text:
        return text
    if fallback_name:
        return str(fallback_name).strip() or TOOLBOX_MAP_DEFAULT_NAME
    return TOOLBOX_MAP_DEFAULT_NAME


def load_toolbox_map_metadata():
    payload = load_json_file(TOOLBOX_MAP_METADATA_PATH)
    if not isinstance(payload, dict):
        return {"maps": {}}
    maps = payload.get("maps")
    if not isinstance(maps, dict):
        payload["maps"] = {}
    return payload


def save_toolbox_map_metadata(payload):
    atomic_write_json(TOOLBOX_MAP_METADATA_PATH, payload)


def rename_observations_for_map(old_map_name, new_map_name):
    old_map_id = _resolve_map_id(old_map_name)
    new_map_id = _ensure_pipeline_run_map(new_map_name)
    return _migrate_observations_to_map(old_map_id, new_map_id)


def get_toolbox_map_metadata(name):
    payload = load_toolbox_map_metadata()
    maps = payload.get("maps") or {}
    entry = maps.get(name)
    return entry if isinstance(entry, dict) else {}


def persist_toolbox_autoload_record(name):
    record = get_toolbox_map_record(name)
    atomic_write_json(
        TOOLBOX_MAP_AUTOLOAD_PATH,
        {
            "name": record["name"],
            "building_name": record["building_name"],
            "updated_at": time.time(),
        },
    )
    return record


def persist_active_toolbox_map_record(record, mode="frozen", observation_map_name=None, recording_enabled=True):
    payload = {
        "mode": mode,
        "name": record["name"],
        "building_name": record["building_name"],
        "recording_enabled": bool(recording_enabled),
        "updated_at": time.time(),
    }
    if observation_map_name:
        payload["observation_map_name"] = observation_map_name
    atomic_write_json(TOOLBOX_MAP_ACTIVE_PATH, payload)
    return payload


def load_active_toolbox_map_record():
    payload = load_json_file(TOOLBOX_MAP_ACTIVE_PATH)
    return payload if isinstance(payload, dict) else {}


def saved_toolbox_map_snapshot_path(name):
    return TOOLBOX_MAP_SAVE_DIR / f"{sanitize_toolbox_map_name(name)}.snapshot.json"


def resolve_toolbox_map_storage_dir(record):
    try:
        return pathlib.Path(record.get("storage_dir") or TOOLBOX_MAP_SAVE_DIR)
    except Exception:
        return TOOLBOX_MAP_SAVE_DIR


def load_saved_toolbox_map_yaml(record):
    yaml_path = resolve_toolbox_map_storage_dir(record) / record["yaml_file"]
    if not yaml_path.exists():
        return None
    payload = {}
    try:
        for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            payload[key.strip()] = value.strip().strip("'\"")
    except Exception:
        return None
    return payload


def parse_yaml_float(payload, key, default=0.0):
    try:
        return float(payload.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_yaml_origin(payload):
    text = str(payload.get("origin") or "").strip()
    if text.startswith("[") and text.endswith("]"):
        parts = [part.strip() for part in text[1:-1].split(",")]
        try:
            return {
                "x": float(parts[0]) if len(parts) > 0 else 0.0,
                "y": float(parts[1]) if len(parts) > 1 else 0.0,
            }
        except (TypeError, ValueError):
            pass
    return {"x": 0.0, "y": 0.0}


def read_pgm_image(path):
    try:
        raw = path.read_bytes()
    except Exception:
        return None

    index = 0

    def next_token():
        nonlocal index
        while index < len(raw):
            char = raw[index:index + 1]
            if char == b"#":
                while index < len(raw) and raw[index:index + 1] not in (b"\n", b"\r"):
                    index += 1
            elif char.isspace():
                index += 1
            else:
                break
        start = index
        while index < len(raw) and not raw[index:index + 1].isspace():
            index += 1
        return raw[start:index].decode("ascii")

    try:
        magic = next_token()
        width = int(next_token())
        height = int(next_token())
        max_value = int(next_token())
    except Exception:
        return None
    if magic != "P5" or width <= 0 or height <= 0 or max_value <= 0:
        return None
    if index < len(raw) and raw[index:index + 1].isspace():
        index += 1
    pixels = raw[index:index + (width * height)]
    if len(pixels) != width * height:
        return None
    return {"width": width, "height": height, "max_value": max_value, "pixels": pixels}


def build_saved_toolbox_map_snapshot_from_yaml(record):
    yaml_payload = load_saved_toolbox_map_yaml(record)
    if not isinstance(yaml_payload, dict):
        return None
    image_name = yaml_payload.get("image")
    if not image_name:
        return None
    storage_dir = resolve_toolbox_map_storage_dir(record)
    pgm_path = pathlib.Path(image_name)
    if not pgm_path.is_absolute():
        pgm_path = storage_dir / pgm_path
    image = read_pgm_image(pgm_path)
    if not isinstance(image, dict):
        return None

    negate = int(parse_yaml_float(yaml_payload, "negate", 0.0))
    occupied_thresh = parse_yaml_float(yaml_payload, "occupied_thresh", 0.65)
    free_thresh = parse_yaml_float(yaml_payload, "free_thresh", 0.25)
    max_value = float(image["max_value"])
    width = image["width"]
    height = image["height"]
    pixels = image["pixels"]

    # Backward compatibility: old slam_tab_manager.py (version < 2) flipped Y
    # when writing the PGM so that row 0 was the top of the image. ROS standard
    # (and slam_toolbox/save_map) writes row 0 as the bottom. Flip back when
    # loading old maps that were saved by the buggy code.
    version = yaml_payload.get("slam_tab_manager_version", "1")
    needs_flip_y = (
        str(storage_dir) == str(SLAM_TAB_SAVE_DIR)
        and version != "2"
    )

    data = []
    for y in range(height):
        row_y = (height - 1 - y) if needs_flip_y else y
        row_start = row_y * width
        for x in range(width):
            pixel = pixels[row_start + x]
            # In ROS map convention, 205 is the standard grey value for unknown.
            # Explicitly treat it as unknown so it doesn't leak into free/occupied
            # when the YAML thresholds (e.g. free_thresh 0.25) misclassify it.
            if pixel == 205:
                data.append(-1)
                continue
            value = (max_value - float(pixel)) / max_value if not negate else float(pixel) / max_value
            if value > occupied_thresh:
                data.append(100)
            elif value < free_thresh:
                data.append(0)
            else:
                data.append(-1)

    return {
        "available": True,
        "generated_at": record["updated_at"] or time.time(),
        "source": f"saved:{record['name']}",
        "mode": "frozen",
        "name": record["name"],
        "building_name": record["building_name"],
        "map": {
            "frame_id": "map",
            "resolution": parse_yaml_float(yaml_payload, "resolution", 0.05),
            "width": image["width"],
            "height": image["height"],
            "origin": parse_yaml_origin(yaml_payload),
            "data": data,
        },
        "robot": None,
        "detections": [],
    }


def _pgm_dimensions(path: pathlib.Path) -> tuple[int, int] | None:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    index = 0
    def next_token():
        nonlocal index
        while index < len(raw):
            char = raw[index:index + 1]
            if char == b"#":
                while index < len(raw) and raw[index:index + 1] not in (b"\n", b"\r"):
                    index += 1
            elif char.isspace():
                index += 1
            else:
                break
        start = index
        while index < len(raw) and not raw[index:index + 1].isspace():
            index += 1
        return raw[start:index].decode("ascii")
    try:
        magic = next_token()
        width = int(next_token())
        height = int(next_token())
        _maxval = int(next_token())
    except Exception:
        return None
    if magic != "P5":
        return None
    return (width, height)


def build_saved_toolbox_map_snapshot(record):
    try:
        payload = load_map_payload_from_path(saved_toolbox_map_snapshot_path(record["name"]))
    except OSError:
        payload = None

    # If a stale .snapshot.json exists (e.g. copied from a live map with different
    # dimensions), detect it and fall back to YAML/PGM.
    snapshot_is_valid = isinstance(payload, dict)
    if snapshot_is_valid:
        source = str(payload.get("source") or "")
        if not source.startswith("saved:"):
            snapshot_is_valid = False
        else:
            map_payload = payload.get("map") if isinstance(payload, dict) else None
            snap_w = int(map_payload.get("width") or 0) if isinstance(map_payload, dict) else 0
            snap_h = int(map_payload.get("height") or 0) if isinstance(map_payload, dict) else 0
            storage_dir = resolve_toolbox_map_storage_dir(record)
            pgm_dims = _pgm_dimensions(storage_dir / f"{record['name']}.pgm")
            if pgm_dims is not None and (snap_w, snap_h) != pgm_dims:
                snapshot_is_valid = False

    if not snapshot_is_valid:
        payload = build_saved_toolbox_map_snapshot_from_yaml(record)
    if not isinstance(payload, dict):
        return None

    payload = dict(payload)
    payload["available"] = True
    payload["mode"] = "frozen"
    payload["name"] = record["name"]
    payload["building_name"] = record["building_name"]
    payload["source"] = f"saved:{record['name']}"
    return payload


def persist_frozen_toolbox_map_snapshot(record):
    payload = build_saved_toolbox_map_snapshot(record)
    if payload is None:
        try:
            TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH.unlink(missing_ok=True)
        except Exception:
            pass
        return None
    atomic_write_json(TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH, payload)
    return payload


def load_active_toolbox_map_payload():
    active_payload = load_active_toolbox_map_record()
    if str(active_payload.get("mode") or "").strip().lower() != "frozen":
        return None
    if not active_payload.get("name"):
        return None
    active_name = sanitize_toolbox_map_name(active_payload.get("name"))
    active_building_name = normalize_toolbox_building_name(
        active_payload.get("building_name"),
        fallback_name=active_name,
    )

    payload = load_map_payload_from_path(TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH)
    if (
        not isinstance(payload, dict)
        or not payload.get("name")
        or (
            payload.get("name")
            and sanitize_toolbox_map_name(payload.get("name")) != active_name
        )
        or (
            str(payload.get("source") or "").startswith("saved:")
            and str(payload.get("source")) != f"saved:{active_name}"
        )
    ):
        record = get_toolbox_map_record(active_name)
        if not record or not record.get("ready_to_load"):
            # Stale active.json pointing to a deleted/non-existent map — clear it
            # so the live snapshot fallback is used and the state stays consistent.
            try:
                TOOLBOX_MAP_ACTIVE_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            return None
        payload = persist_frozen_toolbox_map_snapshot(record)
    if not isinstance(payload, dict):
        # Map exists but snapshot could not be built (e.g. old-style save without
        # YAML/PGM). Preserve active.json so recording state is not lost.
        return None

    payload = dict(payload)
    payload["available"] = True
    payload["mode"] = "frozen"
    payload["name"] = active_name
    payload["building_name"] = active_building_name
    payload.setdefault("source", f"saved:{active_name}")
    return payload


def clear_toolbox_autoload_record():
    try:
        TOOLBOX_MAP_AUTOLOAD_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def delete_saved_toolbox_map(name):
    record = get_toolbox_map_record(name)
    if not any((record["yaml_exists"], record["pgm_exists"], record["posegraph_exists"], record["data_exists"])):
        raise HTTPException(status_code=404, detail=f"No saved toolbox map files found for '{record['name']}'.")

    base = resolve_toolbox_map_storage_dir(record) / record["name"]
    deleted_files = []
    for path in (
        base.with_suffix(".yaml"),
        base.with_suffix(".pgm"),
        base.with_suffix(".posegraph"),
        base.with_suffix(".data"),
    ):
        if delete_file_if_exists(path):
            deleted_files.append(path.name)

    metadata = load_toolbox_map_metadata()
    maps = metadata.get("maps") or {}
    if record["name"] in maps:
        del maps[record["name"]]
        metadata["maps"] = maps
        save_toolbox_map_metadata(metadata)

    autoload_payload = load_json_file(TOOLBOX_MAP_AUTOLOAD_PATH)
    autoload_name = None
    if isinstance(autoload_payload, dict):
        autoload_name = sanitize_toolbox_map_name(autoload_payload.get("name"))
    if autoload_name == record["name"]:
        remaining_maps = list_saved_toolbox_maps()
        if remaining_maps:
            persist_toolbox_autoload_record(remaining_maps[0]["name"])
        else:
            clear_toolbox_autoload_record()

    return {
        "ok": True,
        "deleted": True,
        "name": record["name"],
        "building_name": record["building_name"],
        "deleted_files": deleted_files,
    }


def _build_toolbox_map_record_for_dir(sanitized_name, storage_dir):
    base = pathlib.Path(storage_dir) / sanitized_name
    yaml_path = base.with_suffix(".yaml")
    pgm_path = base.with_suffix(".pgm")
    posegraph_path = base.with_suffix(".posegraph")
    data_path = base.with_suffix(".data")
    metadata = get_toolbox_map_metadata(sanitized_name)
    updated_at = max(
        [
            path.stat().st_mtime
            for path in (yaml_path, pgm_path, posegraph_path, data_path)
            if path.exists()
        ]
        or [0.0]
    )
    return {
        "name": sanitized_name,
        "building_name": normalize_toolbox_building_name(metadata.get("building_name"), fallback_name=sanitized_name),
        "storage_dir": str(storage_dir),
        "yaml_exists": yaml_path.exists(),
        "pgm_exists": pgm_path.exists(),
        "posegraph_exists": posegraph_path.exists(),
        "data_exists": data_path.exists(),
        "ready_to_load": posegraph_path.exists() and data_path.exists(),
        "yaml_file": yaml_path.name,
        "pgm_file": pgm_path.name,
        "posegraph_file": posegraph_path.name,
        "data_file": data_path.name,
        "updated_at": updated_at,
    }


def get_toolbox_map_record(name):
    sanitized_name = sanitize_toolbox_map_name(name)
    best_record = None
    best_score = None
    for storage_dir in TOOLBOX_MAP_SEARCH_DIRS:
        record = _build_toolbox_map_record_for_dir(sanitized_name, storage_dir)
        score = (
            1 if record["ready_to_load"] else 0,
            1 if any((record["yaml_exists"], record["pgm_exists"], record["posegraph_exists"], record["data_exists"])) else 0,
            record["updated_at"],
        )
        if best_score is None or score > best_score:
            best_record = record
            best_score = score
    return best_record


def list_saved_toolbox_maps():
    if not any(storage_dir.exists() for storage_dir in TOOLBOX_MAP_SEARCH_DIRS):
        return []

    names = set()
    for storage_dir in TOOLBOX_MAP_SEARCH_DIRS:
        for pattern in ("*.yaml", "*.pgm", "*.posegraph", "*.data"):
            for path in storage_dir.glob(pattern):
                names.add(path.stem)

    return sorted(
        (get_toolbox_map_record(name) for name in names),
        key=lambda item: (item["updated_at"], item["name"]),
        reverse=True,
    )


def build_toolbox_map_persistence_payload():
    status_payload = load_json_file(TOOLBOX_MAP_STATUS_PATH)
    if not isinstance(status_payload, dict):
        status_payload = {
            "available": False,
            "state": "idle",
            "message": "slam_toolbox map save/load manager status is not available yet.",
        }

    autoload_payload = load_json_file(TOOLBOX_MAP_AUTOLOAD_PATH)
    autoload_name = None
    autoload_building_name = None
    if isinstance(autoload_payload, dict):
        autoload_name = sanitize_toolbox_map_name(autoload_payload.get("name"))
        autoload_building_name = normalize_toolbox_building_name(
            autoload_payload.get("building_name"),
            fallback_name=autoload_name,
        )

    active_payload = load_active_toolbox_map_record()
    if isinstance(status_payload, dict) and status_payload.get("state") == "saved":
        map_record = status_payload.get("map_record") if isinstance(status_payload.get("map_record"), dict) else {}
        saved_name = sanitize_toolbox_map_name(map_record.get("name") or status_payload.get("name"))
        saved_record = get_toolbox_map_record(saved_name)
        old_observation_map_name = active_payload.get("observation_map_name") or active_payload.get("name")
        if old_observation_map_name and saved_record.get("name"):
            rename_observations_for_map(old_observation_map_name, saved_record["name"])
        persist_active_toolbox_map_record(saved_record, mode="frozen", recording_enabled=False)
        snapshot = build_saved_toolbox_map_snapshot(saved_record)
        atomic_write_json(TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH, snapshot)
        active_payload = load_active_toolbox_map_record()
    active_mode = str(active_payload.get("mode") or "idle").strip().lower()
    active_name = sanitize_toolbox_map_name(active_payload.get("name")) if active_payload.get("name") else None
    active_open_until = active_payload.get("map_open_until") if isinstance(active_payload, dict) else None
    active_is_open = False
    try:
        active_is_open = active_open_until is not None and float(active_open_until) >= time.time()
    except (TypeError, ValueError):
        active_is_open = False
    active_building_name = None
    if active_name:
        active_building_name = normalize_toolbox_building_name(
            active_payload.get("building_name"),
            fallback_name=active_name,
        )

    return {
        "available": True,
        "default_name": TOOLBOX_MAP_DEFAULT_NAME,
        "autoload_name": autoload_name,
        "autoload_building_name": autoload_building_name,
        "active_mode": active_mode,
        "active_name": active_name,
        "active_building_name": active_building_name,
        "active_is_open": active_is_open,
        "active_open_until": active_open_until,
        "recording_enabled": active_payload.get("recording_enabled") if isinstance(active_payload, dict) else None,
        "status": status_payload,
        "maps": list_saved_toolbox_maps(),
    }


def _quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation around Z) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _apply_object_standoff(
    object_pose: dict,
    robot_pose: dict | None,
    standoff_m: float = 1.0,
) -> dict:
    """Offset an object goal towards the robot so the robot stops short of the object.

    Computes the vector from the object to the robot, normalises it, and moves
    the goal ``standoff_m`` metres towards the robot.  If the robot is already
    within ``standoff_m`` of the object, the original pose is returned unchanged.
    """
    if robot_pose is None:
        return object_pose
    ox = float(object_pose.get("x", 0.0))
    oy = float(object_pose.get("y", 0.0))
    rx = float(robot_pose.get("x", 0.0))
    ry = float(robot_pose.get("y", 0.0))
    dx = rx - ox
    dy = ry - oy
    dist = math.hypot(dx, dy)
    if dist <= standoff_m or dist < 1e-6:
        return object_pose
    scale = standoff_m / dist
    return {
        "x": ox + dx * scale,
        "y": oy + dy * scale,
        "z": float(object_pose.get("z", 0.0)),
    }


def _compute_approach_standoff_goal(
    raw_goal_pose: dict,
    nav2_plan: dict,
    standoff_m: float = 1.0,
) -> dict | None:
    """Return a goal offset along the final approach direction.

    Uses the last segment of the planned path to determine the direction from
    which the robot arrives, then moves the goal ``standoff_m`` metres back
    along that direction.  Returns None if the path is too short.
    """
    path = nav2_plan.get("path") or []
    if len(path) < 2:
        return None
    p_prev = path[-2]
    p_goal = path[-1]
    gx = float(p_goal.get("x", 0.0))
    gy = float(p_goal.get("y", 0.0))
    px = float(p_prev.get("x", 0.0))
    py = float(p_prev.get("y", 0.0))
    dx = gx - px
    dy = gy - py
    seg_len = math.hypot(dx, dy)
    if seg_len < 1e-6:
        return None
    scale = standoff_m / seg_len
    return {
        "x": gx - dx * scale,
        "y": gy - dy * scale,
        "z": float(raw_goal_pose.get("z", 0.0)),
    }


def _get_robot_start_pose() -> dict | None:
    """Return the current robot pose for client-side A* fallback planning.

    Nav2 should NOT use this — it looks up the robot pose from TF directly.
    This helper is only for the A* fallback when Nav2 is unavailable.
    """
    # Prefer the slam_tab robot pose file because it reflects the live
    # odometry and any set-pose corrections applied by the user in the SLAM
    # UI.  The toolbox snapshot only updates its robot pose when a recording
    # session is active, so it can be stale after set-pose or when idle.
    live_pose = _load_json_file(SLAM_TAB_ROBOT_POSE_PATH)
    if isinstance(live_pose, dict):
        pos = live_pose.get("position")
        orient = live_pose.get("orientation")
        if isinstance(pos, dict) and pos.get("x") is not None and pos.get("y") is not None:
            yaw = 0.0
            if isinstance(orient, dict):
                yaw = _quaternion_to_yaw(
                    float(orient.get("x", 0.0)),
                    float(orient.get("y", 0.0)),
                    float(orient.get("z", 0.0)),
                    float(orient.get("w", 1.0)),
                )
            return {
                "x": float(pos["x"]),
                "y": float(pos["y"]),
                "z": 0.0,
                "yaw": yaw,
            }

    # Fall back to the toolbox snapshot if the slam_tab file is missing.
    toolbox_map_payload = load_map_payload_from_path(TOOLBOX_MAP_PATH)
    if isinstance(toolbox_map_payload, dict):
        robot_pose = toolbox_map_payload.get("robot")
        if isinstance(robot_pose, dict) and robot_pose.get("x") is not None and robot_pose.get("y") is not None:
            return {
                "x": float(robot_pose["x"]),
                "y": float(robot_pose["y"]),
                "z": 0.0,
                "yaw": float(robot_pose.get("yaw", 0.0) or 0.0),
            }

    return None


def _load_slam_tab_map_payload() -> dict | None:
    payload = _load_json_file(SLAM_TAB_MAP_PATH)
    if not isinstance(payload, dict):
        return None

    info = payload.get("info")
    data = payload.get("data")
    if not isinstance(info, dict) or not isinstance(data, list):
        return None

    origin = info.get("origin") or {}
    origin_position = origin.get("position") or {}
    header = payload.get("header") or {}
    return {
        "available": True,
        "source": "slam_tab",
        "map": {
            "frame_id": str(header.get("frame_id") or "map"),
            "resolution": float(info.get("resolution") or 0.0),
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "origin": {
                "x": float(origin_position.get("x") or 0.0),
                "y": float(origin_position.get("y") or 0.0),
            },
            "data": data,
        },
    }


def _slam_tab_response_from_normalized_map_payload(payload: dict) -> dict | None:
    if not isinstance(payload, dict):
        return None

    map_data = payload.get("map")
    if not isinstance(map_data, dict):
        return None

    origin = map_data.get("origin") or {}
    data = map_data.get("data") or []
    if not isinstance(origin, dict) or not isinstance(data, list):
        return None

    return {
        "header": {
            "stamp": {"sec": 0, "nanosec": 0},
            "frame_id": str(map_data.get("frame_id") or "map"),
        },
        "info": {
            "map_load_time": {"sec": 0, "nanosec": 0},
            "resolution": float(map_data.get("resolution") or 0.0),
            "width": int(map_data.get("width") or 0),
            "height": int(map_data.get("height") or 0),
            "origin": {
                "position": {
                    "x": float(origin.get("x") or 0.0),
                    "y": float(origin.get("y") or 0.0),
                    "z": 0.0,
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
        },
        "data": data,
    }


def request_nav2_plan(goal_pose: dict, start_pose: dict | None = None) -> dict:
    request_id = f"plan-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    # Remove any stale request/response files left by a previous failed request
    # so we don't accidentally read old state if the bridge doesn't respond.
    for p in (NAV2_PLAN_REQUEST_PATH, NAV2_PLAN_RESPONSE_PATH):
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    request_payload = {
        "request_id": request_id,
        "goal": {"x": float(goal_pose["x"]), "y": float(goal_pose["y"])},
        "created_at": time.time(),
    }
    if start_pose is not None:
        request_payload["start"] = {
            "x": float(start_pose["x"]),
            "y": float(start_pose["y"]),
            "z": float(start_pose.get("z", 0.0) or 0.0),
            "yaw": float(start_pose.get("yaw", 0.0) or 0.0),
        }
    atomic_write_json(
        NAV2_PLAN_REQUEST_PATH,
        request_payload,
    )
    deadline = time.time() + NAV2_PLAN_TIMEOUT_SEC
    while time.time() < deadline:
        if NAV2_PLAN_RESPONSE_PATH.exists():
            try:
                with NAV2_PLAN_RESPONSE_PATH.open("r", encoding="utf-8") as f:
                    response = json.load(f)
                if isinstance(response, dict) and response.get("request_id") == request_id:
                    if "error" in response:
                        raise HTTPException(status_code=503, detail=f"Nav2 planner: {response['error']}")
                    return response
            except HTTPException:
                raise
            except Exception:
                pass
        time.sleep(NAV2_PLAN_POLL_SEC)
    latest_response = load_json_file(NAV2_PLAN_RESPONSE_PATH)
    if isinstance(latest_response, dict) and latest_response.get("error"):
        latest_request_id = latest_response.get("request_id")
        if latest_request_id and latest_request_id != request_id:
            detail = (
                "Nav2 planner backend did not answer the current request before the timeout. "
                f"Latest backend error (from {latest_request_id}): {latest_response['error']}"
            )
        else:
            detail = f"Nav2 planner: {latest_response['error']}"
        raise HTTPException(status_code=503, detail=detail)
    raise HTTPException(status_code=504, detail="Nav2 planner did not respond within the timeout.")


def _plan_uses_nav2_planner(plan: dict) -> bool:
    planner = plan.get("planner")
    if not isinstance(planner, dict):
        return False
    planner_name = str(planner.get("name") or "").strip().lower()
    return planner_name.startswith("nav2")


def nav2_path_violates_occupancy(map_payload: dict, nav2_plan: dict) -> bool:
    if not isinstance(map_payload, dict) or not isinstance(nav2_plan, dict):
        return False

    map_data = map_payload.get("map") or {}
    resolution = float(map_data.get("resolution") or 0.0)
    width = int(map_data.get("width") or 0)
    height = int(map_data.get("height") or 0)
    origin = map_data.get("origin") or {}
    origin_x = float(origin.get("x") or 0.0)
    origin_y = float(origin.get("y") or 0.0)
    data = list(map_data.get("data") or [])
    path = nav2_plan.get("path") or []

    if resolution <= 0 or width <= 0 or height <= 0 or len(data) != width * height or len(path) < 2:
        return False

    navigation_map = build_blocked_cells(data, width, height, resolution)
    blocked_cells = navigation_map["blocked"]

    try:
        grid_points = [
            world_to_grid(point["x"], point["y"], origin_x, origin_y, resolution)
            for point in path
            if isinstance(point, dict) and point.get("x") is not None and point.get("y") is not None
        ]
    except (KeyError, TypeError, ValueError):
        return False

    if len(grid_points) < 2:
        return False

    for col, row in grid_points:
        if not is_traversable(blocked_cells, width, height, col, row):
            return True

    for start, end in zip(grid_points, grid_points[1:]):
        if not has_line_of_sight(blocked_cells, width, height, start, end):
            return True

    return False


def build_navigation_status_fallback():
    request_payload = load_json_file(NAVIGATION_REQUEST_PATH)
    if not isinstance(request_payload, dict):
        return enrich_navigation_status({"available": False, "state": "idle"})

    payload = {
        "available": True,
        "state": "queued",
        "request_id": request_payload.get("request_id"),
        "created_at": request_payload.get("created_at"),
        "updated_at": request_payload.get("created_at"),
        "message": "Execution request is queued for the ROS navigation executor.",
        "observation": request_payload.get("observation"),
        "plan": request_payload.get("plan"),
        "execution": request_payload.get("execution"),
    }
    if _is_stale_queued_navigation_status(payload):
        payload["queue_stale"] = True
        payload["executor_available"] = False
        payload["message"] = "Execution request is still queued. The ROS navigation executor may not be running."
    return enrich_navigation_status(payload)


def enrich_navigation_status(status_payload: dict) -> dict:
    payload = dict(status_payload)
    payload["server_time"] = time.time()
    cmd_vel_payload = load_json_file(CMD_VEL_STATUS_PATH)
    if isinstance(cmd_vel_payload, dict):
        payload["cmd_vel"] = cmd_vel_payload
    return payload


def _navigation_status_age_sec(status_payload: dict) -> float | None:
    if not isinstance(status_payload, dict):
        return None
    raw_timestamp = status_payload.get("updated_at", status_payload.get("created_at"))
    try:
        timestamp = float(raw_timestamp)
    except (TypeError, ValueError):
        return None
    return max(0.0, time.time() - timestamp)


def _is_stale_queued_navigation_status(status_payload: dict) -> bool:
    if not isinstance(status_payload, dict) or status_payload.get("state") != "queued":
        return False
    age_sec = _navigation_status_age_sec(status_payload)
    return age_sec is not None and age_sec > NAVIGATION_QUEUE_STALE_SEC


def _clear_navigation_request_files() -> None:
    for path in (NAVIGATION_REQUEST_PATH, NAVIGATION_STATUS_PATH, NAVIGATION_CANCEL_REQUEST_PATH):
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def world_to_grid(x, y, origin_x, origin_y, resolution):
    return (
        int(math.floor((float(x) - origin_x) / resolution)),
        int(math.floor((float(y) - origin_y) / resolution)),
    )


def grid_to_world(col, row, origin_x, origin_y, resolution):
    return {
        "x": origin_x + (col + 0.5) * resolution,
        "y": origin_y + (row + 0.5) * resolution,
        "frame_id": "world",
    }


def in_bounds(col, row, width, height):
    return 0 <= col < width and 0 <= row < height


def cell_value(data, width, col, row):
    return int(data[row * width + col])


def meters_to_cells(distance_m, resolution):
    if resolution <= 0:
        return 0
    return max(0, int(math.ceil(float(distance_m) / float(resolution))))


def circular_offsets(radius_cells):
    return [
        (drow, dcol)
        for drow in range(-radius_cells, radius_cells + 1)
        for dcol in range(-radius_cells, radius_cells + 1)
        if math.hypot(dcol, drow) <= radius_cells + 1e-6
    ]


def dilate_cells(cells, radius_cells, width, height):
    if radius_cells <= 0:
        return set(cells)

    result = set()
    offsets = circular_offsets(radius_cells)
    for col, row in cells:
        for drow, dcol in offsets:
            inflated_col = col + dcol
            inflated_row = row + drow
            if in_bounds(inflated_col, inflated_row, width, height):
                result.add((inflated_col, inflated_row))
    return result


def erode_cells(cells, radius_cells, width, height):
    if radius_cells <= 0:
        return set(cells)

    result = set()
    offsets = circular_offsets(radius_cells)
    for col, row in cells:
        keep = True
        for drow, dcol in offsets:
            neighbor_col = col + dcol
            neighbor_row = row + drow
            if not in_bounds(neighbor_col, neighbor_row, width, height) or (neighbor_col, neighbor_row) not in cells:
                keep = False
                break
        if keep:
            result.add((col, row))
    return result


def close_cells(cells, radius_cells, width, height):
    if radius_cells <= 0:
        return set(cells)
    return erode_cells(dilate_cells(cells, radius_cells, width, height), radius_cells, width, height)


def build_blocked_cells(
    data,
    width,
    height,
    resolution,
    occupied_threshold=NAV_OCCUPIED_THRESHOLD,
    free_threshold=NAV_FREE_THRESHOLD,
    obstacle_inflation_radius_m=NAV_OBSTACLE_INFLATION_RADIUS_M,
    unknown_inflation_radius_m=NAV_UNKNOWN_INFLATION_RADIUS_M,
    wall_closing_radius_m=NAV_WALL_CLOSING_RADIUS_M,
    block_unknown=True,
):
    occupied = set()
    uncertain = set()
    unknown = set()
    obstacle_inflation_radius_cells = meters_to_cells(obstacle_inflation_radius_m, resolution)
    unknown_inflation_radius_cells = meters_to_cells(unknown_inflation_radius_m, resolution)
    wall_closing_radius_cells = meters_to_cells(wall_closing_radius_m, resolution)

    for row in range(height):
        for col in range(width):
            value = cell_value(data, width, col, row)
            if value < 0:
                unknown.add((col, row))
            elif value >= occupied_threshold:
                occupied.add((col, row))
            elif value > free_threshold:
                uncertain.add((col, row))

    structural_cells = occupied | uncertain
    structural_cells = close_cells(structural_cells, wall_closing_radius_cells, width, height)

    blocked = dilate_cells(structural_cells, obstacle_inflation_radius_cells, width, height)
    if block_unknown:
        blocked |= dilate_cells(unknown, unknown_inflation_radius_cells, width, height)
        blocked |= unknown
    return {
        "blocked": blocked,
        "occupied": occupied,
        "uncertain": uncertain,
        "unknown": unknown,
        "obstacle_inflation_radius_cells": obstacle_inflation_radius_cells,
        "unknown_inflation_radius_cells": unknown_inflation_radius_cells,
        "wall_closing_radius_cells": wall_closing_radius_cells,
    }


def is_traversable(blocked_cells, width, height, col, row):
    return in_bounds(col, row, width, height) and (col, row) not in blocked_cells


def snap_to_nearest_traversable_cell(blocked_cells, width, height, requested_col, requested_row, max_radius):
    if is_traversable(blocked_cells, width, height, requested_col, requested_row):
        return requested_col, requested_row, False

    best = None
    best_distance = None
    for radius in range(1, max_radius + 1):
        for row in range(max(0, requested_row - radius), min(height, requested_row + radius + 1)):
            for col in range(max(0, requested_col - radius), min(width, requested_col + radius + 1)):
                if max(abs(col - requested_col), abs(row - requested_row)) != radius:
                    continue
                if not is_traversable(blocked_cells, width, height, col, row):
                    continue
                distance = math.hypot(col - requested_col, row - requested_row)
                if best is None or distance < best_distance:
                    best = (col, row)
                    best_distance = distance
        if best is not None:
            return best[0], best[1], True
    return None, None, False


def snap_pose_to_nearest_traversable_map_cell(
    map_payload,
    pose,
    *,
    pose_label,
    max_snap_distance_m=NAV_MAX_SNAP_DISTANCE_M,
    occupied_threshold=NAV_OCCUPIED_THRESHOLD,
    free_threshold=NAV_FREE_THRESHOLD,
    obstacle_inflation_radius_m=0.0,
    unknown_inflation_radius_m=0.0,
    wall_closing_radius_m=0.0,
    allow_unknown=False,
):
    map_data = map_payload.get("map") or {}
    resolution = float(map_data.get("resolution") or 0.0)
    width = int(map_data.get("width") or 0)
    height = int(map_data.get("height") or 0)
    origin = map_data.get("origin") or {}
    origin_x = float(origin.get("x") or 0.0)
    origin_y = float(origin.get("y") or 0.0)
    data = list(map_data.get("data") or [])

    if resolution <= 0 or width <= 0 or height <= 0 or len(data) != width * height:
        return dict(pose), False

    requested_col, requested_row = world_to_grid(
        float(pose["x"]),
        float(pose["y"]),
        origin_x,
        origin_y,
        resolution,
    )
    if not in_bounds(requested_col, requested_row, width, height):
        raise HTTPException(
            status_code=400,
            detail=f"Requested {pose_label} pose is outside the Nav2 global costmap bounds",
        )

    navigation_map = build_blocked_cells(
        data,
        width,
        height,
        resolution,
        occupied_threshold=occupied_threshold,
        free_threshold=free_threshold,
        obstacle_inflation_radius_m=obstacle_inflation_radius_m,
        unknown_inflation_radius_m=unknown_inflation_radius_m,
        wall_closing_radius_m=wall_closing_radius_m,
        block_unknown=not allow_unknown,
    )
    snap_radius_cells = max(1, meters_to_cells(max_snap_distance_m, resolution))
    snapped_col, snapped_row, was_snapped = snap_to_nearest_traversable_cell(
        navigation_map["blocked"],
        width,
        height,
        requested_col,
        requested_row,
        max_radius=snap_radius_cells,
    )
    if snapped_col is None or snapped_row is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"No nearby traversable {pose_label} cell was found on the Nav2 global costmap"
            ),
        )

    snapped_pose = dict(pose)
    if was_snapped:
        snapped_world = grid_to_world(snapped_col, snapped_row, origin_x, origin_y, resolution)
        snapped_pose["x"] = float(snapped_world["x"])
        snapped_pose["y"] = float(snapped_world["y"])
    return snapped_pose, was_snapped


def bresenham_cells(start_col, start_row, end_col, end_row):
    x0 = int(start_col)
    y0 = int(start_row)
    x1 = int(end_col)
    y1 = int(end_row)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    cells = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def has_line_of_sight(blocked_cells, width, height, start_cell, end_cell):
    cells = bresenham_cells(start_cell[0], start_cell[1], end_cell[0], end_cell[1])
    previous = None
    for col, row in cells:
        if not is_traversable(blocked_cells, width, height, col, row):
            return False
        if previous is not None:
            dcol = col - previous[0]
            drow = row - previous[1]
            if dcol != 0 and drow != 0:
                if not is_traversable(blocked_cells, width, height, previous[0] + dcol, previous[1]):
                    return False
                if not is_traversable(blocked_cells, width, height, previous[0], previous[1] + drow):
                    return False
        previous = (col, row)
    return True


def shortcut_grid_path(grid_path, blocked_cells, width, height):
    if len(grid_path) <= 2:
        return list(grid_path)

    smoothed = [grid_path[0]]
    anchor_index = 0
    probe_index = 1

    while probe_index < len(grid_path):
        if has_line_of_sight(blocked_cells, width, height, grid_path[anchor_index], grid_path[probe_index]):
            probe_index += 1
            continue

        smoothed.append(grid_path[probe_index - 1])
        anchor_index = probe_index - 1
        probe_index = anchor_index + 1

    if smoothed[-1] != grid_path[-1]:
        smoothed.append(grid_path[-1])
    return smoothed


def enforce_axis_aligned_goal_approach(grid_path, blocked_cells, width, height):
    if len(grid_path) < 2:
        return list(grid_path)

    previous = grid_path[-2]
    goal = grid_path[-1]
    dcol = goal[0] - previous[0]
    drow = goal[1] - previous[1]

    if dcol == 0 or drow == 0:
        return list(grid_path)

    candidates = [
        (goal[0], previous[1]),
        (previous[0], goal[1]),
    ]

    for intermediate in candidates:
        if intermediate == previous or intermediate == goal:
            continue
        if not is_traversable(blocked_cells, width, height, intermediate[0], intermediate[1]):
            continue
        if not has_line_of_sight(blocked_cells, width, height, previous, intermediate):
            continue
        if not has_line_of_sight(blocked_cells, width, height, intermediate, goal):
            continue

        adjusted = list(grid_path[:-1])
        if adjusted[-1] != intermediate:
            adjusted.append(intermediate)
        adjusted.append(goal)
        return adjusted

    return list(grid_path)


def compute_astar_path(map_payload, start_pose, goal_pose, allow_nearest_reachable_goal=False):
    map_data = map_payload.get("map") or {}
    resolution = float(map_data.get("resolution") or 0.0)
    width = int(map_data.get("width") or 0)
    height = int(map_data.get("height") or 0)
    origin = map_data.get("origin") or {}
    origin_x = float(origin.get("x") or 0.0)
    origin_y = float(origin.get("y") or 0.0)
    data = list(map_data.get("data") or [])

    if resolution <= 0 or width <= 0 or height <= 0 or len(data) != width * height:
        raise HTTPException(status_code=503, detail="Occupancy map is incomplete or invalid")

    navigation_map = build_blocked_cells(data, width, height, resolution)
    blocked_cells = navigation_map["blocked"]
    snap_radius_cells = max(1, meters_to_cells(NAV_MAX_SNAP_DISTANCE_M, resolution))

    requested_start = world_to_grid(start_pose["x"], start_pose["y"], origin_x, origin_y, resolution)
    requested_goal = world_to_grid(goal_pose["x"], goal_pose["y"], origin_x, origin_y, resolution)

    if not in_bounds(requested_start[0], requested_start[1], width, height):
        raise HTTPException(status_code=400, detail="Current robot pose is outside the occupancy map bounds")
    if not in_bounds(requested_goal[0], requested_goal[1], width, height):
        raise HTTPException(status_code=400, detail="Detection robot pose is outside the occupancy map bounds")

    start_col, start_row, start_snapped = snap_to_nearest_traversable_cell(
        blocked_cells, width, height, requested_start[0], requested_start[1], max_radius=snap_radius_cells
    )
    goal_col, goal_row, goal_snapped = snap_to_nearest_traversable_cell(
        blocked_cells, width, height, requested_goal[0], requested_goal[1], max_radius=snap_radius_cells
    )

    if start_col is None or start_row is None:
        raise HTTPException(status_code=400, detail="No nearby traversable start cell was found")
    if goal_col is None or goal_row is None:
        raise HTTPException(status_code=400, detail="No nearby traversable goal cell was found")

    start = (start_col, start_row)
    requested_goal_cell = (goal_col, goal_row)
    goal = requested_goal_cell
    frontier = [(0.0, start)]
    came_from = {start: None}
    g_score = {start: 0.0}
    neighbor_offsets = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for dcol, drow, step_cost in neighbor_offsets:
            next_col = current[0] + dcol
            next_row = current[1] + drow
            if not is_traversable(blocked_cells, width, height, next_col, next_row):
                continue

            if dcol != 0 and drow != 0:
                if not is_traversable(blocked_cells, width, height, current[0] + dcol, current[1]):
                    continue
                if not is_traversable(blocked_cells, width, height, current[0], current[1] + drow):
                    continue

            neighbor = (next_col, next_row)
            tentative_g_score = g_score[current] + step_cost
            if tentative_g_score >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            heuristic = math.hypot(goal[0] - next_col, goal[1] - next_row)
            heapq.heappush(frontier, (tentative_g_score + heuristic, neighbor))

    used_nearest_reachable_goal = False
    if goal not in came_from:
        if not allow_nearest_reachable_goal:
            raise HTTPException(status_code=400, detail="No collision-free path found on the occupancy map")

        reachable_candidates = [cell for cell in g_score.keys() if cell != start]
        if not reachable_candidates:
            reachable_candidates = [start]
        goal = min(
            reachable_candidates,
            key=lambda cell: (
                math.hypot(goal[0] - cell[0], goal[1] - cell[1]),
                g_score[cell],
            ),
        )
        used_nearest_reachable_goal = goal != requested_goal_cell

    grid_path = []
    current = goal
    while current is not None:
        grid_path.append(current)
        current = came_from[current]
    grid_path.reverse()
    smoothed_grid_path = shortcut_grid_path(grid_path, blocked_cells, width, height)

    world_path = [grid_to_world(col, row, origin_x, origin_y, resolution) for col, row in smoothed_grid_path]
    path_length_m = 0.0
    for idx in range(1, len(world_path)):
        path_length_m += math.hypot(
            world_path[idx]["x"] - world_path[idx - 1]["x"],
            world_path[idx]["y"] - world_path[idx - 1]["y"],
        )

    return {
        "path": world_path,
        "path_length_m": path_length_m,
        "num_waypoints": len(world_path),
        "start": {
            "requested_world": {
                "x": float(start_pose["x"]),
                "y": float(start_pose["y"]),
                "z": float(start_pose.get("z", 0.0)),
                "frame_id": "world",
            },
            "requested_grid": {"col": requested_start[0], "row": requested_start[1]},
            "planned_grid": {"col": start_col, "row": start_row},
            "planned_world": grid_to_world(start_col, start_row, origin_x, origin_y, resolution),
            "was_snapped": start_snapped,
        },
        "goal": {
            "requested_world": {
                "x": float(goal_pose["x"]),
                "y": float(goal_pose["y"]),
                "z": float(goal_pose.get("z", 0.0)),
                "frame_id": "world",
            },
            "requested_grid": {"col": requested_goal[0], "row": requested_goal[1]},
            "planned_grid": {"col": goal[0], "row": goal[1]},
            "planned_world": grid_to_world(goal[0], goal[1], origin_x, origin_y, resolution),
            "was_snapped": goal_snapped or used_nearest_reachable_goal,
        },
        "planner": {
            "name": "astar",
            "obstacle_inflation_radius_cells": navigation_map["obstacle_inflation_radius_cells"],
            "obstacle_inflation_radius_m": NAV_OBSTACLE_INFLATION_RADIUS_M,
            "unknown_inflation_radius_cells": navigation_map["unknown_inflation_radius_cells"],
            "unknown_inflation_radius_m": NAV_UNKNOWN_INFLATION_RADIUS_M,
            "wall_closing_radius_cells": navigation_map["wall_closing_radius_cells"],
            "wall_closing_radius_m": NAV_WALL_CLOSING_RADIUS_M,
            "free_threshold": NAV_FREE_THRESHOLD,
            "occupied_threshold": NAV_OCCUPIED_THRESHOLD,
            "max_snap_distance_m": NAV_MAX_SNAP_DISTANCE_M,
            "raw_num_waypoints": len(grid_path),
            "smoothed_num_waypoints": len(smoothed_grid_path),
            "used_nearest_reachable_goal": used_nearest_reachable_goal,
            "occupancy_policy": {
                "traversable": f"occupancy value in [0, {NAV_FREE_THRESHOLD}] and outside inflated obstacle/uncertainty buffers",
                "occupied": f"occupancy value >= {NAV_OCCUPIED_THRESHOLD}",
                "uncertain": f"occupancy value in ({NAV_FREE_THRESHOLD}, {NAV_OCCUPIED_THRESHOLD}) and treated as blocked",
                "unknown": "occupancy value < 0 and treated as blocked",
            },
            "planning_steps": [
                "Build conservative navigation mask",
                "Close narrow wall cracks",
                "Inflate obstacles by robot footprint margin",
                "Inflate unknown space conservatively",
                "Snap start and goal to nearby safe cells",
                "Run A* on traversable cells",
                "Shortcut segments with line-of-sight validation",
            ],
        },
    }


def fetch_interactions(scene_id=None, building=None, object_id=None, both_linked=False):
    query = """
        SELECT
            i.id,
            i.action,
            i.caption,
            COALESCE(subject_obs.scene_id, object_obs.scene_id) AS scene_id,
            i.created_at,
            i.model_source,
            i.subject_bbox,
            i.object_bbox,
            s.timestamp AS scene_timestamp,
            subject_obs.object_id,
            so.class_id,
            subject_obs.id,
            subject_obs.yolo_track_id,
            subject_person.person_id,
            object_obs.object_id,
            oo.class_id,
            object_obs.id,
            object_obs.yolo_track_id,
            object_person.person_id
        FROM interactions i
        LEFT JOIN object_observations subject_obs
            ON subject_obs.id = i.subject_id
        LEFT JOIN object_observations object_obs
            ON object_obs.id = i.object_id
        LEFT JOIN LATERAL (
            SELECT fo.person_id
            FROM face_observations fo
            WHERE fo.object_id = subject_obs.object_id
               OR (
                    fo.scene_id IS NOT DISTINCT FROM subject_obs.scene_id
                    AND fo.yolo_track_id IS NOT NULL
                    AND subject_obs.yolo_track_id IS NOT NULL
                    AND fo.yolo_track_id = subject_obs.yolo_track_id
               )
            ORDER BY fo.created_at DESC, fo.id DESC
            LIMIT 1
        ) subject_person ON TRUE
        LEFT JOIN LATERAL (
            SELECT fo.person_id
            FROM face_observations fo
            WHERE fo.object_id = object_obs.object_id
               OR (
                    fo.scene_id IS NOT DISTINCT FROM object_obs.scene_id
                    AND fo.yolo_track_id IS NOT NULL
                    AND object_obs.yolo_track_id IS NOT NULL
                    AND fo.yolo_track_id = object_obs.yolo_track_id
               )
            ORDER BY fo.created_at DESC, fo.id DESC
            LIMIT 1
        ) object_person ON TRUE
        LEFT JOIN scenes s
            ON s.id = COALESCE(subject_obs.scene_id, object_obs.scene_id)
        LEFT JOIN objects so
            ON so.id = subject_obs.object_id
        LEFT JOIN objects oo
            ON oo.id = object_obs.object_id
    """
    params = []
    conditions = []
    if scene_id is not None:
        conditions.append("(subject_obs.scene_id = %s OR object_obs.scene_id = %s)")
        params.extend([scene_id, scene_id])
    if object_id is not None:
        conditions.append("(subject_obs.object_id = %s OR object_obs.object_id = %s)")
        params.extend([object_id, object_id])
    if both_linked:
        conditions.append("i.subject_id IS NOT NULL AND i.object_id IS NOT NULL")
    map_id = _resolve_map_id(building)
    if map_id is not None:
        conditions.append("(i.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = i.scene_id AND s2.map_id = %s))")
        params.extend([map_id, map_id])
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += """
        ORDER BY COALESCE(s.timestamp, i.created_at) DESC, i.id DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    interactions = []
    for row in rows:
        entry = {
            "id": row[0],
            "action": row[1],
            "caption": row[2],
            "scene_id": row[3],
            "created_at": row[4],
            "model_source": row[5],
            "subject_bbox": row[6],
            "object_bbox": row[7],
            "scene_timestamp": row[8],
            "objects": [],
        }

        if row[9] is not None:
            entry["objects"].append(
                {
                    "role": "subject",
                    "observation_id": row[11],
                    "object_id": row[9],
                    "class_id": row[10],
                    "class_name": class_name_from_id(row[10]),
                    "yolo_track_id": row[12],
                    "image_url": f"/api/observations/{row[11]}/image" if row[11] is not None else None,
                }
            )

        if row[14] is not None:
            entry["objects"].append(
                {
                    "role": "object",
                    "observation_id": row[16],
                    "object_id": row[14],
                    "class_id": row[15],
                    "class_name": class_name_from_id(row[15]),
                    "yolo_track_id": row[17],
                    "image_url": f"/api/observations/{row[16]}/image" if row[16] is not None else None,
                }
            )

        interactions.append(entry)

    return interactions

CLUSTER_MERGE_INTERVAL_HOURS = float(os.environ.get("CLUSTER_MERGE_INTERVAL_HOURS", "24"))


async def _daily_cluster_merge_task():
    """Background task: run convergent cluster merging on a fixed interval."""
    while True:
        await asyncio.sleep(CLUSTER_MERGE_INTERVAL_HOURS * 3600)
        try:
            _run_convergent_merge(similarity_threshold=0.643, min_observations=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[cluster-merge] scheduled run failed: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_daily_cluster_merge_task())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(lifespan=lifespan)

class NoCacheMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers", [])
                headers.append([b"cache-control", b"no-cache, no-store, must-revalidate"])
                headers.append([b"pragma", b"no-cache"])
                headers.append([b"expires", b"0"])
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)

app.add_middleware(NoCacheMiddleware)

# --- Crop gallery endpoint (must be after app = FastAPI) ---
def pil_image_to_base64(img: PILImage.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_objects_name_schema_ready = False

def ensure_objects_name_schema(conn):
    global _objects_name_schema_ready
    if _objects_name_schema_ready:
        return conn
    with conn.cursor() as cur:
        cur.execute("ALTER TABLE objects ADD COLUMN IF NOT EXISTS name TEXT")
        cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS mask_image BYTEA")
        cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS original_cropped_image BYTEA")
        cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION")
        cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION")
        cur.execute("ALTER TABLE object_observations ADD COLUMN IF NOT EXISTS attributes_json JSONB")
        cur.execute("ALTER TABLE scenes ADD COLUMN IF NOT EXISTS original_scene_image BYTEA")
        cur.execute("ALTER TABLE scenes ADD COLUMN IF NOT EXISTS stitched_scene_image BYTEA")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS object_observation_parts (
                id BIGSERIAL PRIMARY KEY,
                observation_id BIGINT REFERENCES object_observations(id) ON DELETE CASCADE,
                object_id BIGINT REFERENCES objects(id) ON DELETE CASCADE,
                part_name TEXT NOT NULL,
                embedding VECTOR(384),
                colors_json JSONB,
                bbox_x_min BIGINT,
                bbox_y_min BIGINT,
                bbox_x_max BIGINT,
                bbox_y_max BIGINT,
                quality_score DOUBLE PRECISION,
                preprocessing_json JSONB,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_object_observation_parts_unique_part
            ON object_observation_parts (observation_id, part_name)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_object_observation_parts_object_part
            ON object_observation_parts (object_id, part_name)
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS robot_paths (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                building_name TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                point_count INT DEFAULT 0,
                path_data JSONB NOT NULL
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_robot_paths_name ON robot_paths(name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_robot_paths_created ON robot_paths(created_at DESC)")
    conn.commit()
    _objects_name_schema_ready = True
    return conn

def get_conn():
    conn = psycopg2.connect(DATABASE_URL)
    return ensure_objects_name_schema(conn)


class VideoPublisherRequest(BaseModel):
    frame_dir: str = Field(default=VIDEO_PUBLISHER_DEFAULT_FRAME_DIR, min_length=1)
    image_stride: int = Field(default=5, ge=1, le=100)
    publish_hz: float = Field(default=1.0, gt=0.0, le=30.0)
    loop: bool = True
    recursive: bool = False
    max_images: int = Field(default=0, ge=0, le=50000)
    rgb_topic: str = Field(default=VIDEO_PUBLISHER_RGB_TOPIC, min_length=1)
    create_run_map: bool = True
    run_map_name: str | None = None
    generate_captions: bool = True


class ClusterTestsetBuildRequest(BaseModel):
    name: str | None = None
    dataset_root: str = Field(default=CLUSTER_TESTSET_DEFAULT_DATASET_ROOT, min_length=1)
    split: str | None = None
    image_count: int = Field(default=50, ge=1)
    seed: int | None = None
    recursive: bool = True


class ClusterYoloDetectionRequest(BaseModel):
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections_per_image: int = Field(default=20, ge=1, le=200)
    class_preset: str = Field(default="all", pattern="^(all|persons|objects|custom)$")
    classes: list[int] | None = None
    segmentation: bool = False


class ClusterUpscaleRequest(BaseModel):
    scale: int = Field(default=2, ge=2, le=8)
    method: str = Field(default="lanczos", pattern="^(lanczos|bicubic|bilinear|nearest)$")


class YoloProbeDetectRequest(BaseModel):
    paths: list[str]
    model_path: str = ""
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_detections_per_image: int = Field(default=30, ge=1, le=500)
    class_preset: str = Field(default="all", pattern="^(all|persons|objects)$")
    segmentation: bool = False
    # pipeline mode
    detect_mode: str = Field(default="standard", pattern="^(standard|person_crop|hand_crop|sahi|preprocessed)$")
    # person_crop / hand_crop options
    person_model_path: str = ""
    person_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    person_pad_px: int = Field(default=30, ge=0, le=300)
    # hand_crop options
    hand_model_path: str = ""   # model used to detect hands inside person crops (blank = same as main)
    hand_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    hand_pad_px: int = Field(default=20, ge=0, le=300)
    hand_class_id: int = Field(default=0, ge=-1, le=79)  # class used as "hand" inside person crop; -1 = any class
    # SAHI options
    sahi_slice_size: int = Field(default=320, ge=64, le=1280)
    sahi_overlap_ratio: float = Field(default=0.2, ge=0.0, le=0.5)
    # preprocessing options (used when detect_mode=preprocessed, but can stack on any mode)
    preprocess_clahe: bool = False
    preprocess_gamma: float = Field(default=1.0, ge=0.1, le=5.0)
    preprocess_sharpen: bool = False
    preprocess_denoise: bool = False
    preprocess_auto_brighten: bool = False


class ClusterExperimentRunRequest(BaseModel):
    name: str | None = None
    variants: list[str] = Field(default_factory=list)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    cluster_method: str = Field(default="greedy", pattern="^(greedy|hdbscan)$")
    cluster_methods: list[str] = Field(default_factory=list)
    post_merge_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    auto_thresholds: bool = False
    stream: bool = False


class ClusterExperimentSaveRequest(BaseModel):
    name: str | None = None
    testset_id: str
    results: list[dict]
    thresholds: list[float | None] = Field(default_factory=list)
    metadata: dict | None = None


class ClusterGroundTruthLabel(BaseModel):
    index: int = Field(ge=0)
    detection_id: str | None = None
    identity: int | str | None = None
    camera: int | None = None
    note: str | None = None


class ClusterGroundTruthUpdateRequest(BaseModel):
    labels: list[ClusterGroundTruthLabel] = Field(default_factory=list)


def render_page(request: Request, template_name: str, active_page: str):
    return templates.TemplateResponse(
        request,
        template_name,
        {
            "active_page": active_page,
            "video_publisher_default_frame_dir": VIDEO_PUBLISHER_DEFAULT_FRAME_DIR,
        },
    )


def _load_vlm_tool_specs() -> list[dict]:
    from agent.tools import TOOL_SCHEMAS

    navigation_tool_names = {
        "get_room_navigation_target",
        "move_to_position",
    }
    tool_specs: list[dict] = []
    for schema in TOOL_SCHEMAS:
        function = schema.get("function") or {}
        tool_name = function.get("name", "")
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        required = set(parameters.get("required") or [])
        tool_specs.append(
            {
                "name": tool_name,
                "category": "navigation" if tool_name in navigation_tool_names else "data_retrieval",
                "description": str(function.get("description") or "").strip(),
                "arguments": [
                    {
                        "name": name,
                        "type": spec.get("type", "any"),
                        "description": str(spec.get("description") or "").strip(),
                        "required": name in required,
                    }
                    for name, spec in properties.items()
                ],
            }
        )
    return sorted(tool_specs, key=lambda item: (item["category"], item["name"]))


def _normalize_home_tab(tab: str | None) -> str:
    value = (tab or "scenes").strip().lower()
    return value if value in VALID_HOME_TABS else "scenes"


def _docker_exec(script: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", "exec", VIDEO_PUBLISHER_CONTAINER, "bash", "-lc", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError as exc:
        return _docker_exec_via_socket(script)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Timed out while talking to the runtime container.") from exc


def _docker_socket_request(method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    sock_path = "/var/run/docker.sock"
    if not os.path.exists(sock_path):
        raise HTTPException(
            status_code=503,
            detail="Docker socket is not mounted in the web service. Recreate the web container first.",
        )

    payload = b""
    if body is not None:
        payload = json.dumps(body).encode("utf-8")

    request_lines = [
        f"{method} {path} HTTP/1.1",
        "Host: docker",
        "Connection: close",
    ]
    if payload:
        request_lines.extend(
            [
                "Content-Type: application/json",
                f"Content-Length: {len(payload)}",
            ]
        )
    request_bytes = ("\r\n".join(request_lines) + "\r\n\r\n").encode("utf-8") + payload

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        try:
            client.connect(sock_path)
        except OSError as exc:
            raise HTTPException(status_code=503, detail=f"Failed to connect to Docker socket: {exc}") from exc
        client.sendall(request_bytes)
        chunks = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)

    raw = b"".join(chunks)
    header_bytes, _, body_bytes = raw.partition(b"\r\n\r\n")
    header_text = header_bytes.decode("utf-8", errors="replace")
    status_line = header_text.splitlines()[0] if header_text else ""
    try:
        status_code = int(status_line.split(" ", 2)[1])
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Unexpected Docker API response: {status_line or 'empty response'}") from exc

    if "transfer-encoding: chunked" in header_text.lower():
        body_bytes = _decode_chunked_body(body_bytes)

    if status_code >= 400:
        error_text = body_bytes.decode("utf-8", errors="replace").strip()
        raise HTTPException(status_code=503, detail=f"Docker API error {status_code}: {error_text}")
    return status_code, body_bytes


def _decode_chunked_body(data: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(data):
        end = data.find(b"\r\n", i)
        if end == -1:
            break
        size_hex = data[i:end].decode("ascii", errors="replace").split(";")[0].strip()
        try:
            chunk_size = int(size_hex, 16)
        except ValueError:
            break
        if chunk_size == 0:
            break
        chunk_start = end + 2
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(data):
            break
        out.extend(data[chunk_start:chunk_end])
        i = chunk_end + 2
    return bytes(out)


def _docker_exec_via_socket(script: str) -> subprocess.CompletedProcess[str]:
    exit_marker = "__COPILOT_DOCKER_EXEC_EXIT_CODE__="
    wrapped_script = (
        f"bash -lc {shlex.quote(script)}\n"
        "exec_rc=$?\n"
        f"printf '\\n{exit_marker}%s\\n' \"$exec_rc\"\n"
        "exit 0\n"
    )
    _, create_body = _docker_socket_request(
        "POST",
        f"/v1.43/containers/{urllib.parse.quote(VIDEO_PUBLISHER_CONTAINER, safe='')}/exec",
        body={
            "AttachStdout": True,
            "AttachStderr": True,
            "Cmd": ["bash", "-lc", wrapped_script],
        },
    )
    try:
        exec_id = json.loads(create_body)["Id"]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to parse Docker exec create response: {create_body!r}") from exc

    _, start_body = _docker_socket_request(
        "POST",
        f"/v1.43/exec/{urllib.parse.quote(exec_id, safe='')}/start",
        body={
            "Detach": False,
            "Tty": False,
        },
    )

    _, inspect_body = _docker_socket_request(
        "GET",
        f"/v1.43/exec/{urllib.parse.quote(exec_id, safe='')}/json",
    )
    stdout = _decode_docker_raw_stream(start_body)
    exit_code = None
    marker_index = stdout.rfind(exit_marker)
    if marker_index != -1:
        marker_value = stdout[marker_index + len(exit_marker):].splitlines()[0].strip()
        try:
            exit_code = int(marker_value)
        except Exception:
            exit_code = None
        stdout = stdout[:marker_index].rstrip("\n")

    if exit_code is None:
        try:
            exit_code = int(json.loads(inspect_body).get("ExitCode", 1))
        except Exception:
            exit_code = 1

    return subprocess.CompletedProcess(
        args=["docker-socket-exec", VIDEO_PUBLISHER_CONTAINER],
        returncode=exit_code,
        stdout=stdout,
        stderr="",
    )

class YoloProbePreviewRequest(BaseModel):
    path: str
    preprocess_clahe: bool = False
    preprocess_gamma: float = Field(default=1.0, ge=0.1, le=5.0)
    preprocess_sharpen: bool = False
    preprocess_denoise: bool = False
    preprocess_auto_brighten: bool = False


def _decode_docker_raw_stream(payload: bytes | str) -> str:
    if isinstance(payload, str):
        data = payload.encode("utf-8", errors="surrogateescape")
    else:
        data = payload
    if len(data) < 8:
        return payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload

    parts: list[bytes] = []
    offset = 0
    parsed_any = False
    while offset + 8 <= len(data):
        header = data[offset : offset + 8]
        stream_type = header[0]
        frame_size = int.from_bytes(header[4:8], "big")
        next_offset = offset + 8 + frame_size
        if stream_type not in (1, 2) or next_offset > len(data):
            if not parsed_any:
                return payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
            break
        parsed_any = True
        parts.append(data[offset + 8 : next_offset])
        offset = next_offset

    if not parsed_any:
        return payload.decode("utf-8", errors="replace") if isinstance(payload, bytes) else payload
    return b"".join(parts).decode("utf-8", errors="replace")


def _to_container_frame_dir(frame_dir: str) -> str:
    candidate = Path(frame_dir).expanduser()
    mappings = [(VIDEO_PUBLISHER_HOST_RUNTIME_ROOT, VIDEO_PUBLISHER_CONTAINER_RUNTIME_ROOT)]
    for mount in VIDEO_PUBLISHER_EXTRA_RUNTIME_MOUNTS.split(";"):
        mount = mount.strip()
        if not mount or "=" not in mount:
            continue
        host_root, container_root = mount.split("=", 1)
        mappings.append((host_root.strip(), container_root.strip()))

    for host_root_value, container_root_value in mappings:
        if not host_root_value or not container_root_value:
            continue
        try:
            relative = candidate.relative_to(Path(host_root_value).expanduser())
            return str(Path(container_root_value) / relative)
        except Exception:
            pass
    return str(candidate)


def _default_pipeline_run_map_name(frame_dir: str) -> str:
    source_name = Path(frame_dir).expanduser().name or "frames"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return sanitize_toolbox_map_name(f"pipeline_{source_name}_{timestamp}")


def _parse_boolish(value):
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "on"}:
        return True
    if text in {"false", "0", "no", "off"}:
        return False
    return None


def _ensure_pipeline_run_map(map_name: str) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO maps (name)
                VALUES (%s)
                ON CONFLICT (name)
                DO UPDATE SET name = EXCLUDED.name
                RETURNING id
                """,
                (map_name,),
            )
            row = cur.fetchone()
        conn.commit()
    return int(row[0])


def _migrate_observations_to_map(old_map_id: int | None, new_map_id: int) -> int:
    """Migrate observations from old_map_id to new_map_id. Returns number of rows updated."""
    if old_map_id is None or old_map_id == new_map_id:
        return 0
    total_updated = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for table in ("scenes", "object_observations", "face_observations", "interactions"):
                cur.execute(
                    f"UPDATE {table} SET map_id = %s WHERE map_id = %s",
                    (new_map_id, old_map_id),
                )
                total_updated += cur.rowcount
        conn.commit()
    return total_updated


def _activate_pipeline_run_map(map_name: str) -> None:
    persist_active_toolbox_map_record(
        {
            "name": map_name,
            "building_name": map_name,
        },
        mode="recording",
        observation_map_name=map_name,
    )


def _video_publisher_status_payload() -> dict:
    script = f"""
PID_FILE={shlex.quote(VIDEO_PUBLISHER_PID_FILE)}
LOG_FILE={shlex.quote(VIDEO_PUBLISHER_LOG_FILE)}
pid=""
status="stopped"
captions_enabled=""
face_detector_available=""
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if [ -r "/proc/$pid/stat" ]; then
      proc_state="$(awk '{{print $3}}' "/proc/$pid/stat" 2>/dev/null || true)"
      if [ "$proc_state" = "Z" ]; then
        status="stale"
      else
        status="running"
      fi
    else
      status="running"
    fi
  else
    status="stale"
  fi
fi
source /opt/ros/humble/setup.bash >/dev/null 2>&1 || true
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash >/dev/null 2>&1 || true
fi
if command -v ros2 >/dev/null 2>&1; then
  caption_param_output="$(timeout 3s ros2 param get /scene_description_node captions_enabled 2>&1 || true)"
  case "$caption_param_output" in
    *True*|*true*) captions_enabled=true ;;
    *False*|*false*) captions_enabled=false ;;
  esac
  if timeout 3s ros2 node list 2>/dev/null | grep -qx "/face_detector_node"; then
    face_detector_available=true
  else
    face_detector_available=false
  fi
fi
printf 'status=%s\\n' "$status"
printf 'pid=%s\\n' "$pid"
printf 'captions_enabled=%s\\n' "$captions_enabled"
printf 'face_detector_available=%s\\n' "$face_detector_available"
if [ -f "$LOG_FILE" ]; then
  echo "__LOG__"
  tail -n 40 "$LOG_FILE"
fi
"""
    result = _docker_exec(script)
    if result.returncode != 0:
        raise HTTPException(
            status_code=503,
            detail=(result.stderr or result.stdout or "Failed to inspect video publisher status.").strip(),
        )

    lines = result.stdout.splitlines()
    log_index = lines.index("__LOG__") if "__LOG__" in lines else len(lines)
    header_lines = lines[:log_index]
    log_tail = "\n".join(lines[log_index + 1:]).strip() if log_index < len(lines) else ""
    parsed = {}
    for line in header_lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()

    payload = {
        "status": parsed.get("status", "unknown"),
        "pid": parsed.get("pid") or None,
        "log_tail": log_tail,
        "container": VIDEO_PUBLISHER_CONTAINER,
        "database_url": _masked_database_url(),
        "last_request": _video_publisher_last_request or None,
        "captions_enabled": _parse_boolish(parsed.get("captions_enabled")),
        "face_detector_available": _parse_boolish(parsed.get("face_detector_available")),
    }
    return payload


def _fetch_table_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    with get_conn() as conn:
        with conn.cursor() as cur:
            for table in DB_TABLES:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = int(cur.fetchone()[0])
    return counts


def _masked_database_url() -> str:
    parsed = urllib.parse.urlsplit(DATABASE_URL)
    if not parsed.scheme or "@" not in parsed.netloc:
        return DATABASE_URL
    userinfo, hostinfo = parsed.netloc.rsplit("@", 1)
    username = userinfo.split(":", 1)[0]
    # Show the externally mapped host port (35432) instead of the internal Docker port (5432)
    hostinfo = hostinfo.replace(":5432", ":35432")
    safe_netloc = f"{username}:***@{hostinfo}"
    return urllib.parse.urlunsplit((parsed.scheme, safe_netloc, parsed.path, parsed.query, parsed.fragment))


def _database_specs_payload() -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            postgres_version = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            vector_row = cur.fetchone()
            vector_version = vector_row[0] if vector_row else None
            cur.execute(
                """
                SELECT table_name, column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public'
                ORDER BY table_name, ordinal_position
                """
            )
            column_rows = cur.fetchall()
            cur.execute(
                """
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
                """
            )
            index_rows = cur.fetchall()

    counts = _fetch_table_counts()
    tables: dict[str, dict] = {
        table: {"name": table, "row_count": counts.get(table, 0), "columns": [], "indexes": []}
        for table in DB_TABLES
    }

    for table_name, column_name, data_type, is_nullable in column_rows:
        tables.setdefault(table_name, {"name": table_name, "row_count": counts.get(table_name, 0), "columns": [], "indexes": []})
        tables[table_name]["columns"].append(
            {
                "name": column_name,
                "type": data_type,
                "nullable": is_nullable == "YES",
            }
        )

    for table_name, index_name, index_def in index_rows:
        tables.setdefault(table_name, {"name": table_name, "row_count": counts.get(table_name, 0), "columns": [], "indexes": []})
        tables[table_name]["indexes"].append(
            {
                "name": index_name,
                "definition": index_def,
            }
        )

    return {
        "status": "ok",
        "database_url": _masked_database_url(),
        "postgres_version": postgres_version,
        "vector_extension_version": vector_version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": [tables[name] for name in sorted(tables.keys())],
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request, tab: str = "scenes"):
    return render_page(request, "index.html", _normalize_home_tab(tab))


@app.get("/clusters", response_class=HTMLResponse)
def clusters_page(request: Request):
    return render_page(request, "clusters.html", "clusters")

@app.get("/cluster-testsets", response_class=HTMLResponse)
def cluster_testsets_page(request: Request):
    return render_page(request, "cluster_testsets.html", "cluster_testsets")


@app.get("/yolo-probe", response_class=HTMLResponse)
def yolo_probe_page(request: Request):
    return render_page(request, "yolo_probe.html", "yolo_probe")


@app.get("/api/yolo-probe/scan")
def yolo_probe_scan(
    folder: str = Query(...),
    limit: int = Query(default=200, ge=1, le=2000),
    recursive: bool = Query(default=True),
):
    resolved = _resolve_dataset_root(folder)
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {resolved}")
    images: list[Path] = []
    iterator = resolved.rglob("*") if recursive else resolved.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
    _YOLO_PROBE_ALLOWED_FOLDERS.add(str(resolved.resolve()))
    total = len(images)
    return {
        "folder": str(resolved),
        "total": total,
        "images": [{"path": str(p), "filename": p.name} for p in images[:limit]],
    }


@app.get("/api/yolo-probe/models")
def yolo_probe_models():
    default = str(CLUSTER_TESTSET_YOLO_MODEL or "").strip()
    search_dirs = [Path("/models")]
    if default:
        search_dirs.append(Path(default).parent)
    seen: set[str] = set()
    models = []
    for d in search_dirs:
        if d.exists():
            for p in sorted(d.glob("*.onnx")):
                key = str(p.resolve())
                if key not in seen:
                    seen.add(key)
                    models.append({"path": key, "name": p.name})
    return {"models": models, "default": default}


@app.get("/api/yolo-probe/image")
def yolo_probe_image(path: str = Query(...)):
    resolved = str(Path(path).resolve())
    if not any(resolved.startswith(folder) for folder in _YOLO_PROBE_ALLOWED_FOLDERS):
        raise HTTPException(status_code=403, detail="Image path is not in a scanned folder")
    p = Path(resolved)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    image_bytes = p.read_bytes()
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=image_bytes, media_type=media_type)




@app.post("/api/yolo-probe/preview-preprocessed")
def yolo_probe_preview_preprocessed(req: YoloProbePreviewRequest):
    """Return the preprocessed version of an image as a PNG."""
    import io
    resolved = str(Path(req.path).resolve())
    if not any(resolved.startswith(folder) for folder in _YOLO_PROBE_ALLOWED_FOLDERS):
        raise HTTPException(status_code=403, detail="Image path is not in a scanned folder")
    p = Path(resolved)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    image = PILImage.open(str(p)).convert("RGB")
    image = _yolo_probe_preprocess(
        image,
        clahe=req.preprocess_clahe,
        gamma=req.preprocess_gamma,
        sharpen=req.preprocess_sharpen,
        denoise=req.preprocess_denoise,
        auto_brighten=req.preprocess_auto_brighten,
    )
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/api/yolo-probe/detect")
def yolo_probe_detect(req: YoloProbeDetectRequest):
    model_path = (req.model_path or "").strip() or str(CLUSTER_TESTSET_YOLO_MODEL or "").strip()
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="No YOLO model configured. Set CLUSTER_TESTSET_YOLO_MODEL env var or pass model_path in request.",
        )
    session = _load_yolo_session(model_path)
    allowed_classes = _resolve_yolo_detection_classes(req.class_preset, None)

    # For person_crop / hand_crop mode, also load the person detection model (may be same model)
    person_model_path = (req.person_model_path or "").strip() or model_path
    person_session = (
        _load_yolo_session(person_model_path)
        if req.detect_mode in ("person_crop", "hand_crop")
        else None
    )
    # For hand_crop mode, load the hand detection model (runs inside person crop)
    hand_model_path = (req.hand_model_path or "").strip() or model_path
    hand_session = _load_yolo_session(hand_model_path) if req.detect_mode == "hand_crop" else None

    do_preprocess = req.preprocess_clahe or abs(req.preprocess_gamma - 1.0) > 0.01 or req.preprocess_sharpen or req.preprocess_denoise or req.preprocess_auto_brighten

    results = []
    for path_str in req.paths:
        resolved = str(Path(path_str).resolve())
        if not any(resolved.startswith(folder) for folder in _YOLO_PROBE_ALLOWED_FOLDERS):
            results.append({"path": path_str, "detections": [], "error": "not_allowed"})
            continue
        p = Path(resolved)
        if not p.exists() or not p.is_file():
            results.append({"path": path_str, "detections": [], "error": "not_found"})
            continue
        try:
            image = PILImage.open(str(p)).convert("RGB")
            if do_preprocess:
                image = _yolo_probe_preprocess(
                    image,
                    clahe=req.preprocess_clahe,
                    gamma=req.preprocess_gamma,
                    sharpen=req.preprocess_sharpen,
                    denoise=req.preprocess_denoise,
                    auto_brighten=req.preprocess_auto_brighten,
                )
            if req.detect_mode == "sahi":
                dets = _run_yolo_sahi(
                    image,
                    session=session,
                    confidence=req.confidence,
                    iou_threshold=req.iou_threshold,
                    max_detections=req.max_detections_per_image,
                    classes=allowed_classes,
                    slice_size=req.sahi_slice_size,
                    overlap_ratio=req.sahi_overlap_ratio,
                )
                results.append({"path": path_str, "detections": dets})
            elif req.detect_mode == "person_crop":
                person_dets, obj_dets = _run_yolo_person_crop_objects(
                    image,
                    person_session=person_session,
                    obj_session=session,
                    person_confidence=req.person_confidence,
                    obj_confidence=req.confidence,
                    iou_threshold=req.iou_threshold,
                    max_obj_detections=req.max_detections_per_image,
                    obj_classes=allowed_classes,
                    pad_px=req.person_pad_px,
                )
                results.append({
                    "path": path_str,
                    "detections": person_dets + obj_dets,
                    "person_count": len(person_dets),
                    "object_count": len(obj_dets),
                })
            elif req.detect_mode == "hand_crop":
                person_dets, hand_dets, obj_dets = _run_yolo_hand_crop_objects(
                    image,
                    person_session=person_session,
                    hand_session=hand_session,
                    obj_session=session,
                    person_confidence=req.person_confidence,
                    hand_confidence=req.hand_confidence,
                    obj_confidence=req.confidence,
                    iou_threshold=req.iou_threshold,
                    max_obj_detections=req.max_detections_per_image,
                    obj_classes=allowed_classes,
                    person_pad_px=req.person_pad_px,
                    hand_pad_px=req.hand_pad_px,
                    hand_class_id=req.hand_class_id,
                )
                results.append({
                    "path": path_str,
                    "detections": person_dets + hand_dets + obj_dets,
                    "person_count": len(person_dets),
                    "hand_count": len(hand_dets),
                    "object_count": len(obj_dets),
                })
            else:
                # standard or preprocessed — same inference path
                dets = _run_yolo_on_pil(
                    image,
                    session=session,
                    confidence=req.confidence,
                    iou_threshold=req.iou_threshold,
                    max_detections=req.max_detections_per_image,
                    classes=allowed_classes,
                    segmentation=req.segmentation,
                )
                results.append({"path": path_str, "detections": dets})
        except Exception as exc:
            results.append({"path": path_str, "detections": [], "error": str(exc)})
    return {"results": results, "model_path": model_path}


@app.get("/cluster-experiments", response_class=HTMLResponse)
def cluster_experiments_page(request: Request):
    return render_page(request, "cluster_experiments.html", "cluster_experiments")

@app.get("/pipeline", response_class=HTMLResponse)
def pipeline_page(request: Request):
    return render_page(request, "pipeline.html", "pipeline")


@app.get("/robot-pipeline", response_class=HTMLResponse)
def robot_pipeline_page(request: Request):
    return render_page(request, "robot_pipeline.html", "robot_pipeline")


@app.get("/db-admin", response_class=HTMLResponse)
def db_admin_page(request: Request):
    return render_page(request, "db_admin.html", "db")



@app.get("/yolo-compare", response_class=HTMLResponse)
def yolo_compare_page(request: Request):
    return render_page(request, "yolo_compare.html", "yolo_compare")


@app.get("/yolo-finetune", response_class=HTMLResponse)
def yolo_finetune_page(request: Request):
    return render_page(request, "yolo_finetune.html", "yolo_finetune")


@app.get("/slam", response_class=HTMLResponse)
def slam_page(request: Request):
    return render_page(request, "slam.html", "slam")


# ---------------------------------------------------------------------------
# YOLO Pose Eval
# ---------------------------------------------------------------------------

_YOLO_POSE_EVAL_ALLOWED_FOLDERS: set[str] = set()
_YOLO_POSE_EVAL_JOBS: dict[str, dict] = {}
_YOLO_POSE_EVAL_JOBS_LOCK = threading.Lock()
_YOLO_POSE_EVAL_RESULTS_DIR = Path("/shared/yolo_pose_eval")


def _persist_job(job_id: str) -> None:
    """Write the lightweight job metadata to disk so it survives restarts."""
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        job = _YOLO_POSE_EVAL_JOBS.get(job_id)
    if job is None:
        return
    job_dir = _YOLO_POSE_EVAL_RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    # Save a lightweight copy (exclude large in-memory summary; it lives in {job_id}.json)
    lightweight = {
        k: v for k, v in job.items() if k != "summary"
    }
    (job_dir / "job.json").write_text(json.dumps(lightweight, indent=2, default=str))


def _load_historical_jobs() -> None:
    """Scan the results directory and load any historical jobs into memory."""
    if not _YOLO_POSE_EVAL_RESULTS_DIR.exists():
        return
    # New format: subdirectories with job.json or result artifacts.
    for job_dir in sorted(_YOLO_POSE_EVAL_RESULTS_DIR.iterdir()):
        if not job_dir.is_dir():
            continue
        job_id = job_dir.name
        if not job_id.startswith("ype-"):
            continue
        job_file = job_dir / "job.json"
        result_json = job_dir / f"{job_id}.json"
        result_csv = job_dir / f"{job_id}.csv"
        try:
            if job_file.exists():
                job = json.loads(job_file.read_text())
                job_id = job.get("id") or job_id
            elif result_json.exists() or result_csv.exists():
                summary = []
                if result_json.exists():
                    loaded = json.loads(result_json.read_text())
                    summary = loaded if isinstance(loaded, list) else []
                job = {
                    "id": job_id,
                    "name": "Historical eval",
                    "status": "done" if result_json.exists() or result_csv.exists() else "unknown",
                    "total_configs": len(summary),
                    "completed_configs": len(summary),
                    "latest_result": None,
                    "error": None,
                    "summary": summary,
                }
            else:
                continue
            viz_dir = job_dir / f"{job_id}_viz"
            if viz_dir.exists() and viz_dir.is_dir() and not job.get("viz_paths"):
                job["viz_paths"] = [f.name for f in sorted(viz_dir.iterdir()) if f.is_file()]
            job.setdefault("summary", [])
            job.setdefault("viz_paths", [])
            if job.get("status") in {"queued", "running"} and not (result_json.exists() or result_csv.exists()):
                job["status"] = "error"
                job["error"] = job.get("error") or "Interrupted before results were written."
                job.pop("progress", None)
            with _YOLO_POSE_EVAL_JOBS_LOCK:
                if job_id not in _YOLO_POSE_EVAL_JOBS:
                    _YOLO_POSE_EVAL_JOBS[job_id] = job
        except Exception:
            continue

    # Old format: flat files {job_id}.json / {job_id}.csv / {job_id}_viz/
    seen_ids = set()
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        seen_ids = set(_YOLO_POSE_EVAL_JOBS.keys())
    for path in sorted(_YOLO_POSE_EVAL_RESULTS_DIR.iterdir()):
        if not path.is_file() or path.suffix != ".json":
            continue
        job_id = path.stem
        if not job_id.startswith("ype-") or job_id in seen_ids:
            continue
        try:
            summary = json.loads(path.read_text())
            viz_dir = _YOLO_POSE_EVAL_RESULTS_DIR / f"{job_id}_viz"
            viz_paths = []
            if viz_dir.exists() and viz_dir.is_dir():
                viz_paths = [f.name for f in sorted(viz_dir.iterdir()) if f.is_file()]
            job = {
                "id": job_id,
                "name": f"Historical eval",
                "status": "done",
                "total_configs": len(summary) if isinstance(summary, list) else 0,
                "completed_configs": len(summary) if isinstance(summary, list) else 0,
                "latest_result": None,
                "error": None,
                "summary": summary if isinstance(summary, list) else [],
                "viz_paths": viz_paths,
            }
            with _YOLO_POSE_EVAL_JOBS_LOCK:
                if job_id not in _YOLO_POSE_EVAL_JOBS:
                    _YOLO_POSE_EVAL_JOBS[job_id] = job
        except Exception:
            continue


def _load_job_summary_from_disk(job_id: str) -> list[dict] | None:
    """Load the full result summary from the per-job JSON file."""
    path = _resolve_yolo_pose_eval_result_path(job_id, ".json")
    if path is None:
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _resolve_yolo_pose_eval_result_path(job_id: str, suffix: str) -> Path | None:
    """Resolve a result file path, supporting both new (subdir) and old (flat) formats."""
    # New format: {job_id}/{job_id}{suffix}
    new_path = _YOLO_POSE_EVAL_RESULTS_DIR / job_id / f"{job_id}{suffix}"
    if new_path.exists():
        return new_path
    # Old format: {job_id}{suffix}
    old_path = _YOLO_POSE_EVAL_RESULTS_DIR / f"{job_id}{suffix}"
    if old_path.exists():
        return old_path
    return None


def _yolo_pose_eval_artifact_flags(job_id: str) -> dict[str, bool]:
    """Return which downloadable result artifacts exist for a job."""
    has_json = _resolve_yolo_pose_eval_result_path(job_id, ".json") is not None
    has_csv = _resolve_yolo_pose_eval_result_path(job_id, ".csv") is not None
    return {
        "has_json": has_json,
        "has_csv": has_csv,
        "has_results": has_json or has_csv,
    }


def _sync_yolo_pose_eval_job_from_artifacts(job: dict) -> dict:
    """Expose completed artifacts even if in-memory job metadata is stale."""
    flags = _yolo_pose_eval_artifact_flags(job["id"])
    job.update(flags)
    if flags["has_results"] and job.get("status") in {"queued", "running"}:
        total = int(job.get("total_configs") or 0)
        completed = int(job.get("completed_configs") or 0)
        if total == 0 or completed >= total:
            job["status"] = "done"
            job.pop("progress", None)
    return job


def _resolve_yolo_pose_eval_viz_path(job_id: str, filename: str) -> Path | None:
    """Resolve a viz image path, supporting both new (subdir) and old (flat) formats."""
    # New format: {job_id}/{job_id}_viz/{filename}
    new_path = _YOLO_POSE_EVAL_RESULTS_DIR / job_id / f"{job_id}_viz" / filename
    if new_path.exists() and new_path.is_file():
        return new_path
    # Old format: {job_id}_viz/{filename}
    old_path = _YOLO_POSE_EVAL_RESULTS_DIR / f"{job_id}_viz" / filename
    if old_path.exists() and old_path.is_file():
        return old_path
    return None


# Load historical jobs on startup
_load_historical_jobs()


class YoloPoseEvalGtLoadRequest(BaseModel):
    path: str


class YoloPoseEvalGtSaveRequest(BaseModel):
    path: str
    data: list[dict]


class YoloPoseEvalAutoLabelRequest(BaseModel):
    image_folder: str
    model: str = "yolov8n-pose.pt"
    model_type: str = "yolo"
    recursive: bool = False


class YoloPoseEvalRunModel(BaseModel):
    path: str
    model_type: str = "yolo"
    text_prompt: str = "person"
    ablation_brightness: bool = False
    ablation_crop_redetect: bool = False
    tracker: str = "none"


class YoloPoseEvalRunRequest(BaseModel):
    models: list[YoloPoseEvalRunModel]
    conf_thresholds: list[float]
    imgsz: list[int]
    device: str = "0"
    classes: list[str] = Field(default_factory=list)
    gt_path: str
    gt_format: str = "xyxy"
    image_folder: str


@app.get("/yolo-pose-eval", response_class=HTMLResponse)
def yolo_pose_eval_page(request: Request):
    return render_page(request, "yolo_pose_eval.html", "yolo_pose_eval")


@app.get("/api/yolo-pose-eval/scan")
def yolo_pose_eval_scan(
    folder: str = Query(...),
    recursive: bool = Query(default=True),
    limit: int = Query(default=500, ge=1, le=2000),
):
    resolved = _resolve_dataset_root(folder)
    if not resolved.exists() or not resolved.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {resolved}")
    images: list[Path] = []
    iterator = resolved.rglob("*") if recursive else resolved.glob("*")
    for path in sorted(iterator):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path.resolve())
    _YOLO_POSE_EVAL_ALLOWED_FOLDERS.add(str(resolved.resolve()))
    total = len(images)
    return {
        "folder": str(resolved),
        "total": total,
        "images": [{"path": str(p), "filename": p.name} for p in images[:limit]],
    }


@app.get("/api/yolo-pose-eval/models")
def yolo_pose_eval_models():
    """List available YOLO (.pt) and ONNX models."""
    seen: set[str] = set()
    models = []
    search_dirs = [Path("/models"), Path("/workspace/models"), Path("/workspace")]
    for d in search_dirs:
        if d.exists():
            for ext in ("*.pt", "*.onnx"):
                for p in sorted(d.glob(ext)):
                    key = str(p.resolve())
                    if key not in seen:
                        seen.add(key)
                        models.append({"path": key, "name": p.name})
    return {"models": models}


@app.get("/api/yolo-pose-eval/image")
def yolo_pose_eval_image(path: str = Query(...)):
    resolved = str(Path(path).resolve())
    if not any(resolved.startswith(folder) for folder in _YOLO_POSE_EVAL_ALLOWED_FOLDERS):
        raise HTTPException(status_code=403, detail="Image path is not in a scanned folder")
    p = Path(resolved)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    image_bytes = p.read_bytes()
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=image_bytes, media_type=media_type)


@app.post("/api/yolo-pose-eval/ground-truth/load")
def yolo_pose_eval_gt_load(req: YoloPoseEvalGtLoadRequest):
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Ground truth file not found: {p}")
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}")
    return {"data": data}


@app.post("/api/yolo-pose-eval/ground-truth")
def yolo_pose_eval_gt_save(req: YoloPoseEvalGtSaveRequest):
    p = Path(req.path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(req.data, indent=2))
    return {"path": str(p)}


@app.post("/api/yolo-pose-eval/auto-label")
def yolo_pose_eval_auto_label(req: YoloPoseEvalAutoLabelRequest):
    import yolo_pose_eval as _ype

    folder = Path(req.image_folder)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")
    image_paths = []
    iterator = folder.rglob("*") if req.recursive else folder.glob("*")
    for p in sorted(iterator):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            image_paths.append(p)
    classes = None
    records = _ype.auto_label_images(
        image_paths=image_paths,
        model_path=req.model,
        conf=0.25,
        imgsz=640,
        device="0",
        classes=classes,
        model_type=req.model_type,
    )
    return {"records": records, "count": len(records)}


def _run_yolo_pose_eval_job(job_id: str, payload: dict):
    import yolo_pose_eval as _ype

    # Start fresh: clear zero-shot model caches so we don't get stuck on CPU
    # from a previous job's OOM fallback.
    for key in list(_ype._MODEL_CACHE.keys()):
        if key.startswith(("owlv2:", "gdino:")):
            del _ype._MODEL_CACHE[key]
    _ype._DEVICE_FALLBACK.clear()

    job_dir = _YOLO_POSE_EVAL_RESULTS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = job_dir / f"{job_id}_viz"

    with _YOLO_POSE_EVAL_JOBS_LOCK:
        job = _YOLO_POSE_EVAL_JOBS[job_id]
        job["status"] = "running"
    _persist_job(job_id)

    try:
        gt_path = Path(payload["gt_path"])
        gt_format = payload.get("gt_format", "xyxy")
        records = _ype.load_ground_truth(gt_path, fmt=gt_format)

        # Filter to images that exist
        records = [r for r in records if r.image_path.exists()]
        if not records:
            raise ValueError("No valid images found in ground truth")

        classes: set[str] | None = set(payload["classes"]) if payload.get("classes") else None

        # Auto-select confidence thresholds per model type when user leaves the field empty
        _DEFAULT_CONF_BY_TYPE: dict[str, list[float]] = {
            "yolo": [0.25, 0.5],
            "owlv2": [0.001, 0.005, 0.01, 0.02],
            "grounding_dino": [0.001, 0.01, 0.05],
        }
        user_conf_thresholds: list[float] = payload.get("conf_thresholds", [])

        combinations: list[_ype.ModelConfig] = []
        for m in payload["models"]:
            model_type = m.get("model_type", "yolo")
            conf_thresholds = user_conf_thresholds if user_conf_thresholds else _DEFAULT_CONF_BY_TYPE.get(model_type, [0.25])
            for conf in conf_thresholds:
                for imgsz in payload["imgsz"]:
                    combinations.append(
                        _ype.ModelConfig(
                            model_path=m["path"],
                            model_type=m.get("model_type", "yolo"),
                            conf=conf,
                            imgsz=imgsz,
                            device=payload.get("device", "0"),
                            iou=0.7,
                            text_prompt=m.get("text_prompt", "person"),
                            ablation_brightness=m.get("ablation_brightness", False),
                            ablation_crop_redetect=m.get("ablation_crop_redetect", False),
                            tracker=m.get("tracker", "none"),
                        )
                    )

        total = len(combinations)
        with _YOLO_POSE_EVAL_JOBS_LOCK:
            _YOLO_POSE_EVAL_JOBS[job_id]["total_configs"] = total
        _persist_job(job_id)

        if total > 50:
            print(f"[WARN] YOLO Pose Eval job {job_id} has {total} configurations. "
                  f"This will take a long time, especially for zero-shot models.", flush=True)

        results: list[_ype.EvalResult] = []
        last_persist_time = time.time()

        for idx, cfg in enumerate(combinations):
            def _progress_callback(completed_images: int, total_images: int):
                with _YOLO_POSE_EVAL_JOBS_LOCK:
                    _YOLO_POSE_EVAL_JOBS[job_id]["progress"] = {
                        "config_index": idx + 1,
                        "total_configs": total,
                        "completed_images": completed_images,
                        "total_images": total_images,
                        "model": Path(cfg.model_path).name,
                        "conf": cfg.conf,
                        "imgsz": cfg.imgsz,
                    }
                _persist_job(job_id)

            try:
                res = _ype.run_yolo_on_dataset(records, cfg, classes, progress_callback=_progress_callback)
                results.append(res)
            except Exception as cfg_exc:
                print(f"[ERROR] Config {idx + 1}/{total} failed for {Path(cfg.model_path).name}: {cfg_exc}", flush=True)
                # Purge zero-shot model cache on failure to avoid corrupted GPU state leaking into next config
                if cfg.model_type in ("owlv2", "grounding_dino"):
                    for key in list(_ype._MODEL_CACHE.keys()):
                        if key.startswith(("owlv2:", "gdino:")):
                            del _ype._MODEL_CACHE[key]
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                # Create a dummy result so the summary still has the right number of rows
                dummy = _ype.EvalResult(config=cfg)
                dummy.error = str(cfg_exc)
                results.append(dummy)

            with _YOLO_POSE_EVAL_JOBS_LOCK:
                _YOLO_POSE_EVAL_JOBS[job_id]["completed_configs"] = idx + 1
                latest = results[-1]
                _YOLO_POSE_EVAL_JOBS[job_id]["latest_result"] = {
                    "model": Path(cfg.model_path).name,
                    "conf": cfg.conf,
                    "imgsz": cfg.imgsz,
                    "tracker": cfg.tracker,
                    "ap50": getattr(latest, "ap50", None),
                }
                if "progress" in _YOLO_POSE_EVAL_JOBS[job_id]:
                    del _YOLO_POSE_EVAL_JOBS[job_id]["progress"]
            _persist_job(job_id)

            # Periodic persistence every 30s in case the process is killed
            if time.time() - last_persist_time > 30:
                _persist_job(job_id)
                last_persist_time = time.time()

            # Clear GPU cache between zero-shot model configs to reduce OOM risk
            if cfg.model_type in ("owlv2", "grounding_dino"):
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

        # Save results
        json_path = job_dir / f"{job_id}.json"
        csv_path = job_dir / f"{job_id}.csv"
        json_path.write_text(json.dumps(_ype.results_to_json(results), indent=2))
        csv_path.write_text(_ype.results_to_csv(results))

        # Generate visualizations
        viz_dir.mkdir(exist_ok=True)
        viz_paths: list[str] = []
        for res in results:
            written = _ype.draw_eval_visualisations(res.per_image, viz_dir, max_images=12)
            for name in written:
                viz_paths.append(name)

        with _YOLO_POSE_EVAL_JOBS_LOCK:
            _YOLO_POSE_EVAL_JOBS[job_id]["status"] = "done"
            _YOLO_POSE_EVAL_JOBS[job_id]["summary"] = _ype.results_to_json(results)
            _YOLO_POSE_EVAL_JOBS[job_id]["viz_paths"] = viz_paths
            _YOLO_POSE_EVAL_JOBS[job_id]["error"] = None
        _persist_job(job_id)

    except Exception as exc:
        with _YOLO_POSE_EVAL_JOBS_LOCK:
            _YOLO_POSE_EVAL_JOBS[job_id]["status"] = "error"
            _YOLO_POSE_EVAL_JOBS[job_id]["error"] = str(exc)
        _persist_job(job_id)


@app.post("/api/yolo-pose-eval/run")
def yolo_pose_eval_run(req: YoloPoseEvalRunRequest):
    job_id = f"ype-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    job = {
        "id": job_id,
        "name": f"Eval {len(req.models)} model(s)",
        "status": "queued",
        "total_configs": 0,
        "completed_configs": 0,
        "latest_result": None,
        "error": None,
        "summary": [],
        "viz_paths": [],
    }
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        _YOLO_POSE_EVAL_JOBS[job_id] = job
    _persist_job(job_id)
    thread = threading.Thread(
        target=_run_yolo_pose_eval_job, args=(job_id, req.model_dump()), daemon=False
    )
    thread.start()
    return {"id": job_id}


@app.get("/api/yolo-pose-eval/jobs")
def yolo_pose_eval_jobs():
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        jobs = []
        for job in _YOLO_POSE_EVAL_JOBS.values():
            lightweight = {k: v for k, v in job.items() if k != "summary"}
            lightweight = _sync_yolo_pose_eval_job_from_artifacts(lightweight)
            jobs.append(lightweight)
    jobs.sort(key=lambda job: str(job.get("id", "")), reverse=True)
    return {"jobs": jobs}


@app.get("/api/yolo-pose-eval/jobs/{job_id}")
def yolo_pose_eval_job(job_id: str):
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        job = _YOLO_POSE_EVAL_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _sync_yolo_pose_eval_job_from_artifacts(job.copy())
    # If summary is empty but results exist on disk, lazy-load them
    if not job.get("summary"):
        summary = _load_job_summary_from_disk(job_id)
        if summary is not None:
            with _YOLO_POSE_EVAL_JOBS_LOCK:
                _YOLO_POSE_EVAL_JOBS[job_id]["summary"] = summary
            job = _YOLO_POSE_EVAL_JOBS[job_id]
            job = _sync_yolo_pose_eval_job_from_artifacts(job.copy())
    return job


@app.delete("/api/yolo-pose-eval/jobs/{job_id}")
def yolo_pose_eval_delete_job(job_id: str):
    with _YOLO_POSE_EVAL_JOBS_LOCK:
        job = _YOLO_POSE_EVAL_JOBS.pop(job_id, None)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # New format: subdirectory
    job_dir = _YOLO_POSE_EVAL_RESULTS_DIR / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    # Old format: flat files
    for suffix in (".json", ".csv"):
        flat_file = _YOLO_POSE_EVAL_RESULTS_DIR / f"{job_id}{suffix}"
        if flat_file.exists():
            flat_file.unlink()
    viz_dir = _YOLO_POSE_EVAL_RESULTS_DIR / f"{job_id}_viz"
    if viz_dir.exists() and viz_dir.is_dir():
        import shutil
        shutil.rmtree(viz_dir)
    return {"deleted": True}


@app.get("/api/yolo-pose-eval/jobs/{job_id}/results.csv")
def yolo_pose_eval_results_csv(job_id: str):
    path = _resolve_yolo_pose_eval_result_path(job_id, ".csv")
    if path is None:
        raise HTTPException(status_code=404, detail="CSV not found")
    return Response(content=path.read_text(), media_type="text/csv")


@app.get("/api/yolo-pose-eval/jobs/{job_id}/results.json")
def yolo_pose_eval_results_json(job_id: str):
    path = _resolve_yolo_pose_eval_result_path(job_id, ".json")
    if path is None:
        raise HTTPException(status_code=404, detail="JSON not found")
    return Response(content=path.read_text(), media_type="application/json")


@app.get("/api/yolo-pose-eval/jobs/{job_id}/config/{idx}")
def yolo_pose_eval_config_detail(job_id: str, idx: int):
    path = _resolve_yolo_pose_eval_result_path(job_id, ".json")
    if path is None:
        raise HTTPException(status_code=404, detail="Results not found")
    try:
        data = json.loads(path.read_text())
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to parse results")
    if not isinstance(data, list) or idx < 0 or idx >= len(data):
        raise HTTPException(status_code=400, detail="Config index out of range")
    return {"per_image": data[idx].get("per_image", [])}


@app.get("/api/yolo-pose-eval/jobs/{job_id}/viz/{filename}")
def yolo_pose_eval_viz(job_id: str, filename: str):
    path = _resolve_yolo_pose_eval_viz_path(job_id, filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Visualization not found")
    image_bytes = path.read_bytes()
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=image_bytes, media_type=media_type)


@app.get("/vlm-tools", response_class=HTMLResponse)
def vlm_tools_page(request: Request):
    return templates.TemplateResponse(
        request,
        "vlm_tools.html",
        {
            "active_page": "vlm_tools",
            "tool_specs": _load_vlm_tool_specs(),
        },
    )




@app.get("/scene-change-dataset", response_class=HTMLResponse)
def scene_change_dataset_page(request: Request):
    return render_page(request, "scene_change_dataset.html", "scene_change_dataset")


@app.get("/scene-change-strategy", response_class=HTMLResponse)
def scene_change_strategy_page(request: Request):
    return render_page(request, "scene_change_strategy.html", "scene_change_strategy")


class VlmSelectionRequest(BaseModel):
    option_id: str


@app.get("/api/vlm/model")
def get_vlm_model():
    from agent.vlm_client import describe_active_vlm

    return describe_active_vlm()


@app.get("/api/vlm/options")
def get_vlm_options():
    from agent.vlm_client import get_vlm_selection_state

    return get_vlm_selection_state()


@app.post("/api/vlm/selection")
def set_vlm_selection(req: VlmSelectionRequest):
    from agent.vlm_client import set_active_vlm_option

    try:
        return set_active_vlm_option(req.option_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/video_publisher/status")
def get_video_publisher_status():
    return _video_publisher_status_payload()


@app.post("/api/pipeline/video_publisher/start")
def start_video_publisher(request: VideoPublisherRequest):
    frame_dir = request.frame_dir.strip()
    if not frame_dir:
        raise HTTPException(status_code=400, detail="frame_dir must not be empty.")

    container_frame_dir = _to_container_frame_dir(frame_dir)
    run_map_name = None
    run_map_id = None
    if request.create_run_map:
        requested_map_name = str(request.run_map_name or "").strip()
        run_map_name = sanitize_toolbox_map_name(requested_map_name) if requested_map_name else _default_pipeline_run_map_name(frame_dir)
        run_map_id = _ensure_pipeline_run_map(run_map_name)
        _activate_pipeline_run_map(run_map_name)

    loop_value = "true" if request.loop else "false"
    recursive_value = "true" if request.recursive else "false"
    captions_enabled_value = "true" if request.generate_captions else "false"
    start_script = f"""
set -e
PID_FILE={shlex.quote(VIDEO_PUBLISHER_PID_FILE)}
LOG_FILE={shlex.quote(VIDEO_PUBLISHER_LOG_FILE)}
FACE_PID_FILE={shlex.quote(FACE_DETECTOR_PID_FILE)}
FRAME_DIR={shlex.quote(container_frame_dir)}
CAPTIONS_ENABLED={captions_enabled_value}
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    if [ -r "/proc/$pid/stat" ]; then
      proc_state="$(awk '{{print $3}}' "/proc/$pid/stat" 2>/dev/null || true)"
      if [ "$proc_state" != "Z" ]; then
        echo "already_running"
        exit 20
      fi
    else
      echo "already_running"
      exit 20
    fi
  fi
  rm -f "$PID_FILE"
fi
if [ ! -d "$FRAME_DIR" ]; then
  echo "missing_frame_dir:$FRAME_DIR"
  exit 12
fi
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
if ! timeout 15s ros2 param set /scene_description_node captions_enabled "$CAPTIONS_ENABLED" >/tmp/bordsupr_caption_param_set.log 2>&1; then
  cat /tmp/bordsupr_caption_param_set.log >&2 || true
fi
export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:/usr/lib/python3/dist-packages:${{PYTHONPATH:-}}
if ! timeout 15s ros2 node list 2>/dev/null | grep -qx "/face_detector_node"; then
  echo "face_detector_node is not running; starting it so this folder run records face detections." >> "$LOG_FILE"
  CONFIG_FILE="$(ros2 pkg prefix bordsupr)/share/bordsupr/config/config.yaml"
  nohup ros2 run bordsupr face_detector_node --ros-args --params-file "$CONFIG_FILE" >> "$LOG_FILE" 2>&1 &
  echo $! > "$FACE_PID_FILE"
  sleep 3
fi
if ! timeout 15s ros2 node list 2>/dev/null | grep -qx "/face_detector_node"; then
  echo "face_detector_unavailable"
  exit 23
fi
nohup python3.10 {shlex.quote(VIDEO_PUBLISHER_SCRIPT_PATH)} --ros-args \
  -p rgb_topic:={shlex.quote(request.rgb_topic)} \
  -p image_folder:="$FRAME_DIR" \
  -p publish_hz:={request.publish_hz} \
  -p image_stride:={request.image_stride} \
  -p loop:={loop_value} \
  -p recursive:={recursive_value} \
  -p max_images:={request.max_images} \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
"""
    result = _docker_exec(start_script)
    if result.returncode == 20:
        raise HTTPException(status_code=409, detail="Video publisher is already running.")
    if result.returncode == 23:
        raise HTTPException(
            status_code=503,
            detail=(
                "Face detector node is not running and could not be started. "
                "Check the bordsupr container log for InsightFace/ONNXRuntime startup errors."
            ),
        )
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(result.stderr or result.stdout or "Failed to start video publisher.").strip(),
        )

    _video_publisher_last_request.clear()
    _video_publisher_last_request.update(
        {
            "frame_dir": frame_dir,
            "container_frame_dir": container_frame_dir,
            "image_stride": request.image_stride,
            "publish_hz": request.publish_hz,
            "loop": request.loop,
            "recursive": request.recursive,
            "max_images": request.max_images,
            "rgb_topic": request.rgb_topic,
            "run_map_name": run_map_name,
            "run_map_id": run_map_id,
            "generate_captions": request.generate_captions,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    return _video_publisher_status_payload()


@app.post("/api/pipeline/video_publisher/stop")
def stop_video_publisher():
    stop_script = f"""
PID_FILE={shlex.quote(VIDEO_PUBLISHER_PID_FILE)}
if [ -f "$PID_FILE" ]; then
  pid="$(cat "$PID_FILE" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
fi
"""
    result = _docker_exec(stop_script)
    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail=(result.stderr or result.stdout or "Failed to stop video publisher.").strip(),
        )
    return _video_publisher_status_payload()


# ---------------------------------------------------------------------------
# Robot Pipeline Simulator endpoints
# ---------------------------------------------------------------------------

@app.post("/api/robot-pipeline/start")
def robot_pipeline_start(payload: dict | None = None):
    """Start a robot pipeline simulation run."""
    from agent.robot_pipeline import get_runner

    payload = payload or {}
    frame_dir = str(payload.get("frame_dir") or "").strip()
    if not frame_dir:
        raise HTTPException(status_code=400, detail="frame_dir is required.")

    rooms_csv = str(payload.get("rooms") or "").strip()
    room_list = [r.strip() for r in rooms_csv.split(",") if r.strip()]
    if not room_list:
        raise HTTPException(status_code=400, detail="rooms must not be empty.")

    frames_per_room = max(1, int(payload.get("frames_per_room") or 5))
    poll_interval = max(1.0, float(payload.get("poll_interval_seconds") or 5.0))
    pause_on_stay = bool(payload.get("pause_on_stay", True))
    stride = max(1, int(payload.get("stride") or 5))
    hz = max(0.1, float(payload.get("hz") or 1.0))
    generate_captions = bool(payload.get("generate_captions", True))
    strategy_type = str(payload.get("strategy_type") or "v4").strip().lower()
    if strategy_type not in ("v4", "v5", "v6"):
        strategy_type = "v4"

    runner = get_runner()
    try:
        run_id = runner.start(
            frame_dir=frame_dir,
            room_list=room_list,
            frames_per_room=frames_per_room,
            poll_interval_seconds=poll_interval,
            pause_on_stay=pause_on_stay,
            stride=stride,
            hz=hz,
            generate_captions=generate_captions,
            strategy_type=strategy_type,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {exc}")

    return {"run_id": run_id, "status": "started"}


@app.post("/api/robot-pipeline/stop")
def robot_pipeline_stop():
    """Stop the current robot pipeline simulation run."""
    from agent.robot_pipeline import get_runner

    runner = get_runner()
    result = runner.stop()
    return result


@app.get("/api/robot-pipeline/status")
def robot_pipeline_status():
    """Get the current robot pipeline simulation status."""
    from agent.robot_pipeline import get_runner

    runner = get_runner()
    return runner.get_status()


@app.get("/api/robot-pipeline/stream")
async def robot_pipeline_stream():
    """Stream robot pipeline simulation steps via SSE."""
    import asyncio
    import json as _json

    from agent.robot_pipeline import get_runner

    runner = get_runner()
    q = runner.get_sse_queue()
    loop = asyncio.get_running_loop()

    async def event_generator():
        while True:
            try:
                event = await loop.run_in_executor(None, q.get, True, 1.0)
            except Exception:
                # queue.get with timeout raises Empty on timeout
                event = None

            if event is None:
                # Check if runner is still running
                status = runner.get_status()
                if status["status"] not in ("running", "idle"):
                    yield f"event: done\ndata: {_json.dumps({'finished': True, 'status': status['status']})}\n\n"
                    break
                continue

            event_type = event.get("type", "step")
            if event_type == "done":
                yield f"event: done\ndata: {_json.dumps({'finished': True, 'status': event.get('status')})}\n\n"
                break
            elif event_type == "error":
                yield f"event: error\ndata: {_json.dumps({'error': event.get('error')})}\n\n"
            else:
                yield f"event: {event_type}\ndata: {_json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/robot-pipeline/runs")
def robot_pipeline_runs(limit: int = Query(20, ge=1, le=100)):
    """List past robot pipeline simulation runs."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, created_at,
                    (SELECT COUNT(*) FROM navigation_decisions WHERE map_id = maps.id) AS decision_count,
                    (SELECT COUNT(*) FROM robot_visits WHERE map_id = maps.id) AS visit_count
                FROM maps
                WHERE name LIKE 'pipeline_run_%%'
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return {
        "runs": [
            {
                "run_id": row[0],
                "map_name": row[0],
                "created_at": row[1].isoformat() if row[1] else None,
                "decision_count": row[2],
                "visit_count": row[3],
            }
            for row in rows
        ]
    }


@app.get("/api/robot-pipeline/runs/{run_id}")
def robot_pipeline_run_detail(run_id: str):
    """Get detailed results for a specific pipeline run."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            # Resolve map_id
            cur.execute("SELECT id FROM maps WHERE name = %s", (run_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
            map_id = row[0]

            # Get decisions
            cur.execute(
                """
                SELECT step_number, target_room, scene_changed, change_severity,
                       activities_changed, action, reasoning, tool_calls_json,
                       created_at
                FROM navigation_decisions
                WHERE map_id = %s
                ORDER BY step_number
                """,
                (map_id,),
            )
            decisions = [
                {
                    "step_number": r[0],
                    "target_room": r[1],
                    "scene_changed": r[2],
                    "change_severity": r[3],
                    "activities_changed": r[4],
                    "action": r[5],
                    "reasoning": r[6],
                    "tool_calls": r[7],
                    "created_at": r[8].isoformat() if r[8] else None,
                }
                for r in cur.fetchall()
            ]

            # Get visits
            cur.execute(
                """
                SELECT room_name, arrived_at, departed_at, scene_count
                FROM robot_visits
                WHERE map_id = %s
                ORDER BY arrived_at
                """,
                (map_id,),
            )
            visits = [
                {
                    "room_name": r[0],
                    "arrived_at": r[1].isoformat() if r[1] else None,
                    "departed_at": r[2].isoformat() if r[2] else None,
                    "scene_count": r[3],
                }
                for r in cur.fetchall()
            ]

    return {
        "run_id": run_id,
        "map_id": map_id,
        "decisions": decisions,
        "visits": visits,
    }


@app.get("/api/buildings")
def get_buildings():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM maps ORDER BY name")
            rows = cur.fetchall()
    return [{"name": r[0]} for r in rows]


@app.get("/api/db/specs")
def get_db_specs():
    return _database_specs_payload()


@app.post("/api/db/reset")
def reset_database():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE TABLE
                    face_observations,
                    object_observations,
                    interactions,
                    objects,
                    scenes,
                    rooms,
                    maps
                RESTART IDENTITY CASCADE
                """
            )
    return {
        "status": "ok",
        "message": "Database emptied.",
        "counts": _fetch_table_counts(),
    }


class RoomPayload(BaseModel):
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float | None = None
    y3: float | None = None
    x4: float | None = None
    y4: float | None = None


class SaveRoomsRequest(BaseModel):
    map_name: str
    rooms: list[RoomPayload] = Field(default_factory=list)


class RoomAddRequest(BaseModel):
    map_name: str
    name: str
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float | None = None
    y3: float | None = None
    x4: float | None = None
    y4: float | None = None


class RoomEntryPointRequest(BaseModel):
    map_name: str
    room_name: str
    entry_x: float
    entry_y: float


def active_room_map_name():
    active_payload = load_active_toolbox_map_record()
    if str(active_payload.get("mode") or "").strip().lower() == "frozen" and active_payload.get("name"):
        return sanitize_toolbox_map_name(active_payload.get("name"))
    return None


def fetch_rooms_for_map(map_name=None, all_maps=False):
    requested_map_name = sanitize_toolbox_map_name(map_name) if map_name else active_room_map_name()
    requested_map_id = _resolve_map_id(requested_map_name)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if all_maps:
                cur.execute("""
                    SELECT r.id, m.name, r.name, r.x1, r.y1, r.x2, r.y2, r.x3, r.y3, r.x4, r.y4, r.entry_x, r.entry_y
                    FROM rooms r
                    LEFT JOIN maps m ON m.id = r.map_id
                    ORDER BY m.name NULLS LAST, r.id
                """)
            elif requested_map_id is not None:
                cur.execute("""
                    SELECT r.id, m.name, r.name, r.x1, r.y1, r.x2, r.y2, r.x3, r.y3, r.x4, r.y4, r.entry_x, r.entry_y
                    FROM rooms r
                    LEFT JOIN maps m ON m.id = r.map_id
                    WHERE r.map_id = %s
                    ORDER BY r.id
                """, (requested_map_id,))
            else:
                cur.execute("""
                    SELECT r.id, m.name, r.name, r.x1, r.y1, r.x2, r.y2, r.x3, r.y3, r.x4, r.y4, r.entry_x, r.entry_y
                    FROM rooms r
                    LEFT JOIN maps m ON m.id = r.map_id
                    WHERE r.map_id IS NULL
                    ORDER BY r.id
                """)
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "map_name": r[1],
            "name": r[2],
            "x1": r[3],
            "y1": r[4],
            "x2": r[5],
            "y2": r[6],
            "x3": r[7],
            "y3": r[8],
            "x4": r[9],
            "y4": r[10],
            "entry_x": r[11],
            "entry_y": r[12],
        }
        for r in rows
    ]


@app.get("/api/rooms")
def get_rooms(
    map_name: str | None = None,
    all_maps: bool = False,
):
    return fetch_rooms_for_map(map_name=map_name, all_maps=all_maps)


@app.post("/api/rooms/save")
def save_rooms(request: SaveRoomsRequest):
    map_name = sanitize_toolbox_map_name(request.map_name)
    map_id = _resolve_map_id(map_name)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if map_id is not None:
                cur.execute("DELETE FROM rooms WHERE map_id = %s", (map_id,))
            else:
                cur.execute("DELETE FROM rooms WHERE map_id IS NULL AND map_name = %s", (map_name,))
            for room in request.rooms:
                room_name = str(room.name or "").strip()
                if not room_name:
                    continue
                cur.execute(
                    """
                    INSERT INTO rooms (map_id, map_name, name, x1, y1, x2, y2, x3, y3, x4, y4)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        map_id,
                        map_name,
                        room_name,
                        room.x1,
                        room.y1,
                        room.x2,
                        room.y2,
                        room.x3,
                        room.y3,
                        room.x4,
                        room.y4,
                    ),
                )
        conn.commit()
    return {"ok": True, "map_name": map_name, "rooms": fetch_rooms_for_map(map_name=map_name)}


@app.post("/api/rooms/add")
def add_room(request: RoomAddRequest):
    map_name = sanitize_toolbox_map_name(request.map_name)
    map_id = _resolve_map_id(map_name)
    room_name = str(request.name or "").strip()
    if not room_name:
        raise HTTPException(status_code=400, detail="Room name is required")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rooms (map_id, map_name, name, x1, y1, x2, y2, x3, y3, x4, y4)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    map_id,
                    map_name,
                    room_name,
                    request.x1,
                    request.y1,
                    request.x2,
                    request.y2,
                    request.x3,
                    request.y3,
                    request.x4,
                    request.y4,
                ),
            )
            row = cur.fetchone()
            conn.commit()
    return {"ok": True, "id": row[0] if row else None, "room": fetch_rooms_for_map(map_name=map_name)}


@app.post("/api/rooms/set_entry_point")
def set_room_entry_point(request: RoomEntryPointRequest):
    map_name = sanitize_toolbox_map_name(request.map_name)
    map_id = _resolve_map_id(map_name)
    room_name = str(request.room_name or "").strip()
    if not map_id:
        raise HTTPException(status_code=404, detail=f"Map '{map_name}' not found")
    if not room_name:
        raise HTTPException(status_code=400, detail="room_name is required")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE rooms
                SET entry_x = %s, entry_y = %s
                WHERE map_id = %s AND name = %s
                RETURNING id
                """,
                (request.entry_x, request.entry_y, map_id, room_name),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Room '{room_name}' not found on map '{map_name}'")
            conn.commit()
    return {"ok": True, "room": room_name, "entry_x": request.entry_x, "entry_y": request.entry_y}


@app.get("/api/scenes")
def get_scenes(limit: int | None = Query(default=None, ge=1, le=500), offset: int = Query(default=0, ge=0), building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if limit is None:
                if map_id is not None:
                    cur.execute("""
                        SELECT id, x, y, caption, timestamp, original_scene_image IS NOT NULL, stitched_scene_image IS NOT NULL
                        FROM scenes
                        WHERE map_id = %s
                        ORDER BY timestamp DESC
                    """, (map_id,))
                else:
                    cur.execute("""
                        SELECT id, x, y, caption, timestamp, original_scene_image IS NOT NULL, stitched_scene_image IS NOT NULL
                        FROM scenes
                        ORDER BY timestamp DESC
                    """)
                rows = cur.fetchall()
                total = len(rows)
            else:
                if map_id is not None:
                    cur.execute("SELECT COUNT(*) FROM scenes WHERE map_id = %s", (map_id,))
                    total = cur.fetchone()[0]
                    cur.execute("""
                        SELECT id, x, y, caption, timestamp, original_scene_image IS NOT NULL, stitched_scene_image IS NOT NULL
                        FROM scenes
                        WHERE map_id = %s
                        ORDER BY timestamp DESC
                        LIMIT %s OFFSET %s
                    """, (map_id, limit, offset))
                else:
                    cur.execute("SELECT COUNT(*) FROM scenes")
                    total = cur.fetchone()[0]
                    cur.execute("""
                        SELECT id, x, y, caption, timestamp, original_scene_image IS NOT NULL, stitched_scene_image IS NOT NULL
                        FROM scenes
                        ORDER BY timestamp DESC
                        LIMIT %s OFFSET %s
                    """, (limit, offset))
                rows = cur.fetchall()
    scene_ids = [r[0] for r in rows]
    objects_by_scene: dict[int, list] = {}
    if scene_ids:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT oo.scene_id, o.id, o.name, o.class_id, COUNT(*) AS detections
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE oo.scene_id = ANY(%s)
                    GROUP BY oo.scene_id, o.id, o.name, o.class_id
                    ORDER BY oo.scene_id, detections DESC
                """, (scene_ids,))
                for scene_id, object_id, obj_name, class_id, count in cur.fetchall():
                    objects_by_scene.setdefault(scene_id, []).append({
                        "object_id": object_id,
                        "name": obj_name,
                        "class_id": class_id,
                        "class_name": class_name_from_id(class_id),
                        "count": int(count),
                    })
    items = [
        {
            "id": r[0],
            "x": r[1],
            "y": r[2],
            "caption": r[3],
            "timestamp": r[4],
            "has_original_image": r[5],
            "original_image_url": f"/api/scenes/{r[0]}/original_image" if r[5] else None,
            "has_stitched_image": r[6],
            "stitched_image_url": f"/api/scenes/{r[0]}/stitched_image" if r[6] else None,
            "objects": objects_by_scene.get(r[0], []),
        }
        for r in rows
    ]
    if limit is None:
        return items
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


@app.get("/api/interactions")
def get_interactions(building: str | None = Query(default=None), both_linked: bool = Query(default=False)):
    return fetch_interactions(building=building, both_linked=both_linked)


@app.get("/api/scenes/{scene_id}/interactions")
def get_scene_interactions(scene_id: int, building: str | None = Query(default=None)):
    return fetch_interactions(scene_id=scene_id, building=building)


@app.delete("/api/scenes/{scene_id}")
def delete_scene(scene_id: int):
    """Delete a single scene and its cascading observations.

    object_observations are deleted via ON DELETE CASCADE.
    face_observations and interactions have their scene_id set to NULL.
    Orphaned objects (zero observations left) are also removed.
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scenes WHERE id = %s RETURNING id", (scene_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Scene {scene_id} not found.")

            # Remove objects that now have zero observations anywhere
            cur.execute(
                """
                DELETE FROM objects
                WHERE id NOT IN (
                    SELECT DISTINCT object_id FROM object_observations WHERE object_id IS NOT NULL
                )
                """
            )
            orphaned_deleted = cur.rowcount

        conn.commit()

    return {
        "ok": True,
        "scene_id": scene_id,
        "deleted_orphaned_objects": orphaned_deleted,
        "message": f"Scene {scene_id} deleted.",
    }


@app.get("/api/objects")
def get_objects(
    limit: int | None = Query(default=None, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    min_observations: int = Query(default=0, ge=0),
    building: str | None = Query(default=None),
    class_id: int | None = Query(default=None),
    sort: str = Query(default="last_seen"),
    object_id: int | None = Query(default=None),
    name: str | None = Query(default=None),
):
    order_by = "observation_count DESC, o.id ASC" if sort == "observations" else "last_seen_at DESC NULLS LAST"
    map_id = _resolve_map_id(building)
    join_type = "INNER JOIN" if map_id is not None else "LEFT JOIN"
    building_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))" if map_id is not None else ""
    # Optional single-class filter (e.g. only person=0, only couch, ...).
    class_sql = " AND o.class_id = %s" if class_id is not None else ""
    # Optional exact object-id search.
    id_sql = " AND o.id = %s" if object_id is not None else ""
    # Optional case-insensitive substring search on the tracked object name.
    name_filter = (name or "").strip()
    name_sql = " AND o.name ILIKE %s" if name_filter else ""
    name_param = f"%{name_filter}%"
    with get_conn() as conn:
        with conn.cursor() as cur:
            if limit is None:
                params = []
                if map_id is not None:
                    params.extend([map_id, map_id])
                if class_id is not None:
                    params.append(class_id)
                if object_id is not None:
                    params.append(object_id)
                if name_filter:
                    params.append(name_param)
                params.append(min_observations)
                cur.execute(f"""
                    SELECT
                        o.id,
                        o.class_id,
                        o.name,
                        o.created_at,
                        COUNT(oo.id) AS observation_count,
                        COUNT(oo.id) FILTER (
                            WHERE EXISTS (
                                SELECT 1
                                FROM face_observations fo
                                WHERE fo.object_id = oo.object_id
                                   OR (
                                        fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                                        AND fo.yolo_track_id IS NOT NULL
                                        AND oo.yolo_track_id IS NOT NULL
                                        AND fo.yolo_track_id = oo.yolo_track_id
                                   )
                            )
                        ) AS recognized_observation_count,
                        MAX(oo.created_at) AS last_seen_at,
                        MIN(oo.created_at) AS first_seen_at,
                        MAX(oo.confidence) AS max_confidence,
                        (
                            ARRAY_AGG(oo.yolo_track_id ORDER BY oo.created_at DESC)
                            FILTER (WHERE oo.yolo_track_id IS NOT NULL)
                        )[1] AS latest_yolo_track_id,
                        (
                            SELECT COUNT(DISTINCT i.id)
                            FROM interactions i
                            WHERE EXISTS (
                                SELECT 1 FROM object_observations oo2
                                WHERE oo2.object_id = o.id AND oo2.id = i.subject_id
                            ) OR EXISTS (
                                SELECT 1 FROM object_observations oo2
                                WHERE oo2.object_id = o.id AND oo2.id = i.object_id
                            )
                        ) AS interaction_count
                    FROM objects o
                    {join_type} object_observations oo
                        ON oo.object_id = o.id
                        {building_sql}
                    WHERE 1=1{class_sql}{id_sql}{name_sql}
                    GROUP BY o.id, o.class_id, o.name, o.created_at
                    HAVING COUNT(oo.id) >= %s
                    ORDER BY {order_by}
                """, tuple(params))
                rows = cur.fetchall()
                total = len(rows)
            else:
                count_params = []
                if map_id is not None:
                    count_params.extend([map_id, map_id])
                if class_id is not None:
                    count_params.append(class_id)
                if object_id is not None:
                    count_params.append(object_id)
                if name_filter:
                    count_params.append(name_param)
                count_params.append(min_observations)
                cur.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM (
                        SELECT o.id
                        FROM objects o
                        {join_type} object_observations oo ON oo.object_id = o.id
                        {building_sql}
                        WHERE 1=1{class_sql}{id_sql}{name_sql}
                        GROUP BY o.id, o.class_id, o.name, o.created_at
                        HAVING COUNT(oo.id) >= %s
                    ) filtered_objects
                    """,
                    tuple(count_params),
                )
                total = cur.fetchone()[0]
                query_params = []
                if map_id is not None:
                    query_params.extend([map_id, map_id])
                if class_id is not None:
                    query_params.append(class_id)
                if object_id is not None:
                    query_params.append(object_id)
                if name_filter:
                    query_params.append(name_param)
                query_params.append(min_observations)
                query_params.extend([limit, offset])
                cur.execute(f"""
                    SELECT
                        o.id,
                        o.class_id,
                        o.name,
                        o.created_at,
                        COUNT(oo.id) AS observation_count,
                        MAX(oo.created_at) AS last_seen_at,
                        MIN(oo.created_at) AS first_seen_at,
                        MAX(oo.confidence) AS max_confidence,
                        (
                            ARRAY_AGG(oo.yolo_track_id ORDER BY oo.created_at DESC)
                            FILTER (WHERE oo.yolo_track_id IS NOT NULL)
                        )[1] AS latest_yolo_track_id
                    FROM objects o
                    {join_type} object_observations oo
                        ON oo.object_id = o.id
                        {building_sql}
                    WHERE 1=1{class_sql}{id_sql}{name_sql}
                    GROUP BY o.id, o.class_id, o.name, o.created_at
                    HAVING COUNT(oo.id) >= %s
                    ORDER BY {order_by}
                    LIMIT %s OFFSET %s
                """, tuple(query_params))
                rows = cur.fetchall()
                # Expensive per-object stats are computed only for the page's objects.
                page_ids = [r[0] for r in rows]
                recognized_counts: dict = {}
                interaction_counts: dict = {}
                if page_ids:
                    cur.execute("""
                        SELECT oo.object_id, COUNT(*)
                        FROM object_observations oo
                        WHERE oo.object_id = ANY(%s)
                          AND (
                              EXISTS (SELECT 1 FROM face_observations fo WHERE fo.object_id = oo.object_id)
                              OR EXISTS (
                                  SELECT 1 FROM face_observations fo
                                  WHERE fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                                    AND fo.yolo_track_id IS NOT NULL
                                    AND oo.yolo_track_id IS NOT NULL
                                    AND fo.yolo_track_id = oo.yolo_track_id
                              )
                          )
                        GROUP BY oo.object_id
                    """, (page_ids,))
                    recognized_counts = {r[0]: int(r[1]) for r in cur.fetchall()}
                    cur.execute("""
                        SELECT oid, COUNT(DISTINCT iid) FROM (
                            SELECT oo2.object_id AS oid, i.id AS iid
                            FROM interactions i
                            JOIN object_observations oo2 ON oo2.id = i.subject_id
                            WHERE oo2.object_id = ANY(%s)
                            UNION
                            SELECT oo2.object_id AS oid, i.id AS iid
                            FROM interactions i
                            JOIN object_observations oo2 ON oo2.id = i.object_id
                            WHERE oo2.object_id = ANY(%s)
                        ) u
                        GROUP BY oid
                    """, (page_ids, page_ids))
                    interaction_counts = {r[0]: int(r[1]) for r in cur.fetchall()}
                rows = [
                    (r[0], r[1], r[2], r[3], r[4],
                     recognized_counts.get(r[0], 0),
                     r[5], r[6], r[7], r[8],
                     interaction_counts.get(r[0], 0))
                    for r in rows
                ]
    items = [
        {
            "id": r[0],
            "class_id": r[1],
            "class_name": class_name_from_id(r[1]),
            "name": r[2],
            "created_at": r[3],
            "observation_count": r[4],
            "recognized_observation_count": r[5],
            "last_seen_at": r[6],
            "first_seen_at": r[7],
            "max_confidence": float(r[8]) if r[8] is not None else None,
            "yolo_track_id": r[9],
            "interaction_count": r[10] or 0,
        }
        for r in rows
    ]
    if limit is None:
        return items
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "min_observations": min_observations,
        "has_more": offset + len(items) < total,
    }

@app.get("/api/object-classes")
def get_object_classes(building: str | None = Query(default=None)):
    """Distinct object classes present in the DB, for populating a class filter.

    Returns a list of {class_id, class_name, cluster_count} sorted by count desc.
    """
    map_id = _resolve_map_id(building)
    join_type = "INNER JOIN" if map_id is not None else "LEFT JOIN"
    building_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))" if map_id is not None else ""
    params = []
    if map_id is not None:
        params.extend([map_id, map_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT o.class_id, COUNT(DISTINCT o.id) AS cluster_count
                FROM objects o
                {join_type} object_observations oo
                    ON oo.object_id = o.id
                    {building_sql}
                GROUP BY o.class_id
                ORDER BY cluster_count DESC, o.class_id ASC
            """, tuple(params))
            rows = cur.fetchall()
    return [
        {
            "class_id": r[0],
            "class_name": class_name_from_id(r[0]) or (f"class {r[0]}" if r[0] is not None else "unknown"),
            "cluster_count": r[1],
        }
        for r in rows
    ]

@app.get("/api/objects/{object_id}/observations")
def get_object_observations(object_id: str, building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            building_sql = ""
            params = [object_id]
            if map_id is not None:
                building_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
                params.extend([map_id, map_id])
            cur.execute(f"""
                SELECT
                    oo.id,
                    oo.object_id,
                    oo.scene_id,
                    oo.x,
                    oo.y,
                    oo.z,
                    oo.created_at,
                    vector_dims(oo.embedding) AS embedding_dim,
                    LEFT(oo.embedding::text, 120) AS embedding_preview,
                    s.caption,
                    s.x,
                    s.y,
                    s.timestamp,
                    oo.yolo_track_id AS yolo_track_id,
                    obs_person.person_id,
                    oo.robot_x,
                    oo.robot_y,
                    oo.robot_z,
                    oo.rel_x,
                    oo.rel_y,
                    oo.rel_z,
                    oo.position_source,
                    CASE WHEN oo.position_source = 'dynosam' THEN TRUE ELSE FALSE END AS has_dynosam_position,
                    CASE WHEN oo.position_source = 'depth' THEN TRUE ELSE FALSE END AS has_depth_position,
                    oo.detection_backend,
                    oo.embedding_backend
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                LEFT JOIN LATERAL (
                    SELECT fo.person_id
                    FROM face_observations fo
                    WHERE fo.object_id = oo.object_id
                       OR (
                            fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                            AND fo.yolo_track_id IS NOT NULL
                            AND oo.yolo_track_id IS NOT NULL
                            AND fo.yolo_track_id = oo.yolo_track_id
                       )
                    ORDER BY fo.created_at DESC, fo.id DESC
                    LIMIT 1
                ) obs_person ON TRUE
                WHERE oo.object_id = %s
                  {building_sql}
                ORDER BY oo.created_at DESC
            """, tuple(params))
            rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "object_id": r[1],
            "scene_id": r[2],
            "x": r[3],
            "y": r[4],
            "z": r[5],
            "created_at": r[6],
            "embedding_dim": r[7],
            "embedding_preview": r[8],
            "scene_caption": r[9],
            "scene_x": r[10],
            "scene_y": r[11],
            "scene_timestamp": r[12],
            "yolo_track_id": r[13],
            "person_id": r[14],
            "robot_x": float(r[15]) if r[15] is not None else None,
            "robot_y": float(r[16]) if r[16] is not None else None,
            "robot_z": float(r[17]) if r[17] is not None else None,
            "rel_x": float(r[18]) if r[18] is not None else None,
            "rel_y": float(r[19]) if r[19] is not None else None,
            "rel_z": float(r[20]) if r[20] is not None else None,
            "position_source": r[21],
            "has_dynosam_position": bool(r[22]),
            "has_depth_position": bool(r[23]),
            "detection_backend": r[24],
            "embedding_backend": r[25],
            "image_url": f"/api/observations/{r[0]}/image",
            "original_image_url": f"/api/observations/{r[0]}/original_image",
        }
        for r in rows
    ]


@app.get("/api/detections/latest-observation")
def get_latest_observation_by_detection(
    track_id: int = Query(...),
    instance_id: int | None = Query(default=None),
    building: str | None = Query(default=None),
):
    """Return the latest DB observation for a given YOLO track ID (and optional instance_id)."""
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            building_sql = ""
            params = [track_id]
            if map_id is not None:
                building_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
                params.extend([map_id, map_id])
            cur.execute(f"""
                SELECT
                    oo.id,
                    oo.object_id,
                    oo.scene_id,
                    oo.x,
                    oo.y,
                    oo.z,
                    oo.created_at,
                    oo.yolo_track_id,
                    oo.position_source,
                    s.caption,
                    oo.confidence,
                    oo.quality_score
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.yolo_track_id = %s::text
                  {building_sql}
                ORDER BY oo.created_at DESC
                LIMIT 1
            """, tuple(params))
            row = cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No observation found for this detection")
    return {
        "id": row[0],
        "object_id": row[1],
        "scene_id": row[2],
        "x": row[3],
        "y": row[4],
        "z": row[5],
        "created_at": row[6],
        "yolo_track_id": row[7],
        "position_source": row[8],
        "scene_caption": row[9],
        "confidence": float(row[10]) if row[10] is not None else None,
        "quality_score": float(row[11]) if row[11] is not None else None,
        "image_url": f"/api/observations/{row[0]}/image",
        "original_image_url": f"/api/observations/{row[0]}/original_image",
    }


@app.get("/api/objects/{object_id}/faces")
def get_object_faces(object_id: str, limit: int = Query(default=120, ge=1, le=500), building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    clean_object_id = str(object_id).strip()
    with get_conn() as conn:
        with conn.cursor() as cur:
            building_sql = ""
            params = [clean_object_id, clean_object_id]
            if map_id is not None:
                building_sql = " AND (fo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = fo.scene_id AND s2.map_id = %s))"
                params.extend([map_id, map_id])
            params.append(limit)
            cur.execute(
                f"""
                WITH matched_faces AS (
                    SELECT DISTINCT ON (fo.id)
                        fo.id,
                        fo.person_id,
                        fo.scene_id,
                        fo.object_id,
                        fo.yolo_track_id,
                        fo.person_x_min,
                        fo.person_y_min,
                        fo.person_x_max,
                        fo.person_y_max,
                        fo.face_x_min,
                        fo.face_y_min,
                        fo.face_x_max,
                        fo.face_y_max,
                        fo.score,
                        fo.created_at
                    FROM face_observations fo
                    LEFT JOIN object_observations oo
                      ON fo.yolo_track_id IS NOT NULL
                     AND btrim(fo.yolo_track_id) <> ''
                     AND oo.yolo_track_id = fo.yolo_track_id
                     AND (
                        fo.scene_id IS NULL
                        OR oo.scene_id = fo.scene_id
                     )
                    WHERE (
                        fo.object_id::text = %s
                        OR oo.object_id::text = %s
                    )
                      {building_sql}
                    ORDER BY fo.id, fo.created_at DESC
                )
                SELECT *
                FROM matched_faces
                ORDER BY created_at DESC
                LIMIT %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    items = [
        {
            "id": r[0],
            "person_id": r[1],
            "scene_id": r[2],
            "object_id": r[3],
            "yolo_track_id": r[4],
            "person_bbox": [r[5], r[6], r[7], r[8]],
            "face_bbox": [r[9], r[10], r[11], r[12]],
            "score": r[13],
            "created_at": r[14],
            "image_url": f"/api/faces/{r[0]}/image",
        }
        for r in rows
    ]
    return items


@app.get("/api/objects/{object_id}/interactions")
def get_object_interactions_api(object_id: str, building: str | None = Query(default=None)):
    """Return all interactions where the object participated as either subject or object."""
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            building_sql = ""
            params = [object_id, object_id]
            if map_id is not None:
                building_sql = " AND i.map_id = %s"
                params.append(map_id)
            cur.execute(f"""
                SELECT
                    i.id,
                    i.action,
                    i.caption,
                    COALESCE(subject_obs.scene_id, object_obs.scene_id) AS scene_id,
                    i.created_at,
                    i.model_source,
                    i.subject_bbox,
                    i.object_bbox,
                    s.timestamp AS scene_timestamp,
                    subject_obs.object_id,
                    so.class_id,
                    subject_obs.id,
                    subject_obs.yolo_track_id,
                    subject_person.person_id,
                    object_obs.object_id,
                    oo.class_id,
                    object_obs.id,
                    object_obs.yolo_track_id,
                    object_person.person_id
                FROM interactions i
                LEFT JOIN object_observations subject_obs
                    ON subject_obs.id = i.subject_id
                LEFT JOIN object_observations object_obs
                    ON object_obs.id = i.object_id
                LEFT JOIN LATERAL (
                    SELECT fo.person_id
                    FROM face_observations fo
                    WHERE fo.object_id = subject_obs.object_id
                       OR (
                            fo.scene_id IS NOT DISTINCT FROM subject_obs.scene_id
                            AND fo.yolo_track_id IS NOT NULL
                            AND subject_obs.yolo_track_id IS NOT NULL
                            AND fo.yolo_track_id = subject_obs.yolo_track_id
                       )
                    ORDER BY fo.created_at DESC, fo.id DESC
                    LIMIT 1
                ) subject_person ON TRUE
                LEFT JOIN LATERAL (
                    SELECT fo.person_id
                    FROM face_observations fo
                    WHERE fo.object_id = object_obs.object_id
                       OR (
                            fo.scene_id IS NOT DISTINCT FROM object_obs.scene_id
                            AND fo.yolo_track_id IS NOT NULL
                            AND object_obs.yolo_track_id IS NOT NULL
                            AND fo.yolo_track_id = object_obs.yolo_track_id
                       )
                    ORDER BY fo.created_at DESC, fo.id DESC
                    LIMIT 1
                ) object_person ON TRUE
                LEFT JOIN scenes s
                    ON s.id = COALESCE(subject_obs.scene_id, object_obs.scene_id)
                LEFT JOIN objects so
                    ON so.id = subject_obs.object_id
                LEFT JOIN objects oo
                    ON oo.id = object_obs.object_id
                WHERE (subject_obs.object_id = %s OR object_obs.object_id = %s)
                  {building_sql}
                ORDER BY COALESCE(s.timestamp, i.created_at) DESC, i.id DESC
            """, tuple(params))
            rows = cur.fetchall()

    interactions = []
    for row in rows:
        entry = {
            "id": row[0],
            "action": row[1],
            "caption": row[2],
            "scene_id": row[3],
            "created_at": row[4],
            "model_source": row[5],
            "subject_bbox": row[6],
            "object_bbox": row[7],
            "scene_timestamp": row[8],
            "objects": [],
        }

        if row[9] is not None:
            entry["objects"].append(
                {
                    "role": "subject",
                    "observation_id": row[11],
                    "object_id": row[9],
                    "class_id": row[10],
                    "class_name": class_name_from_id(row[10]),
                    "yolo_track_id": row[12],
                    "image_url": f"/api/observations/{row[11]}/image" if row[11] is not None else None,
                }
            )

        if row[13] is not None:
            entry["objects"].append(
                {
                    "role": "object",
                    "observation_id": row[15],
                    "object_id": row[13],
                    "class_id": row[14],
                    "class_name": class_name_from_id(row[14]),
                    "yolo_track_id": row[16],
                    "image_url": f"/api/observations/{row[15]}/image" if row[15] is not None else None,
                }
            )

        interactions.append(entry)

    return interactions


@app.get("/api/observation_topo/observations")
def get_observation_topo_observations(object_id: str | None = None, building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    query = """
        SELECT
            ranked.id,
            ranked.object_id,
            ranked.class_id,
            ranked.x,
            ranked.y,
            ranked.z,
            ranked.created_at,
            ranked.scene_id,
            ranked.yolo_track_id
        FROM (
            SELECT
                oo.id,
                oo.object_id,
                oo.class_id,
                oo.x,
                oo.y,
                oo.z,
                oo.created_at,
                oo.scene_id,
                oo.yolo_track_id,
                obs_person.person_id,
                ROW_NUMBER() OVER (
                    PARTITION BY oo.object_id
                    ORDER BY oo.created_at DESC, oo.id DESC
                ) AS row_num
            FROM object_observations oo
            LEFT JOIN LATERAL (
                SELECT fo.person_id
                FROM face_observations fo
                WHERE fo.object_id = oo.object_id
                   OR (
                        fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                        AND fo.yolo_track_id IS NOT NULL
                        AND oo.yolo_track_id IS NOT NULL
                        AND fo.yolo_track_id = oo.yolo_track_id
                   )
                ORDER BY fo.created_at DESC, fo.id DESC
                LIMIT 1
            ) obs_person ON TRUE
            WHERE oo.x IS NOT NULL
              AND oo.y IS NOT NULL
    """
    params = []
    if map_id is not None:
        query += " AND oo.map_id = %s"
        params.append(map_id)
    if object_id is not None and str(object_id).strip():
        query += " AND oo.object_id = %s"
        params.append(str(object_id).strip())
    query += """
        ) ranked
        WHERE ranked.row_num = 1
        ORDER BY ranked.created_at DESC, ranked.id DESC
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "object_id": r[1],
            "class_id": r[2],
            "class_name": class_name_from_id(r[2]),
            "x": float(r[3]) if r[3] is not None else None,
            "y": float(r[4]) if r[4] is not None else None,
            "z": float(r[5]) if r[5] is not None else None,
            "created_at": r[6],
            "scene_id": r[7],
            "yolo_track_id": r[8],
        }
        for r in rows
    ]


@app.get("/api/lidar_occupancy_map")
def get_lidar_occupancy_map():
    payload = load_lidar_occupancy_map_payload()
    if payload is not None:
        return payload
    return {"available": False, "reason": _map_unavailable_reason(LIDAR_OCCUPANCY_MAP_PATH, "map_not_generated")}


@app.get("/api/toolbox_map")
def get_toolbox_map():
    active_payload = load_active_toolbox_map_payload()
    if active_payload is not None:
        return active_payload

    payload = load_map_payload_from_path(TOOLBOX_MAP_PATH)
    if payload is not None:
        return payload
    return {"available": False, "reason": _map_unavailable_reason(TOOLBOX_MAP_PATH, "toolbox_map_not_generated")}


@app.get("/api/nav2_global_costmap")
def get_nav2_global_costmap():
    payload = load_map_payload_from_path(NAV2_GLOBAL_COSTMAP_PATH)
    if payload is not None:
        return payload
    return {"available": False, "reason": _map_unavailable_reason(NAV2_GLOBAL_COSTMAP_PATH, "nav2_costmap_not_generated")}


@app.get("/api/toolbox_map/persistence")
def get_toolbox_map_persistence():
    return build_toolbox_map_persistence_payload()


@app.post("/api/toolbox_map/open_heartbeat")
def toolbox_map_open_heartbeat():
    active = load_active_toolbox_map_record()
    if not isinstance(active, dict) or not (active.get("name") or active.get("observation_map_name")):
        return {"ok": False, "active_is_open": False, "reason": "no_active_map"}

    now = time.time()
    active["map_opened_at"] = now
    active["map_open_until"] = now + TOOLBOX_MAP_OPEN_HEARTBEAT_TTL_SEC
    active["map_open_source"] = "web_map_tab"
    atomic_write_json(TOOLBOX_MAP_ACTIVE_PATH, active)
    return {
        "ok": True,
        "active_is_open": True,
        "open_until": active["map_open_until"],
        "recording_enabled": bool(active.get("recording_enabled")),
    }


@app.post("/api/toolbox_map/save")
def save_toolbox_map(
    name: str | None = Query(default=None),
    building_name: str | None = Query(default=None),
):
    requested_name = building_name if building_name is not None else name
    map_name = sanitize_toolbox_map_name(requested_name)
    resolved_building_name = normalize_toolbox_building_name(building_name, fallback_name=requested_name or map_name)
    request_id = str(uuid.uuid4())
    created_at = time.time()
    atomic_write_json(
        TOOLBOX_MAP_REQUEST_PATH,
        {
            "request_id": request_id,
            "created_at": created_at,
            "action": "save",
            "name": map_name,
            "building_name": resolved_building_name,
        },
    )
    new_map_id = _ensure_pipeline_run_map(resolved_building_name)

    # Migrate observations from the previously active pipeline-run map so they
    # appear under the newly saved building name.
    active_record = load_active_toolbox_map_record()
    old_map_name = None
    if active_record:
        mode = str(active_record.get("mode") or "recording").strip().lower()
        if mode == "frozen":
            old_map_name = active_record.get("name")
        else:
            old_map_name = active_record.get("observation_map_name") or active_record.get("name")
    old_map_id = _resolve_map_id(old_map_name)
    migrated = _migrate_observations_to_map(old_map_id, new_map_id)
    if migrated:
        logging.info(f"Migrated {migrated} observations from map '{old_map_name}' to '{resolved_building_name}'")

    # Migrate rooms to the new map name so they remain discoverable.
    if old_map_name and old_map_name != resolved_building_name:
        try:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if old_map_id is not None:
                        cur.execute(
                            "UPDATE rooms SET map_id = %s, map_name = %s WHERE map_id = %s",
                            (new_map_id, resolved_building_name, old_map_id),
                        )
                    else:
                        cur.execute(
                            "UPDATE rooms SET map_id = %s, map_name = %s WHERE map_id IS NULL AND map_name = %s",
                            (new_map_id, resolved_building_name, old_map_name),
                        )
                    if cur.rowcount:
                        logging.info(f"Migrated {cur.rowcount} rooms from map '{old_map_name}' to '{resolved_building_name}'")
                conn.commit()
        except Exception:
            logging.exception("Failed to migrate rooms to new map name")

    # Keep the active record pointing at the new map so future observations
    # are tagged consistently.
    if active_record and active_record.get("mode") == "recording":
        active_record["observation_map_name"] = resolved_building_name
        atomic_write_json(TOOLBOX_MAP_ACTIVE_PATH, active_record)

    return {
        "ok": True,
        "queued": True,
        "request_id": request_id,
        "action": "save",
        "name": map_name,
        "building_name": resolved_building_name,
        "created_at": created_at,
    }


@app.post("/api/toolbox_map/load")
def load_toolbox_map(name: str | None = Query(default=None)):
    map_name = sanitize_toolbox_map_name(name)
    record = get_toolbox_map_record(map_name)
    if not record["ready_to_load"]:
        raise HTTPException(status_code=404, detail=f"No saved slam_toolbox posegraph found for '{map_name}'.")

    record = persist_toolbox_autoload_record(map_name)
    persist_active_toolbox_map_record(record, mode="frozen", recording_enabled=False)
    persist_frozen_toolbox_map_snapshot(record)
    request_id = str(uuid.uuid4())
    created_at = time.time()
    atomic_write_json(
        TOOLBOX_MAP_REQUEST_PATH,
        {
            "request_id": request_id,
            "created_at": created_at,
            "action": "load",
            "name": record["name"],
            "building_name": record["building_name"],
        },
    )
    return {
        "ok": True,
        "queued": True,
        "request_id": request_id,
        "action": "load",
        "name": record["name"],
        "building_name": record["building_name"],
        "message": f"Loading saved map for building '{record['building_name']}'.",
        "created_at": created_at,
    }


@app.post("/api/toolbox_map/delete")
def delete_toolbox_map(name: str | None = Query(default=None)):
    map_name = sanitize_toolbox_map_name(name)
    payload = delete_saved_toolbox_map(map_name)
    return {
        **payload,
        "message": f"Deleted saved map for building '{payload['building_name']}'.",
        "created_at": time.time(),
    }


@app.post("/api/toolbox_map/start_recording")
def start_toolbox_map_recording(name: str | None = Query(default=None)):
    map_name = sanitize_toolbox_map_name(name)
    record = get_toolbox_map_record(map_name)
    if not record["ready_to_load"]:
        raise HTTPException(status_code=404, detail=f"No saved map found for '{map_name}'.")
    persist_active_toolbox_map_record(record, mode="frozen", recording_enabled=True)
    return {
        "ok": True,
        "name": record["name"],
        "building_name": record["building_name"],
        "recording_enabled": True,
    }


@app.post("/api/toolbox_map/stop_recording")
def stop_toolbox_map_recording():
    active = load_active_toolbox_map_record()
    if isinstance(active, dict) and active.get("name"):
        record = {
            "name": active["name"],
            "building_name": active.get("building_name") or active["name"],
        }
        persist_active_toolbox_map_record(record, mode="frozen", recording_enabled=False)
    else:
        atomic_write_json(TOOLBOX_MAP_ACTIVE_PATH, {"recording_enabled": False, "updated_at": time.time()})
    return {"ok": True, "recording_enabled": False}


@app.get("/api/lidar_detection_markers")
def get_lidar_detection_markers(limit: int | None = None, building: str | None = Query(default=None)):
    safe_limit = None
    if limit is not None:
        try:
            safe_limit = max(1, min(int(limit), 5000))
        except (TypeError, ValueError):
            safe_limit = None
    map_id = _resolve_map_id(building)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.columns
                        WHERE table_name = 'object_observations'
                          AND column_name = 'robot_x'
                    )
                    """
                )
                has_robot_columns = bool(cur.fetchone()[0])
                building_sql = ""
                if map_id is not None:
                    building_sql = " AND oo.map_id = %s"

                if has_robot_columns:
                    query = f"""
                        WITH ranked_observations AS (
                            SELECT
                                oo.id,
                                oo.object_id,
                                oo.class_id,
                                oo.yolo_track_id,
                                COALESCE(oo.x, oo.robot_x, s.x) AS x,
                                COALESCE(oo.y, oo.robot_y, s.y) AS y,
                                COALESCE(oo.z, oo.robot_z, 0.0) AS z,
                                (
                                    oo.x IS NOT NULL
                                    AND oo.y IS NOT NULL
                                    AND (
                                        oo.robot_x IS NULL
                                        OR oo.robot_y IS NULL
                                        OR oo.robot_z IS NULL
                                        OR ABS(oo.x - oo.robot_x) > 1e-6
                                        OR ABS(oo.y - oo.robot_y) > 1e-6
                                        OR ABS(COALESCE(oo.z, 0.0) - COALESCE(oo.robot_z, 0.0)) > 1e-6
                                    )
                                ) AS has_dynosam_position,
                                oo.position_source,
                                oo.created_at,
                                oo.scene_id,
                                s.timestamp,
                                ROW_NUMBER() OVER (
                                    PARTITION BY oo.object_id
                                    ORDER BY COALESCE(s.timestamp, oo.created_at) DESC, oo.id DESC
                                ) AS row_num
                            FROM object_observations oo
                            LEFT JOIN scenes s ON s.id = oo.scene_id
                            WHERE COALESCE(oo.x, oo.robot_x, s.x) IS NOT NULL
                              AND COALESCE(oo.y, oo.robot_y, s.y) IS NOT NULL
                              {building_sql}
                        )
                        SELECT
                            id,
                            object_id,
                            class_id,
                            yolo_track_id,
                            x,
                            y,
                            z,
                            has_dynosam_position,
                            position_source,
                            created_at,
                            scene_id,
                            timestamp
                        FROM ranked_observations
                        WHERE row_num = 1
                        ORDER BY COALESCE(timestamp, created_at) DESC, id DESC
                    """
                    params = []
                    if map_id is not None:
                        params.append(map_id)
                    if safe_limit is not None:
                        query += " LIMIT %s"
                        params.append(safe_limit)
                    cur.execute(query, params)
                else:
                    query = f"""
                        WITH ranked_observations AS (
                            SELECT
                                oo.id,
                                oo.object_id,
                                oo.class_id,
                                oo.yolo_track_id,
                                s.x,
                                s.y,
                                NULL::DOUBLE PRECISION AS z,
                                FALSE AS has_dynosam_position,
                                oo.position_source,
                                oo.created_at,
                                oo.scene_id,
                                s.timestamp,
                                ROW_NUMBER() OVER (
                                    PARTITION BY oo.object_id
                                    ORDER BY COALESCE(s.timestamp, oo.created_at) DESC, oo.id DESC
                                ) AS row_num
                            FROM object_observations oo
                            LEFT JOIN scenes s ON s.id = oo.scene_id
                            WHERE s.x IS NOT NULL
                              AND s.y IS NOT NULL
                              {building_sql}
                        )
                        SELECT
                            id,
                            object_id,
                            class_id,
                            yolo_track_id,
                            x,
                            y,
                            z,
                            has_dynosam_position,
                            position_source,
                            created_at,
                            scene_id,
                            timestamp
                        FROM ranked_observations
                        WHERE row_num = 1
                        ORDER BY COALESCE(timestamp, created_at) DESC, id DESC
                    """
                    params = []
                    if map_id is not None:
                        params.append(map_id)
                    if safe_limit is not None:
                        query += " LIMIT %s"
                        params.append(safe_limit)
                    cur.execute(query, params)
                rows = cur.fetchall()
    except Exception:
        return []

    return [
        {
            "observation_id": r[0],
            "object_id": r[1],
            "class_id": r[2],
            "class_name": class_name_from_id(r[2]),
            "yolo_track_id": r[3],
            "x": float(r[4]) if r[4] is not None else None,
            "y": float(r[5]) if r[5] is not None else None,
            "z": float(r[6]) if r[6] is not None else None,
            "has_dynosam_position": bool(r[7]),
            "position_source": r[8],
            "created_at": r[9],
            "scene_id": r[10],
            "scene_timestamp": r[11],
            "image_url": f"/api/observations/{r[0]}/image",
        }
        for r in rows
    ]


@app.post("/api/lidar_detection_markers/{observation_id}/plan")
def plan_lidar_detection_navigation(observation_id: int):
    stale_pending = assert_slam_pose_not_pending_for_nav2(allow_stale_for_planning=True)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    oo.id,
                    oo.object_id,
                    oo.class_id,
                    oo.yolo_track_id,
                    oo.position_source,
                    oo.x AS goal_x,
                    oo.y AS goal_y,
                    oo.z AS goal_z,
                    oo.created_at,
                    s.timestamp
                FROM object_observations oo
                LEFT JOIN scenes s ON s.id = oo.scene_id
                WHERE oo.id = %s
                LIMIT 1
                """,
                (observation_id,),
            )
            row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Persisted detection observation not found")
    if row[4] not in {"dynosam", "depth"}:
        raise HTTPException(
            status_code=400,
            detail="Persisted detection does not have navigation-safe coordinates",
        )
    if row[5] is None or row[6] is None:
        raise HTTPException(status_code=400, detail="Persisted detection does not have stored map coordinates")

    raw_goal_pose = {"x": float(row[5]), "y": float(row[6]), "z": float(row[7] or 0.0)}
    # While a stale set-pose confirmation is still pending, use the SLAM tab's
    # provisional robot pose for preview planning so the user can inspect a
    # path from the requested start pose. Execution remains blocked until
    # localization confirms in live TF.
    start_pose = _get_robot_start_pose() if stale_pending else None
    # Stop 1 metre short of the object, offset along the approach direction.
    temp_plan = request_nav2_plan(raw_goal_pose, start_pose=start_pose)
    goal_pose = _compute_approach_standoff_goal(raw_goal_pose, temp_plan, standoff_m=1.0)
    if goal_pose is None:
        robot_pose = _get_robot_start_pose()
        goal_pose = _apply_object_standoff(raw_goal_pose, robot_pose, standoff_m=1.0)
    plan = request_nav2_plan(goal_pose, start_pose=start_pose)
    map_source = "nav2_smac_2d"
    navigation_execution = {
        "available": True,
        "status": "awaiting_acceptance",
        "reason": (
            "Accepting this preview sends the goal pose to the Nav2 NavigateToPose action, "
            "which handles path planning and obstacle avoidance via the slam_toolbox map."
        ),
        "existing_interfaces": [
            {
                "type": "ros_action",
                "name": "/navigate_to_pose",
                "package": "nav2_msgs/action/NavigateToPose",
                "mode": "Nav2 full navigation stack (planner + controller + recovery)",
            },
        ],
    }

    return {
        "ok": True,
        "observation": {
            "id": row[0],
            "object_id": row[1],
            "class_id": row[2],
            "class_name": class_name_from_id(row[2]),
            "yolo_track_id": row[3],
            "position_source": row[4],
            "created_at": row[8],
            "scene_timestamp": row[9],
        },
        "map_source": map_source,
        "map_generated_at": None,
        "plan": plan,
        "navigation_execution": navigation_execution,
    }


@app.post("/api/map/plan_custom_path")
def plan_custom_path(payload: dict):
    stale_pending = assert_slam_pose_not_pending_for_nav2(allow_stale_for_planning=True)

    start = payload.get("start")
    goal = payload.get("goal")
    if not isinstance(goal, dict) or goal.get("x") is None or goal.get("y") is None:
        raise HTTPException(status_code=400, detail="Missing or invalid goal coordinates")

    explicit_start = isinstance(start, dict) and start.get("x") is not None and start.get("y") is not None
    if explicit_start:
        start_pose = {
            "x": float(start["x"]),
            "y": float(start["y"]),
            "z": float(start.get("z", 0.0) or 0.0),
            "yaw": float(start.get("yaw", 0.0) or 0.0),
        }
    elif stale_pending:
        start_pose = _get_robot_start_pose()
    else:
        # Don't pass a file-based start pose to Nav2 — let it look up the
        # current robot pose from TF, which is always live and correct.
        start_pose = None

    raw_goal_pose = {"x": float(goal["x"]), "y": float(goal["y"]), "z": 0.0}

    # Apply object standoff if requested (e.g., agent navigating to an object).
    # We first plan to the raw goal to discover the approach direction, then
    # offset the goal along that direction so the robot stops before the object.
    standoff_m = float(payload.get("standoff_m") or 0.0)
    if standoff_m > 0.0:
        temp_plan = request_nav2_plan(raw_goal_pose, start_pose=start_pose)
        goal_pose = _compute_approach_standoff_goal(raw_goal_pose, temp_plan, standoff_m=standoff_m)
        if goal_pose is None:
            robot_pose = _get_robot_start_pose()
            goal_pose = _apply_object_standoff(raw_goal_pose, robot_pose, standoff_m=standoff_m)
    else:
        goal_pose = raw_goal_pose

    nav2_costmap_payload = load_map_payload_from_path(NAV2_GLOBAL_COSTMAP_PATH)
    if isinstance(nav2_costmap_payload, dict):
        if start_pose is not None:
            start_pose, _ = snap_pose_to_nearest_traversable_map_cell(
                nav2_costmap_payload,
                start_pose,
                pose_label="start",
                allow_unknown=True,
            )
        goal_pose, _ = snap_pose_to_nearest_traversable_map_cell(
            nav2_costmap_payload,
            goal_pose,
            pose_label="goal",
        )

    plan = request_nav2_plan(goal_pose, start_pose=start_pose)

    return {
        "ok": True,
        "source": "custom",
        "observation": None,
        "map_source": "nav2_smac_2d",
        "map_generated_at": None,
        "plan": plan,
        "navigation_execution": {
            "available": True,
            "status": "awaiting_acceptance",
            "reason": (
                "Accepting this preview sends the goal pose to the Nav2 NavigateToPose action, "
                "which handles path planning and obstacle avoidance via the slam_toolbox map."
            ),
        },
    }


@app.post("/api/navigation/execute")
def execute_navigation_plan(payload: dict):
    assert_slam_pose_not_pending_for_nav2()

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Execution payload must be a JSON object")

    plan = payload.get("plan")
    if not isinstance(plan, dict):
        raise HTTPException(status_code=400, detail="Execution payload is missing a plan")

    path = plan.get("path")
    if not isinstance(path, list) or len(path) < 2:
        raise HTTPException(status_code=400, detail="Execution payload must include at least two path waypoints")
    if not _plan_uses_nav2_planner(plan):
        raise HTTPException(
            status_code=409,
            detail=(
                "This path was not produced by Nav2, so it is preview-only and cannot be executed. "
                "Replan after the robot pose is localized in free Nav2 costmap space."
            ),
        )

    status_payload = load_json_file(NAVIGATION_STATUS_PATH)
    if _is_stale_queued_navigation_status(status_payload):
        _clear_navigation_request_files()
        status_payload = None

    if isinstance(status_payload, dict) and status_payload.get("state") in {"queued", "running"}:
        raise HTTPException(
            status_code=409,
            detail=f"Navigation executor is busy with request {status_payload.get('request_id') or 'unknown'}",
        )

    request_id = f"nav-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    created_at = time.time()
    execution_payload = {
        "request_id": request_id,
        "created_at": created_at,
        "observation": payload.get("observation"),
        "plan": {
            "path": path,
            "path_length_m": float(plan.get("path_length_m") or 0.0),
            "num_waypoints": int(plan.get("num_waypoints") or len(path)),
            "start": plan.get("start"),
            "goal": plan.get("goal"),
            "planner": plan.get("planner"),
        },
        "execution": {
            "mode": "nav2_navigate_to_pose",
            "frame_id": "world",
            "accepted_from": "web_ui",
            "debug_publish_cmd_vel_on_failure": bool(payload.get("debug_publish_cmd_vel_on_failure")),
        },
    }
    atomic_write_json(NAVIGATION_REQUEST_PATH, execution_payload)

    queued_status = {
        "available": True,
        "state": "queued",
        "request_id": request_id,
        "created_at": created_at,
        "updated_at": created_at,
        "message": "Navigation request accepted and queued for execution.",
        "observation": execution_payload.get("observation"),
        "plan": execution_payload.get("plan"),
        "execution": execution_payload.get("execution"),
    }
    atomic_write_json(NAVIGATION_STATUS_PATH, queued_status)
    return enrich_navigation_status(queued_status)


class ObjectArrivalVerifyRequest(BaseModel):
    object_id: str
    object_name: str | None = None
    exclude: list[list[float]] | None = None  # [x, y] spots already tried
    since_ts: float | None = None             # epoch seconds when the journey started
    window_s: int = 90                        # detection freshness window
    cluster_m: float = 2.0                    # spatial clustering of past sightings


def _room_name_for_xy(rooms: list, x: float, y: float) -> str | None:
    for row in rooms:
        name = row[0]
        xs = [v for v in row[1:8:2] if v is not None]
        ys = [v for v in row[2:9:2] if v is not None]
        if xs and ys and min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
            return name
    return None


@app.post("/api/navigation/verify_object_arrival")
def verify_object_arrival(req: ObjectArrivalVerifyRequest):
    """Verify (after a 'go to object' navigation completed) whether the target
    object was re-perceived; if not, propose the next historical sighting spot.

    Detection-based: a fresh `object_observations` row means the perception
    pipeline currently sees the object. Candidates are recency-ordered spatial
    clusters of the object's sighting history, minus the spots in `exclude`.
    """
    window_s = max(15, min(int(req.window_s), 600))
    cluster_m = max(0.5, min(float(req.cluster_m), 10.0))
    exclude = [
        (float(p[0]), float(p[1]))
        for p in (req.exclude or [])
        if isinstance(p, (list, tuple)) and len(p) >= 2
    ]
    robot_pose = _get_robot_start_pose()

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            # Guard: without a live perception pipeline 'not seen' is meaningless.
            cur.execute("SELECT max(timestamp) FROM scenes")
            last_scene = cur.fetchone()[0]
            if last_scene is not None and last_scene.tzinfo is None:
                last_scene = last_scene.replace(tzinfo=timezone.utc)
            if last_scene is None or (datetime.now(timezone.utc) - last_scene).total_seconds() > 300:
                return {"status": "unverifiable", "reason": "perception pipeline stale", "robot_pose": robot_pose}

            cur.execute(
                """
                SELECT x, y, created_at
                FROM object_observations
                WHERE object_id = %s AND created_at > now() - make_interval(secs => %s)
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (req.object_id, window_s),
            )
            recent = cur.fetchall()
            if req.since_ts:
                since_dt = datetime.fromtimestamp(float(req.since_ts) - 15.0, tz=timezone.utc)
                recent = [
                    r for r in recent
                    if (r[2] if r[2].tzinfo else r[2].replace(tzinfo=timezone.utc)) >= since_dt
                ]
            if recent:
                x, y, seen_at = recent[0]
                return {
                    "status": "seen",
                    "evidence": {
                        "x": float(x) if x is not None else None,
                        "y": float(y) if y is not None else None,
                        "seen_at": str(seen_at),
                        "detections": len(recent),
                    },
                    "robot_pose": robot_pose,
                }

            # Next candidate: recency-ordered spatial clusters of sighting history.
            cur.execute(
                """
                SELECT x, y, created_at
                FROM object_observations
                WHERE object_id = %s AND x IS NOT NULL AND y IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 300
                """,
                (req.object_id,),
            )
            rows = cur.fetchall()
            cur.execute("SELECT name, x1, y1, x2, y2, x3, y3, x4, y4 FROM rooms")
            rooms = cur.fetchall()
    finally:
        conn.close()

    clusters: list[dict] = []
    for x, y, ts in rows:
        fx, fy = float(x), float(y)
        for cluster in clusters:
            if math.hypot(cluster["x"] - fx, cluster["y"] - fy) <= cluster_m:
                cluster["observations"] += 1
                break
        else:
            clusters.append({"x": fx, "y": fy, "last_seen_at": ts, "observations": 1})

    remaining = [
        c for c in clusters
        if not any(math.hypot(c["x"] - ex, c["y"] - ey) <= cluster_m for ex, ey in exclude)
    ]
    if not remaining:
        return {"status": "exhausted", "candidates_remaining": 0, "robot_pose": robot_pose}

    candidate = remaining[0]
    return {
        "status": "not_seen",
        "next_candidate": {
            "x": candidate["x"],
            "y": candidate["y"],
            "room": _room_name_for_xy(rooms, candidate["x"], candidate["y"]),
            "last_seen_at": str(candidate["last_seen_at"]),
            "observations": candidate["observations"],
        },
        "candidates_remaining": len(remaining),
        "robot_pose": robot_pose,
    }


@app.post("/api/navigation/stop")
def stop_navigation():
    status_payload = load_json_file(NAVIGATION_STATUS_PATH)
    if not isinstance(status_payload, dict) or status_payload.get("state") not in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="No active navigation to stop.")

    if status_payload.get("state") == "queued":
        request_id = status_payload.get("request_id")
        _clear_navigation_request_files()
        cleared_status = {
            "available": True,
            "state": "idle",
            "request_id": request_id,
            "updated_at": time.time(),
            "message": "Cleared queued navigation request before execution started.",
        }
        atomic_write_json(NAVIGATION_STATUS_PATH, cleared_status)
        return enrich_navigation_status(cleared_status)

    request_id = status_payload.get("request_id")
    cancel_payload = {
        "request_id": request_id,
        "created_at": time.time(),
    }
    atomic_write_json(NAVIGATION_CANCEL_REQUEST_PATH, cancel_payload)

    # Update status immediately so the UI reflects the stop request
    status_payload["state"] = "stopping"
    status_payload["message"] = "Stop requested — waiting for Nav2 to cancel…"
    status_payload["updated_at"] = time.time()
    atomic_write_json(NAVIGATION_STATUS_PATH, status_payload)
    return enrich_navigation_status(status_payload)


@app.get("/api/navigation/status")
def get_navigation_status():
    status_payload = load_json_file(NAVIGATION_STATUS_PATH)
    if isinstance(status_payload, dict):
        if _is_stale_queued_navigation_status(status_payload):
            status_payload = dict(status_payload)
            status_payload["queue_stale"] = True
            status_payload["executor_available"] = False
            status_payload["message"] = (
                "Navigation request is still queued. The ROS navigation executor may not be running."
            )
        status_payload.setdefault("available", True)
        return enrich_navigation_status(status_payload)
    return build_navigation_status_fallback()# ---------------------------------------------------------------------------
# Dataset analysis
# ---------------------------------------------------------------------------

@app.get("/api/scene-change-dataset")
def scene_change_dataset(
    day: str = Query("2026-05-26"),
    room: str = Query(""),
    change_type: str = Query(""),
    dataset: str = Query("default"),
):
    """Return scene-change dataset for a given day with optional filters."""
    import json
    from pathlib import Path

    data_dir = DATASET_BACKUPS.get(dataset)
    if data_dir is None or not data_dir.exists():
        # Fallback to default paths
        candidate_dirs = [SCENE_CHANGE_DATA_DIR]
        data_dir = None
        for candidate in candidate_dirs:
            if candidate.exists():
                data_dir = candidate
                break
    if data_dir is None:
        raise HTTPException(status_code=404, detail="scene-change dataset directory not found")

    scene_path = data_dir / f"scene_changes_{day}.json"
    if not scene_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {scene_path}")

    with open(scene_path, "r", encoding="utf-8") as f:
        scenes = json.load(f)

    # Load interactions if available
    interactions_path = data_dir / f"synthetic_interactions_{day}.json"
    interactions_by_key = {}
    if interactions_path.exists():
        with open(interactions_path, "r", encoding="utf-8") as f:
            inter_data = json.load(f)
        for item in inter_data:
            key = (item.get("room"), item.get("time"))
            interactions_by_key.setdefault(key, []).append(item)

    rooms = ["kitchen", "storage", "office 1", "office 2", "office 3", "meeting room"]
    changes_per_room = {r: 0 for r in rooms}
    total_changes = 0
    major_changes = 0
    minor_changes = 0

    for s in scenes:
        if s["change_type"] in ("minor_change", "major_change"):
            changes_per_room[s["room"]] = changes_per_room.get(s["room"], 0) + 1
            total_changes += 1
        if s["change_type"] == "major_change":
            major_changes += 1
        elif s["change_type"] == "minor_change":
            minor_changes += 1

    # Build change matrix with detailed deltas
    timestamps = sorted(list(set(s["time"] for s in scenes)))
    matrix = {"rooms": rooms, "timestamps": timestamps, "changes": [], "deltas": []}
    for r in rooms:
        change_row = []
        delta_row = []
        for t in timestamps:
            entry = next((s for s in scenes if s["room"] == r and s["time"] == t), None)
            if entry:
                change_row.append(entry["change_type"])
                delta_row.append({
                    "people_added": entry.get("people_added", []),
                    "people_removed": entry.get("people_removed", []),
                    "objects_added": entry.get("objects_added", []),
                    "objects_removed": entry.get("objects_removed", []),
                    "activities_changed": entry.get("activities_changed", False),
                    "activity_change_severity": entry.get("activity_change_severity", "none"),
                    "activities_added": entry.get("activities_added", []),
                    "activities_removed": entry.get("activities_removed", []),
                })
            else:
                change_row.append("no_change")
                delta_row.append({
                    "people_added": [], "people_removed": [],
                    "objects_added": [], "objects_removed": [],
                    "activities_changed": False, "activity_change_severity": "none",
                    "activities_added": [], "activities_removed": [],
                })
        matrix["changes"].append(change_row)
        matrix["deltas"].append(delta_row)

    # Filter scenes for response
    filtered = scenes
    if room:
        filtered = [s for s in filtered if s["room"] == room]
    if change_type:
        filtered = [s for s in filtered if s["change_type"] == change_type]

    # Attach interactions to filtered scenes
    for s in filtered:
        s["interactions"] = interactions_by_key.get((s.get("room"), s.get("time")), [])

    return {
        "dataset_path": str(scene_path),
        "day": day,
        "total_scenes": len(scenes),
        "total_changes": total_changes,
        "major_changes": major_changes,
        "minor_changes": minor_changes,
        "rooms": rooms,
        "changes_per_room": changes_per_room,
        "matrix": matrix,
        "scenes": filtered,
    }


@app.get("/api/scene-change-results")
def list_scene_change_results():
    """List saved scene-change strategy result files."""
    from pathlib import Path

    # Check both current path and legacy /reports path for backward compat
    all_files = set()
    for candidate in [_get_reports_dir(), Path("/reports")]:
        if candidate.exists():
            for pattern in ("scene_change_strategy_*.json", "scene_change_ablation_*.json"):
                for f in candidate.glob(pattern):
                    all_files.add(f.name)
    files = sorted(all_files, reverse=True)
    return {"files": files[:50]}


@app.get("/api/scene-change-results/{filename}")
def load_scene_change_result(filename: str):
    """Load a saved scene-change strategy result file."""
    import json
    from pathlib import Path

    # Try current path first, then legacy /reports path
    for candidate in [_get_reports_dir(), Path("/reports")]:
        file_path = (candidate / filename).resolve()
        if not str(file_path).startswith(str(candidate.resolve())):
            continue
        if file_path.exists():
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise HTTPException(status_code=404, detail="File not found")


def _get_reports_dir() -> Path:
    """Resolve the reports directory for both host and container environments."""
    from pathlib import Path
    file_parent = Path(__file__).resolve().parent
    if file_parent.name == "frontend":
        # Host layout: bordsupr/frontend/app.py -> project root is 2 levels up
        return file_parent.parent.parent / "reports"
    # Container layout: /app/app.py -> reports is /app/reports
    return file_parent / "reports"


@app.get("/api/scene-change-results-debug")
def scene_change_results_debug():
    """Debug endpoint: show reports directory candidates and their contents."""
    from pathlib import Path
    candidates = [_get_reports_dir(), Path("/reports"), Path("reports")]
    result = {}
    for c in candidates:
        result[str(c)] = {
            "exists": c.exists(),
            "is_dir": c.is_dir() if c.exists() else False,
            "files": sorted([f.name for f in c.glob("scene_change_strategy_*.json")] + [f.name for f in c.glob("scene_change_ablation_*.json")]) if c.exists() else [],
        }
    # Also show __file__ and parent for diagnostics
    result["__file__"] = str(Path(__file__).resolve())
    result["__file__parent"] = str(Path(__file__).resolve().parent)
    return result


def _save_scene_change_results_to_file(results_dict: dict, seed: int | None, strategies: list[str]) -> str:
    """Save scene-change strategy results to a timestamped JSON file in reports/."""
    from datetime import datetime, timezone
    from pathlib import Path
    import json

    output_data = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "strategies": strategies,
        "results": results_dict,
    }

    reports_dir = _get_reports_dir()
    reports_dir.mkdir(parents=True, exist_ok=True)
    file_path = reports_dir / f"scene_change_strategy_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # Verify the file was actually written
    if file_path.exists():
        size = file_path.stat().st_size
        logging.info(f"Saved scene-change results to {file_path} ({size} bytes)")
    else:
        logging.warning(f"File {file_path} does not exist after writing!")

    return str(file_path)


@app.post("/api/scene-change-strategy/run")
def scene_change_strategy_run(payload: dict | None = None):
    """Run scene-change strategies and return comparison."""
    import sys
    from pathlib import Path
    payload = payload or {}
    strategies = payload.get("strategies") or ["fixed_10min", "random", "frequency", "greedy", "agent_scene_change"]
    seed = payload.get("seed")
    no_tools = payload.get("no_tools", False)

    # Ensure curiosity module is on path
    from pathlib import Path
    candidate_dirs = [
        Path(__file__).resolve().parent.parent.parent / "curiosity",
        Path("/app") / "curiosity",
    ]
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from scene_change_simulator import run_scene_change_round_robin

    try:
        results = run_scene_change_round_robin(strategies=strategies, seed=seed, no_tools=no_tools)
    except RuntimeError as exc:
        error_msg = str(exc)
        if "VLM" in error_msg or "vlm" in error_msg:
            return JSONResponse(
                status_code=503,
                content={"detail": f"VLM backend unavailable or returned invalid response. {error_msg}"},
            )
        return JSONResponse(
            status_code=500,
            content={"detail": f"Simulation failed: {error_msg}"},
        )
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Simulation failed: {exc}"},
        )

    # Compute theoretical max recall from the dataset (same for all strategies)
    try:
        from scene_change_simulator import SceneChangeTimeLog
        _sc_log = SceneChangeTimeLog()
        _theoretical_max_recall = round(_sc_log.theoretical_max_recall, 4)
    except Exception:
        _theoretical_max_recall = 0.0

    def _to_dict(r):
        return {
            "strategy": r.strategy,
            "total_changes": r.total_changes,
            "total_major_changes": r.total_major_changes,
            "observed_changes": r.observed_changes,
            "total_visits": r.total_visits,
            "visits_with_changes": r.visits_with_changes,
            "change_recall": round(r.change_recall, 4),
            "change_precision": round(r.change_precision, 4),
            "change_f1": round(r.change_f1, 4),
            "hit_rate": round(r.hit_rate, 4),
            "theoretical_max_recall": _theoretical_max_recall,
            "scene_change_tp": r.scene_change_tp,
            "scene_change_tn": r.scene_change_tn,
            "scene_change_fp": r.scene_change_fp,
            "scene_change_fn": r.scene_change_fn,
            "scene_change_accuracy": round(r.scene_change_accuracy, 4),
            "no_change_accuracy": round(getattr(r, "no_change_accuracy", 0.0), 4),
            "change_accuracy": round(getattr(r, "change_accuracy", 0.0), 4),
            "minor_tp": getattr(r, "minor_tp", 0),
            "minor_fp": getattr(r, "minor_fp", 0),
            "minor_fn": getattr(r, "minor_fn", 0),
            "major_tp": getattr(r, "major_tp", 0),
            "major_fp": getattr(r, "major_fp", 0),
            "major_fn": getattr(r, "major_fn", 0),
            "minor_recall": round(getattr(r, "minor_recall", 0.0), 4),
            "minor_precision": round(getattr(r, "minor_precision", 0.0), 4),
            "major_recall": round(getattr(r, "major_recall", 0.0), 4),
            "major_precision": round(getattr(r, "major_precision", 0.0), 4),
            "activity_change_tp": getattr(r, "activity_change_tp", 0),
            "activity_change_tn": getattr(r, "activity_change_tn", 0),
            "activity_change_fp": getattr(r, "activity_change_fp", 0),
            "activity_change_fn": getattr(r, "activity_change_fn", 0),
            "activity_change_accuracy": round(getattr(r, "activity_change_accuracy", 0.0), 4),
            "navigation_to_changed_room": r.navigation_to_changed_room,
            "total_moves": r.total_moves,
            "navigation_precision": round(r.navigation_precision, 4),
            "exploration_coverage": round(r.exploration_coverage, 4),
            "vlm_calls_made": r.vlm_calls_made,
            "elapsed_seconds": r.elapsed_seconds,
            "cross_room_misses": getattr(r, "cross_room_misses", 0),
            "cross_room_miss_rate": round(getattr(r, "cross_room_miss_rate", 0.0), 4),
            "total_event_timesteps": getattr(r, "total_event_timesteps", 0),
            "event_timesteps_caught": getattr(r, "event_timesteps_caught", 0),
            "event_timestep_rate": round(getattr(r, "event_timestep_rate", 0.0), 4),
            "avg_detection_latency_minutes": getattr(r, "avg_detection_latency_minutes", 0.0),
            "median_detection_latency_minutes": getattr(r, "median_detection_latency_minutes", 0.0),
            "max_detection_latency_minutes": getattr(r, "max_detection_latency_minutes", 0.0),
            "changes_never_detected": getattr(r, "changes_never_detected", 0),
            "latency_histogram": getattr(r, "latency_histogram", {}),
            "detection_latency_histogram": getattr(r, "detection_latency_histogram", {}),
            "avg_detection_latency_steps": getattr(r, "avg_detection_latency_steps", 0.0),
            "median_detection_latency_steps": getattr(r, "median_detection_latency_steps", 0.0),
            "max_detection_latency_steps": getattr(r, "max_detection_latency_steps", 0),
            "changes_detected_immediately": getattr(r, "changes_detected_immediately", 0),
            "per_room_latency": getattr(r, "per_room_latency", {}),
            "cumulative_detection_curve": getattr(r, "cumulative_detection_curve", {}),
            "catchable_changes": getattr(r, "catchable_changes", 0),
            "detected_among_catchable": getattr(r, "detected_among_catchable", 0.0),
            "absent_changes": getattr(r, "absent_changes", 0),
            "blind_changes": getattr(r, "blind_changes", 0),
            "conditional_latency_histogram": getattr(r, "conditional_latency_histogram", {}),
            "visit_opportunity_histogram": getattr(r, "visit_opportunity_histogram", {}),
            "path_max_changes": getattr(r, "path_max_changes", 0),
            "theoretical_max_changes": getattr(r, "theoretical_max_changes", 0),
            "detection_efficiency": round(getattr(r, "detection_efficiency", 0.0), 4),
            "navigation_efficiency": round(getattr(r, "navigation_efficiency", 0.0), 4),
            "normalized_recall": round(getattr(r, "normalized_recall", 0.0), 4),
            "optimal_path_changes": getattr(r, "optimal_path_changes", 0),
            "optimal_path_moves": getattr(r, "optimal_path_moves", 0),
            "optimal_path_efficiency": round(getattr(r, "optimal_path_efficiency", 0.0), 4),
            "robot_memory_tp": getattr(r, "robot_memory_tp", 0),
            "robot_memory_tn": getattr(r, "robot_memory_tn", 0),
            "robot_memory_fp": getattr(r, "robot_memory_fp", 0),
            "robot_memory_fn": getattr(r, "robot_memory_fn", 0),
            "robot_memory_accuracy": round(getattr(r, "robot_memory_accuracy", 0.0), 4),
            "robot_memory_precision": round(getattr(r, "robot_memory_precision", 0.0), 4),
            "robot_memory_recall": round(getattr(r, "robot_memory_recall", 0.0), 4),
            "robot_memory_f1": round(getattr(r, "robot_memory_f1", 0.0), 4),
            "visits": [
                {
                    "visit": v.visit_number,
                    "room": v.room,
                    "start": v.start.strftime("%H:%M"),
                    "end": v.end.strftime("%H:%M"),
                    "changes_observed": v.changes_observed,
                    "dwell_seconds": v.dwell_seconds,
                }
                for v in r.visits
            ],
        }

    # Save results to DB
    try:
        import sys
        from pathlib import Path
        agent_dir = Path(__file__).resolve().parent / "agent"
        if str(agent_dir) not in sys.path:
            sys.path.insert(0, str(agent_dir))
        from eval_db import save_scene_change_result
        batch_id = f"sc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for s, r in results.items():
            try:
                save_scene_change_result(batch_id, r)
            except Exception as e:
                logging.warning(f"Failed to save scene-change result for {s}: {e}")
    except Exception as e:
        logging.warning(f"Failed to import eval_db for scene-change results: {e}")

    results_dict = {s: _to_dict(r) for s, r in results.items()}
    file_path = _save_scene_change_results_to_file(results_dict, seed, strategies)

    return {
        "strategies": strategies,
        "seed": seed,
        "saved_to": file_path,
        "results": results_dict,
    }


@app.get("/api/scene-change-strategy/stream")
def scene_change_strategy_stream(
    strategies: str = Query(""),
    seed: int = Query(42),
    dataset: str = Query("default"),
    no_tools: bool = Query(False),
):
    """Stream scene-change strategy simulation step-by-step via SSE."""
    import sys
    import json
    import asyncio
    from pathlib import Path

    strategy_list = [s.strip() for s in strategies.split(",") if s.strip()]
    if not strategy_list:
        strategy_list = ["fixed_10min", "random", "frequency", "greedy", "agent_scene_change"]

    # Ensure curiosity module is on path
    candidate_dirs = [
        Path(__file__).resolve().parent.parent.parent / "curiosity",
        Path("/app") / "curiosity",
    ]
    for candidate in candidate_dirs:
        if candidate.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break

    from scene_change_simulator import run_scene_change_simulation, SceneChangeSimulationResult

    def _step_to_dict(step) -> dict | None:
        if step is None:
            return None

        def _cap(val, max_len=500):
            if isinstance(val, str) and len(val) > max_len:
                return val[:max_len] + f"\n... ({len(val) - max_len} chars truncated)"
            return val

        def _cap_list(val, max_items=20):
            if isinstance(val, list) and len(val) > max_items:
                return val[:max_items] + [f"... ({len(val) - max_items} more items)"]
            return val

        def _sanitize_json(val):
            """Replace NaN/Infinity with strings so JavaScript JSON.parse accepts it."""
            import math
            if isinstance(val, float):
                if math.isinf(val):
                    return "inf" if val > 0 else "-inf"
                if math.isnan(val):
                    return "nan"
                return val
            if isinstance(val, dict):
                return {k: _sanitize_json(v) for k, v in val.items()}
            if isinstance(val, list):
                return [_sanitize_json(v) for v in val]
            return val

        return _sanitize_json({
            "time_str": step.time_str,
            "room": step.room,
            "action": step.action,
            "target_room": step.target_room,
            "gt_change_type": step.gt_change_type,
            "vlm_detected_change": step.vlm_detected_change,
            "vlm_change": step.vlm_change,
            "vlm_detected_activity_change": step.vlm_detected_activity_change,
            "people_present": _cap_list(step.people_present, 20),
            "objects_present": _cap_list(step.objects_present, 20),
            "decision": step.decision,
            "prompt": _cap(step.prompt, 4000),
            "raw_response": _cap(step.raw_response, 1200),
            "system_prompt": _cap(step.system_prompt, 2000),
            "scene_text": _cap(step.scene_text, 600),
            "interactions": _cap_list(step.interactions, 10),
            "gt_activity_changed": step.gt_activity_changed,
            "tool_calls": step.tool_calls,
            "error": _cap(step.error, 500),
        })

    # Resolve dataset directory
    dataset_dirs = {
        "default": DATASET_BACKUPS.get("default", SCENE_CHANGE_DATA_DIR),
        "adversarial": DATASET_BACKUPS.get("adversarial", SCENE_CHANGE_DATA_DIR),
        "regime": DATASET_BACKUPS.get("regime", SCENE_CHANGE_DATA_DIR),
    }
    data_dir = dataset_dirs.get(dataset, SCENE_CHANGE_DATA_DIR)

    # Compute theoretical max recall from the dataset (same for all strategies)
    try:
        from scene_change_simulator import SceneChangeTimeLog
        _sc_log = SceneChangeTimeLog(data_dir=data_dir)
        _theoretical_max_recall_stream = round(_sc_log.theoretical_max_recall, 4)
    except Exception:
        _theoretical_max_recall_stream = 0.0

    def _result_to_dict(r) -> dict:
        return {
            "strategy": r.strategy,
            "total_changes": r.total_changes,
            "total_major_changes": r.total_major_changes,
            "observed_changes": r.observed_changes,
            "total_visits": r.total_visits,
            "visits_with_changes": r.visits_with_changes,
            "change_recall": round(r.change_recall, 4),
            "change_precision": round(r.change_precision, 4),
            "change_f1": round(r.change_f1, 4),
            "hit_rate": round(r.hit_rate, 4),
            "theoretical_max_recall": _theoretical_max_recall_stream,
            "scene_change_tp": r.scene_change_tp,
            "scene_change_tn": r.scene_change_tn,
            "scene_change_fp": r.scene_change_fp,
            "scene_change_fn": r.scene_change_fn,
            "scene_change_accuracy": round(r.scene_change_accuracy, 4),
            "no_change_accuracy": round(getattr(r, "no_change_accuracy", 0.0), 4),
            "change_accuracy": round(getattr(r, "change_accuracy", 0.0), 4),
            "minor_tp": getattr(r, "minor_tp", 0),
            "minor_fp": getattr(r, "minor_fp", 0),
            "minor_fn": getattr(r, "minor_fn", 0),
            "major_tp": getattr(r, "major_tp", 0),
            "major_fp": getattr(r, "major_fp", 0),
            "major_fn": getattr(r, "major_fn", 0),
            "minor_recall": round(getattr(r, "minor_recall", 0.0), 4),
            "minor_precision": round(getattr(r, "minor_precision", 0.0), 4),
            "major_recall": round(getattr(r, "major_recall", 0.0), 4),
            "major_precision": round(getattr(r, "major_precision", 0.0), 4),
            "activity_change_tp": getattr(r, "activity_change_tp", 0),
            "activity_change_tn": getattr(r, "activity_change_tn", 0),
            "activity_change_fp": getattr(r, "activity_change_fp", 0),
            "activity_change_fn": getattr(r, "activity_change_fn", 0),
            "activity_change_accuracy": round(getattr(r, "activity_change_accuracy", 0.0), 4),
            "navigation_to_changed_room": r.navigation_to_changed_room,
            "total_moves": r.total_moves,
            "navigation_precision": round(r.navigation_precision, 4),
            "exploration_coverage": round(r.exploration_coverage, 4),
            "vlm_calls_made": r.vlm_calls_made,
            "elapsed_seconds": r.elapsed_seconds,
            "cross_room_misses": getattr(r, "cross_room_misses", 0),
            "cross_room_miss_rate": round(getattr(r, "cross_room_miss_rate", 0.0), 4),
            "total_event_timesteps": getattr(r, "total_event_timesteps", 0),
            "event_timesteps_caught": getattr(r, "event_timesteps_caught", 0),
            "event_timestep_rate": round(getattr(r, "event_timestep_rate", 0.0), 4),
            "avg_detection_latency_minutes": getattr(r, "avg_detection_latency_minutes", 0.0),
            "median_detection_latency_minutes": getattr(r, "median_detection_latency_minutes", 0.0),
            "max_detection_latency_minutes": getattr(r, "max_detection_latency_minutes", 0.0),
            "changes_never_detected": getattr(r, "changes_never_detected", 0),
            "latency_histogram": getattr(r, "latency_histogram", {}),
            "detection_latency_histogram": getattr(r, "detection_latency_histogram", {}),
            "avg_detection_latency_steps": getattr(r, "avg_detection_latency_steps", 0.0),
            "median_detection_latency_steps": getattr(r, "median_detection_latency_steps", 0.0),
            "max_detection_latency_steps": getattr(r, "max_detection_latency_steps", 0),
            "changes_detected_immediately": getattr(r, "changes_detected_immediately", 0),
            "per_room_latency": getattr(r, "per_room_latency", {}),
            "cumulative_detection_curve": getattr(r, "cumulative_detection_curve", {}),
            "catchable_changes": getattr(r, "catchable_changes", 0),
            "detected_among_catchable": getattr(r, "detected_among_catchable", 0.0),
            "absent_changes": getattr(r, "absent_changes", 0),
            "blind_changes": getattr(r, "blind_changes", 0),
            "conditional_latency_histogram": getattr(r, "conditional_latency_histogram", {}),
            "visit_opportunity_histogram": getattr(r, "visit_opportunity_histogram", {}),
            "path_max_changes": getattr(r, "path_max_changes", 0),
            "theoretical_max_changes": getattr(r, "theoretical_max_changes", 0),
            "detection_efficiency": round(getattr(r, "detection_efficiency", 0.0), 4),
            "navigation_efficiency": round(getattr(r, "navigation_efficiency", 0.0), 4),
            "normalized_recall": round(getattr(r, "normalized_recall", 0.0), 4),
            "optimal_path_changes": getattr(r, "optimal_path_changes", 0),
            "optimal_path_moves": getattr(r, "optimal_path_moves", 0),
            "optimal_path_efficiency": round(getattr(r, "optimal_path_efficiency", 0.0), 4),
            "robot_memory_tp": getattr(r, "robot_memory_tp", 0),
            "robot_memory_tn": getattr(r, "robot_memory_tn", 0),
            "robot_memory_fp": getattr(r, "robot_memory_fp", 0),
            "robot_memory_fn": getattr(r, "robot_memory_fn", 0),
            "robot_memory_accuracy": round(getattr(r, "robot_memory_accuracy", 0.0), 4),
            "robot_memory_precision": round(getattr(r, "robot_memory_precision", 0.0), 4),
            "robot_memory_recall": round(getattr(r, "robot_memory_recall", 0.0), 4),
            "robot_memory_f1": round(getattr(r, "robot_memory_f1", 0.0), 4),
            "visits": [
                {
                    "visit": v.visit_number,
                    "room": v.room,
                    "start": v.start.strftime("%H:%M"),
                    "end": v.end.strftime("%H:%M"),
                    "changes_observed": v.changes_observed,
                    "dwell_seconds": v.dwell_seconds,
                }
                for v in r.visits
            ],
        }

    async def event_generator():
        import asyncio
        import time
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sim_results: dict[str, Any] = {}
        saved_results: dict[str, Any] = {}
        remaining = len(strategy_list)
        tasks = []
        completed_strategies: set[str] = set()

        for strategy in strategy_list:
            def make_callbacks(strat):
                import time as _time

                def progress_callback(payload):
                    step_dict = _step_to_dict(payload.get("step"))
                    event = {
                        "type": "step",
                        "strategy": payload["strategy"],
                        "steps_processed": payload["steps_processed"],
                        "total_steps": payload["total_steps"],
                        "current_room": payload["current_room"],
                        "changes_so_far": payload["changes_so_far"],
                        "step": step_dict,
                    }
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, event)
                    except Exception:
                        pass
                    _time.sleep(0.001)

                def run_sim():
                    try:
                        result = run_scene_change_simulation(strat, seed=seed, progress_callback=progress_callback, data_dir=data_dir, no_tools=no_tools)
                        sim_results[strat] = result
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "sim_done", "strategy": strat})
                    except Exception as exc:
                        import traceback
                        tb = traceback.format_exc()
                        logging.error("[scene-change-stream] Strategy %s crashed:\n%s", strat, tb)
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "sim_error", "strategy": strat, "error": str(exc), "traceback": tb})

                return run_sim

            tasks.append(loop.run_in_executor(None, make_callbacks(strategy)))

        # Stream events from all running strategies with timeout protection
        last_activity = time.time()
        while remaining > 0:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1800.0)
                last_activity = time.time()
                if event.get("type") == "sim_done":
                    remaining -= 1
                    completed_strategies.add(event["strategy"])
                    strat = event["strategy"]
                    result = sim_results.get(strat)
                    if result:
                        result_dict = _result_to_dict(result)
                        saved_results[strat] = result_dict
                        yield f"event: complete\ndata: {json.dumps({'strategy': strat, 'result': result_dict})}\n\n"
                elif event.get("type") == "sim_error":
                    remaining -= 1
                    completed_strategies.add(event["strategy"])
                    yield f"event: error\ndata: {json.dumps({'strategy': event['strategy'], 'error': event.get('error'), 'traceback': event.get('traceback')})}\n\n"
                else:
                    yield f"event: step\ndata: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                # No events for 360s — mark any unfinished strategies as timed out
                stalled = [s for s in strategy_list if s not in completed_strategies]
                if stalled:
                    for strat in stalled:
                        remaining -= 1
                        completed_strategies.add(strat)
                        yield f"event: error\ndata: {json.dumps({'strategy': strat, 'error': 'Simulation timed out — no progress for 1800 seconds. The VLM server may be overloaded, still loading the model, or unresponsive.', 'error_type': 'timeout'})}\n\n"
                else:
                    # All strategies actually finished but we missed the events somehow
                    remaining = 0
                break

            # Send keepalive ping every 15 seconds to prevent browser disconnect
            if time.time() - last_activity > 15.0:
                yield ": ping\n\n"
                last_activity = time.time()

        # Wait for threads to finish cleanly
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=10.0)
            except Exception:
                pass

        # Persist results to a timestamped file
        file_path = ""
        if saved_results:
            try:
                file_path = _save_scene_change_results_to_file(saved_results, seed, strategy_list)
            except Exception as e:
                logging.warning(f"Failed to save scene-change strategy results to file: {e}")

        yield f"event: done\ndata: {json.dumps({'finished': True, 'saved_to': file_path})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _resolve_dataset_path(path_str: str) -> Path:
    path = Path(path_str)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent.parent.parent.parent / path)
        candidates.append(Path(__file__).resolve().parent.parent / path)
    candidates.append(Path("/app") / path)
    candidates.append(Path("/workspace") / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


@app.post("/api/scenes/clear")
def clear_scenes_for_building(building: str | None = Query(default=None)):
    """Delete all scenes (and cascading object_observations) for a specific building/map.
    Keeps the map, rooms, navigation decisions, and robot visits intact.
    Also removes orphaned objects that no longer have any observations.
    """
    from agent.tools import _active_map_name, _resolve_map_id

    map_name = str(building or "").strip() or _active_map_name()
    if not map_name:
        raise HTTPException(status_code=400, detail="No building name provided and no active building found.")

    with get_conn() as conn:
        with conn.cursor() as cur:
            map_id = _resolve_map_id(cur, map_name)
            if map_id is None:
                raise HTTPException(status_code=404, detail=f"Building '{map_name}' not found.")

            cur.execute("DELETE FROM scenes WHERE map_id = %s", (map_id,))
            deleted_scenes = cur.rowcount

            # Remove objects that now have zero observations anywhere
            cur.execute(
                """
                DELETE FROM objects
                WHERE id NOT IN (
                    SELECT DISTINCT object_id FROM object_observations WHERE object_id IS NOT NULL
                )
                """
            )
            orphaned_deleted = cur.rowcount

        conn.commit()

    return {
        "ok": True,
        "building": map_name,
        "deleted_scenes": deleted_scenes,
        "deleted_orphaned_objects": orphaned_deleted,
        "message": f"Cleared {deleted_scenes} scenes for '{map_name}'.",
    }


@app.get("/api/map")
def get_map_snapshot():
    payload = load_map_payload_from_path(MAP_SNAPSHOT_PATH)
    if payload is None:
        if _get_suppressed_map_entry(MAP_SNAPSHOT_PATH) is not None:
            raise HTTPException(status_code=404, detail="Map snapshot was cleared by reset")
        raise HTTPException(status_code=404, detail="Map snapshot not available")
    return payload


@app.get("/api/lidar_scan")
def get_lidar_scan():
    if not os.path.exists(VELODYNE_SCAN_PATH):
        return {"available": False, "reason": "scan_not_generated"}

    with open(VELODYNE_SCAN_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload = normalize_world_payload(payload)
    payload["source"] = os.path.basename(VELODYNE_SCAN_PATH)
    return payload


def _parse_vector_text(value) -> list[float]:
    if value is None:
        return []
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(part) for part in text.split(",") if part.strip()]


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _centroid_from_rows(rows: list[tuple[str, float | None]], *, quality_weighted: bool) -> list[float]:
    vectors = []
    weights = []
    for embedding_text, quality in rows:
        vector = _parse_vector_text(embedding_text)
        if not vector:
            continue
        weight = 1.0
        if quality_weighted:
            weight = max(0.05, min(1.0, float(quality) if quality is not None else 0.5))
        vectors.append(vector)
        weights.append(weight)
    if not vectors:
        return []
    dim = len(vectors[0])
    totals = [0.0] * dim
    weight_sum = 0.0
    for vector, weight in zip(vectors, weights):
        if len(vector) != dim:
            continue
        for idx, value in enumerate(vector):
            totals[idx] += value * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return []
    return [value / weight_sum for value in totals]


def _cluster_centroid(
    cur,
    cluster_id: str,
    *,
    map_id: int | None,
    quality_weighted: bool,
    part_name: str | None = None,
) -> tuple[list[float], int]:
    map_filter = " AND oo.map_id = %s" if map_id is not None else ""
    if part_name:
        cur.execute(
            f"""
            SELECT oop.embedding::text, oop.quality_score
            FROM object_observation_parts oop
            JOIN object_observations oo ON oo.id = oop.observation_id
            WHERE oo.object_id::text = %s
              AND oop.part_name = %s
              AND oop.embedding IS NOT NULL
              {map_filter}
            """,
            tuple([cluster_id, part_name] + ([map_id] if map_id is not None else [])),
        )
    else:
        cur.execute(
            f"""
            SELECT oo.embedding::text, oo.quality_score
            FROM object_observations oo
            WHERE oo.object_id::text = %s
              AND oo.embedding IS NOT NULL
              {map_filter}
            """,
            tuple([cluster_id] + ([map_id] if map_id is not None else [])),
        )
    rows = cur.fetchall()
    return _centroid_from_rows(rows, quality_weighted=quality_weighted), len(rows)


def _part_similarities(cur, cluster_a: str, cluster_b: str, *, map_id: int | None, quality_weighted: bool) -> dict:
    results = {}
    for part_name in ("upper_body", "lower_body", "feet"):
        centroid_a, count_a = _cluster_centroid(
            cur, cluster_a, map_id=map_id, quality_weighted=quality_weighted, part_name=part_name
        )
        centroid_b, count_b = _cluster_centroid(
            cur, cluster_b, map_id=map_id, quality_weighted=quality_weighted, part_name=part_name
        )
        results[part_name] = {
            "similarity": _cosine_similarity(centroid_a, centroid_b) if centroid_a and centroid_b else None,
            "count_a": count_a,
            "count_b": count_b,
        }
    return results


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _coarse_color_name_rgb(red: int, green: int, blue: int) -> str:
    hue, sat, value = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    hue_deg = hue * 360.0
    value_255 = value * 255.0
    sat_255 = sat * 255.0
    if value_255 < 45:
        return "black"
    if value_255 > 215 and sat_255 < 35:
        return "white"
    if sat_255 < 30:
        return "gray"
    if hue_deg < 20 or hue_deg >= 340:
        return "red"
    if hue_deg < 45:
        return "orange"
    if hue_deg < 68:
        return "yellow"
    if hue_deg < 164:
        return "green"
    if hue_deg < 200:
        return "cyan"
    if hue_deg < 256:
        return "blue"
    if hue_deg < 300:
        return "purple"
    return "pink"


def _quality_from_pil(image: PILImage.Image, mask: PILImage.Image | None = None, confidence: float | None = None) -> dict:
    rgb = image.convert("RGB")
    gray = rgb.convert("L")
    width, height = rgb.size
    masked_gray = gray
    if mask is not None and mask.size == rgb.size:
        masked_gray = PILImage.composite(gray, PILImage.new("L", rgb.size, 0), mask.convert("L"))
    stat = ImageStat.Stat(masked_gray)
    mean_brightness = float(stat.mean[0]) if stat.mean else 0.0
    brightness_score = _clamp01(1.0 - abs(mean_brightness - 128.0) / 128.0)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_stat = ImageStat.Stat(edge)
    blur_score = _clamp01((float(edge_stat.var[0]) if edge_stat.var else 0.0) / 2500.0)
    size_score = _clamp01((width * height) / float(160 * 320))
    mask_coverage = 1.0
    if mask is not None and mask.size == rgb.size:
        mask_l = mask.convert("L")
        mask_coverage = _clamp01(sum(1 for pixel in mask_l.getdata() if pixel > 0) / float(width * height))
    conf = _clamp01(float(confidence) if confidence is not None else 0.65)
    quality = 0.30 * blur_score + 0.25 * brightness_score + 0.25 * size_score + 0.10 * mask_coverage + 0.10 * conf
    return {
        "quality_score": _clamp01(quality),
        "blur_score": blur_score,
        "brightness_score": brightness_score,
        "mean_brightness": mean_brightness,
        "size_score": size_score,
        "width": width,
        "height": height,
        "mask_coverage": mask_coverage,
        "detection_confidence": conf,
        "extractor": "pillow_backfill",
    }


def _dominant_colors_from_pil(image: PILImage.Image, mask: PILImage.Image | None = None, max_colors: int = 4) -> dict:
    rgb = image.convert("RGB")
    if max(rgb.size) > 160:
        rgb.thumbnail((160, 160))
        if mask is not None:
            mask = mask.convert("L").resize(rgb.size, PIL_RESAMPLE_NEAREST)
    pixels = list(rgb.getdata())
    mask_pixels = list(mask.convert("L").getdata()) if mask is not None and mask.size == rgb.size else None
    counts: dict[str, int] = {}
    total = 0
    for idx, (red, green, blue) in enumerate(pixels):
        if mask_pixels is not None and mask_pixels[idx] <= 0:
            continue
        value = max(red, green, blue)
        if value < 25 or value > 245:
            continue
        name = _coarse_color_name_rgb(red, green, blue)
        counts[name] = counts.get(name, 0) + 1
        total += 1
    if total <= 0:
        return {"colors": [], "histogram": {}, "pixel_count": 0}
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return {
        "colors": [{"name": name, "fraction": round(count / total, 4)} for name, count in ranked[:max_colors]],
        "histogram": {name: round(count / total, 4) for name, count in ranked},
        "pixel_count": total,
    }


def _extract_attributes_from_image_bytes(
    crop_bytes: bytes,
    mask_bytes: bytes | None,
    *,
    class_id: int | None,
    confidence: float | None,
    person_class_id: int = 0,
) -> tuple[dict, float | None]:
    if not crop_bytes:
        return {"enabled": False, "error": "missing_crop"}, None
    image = PILImage.open(io.BytesIO(crop_bytes)).convert("RGB")
    mask = None
    if mask_bytes:
        try:
            mask = PILImage.open(io.BytesIO(mask_bytes)).convert("L")
            if mask.size != image.size:
                mask = mask.resize(image.size, PIL_RESAMPLE_NEAREST)
        except Exception:
            mask = None
    quality = _quality_from_pil(image, mask, confidence)
    attributes = {
        "enabled": True,
        "class_id": class_id,
        "extractors": {
            "quality": True,
            "color": True,
            "person_parts": True,
            "part_embeddings": False,
            "super_resolution": False,
            "segmentation_mode": "mask",
            "source": "website_backfill",
        },
        "quality": quality,
        "quality_score": quality["quality_score"],
        "colors": _dominant_colors_from_pil(image, mask),
    }
    if class_id is not None and int(class_id) == int(person_class_id):
        width, height = image.size
        splits = {
            "upper_body": (0, int(round(height * 0.48))),
            "lower_body": (int(round(height * 0.40)), int(round(height * 0.86))),
            "feet": (int(round(height * 0.78)), height),
        }
        parts = {}
        for name, (y1, y2) in splits.items():
            y1 = max(0, min(height, y1))
            y2 = max(y1 + 1, min(height, y2))
            part_img = image.crop((0, y1, width, y2))
            part_mask = mask.crop((0, y1, width, y2)) if mask is not None else None
            part_quality = _quality_from_pil(part_img, part_mask, confidence)
            parts[name] = {
                "bbox": [0, y1, width, y2],
                "quality": part_quality,
                "quality_score": part_quality["quality_score"],
                "colors": _dominant_colors_from_pil(part_img, part_mask),
            }
        attributes["parts"] = parts
    return attributes, quality["quality_score"]


def _backfill_cluster_attributes(cur, object_id: str, *, map_id: int | None = None, limit: int = 200) -> int:
    map_filter = " AND map_id = %s" if map_id is not None else ""
    params = [object_id] + ([map_id] if map_id is not None else []) + [limit]
    cur.execute(
        f"""
        SELECT id, class_id, confidence, cropped_image, mask_image
        FROM object_observations
        WHERE object_id::text = %s
          AND (quality_score IS NULL OR attributes_json IS NULL)
          {map_filter}
        ORDER BY created_at DESC
        LIMIT %s
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    updated = 0
    for row in rows:
        observation_id, class_id, confidence, crop_value, mask_value = row
        crop_bytes = bytes(crop_value) if crop_value is not None else b""
        mask_bytes = bytes(mask_value) if mask_value is not None else None
        try:
            attributes, quality_score = _extract_attributes_from_image_bytes(
                crop_bytes,
                mask_bytes,
                class_id=class_id,
                confidence=confidence,
            )
        except Exception as exc:
            attributes, quality_score = {"enabled": False, "error": str(exc)}, None
        cur.execute(
            """
            UPDATE object_observations
            SET attributes_json = %s::jsonb,
                quality_score = %s
            WHERE id = %s
            """,
            (json.dumps(attributes), quality_score, observation_id),
        )
        updated += 1
    return updated


MARKET1501_RE = re.compile(r"^(-?\d+)_c(\d+)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COCO_CLASSES = [
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
YOLO_PERSON_CLASS_ID = 0
YOLO_OBJECT_CLASS_IDS = [idx for idx in range(len(COCO_CLASSES)) if idx != YOLO_PERSON_CLASS_ID]

# Portable COCO class ids (can be carried far from their last known position).
# Mirrors portable_object_class_ids in bordsupr/runtime config.yaml. Static
# classes (non-person, non-portable) are position-aware during reclustering.
YOLO_PORTABLE_CLASS_IDS = {
    24, 25, 26, 27, 28, 39, 40, 41, 42, 43, 44, 45,
    63, 64, 65, 66, 67, 73, 74, 75, 76, 77, 78, 79,
}
# 3D distance (m) within which two observations of a STATIC class are considered
# the same physical object during reclustering. Mirrors the runtime
# static_object_max_position_distance_m, relaxed slightly for batch clustering.
RECLUSTER_STATIC_POSITION_THRESHOLD_M = 1.5
# 3D distance (m) below which two STATIC-class clusters are MERGED even if their
# embeddings are below the similarity threshold (same physical object seen from
# divergent viewpoints). Set well below the split threshold so only near-coincident
# clusters merge.
RECLUSTER_STATIC_MERGE_DISTANCE_M = 0.5
_AUTO_THRESHOLD_FALLBACKS = {
    "dinov3": [0.50, 0.55, 0.60],
    "vit": [0.50, 0.55, 0.60],
    "osnet": [0.575, 0.60, 0.625],
    "osnet_finetuned": [0.575, 0.60, 0.625],
    "osnet_finetuned_improved": [0.643],
    "convnext": [0.50, 0.55, 0.60],
    "efficientnetv2": [0.50, 0.55, 0.60],
    "deit": [0.50, 0.55, 0.60],
    "swin": [0.50, 0.55, 0.60],
    "clip": [0.50, 0.55, 0.60],
    "sam2": [0.50, 0.55, 0.60],
}


CLUSTER_VARIANTS = {
    "whole_embedding": {
        "label": "Whole Image Embedding",
        "description": "Full detection crop DINOv3 embedding baseline",
        "threshold": 0.965,
    },
    "whole_embedding_wb_bright": {
        "label": "Whole Image + WB + Bright",
        "description": "Full detection crop embedding with white balance and brightness normalization",
        "threshold": 0.965,
    },
    "upscaled_lanczos": {
        "label": "Lanczos Upscaled",
        "description": "Lanczos-resized crop before feature extraction",
        "threshold": 0.965,
    },
    "vit_whole": {
        "label": "ViT Whole Image",
        "description": "Standard ViT-base supervised ImageNet embedding",
        "threshold": 0.600,
    },
    "vit_whole_wb_bright": {
        "label": "ViT Whole Image + WB + Bright",
        "description": "ViT-base whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "osnet_whole": {
        "label": "OSNet Whole Image",
        "description": "OSNet x1.0 MSMT17-trained whole-image CNN embedding (256x128)",
        "threshold": 0.600,
    },
    "osnet_whole_wb_bright": {
        "label": "OSNet Whole Image + WB + Bright",
        "description": "OSNet whole-image CNN embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "osnet_finetuned_whole": {
        "label": "OSNet Fine-tuned Whole Image",
        "description": "Fine-tuned OSNet x1.0 whole-image CNN embedding",
        "threshold": 0.600,
    },
    "osnet_finetuned_whole_wb_bright": {
        "label": "OSNet Fine-tuned Whole Image + WB + Bright",
        "description": "Fine-tuned OSNet whole-image CNN embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "osnet_finetuned_upscaled_esrgan": {
        "label": "OSNet Fine-tuned Real-ESRGAN Upscaled",
        "description": "Fine-tuned OSNet with Real-ESRGAN upscaling",
        "threshold": 0.600,
    },
    "osnet_finetuned_improved_whole": {
        "label": "OSNet Fine-tuned Improved Whole Image",
        "description": "Improved OSNet checkpoint trained with identity-balanced batches and pairwise-F1 selection",
        "threshold": 0.643,
    },
    "osnet_finetuned_improved_whole_wb_bright": {
        "label": "OSNet Fine-tuned Improved Whole Image + WB + Bright",
        "description": "Improved fine-tuned OSNet whole-image embedding with white balance and brightness normalization",
        "threshold": 0.643,
    },
    "convnext_whole": {
        "label": "ConvNeXt-Tiny Whole Image",
        "description": "ConvNeXt-Tiny whole-image embedding",
        "threshold": 0.600,
    },
    "convnext_whole_wb_bright": {
        "label": "ConvNeXt-Tiny Whole Image + WB + Bright",
        "description": "ConvNeXt-Tiny whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "efficientnetv2_whole": {
        "label": "EfficientNetV2 Whole Image",
        "description": "EfficientNetV2 whole-image embedding",
        "threshold": 0.600,
    },
    "efficientnetv2_whole_wb_bright": {
        "label": "EfficientNetV2 Whole Image + WB + Bright",
        "description": "EfficientNetV2 whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "deit_whole": {
        "label": "DeiT Whole Image",
        "description": "DeiT small whole-image embedding",
        "threshold": 0.600,
    },
    "deit_whole_wb_bright": {
        "label": "DeiT Whole Image + WB + Bright",
        "description": "DeiT whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "swin_whole": {
        "label": "Swin Whole Image",
        "description": "Swin Transformer whole-image embedding",
        "threshold": 0.600,
    },
    "swin_whole_wb_bright": {
        "label": "Swin Whole Image + WB + Bright",
        "description": "Swin Transformer whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "clip_whole": {
        "label": "CLIP Whole Image",
        "description": "CLIP whole-image embedding",
        "threshold": 0.600,
    },
    "clip_whole_wb_bright": {
        "label": "CLIP Whole Image + WB + Bright",
        "description": "CLIP whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
    "sam2_whole": {
        "label": "SAM2 Whole Image",
        "description": "SAM 2 vision encoder whole-image embedding",
        "threshold": 0.600,
    },
    "sam2_whole_wb_bright": {
        "label": "SAM2 Whole Image + WB + Bright",
        "description": "SAM 2 vision encoder whole-image embedding with white balance and brightness normalization",
        "threshold": 0.600,
    },
}

_CLUSTER_MODEL_FAMILY_SPECS = {
    "dinov3": {"prefix": "", "label": "", "threshold": 0.600},
    "vit": {"prefix": "vit_", "label": "ViT ", "threshold": 0.600},
    "osnet": {"prefix": "osnet_", "label": "OSNet ", "threshold": 0.600},
    "osnet_finetuned": {"prefix": "osnet_finetuned_", "label": "OSNet Fine-tuned ", "threshold": 0.600},
    "osnet_finetuned_improved": {
        "prefix": "osnet_finetuned_improved_",
        "label": "OSNet Fine-tuned Improved ",
        "threshold": 0.643,
    },
    "convnext": {"prefix": "convnext_", "label": "ConvNeXt-Tiny ", "threshold": 0.600},
    "efficientnetv2": {"prefix": "efficientnetv2_", "label": "EfficientNetV2 ", "threshold": 0.600},
    "deit": {"prefix": "deit_", "label": "DeiT ", "threshold": 0.600},
    "swin": {"prefix": "swin_", "label": "Swin ", "threshold": 0.600},
    "clip": {"prefix": "clip_", "label": "CLIP ", "threshold": 0.600},
    "sam2": {"prefix": "sam2_", "label": "SAM2 ", "threshold": 0.600},
}

_CLUSTER_STRATEGY_SPECS = {
    "upscaled_lanczos": {
        "label": "Lanczos Upscaled",
        "description": "Lanczos-resized crop before feature extraction",
        "threshold": 0.965,
    },
    "upscaled_esrgan": {
        "label": "Real-ESRGAN Upscaled",
        "description": "Real-ESRGAN upscaled crop before feature extraction",
        "threshold": 0.600,
    },
}


def _cluster_variant_id(family_id: str, base_variant: str) -> str:
    if family_id == "dinov3":
        return base_variant
    return f"{_CLUSTER_MODEL_FAMILY_SPECS[family_id]['prefix']}{base_variant}"


for _family_id, _family_spec in _CLUSTER_MODEL_FAMILY_SPECS.items():
    for _base_variant, _strategy_spec in _CLUSTER_STRATEGY_SPECS.items():
        _variant_id = _cluster_variant_id(_family_id, _base_variant)
        CLUSTER_VARIANTS.setdefault(
            _variant_id,
            {
                "label": f"{_family_spec['label']}{_strategy_spec['label']}",
                "description": _strategy_spec["description"],
                "threshold": _family_spec["threshold"],
            },
        )


def _cluster_store_dir() -> Path:
    path = CLUSTER_TESTSET_STORE_DIR
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _experiment_store_dir() -> Path:
    path = CLUSTER_TESTSET_STORE_DIR / "experiments"
    if not path.is_absolute():
        path = Path.cwd() / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def _experiment_path(exp_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(exp_id or "").strip()).strip("-")
    return _experiment_store_dir() / f"{safe}.json"


def _safe_testset_id(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value or "").strip()).strip("-")
    if not safe:
        safe = f"testset-{int(time.time())}"
    return safe[:96]


def _testset_path(testset_id: str) -> Path:
    return _cluster_store_dir() / f"{_safe_testset_id(testset_id)}.json"


def _load_testset(testset_id: str) -> dict:
    path = _testset_path(testset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Testset not found")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_testset(payload: dict) -> None:
    path = _testset_path(payload["id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _resolve_dataset_root(dataset_root: str) -> Path:
    raw = str(dataset_root or "").strip()
    translations = [
        (
            VIDEO_PUBLISHER_HOST_RUNTIME_ROOT,
            "/opt/bordsupr_runtime",
        ),
        (
            VIDEO_PUBLISHER_HOST_RUNTIME_ROOT,
            "/workspace/src",
        ),
    ]
    candidates = [raw]
    for source_prefix, dest_prefix in translations:
        if raw.startswith(source_prefix):
            candidates.append(dest_prefix + raw[len(source_prefix):])

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path.exists():
            return path

    path = Path(candidates[-1] if candidates else raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


def _dataset_root_candidates(dataset_root: str) -> list[str]:
    raw = str(dataset_root or "").strip()
    candidates = [raw]
    for source_prefix, dest_prefix in (
        (VIDEO_PUBLISHER_HOST_RUNTIME_ROOT, "/opt/bordsupr_runtime"),
        (VIDEO_PUBLISHER_HOST_RUNTIME_ROOT, "/workspace/src"),
    ):
        if raw.startswith(source_prefix):
            candidates.append(dest_prefix + raw[len(source_prefix):])
    return list(dict.fromkeys(candidates))


def _discover_dataset_images(dataset_root: Path, *, split: str | None, recursive: bool) -> list[Path]:
    roots = [dataset_root]
    if split:
        split_path = dataset_root / split
        if split_path.exists():
            roots = [split_path]
    images: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        iterator = root.rglob("*") if recursive else root.glob("*")
        for path in iterator:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path.resolve())
    return sorted(images)


def _market_label_for_path(path: str) -> dict:
    name = os.path.basename(path)
    match = MARKET1501_RE.match(name)
    if not match:
        return {"identity": None, "camera": None}
    pid = int(match.group(1))
    return {
        "identity": pid if pid > 0 else None,
        "camera": int(match.group(2)),
    }


def _testset_item_payload(path: Path, dataset_root: Path, idx: int) -> dict:
    label = _market_label_for_path(str(path))
    try:
        rel_path = str(path.relative_to(dataset_root))
    except Exception:
        rel_path = path.name
    return {
        "id": f"img-{idx:05d}",
        "path": str(path),
        "relative_path": rel_path,
        "filename": path.name,
        "label": label,
    }


def _image_bytes_for_testset_item(testset: dict, image_index: int) -> bytes:
    images = testset.get("images") or []
    if image_index < 0 or image_index >= len(images):
        raise HTTPException(status_code=404, detail="Image index not found")
    path = Path(images[image_index].get("path") or "")
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Image file not found")
    return path.read_bytes()


def _detection_by_id(testset: dict, detection_id: str) -> tuple[dict, dict]:
    for image in testset.get("images") or []:
        for detection in image.get("detections") or []:
            if str(detection.get("id")) == str(detection_id):
                return image, detection
    raise HTTPException(status_code=404, detail="Detection not found")


def _has_ground_truth_label(record: dict) -> bool:
    identity = (record.get("label") or {}).get("identity")
    return identity is not None and str(identity).strip() != ""


def _sample_records_for_testset(testset: dict, *, labeled_only: bool = False) -> list[dict]:
    records = []
    has_detections = any((image.get("detections") or []) for image in testset.get("images") or [])
    if has_detections:
        for image_idx, image in enumerate(testset.get("images") or []):
            for detection in image.get("detections") or []:
                records.append(
                    {
                        "kind": "detection",
                        "image_index": image_idx,
                        "detection_id": detection.get("id"),
                        "path": image.get("path"),
                        "bbox": detection.get("bbox"),
                        "filename": f"{image.get('filename')}:{detection.get('class_name')}",
                        "label": detection.get("label") or {},
                        "class_id": detection.get("class_id"),
                        "class_name": detection.get("class_name"),
                    }
                )
        return [record for record in records if _has_ground_truth_label(record)] if labeled_only else records
    for image_idx, image in enumerate(testset.get("images") or []):
        records.append(
            {
                "kind": "image",
                "image_index": image_idx,
                "path": image.get("path"),
                "bbox": None,
                "filename": image.get("filename"),
                "label": image.get("label") or {},
            }
        )
    return [record for record in records if _has_ground_truth_label(record)] if labeled_only else records


_DINOV3_MODEL = None
_DINOV3_PROCESSOR = None
_DINOV3_MODEL_ERROR: Exception | None = None
_DINOV3_MODEL_LOCK = threading.Lock()

_VIT_MODEL = None
_VIT_PROCESSOR = None
_VIT_MODEL_ERROR: Exception | None = None
_VIT_MODEL_LOCK = threading.Lock()

_OSNET_MODEL = None
_OSNET_MODEL_ERROR: Exception | None = None
_OSNET_MODEL_LOCK = threading.Lock()

_OSNET_FINETUNED_MODEL = None
_OSNET_FINETUNED_MODEL_ERROR: Exception | None = None
_OSNET_FINETUNED_MODEL_LOCK = threading.Lock()

_OSNET_FINETUNED_IMPROVED_MODEL = None
_OSNET_FINETUNED_IMPROVED_MODEL_ERROR: Exception | None = None
_OSNET_FINETUNED_IMPROVED_MODEL_LOCK = threading.Lock()

_CONVNEXT_MODEL = None
_CONVNEXT_MODEL_ERROR: Exception | None = None
_CONVNEXT_MODEL_LOCK = threading.Lock()

_EFFICIENTNETV2_MODEL = None
_EFFICIENTNETV2_MODEL_ERROR: Exception | None = None
_EFFICIENTNETV2_MODEL_LOCK = threading.Lock()

_DEIT_MODEL = None
_DEIT_MODEL_ERROR: Exception | None = None
_DEIT_MODEL_LOCK = threading.Lock()

_FASTREID_MODEL = None
_FASTREID_MODEL_ERROR: Exception | None = None
_FASTREID_MODEL_LOCK = threading.Lock()

_SWIN_MODEL = None
_SWIN_PROCESSOR = None
_SWIN_MODEL_ERROR: Exception | None = None
_SWIN_MODEL_LOCK = threading.Lock()

_HRNET_MODEL = None
_HRNET_MODEL_ERROR: Exception | None = None
_HRNET_MODEL_LOCK = threading.Lock()

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_MODEL_ERROR: Exception | None = None
_CLIP_MODEL_LOCK = threading.Lock()

_SAM2_MODEL = None
_SAM2_PROCESSOR = None
_SAM2_MODEL_ERROR: Exception | None = None
_SAM2_MODEL_LOCK = threading.Lock()

_CLUSTER_EXPERIMENT_JOBS: dict[str, dict] = {}
_CLUSTER_EXPERIMENT_JOBS_LOCK = threading.Lock()

_RECLUSTER_JOBS: dict[str, dict] = {}
_RECLUSTER_JOBS_LOCK = threading.Lock()

_FACE_RECLUSTER_JOBS: dict[str, dict] = {}
_FACE_RECLUSTER_JOBS_LOCK = threading.Lock()

_RECOMPUTE_JOBS: dict[str, dict] = {}
_RECOMPUTE_JOBS_LOCK = threading.Lock()

_INTERACTION_RELOAD_JOBS: dict[str, dict] = {}
_INTERACTION_RELOAD_JOBS_LOCK = threading.Lock()
_INTERACTION_YOLO = None
_INTERACTION_YOLO_PATH = "/shared/yolo11m-seg.pt"


def _interaction_yolo_model():
    """Lazy-load & cache the ultralytics YOLO model for offline interaction detection."""
    global _INTERACTION_YOLO
    if _INTERACTION_YOLO is None:
        import os
        os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
        from ultralytics import YOLO
        if not Path(_INTERACTION_YOLO_PATH).exists():
            raise RuntimeError(f"YOLO model not found: {_INTERACTION_YOLO_PATH}")
        _INTERACTION_YOLO = YOLO(_INTERACTION_YOLO_PATH)
    return _INTERACTION_YOLO


def _interaction_detect(image_bgr, conf: float = 0.25) -> list[dict]:
    """Run ultralytics YOLO on a BGR image; return detections with per-frame label ids."""
    model = _interaction_yolo_model()
    result = model.predict(image_bgr, conf=conf, verbose=False)[0]
    names = result.names or {}
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections
    for idx, box in enumerate(boxes, start=1):
        try:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
        except Exception:
            continue
        detections.append({
            "label_id": str(idx),
            "class_id": cls_id,
            "class_name": str(names.get(cls_id, cls_id)),
            "x_min": x1, "y_min": y1, "x_max": x2, "y_max": y2,
        })
    return detections


def _interaction_annotate(image_bgr, detections: list[dict]):
    """Draw ID boxes + labels (port of interaction_description_node._annotate_frame)."""
    import cv2
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]
    sorted_dets = sorted(detections, key=lambda d: (d["x_max"] - d["x_min"]) * (d["y_max"] - d["y_min"]))
    for idx, det in enumerate(sorted_dets):
        x1, y1, x2, y2 = int(det["x_min"]), int(det["y_min"]), int(det["x_max"]), int(det["y_max"])
        label = f"ID {det['label_id']} | {det['class_name']}"
        hue = int((idx * 35) % 180)
        color_bgr = tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 210, 255]]]), cv2.COLOR_HSV2BGR)[0][0])
        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color_bgr, 2)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_x = max(0, min(x1, w - text_w - 12))
        label_y = max(text_h + 8, y1 - 6)
        cv2.rectangle(annotated, (label_x, label_y - text_h - 6), (label_x + text_w + 10, label_y + 2), color_bgr, -1)
        brightness = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
        text_color = (0, 0, 0) if brightness > 135 else (255, 255, 255)
        cv2.putText(annotated, label, (label_x + 5, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA)
    return annotated


def _interaction_build_prompt(detections: list[dict]) -> str:
    lines = [
        f"ID {d['label_id']}: {d['class_name']} at "
        f"bbox [{int(d['x_min'])}, {int(d['y_min'])}, {int(d['x_max'])}, {int(d['y_max'])}]"
        for d in detections
    ]
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


def _interaction_parse_response(response_text: str) -> list[dict]:
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
            return []
        try:
            payload = json.loads(brace_match.group(1))
        except json.JSONDecodeError:
            return []
    interactions = payload.get("interactions", [])
    if not isinstance(interactions, list):
        return []
    return [item for item in interactions if isinstance(item, dict)]


def _interaction_parse_confidence(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if value != value:
        return 0.0
    return max(0.0, min(1.0, value))


def _interaction_query_vlm(prompt: str, bgr_image) -> str:
    import cv2
    from agent.vlm_client import get_vlm_client, get_vlm_model as _vc_model
    ok, buffer = cv2.imencode(".jpg", bgr_image)
    if not ok:
        raise RuntimeError("Failed to encode annotated frame to JPEG")
    image_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    client = get_vlm_client()
    # Interaction JSON extraction must NOT use the reasoning block: it is slow (long
    # internal trace before the answer) and can break strict JSON parsing. The live
    # pipeline (interaction_description_node) sets vlm_disable_thinking=true; mirror that
    # here by forcing enable_thinking=False regardless of the active option's default
    # (the default option leaves thinking=None -> server default = reasoning ON for 9B,
    # which is why reload was much slower than the live pipeline).
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    completion = client.chat.completions.create(
        model=_vc_model(),
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        }],
        extra_body=extra_body,
    )
    return str(completion.choices[0].message.content or "").strip()


def _interaction_link_box_to_observation(cur, scene_id: int, class_id: int, bbox: dict, min_iou: float = 0.3):
    """Return the object_observations.id in this scene (same class) with max IoU to bbox."""
    cur.execute(
        """
        SELECT id, bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max
        FROM object_observations
        WHERE scene_id = %s AND class_id = %s
          AND bbox_x_min IS NOT NULL AND bbox_x_max IS NOT NULL
        """,
        (scene_id, class_id),
    )
    best_id = None
    best_iou = 0.0
    det_box = [bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]]
    for row in cur.fetchall():
        obs_box = [float(row[1]), float(row[2]), float(row[3]), float(row[4])]
        iou = _bbox_iou(det_box, obs_box)
        if iou > best_iou:
            best_iou = iou
            best_id = int(row[0])
    if best_id is not None and best_iou >= min_iou:
        return best_id
    return None


def _run_interaction_reload_job(job_id: str, map_id: int, min_confidence: float) -> None:
    import time as _time
    import cv2
    with _INTERACTION_RELOAD_JOBS_LOCK:
        _INTERACTION_RELOAD_JOBS[job_id]["status"] = "running"
        _INTERACTION_RELOAD_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    def _update(**fields):
        with _INTERACTION_RELOAD_JOBS_LOCK:
            job = _INTERACTION_RELOAD_JOBS.get(job_id)
            if job is not None:
                job.update(fields)

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, original_scene_image FROM scenes "
                    "WHERE map_id = %s AND original_scene_image IS NOT NULL ORDER BY id",
                    (map_id,),
                )
                scenes = cur.fetchall()
        _update(total_steps=len(scenes), phase="deleting old interactions")

        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM interactions WHERE map_id = %s", (map_id,))
            conn.commit()

        inserted = 0
        skipped_scenes = 0
        processed = 0
        _update(phase="regenerating")
        for scene_id, image_bytes in scenes:
            processed += 1
            _update(steps_done=processed)
            try:
                np_arr = np.frombuffer(bytes(image_bytes), dtype=np.uint8)
                bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if bgr is None:
                    skipped_scenes += 1
                    continue
                detections = _interaction_detect(bgr, conf=0.25)
                persons = [d for d in detections if d["class_id"] == 0]
                if not persons:
                    skipped_scenes += 1
                    continue
                det_by_id = {d["label_id"]: d for d in detections}
                annotated = _interaction_annotate(bgr, detections)
                prompt = _interaction_build_prompt(detections)
                response_text = _interaction_query_vlm(prompt, annotated)
                parsed = _interaction_parse_response(response_text)

                with get_conn() as conn:
                    with conn.cursor() as cur:
                        for interaction in parsed:
                            subject_id = re.sub(r"\D", "", str(interaction.get("subject_id", "") or ""))
                            target_id = re.sub(r"\D", "", str(interaction.get("target_id", "") or ""))
                            action = str(interaction.get("action", "") or "").strip()
                            caption = str(interaction.get("caption", "") or "").strip()
                            confidence = _interaction_parse_confidence(interaction.get("confidence"))
                            if not subject_id or not action:
                                continue
                            subject_det = det_by_id.get(subject_id)
                            if subject_det is None or subject_det["class_id"] != 0:
                                continue
                            if confidence < min_confidence:
                                continue
                            target_det = det_by_id.get(target_id) if target_id else None
                            if not caption:
                                if target_det is not None:
                                    caption = f"person {subject_id} is {action} {target_det['class_name']} {target_id}."
                                else:
                                    caption = f"person {subject_id} is {action}."
                            subject_obs = _interaction_link_box_to_observation(cur, scene_id, 0, subject_det)
                            object_obs = (
                                _interaction_link_box_to_observation(cur, scene_id, target_det["class_id"], target_det)
                                if target_det is not None else None
                            )
                            subject_bbox = {"x_min": int(subject_det["x_min"]), "y_min": int(subject_det["y_min"]),
                                            "x_max": int(subject_det["x_max"]), "y_max": int(subject_det["y_max"])}
                            object_bbox = ({"x_min": int(target_det["x_min"]), "y_min": int(target_det["y_min"]),
                                            "x_max": int(target_det["x_max"]), "y_max": int(target_det["y_max"])}
                                           if target_det is not None else None)
                            cur.execute(
                                """
                                INSERT INTO interactions (
                                    action, caption, model_source, subject_bbox, object_bbox,
                                    subject_id, object_id, scene_id, map_id, confidence
                                ) VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
                                """,
                                (
                                    action, caption, "interaction_reload",
                                    json.dumps(subject_bbox),
                                    json.dumps(object_bbox) if object_bbox is not None else None,
                                    subject_obs, object_obs, scene_id, map_id, confidence,
                                ),
                            )
                            inserted += 1
                    conn.commit()
            except Exception as scene_exc:
                skipped_scenes += 1
                _update(last_error=f"scene {scene_id}: {scene_exc}")
                continue

        _update(
            status="done",
            finished_at=_time.strftime("%Y-%m-%d %H:%M:%S"),
            phase="done",
            result={
                "scenes_total": len(scenes),
                "scenes_processed": processed,
                "scenes_skipped": skipped_scenes,
                "interactions_inserted": inserted,
            },
        )
    except Exception as exc:
        _update(status="error", finished_at=_time.strftime("%Y-%m-%d %H:%M:%S"), error=str(exc))


@app.post("/api/interactions/reload")
def reload_interactions(
    building: str | None = Query(default=None),
    min_confidence: float = Query(default=0.4, ge=0.0, le=1.0),
):
    """Regenerate all interactions for a map from stored scene images (background job).

    Deletes the map's existing interactions, then for each scene with a stored
    original_scene_image runs YOLO + VLM and re-inserts interactions, re-linking
    subject/object to the scene's existing object_observations by IoU. Poll
    GET /api/interactions/reload/jobs/{job_id} for progress."""
    map_id = _resolve_map_id(building)
    if map_id is None:
        raise HTTPException(status_code=400, detail="A valid building/map is required to reload interactions.")
    job_id = f"ireload-{uuid.uuid4().hex[:8]}"
    with _INTERACTION_RELOAD_JOBS_LOCK:
        _INTERACTION_RELOAD_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "map_id": map_id,
            "building": building,
            "min_confidence": min_confidence,
            "steps_done": 0,
            "total_steps": 0,
            "phase": "queued",
        }
    threading.Thread(target=_run_interaction_reload_job, args=(job_id, map_id, min_confidence), daemon=True).start()
    return {"job_id": job_id, "map_id": map_id}


@app.get("/api/interactions/reload/jobs/{job_id}")
def get_interaction_reload_job(job_id: str):
    with _INTERACTION_RELOAD_JOBS_LOCK:
        job = _INTERACTION_RELOAD_JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job_id")
        return dict(job)


_REALESRGAN_MODEL = None
_REALESRGAN_ERROR: Exception | None = None
_REALESRGAN_LOCK = threading.Lock()


def _load_dinov3_model():
    global _DINOV3_MODEL, _DINOV3_PROCESSOR, _DINOV3_MODEL_ERROR
    if _DINOV3_MODEL is not None:
        return _DINOV3_MODEL, _DINOV3_PROCESSOR
    if _DINOV3_MODEL_ERROR is not None:
        return None, None
    with _DINOV3_MODEL_LOCK:
        if _DINOV3_MODEL is not None:
            return _DINOV3_MODEL, _DINOV3_PROCESSOR
        if _DINOV3_MODEL_ERROR is not None:
            return None, None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _DINOV3_MODEL = model
            _DINOV3_PROCESSOR = processor
            return model, processor
        except Exception as exc:
            _DINOV3_MODEL_ERROR = exc
            return None, None


def _load_vit_model():
    global _VIT_MODEL, _VIT_PROCESSOR, _VIT_MODEL_ERROR
    if _VIT_MODEL is not None:
        return _VIT_MODEL, _VIT_PROCESSOR
    if _VIT_MODEL_ERROR is not None:
        return None, None
    with _VIT_MODEL_LOCK:
        if _VIT_MODEL is not None:
            return _VIT_MODEL, _VIT_PROCESSOR
        if _VIT_MODEL_ERROR is not None:
            return None, None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_name = "google/vit-base-patch16-224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _VIT_MODEL = model
            _VIT_PROCESSOR = processor
            return model, processor
        except Exception as exc:
            _VIT_MODEL_ERROR = exc
            return None, None


def _load_osnet_model():
    global _OSNET_MODEL, _OSNET_MODEL_ERROR
    if _OSNET_MODEL is not None:
        return _OSNET_MODEL
    if _OSNET_MODEL_ERROR is not None:
        return None
    with _OSNET_MODEL_LOCK:
        if _OSNET_MODEL is not None:
            return _OSNET_MODEL
        if _OSNET_MODEL_ERROR is not None:
            return None
        try:
            import torch
            from osnet_model import osnet_x1_0

            model = osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
            # Download MSMT17-trained weights from HuggingFace
            weights_url = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"
            cache_dir = Path.home() / ".cache" / "torch" / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / "osnet_x1_0_msmt17.pth"
            if not cached_file.exists():
                urllib.request.urlretrieve(weights_url, str(cached_file))
            state_dict = torch.load(str(cached_file), map_location="cpu")
            # Handle DataParallel prefix and classifier mismatch
            model_state = model.state_dict()
            new_state = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                if key in model_state and model_state[key].shape == v.shape:
                    new_state[key] = v
            model.load_state_dict(new_state, strict=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _OSNET_MODEL = model
            return model
        except Exception as exc:
            _OSNET_MODEL_ERROR = exc
            return None


def _load_osnet_finetuned_model():
    global _OSNET_FINETUNED_MODEL, _OSNET_FINETUNED_MODEL_ERROR
    if _OSNET_FINETUNED_MODEL is not None:
        return _OSNET_FINETUNED_MODEL
    if _OSNET_FINETUNED_MODEL_ERROR is not None:
        return None
    with _OSNET_FINETUNED_MODEL_LOCK:
        if _OSNET_FINETUNED_MODEL is not None:
            return _OSNET_FINETUNED_MODEL
        if _OSNET_FINETUNED_MODEL_ERROR is not None:
            return None
        try:
            import torch
            from osnet_model import osnet_x1_0

            model = osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
            finetuned_path = Path("/shared/cluster_testsets/osnet_finetuned_persons_big_2.pth")
            if not finetuned_path.exists():
                raise FileNotFoundError(f"Fine-tuned checkpoint not found: {finetuned_path}")
            state_dict = torch.load(str(finetuned_path), map_location="cpu")
            model_state = model.state_dict()
            new_state = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                if key in model_state and model_state[key].shape == v.shape:
                    new_state[key] = v
            model.load_state_dict(new_state, strict=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _OSNET_FINETUNED_MODEL = model
            return model
        except Exception as exc:
            _OSNET_FINETUNED_MODEL_ERROR = exc
            return None


def _load_osnet_finetuned_improved_model():
    global _OSNET_FINETUNED_IMPROVED_MODEL, _OSNET_FINETUNED_IMPROVED_MODEL_ERROR
    if _OSNET_FINETUNED_IMPROVED_MODEL is not None:
        return _OSNET_FINETUNED_IMPROVED_MODEL
    if _OSNET_FINETUNED_IMPROVED_MODEL_ERROR is not None:
        return None
    with _OSNET_FINETUNED_IMPROVED_MODEL_LOCK:
        if _OSNET_FINETUNED_IMPROVED_MODEL is not None:
            return _OSNET_FINETUNED_IMPROVED_MODEL
        if _OSNET_FINETUNED_IMPROVED_MODEL_ERROR is not None:
            return None
        try:
            import torch
            from osnet_model import osnet_x1_0

            model = osnet_x1_0(num_classes=1, pretrained=False, loss="softmax")
            finetuned_path = Path("/shared/cluster_testsets/osnet_finetuned_persons_big_2_improved.pth")
            if not finetuned_path.exists():
                raise FileNotFoundError(f"Fine-tuned checkpoint not found: {finetuned_path}")
            state_dict = torch.load(str(finetuned_path), map_location="cpu")
            model_state = model.state_dict()
            new_state = {}
            for k, v in state_dict.items():
                key = k[7:] if k.startswith("module.") else k
                if key in model_state and model_state[key].shape == v.shape:
                    new_state[key] = v
            model.load_state_dict(new_state, strict=False)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _OSNET_FINETUNED_IMPROVED_MODEL = model
            return model
        except Exception as exc:
            _OSNET_FINETUNED_IMPROVED_MODEL_ERROR = exc
            return None


def _load_timm_model(model_name: str, global_var_name: str, error_var_name: str, lock: threading.Lock):
    """Generic timm model loader using closure over globals."""
    model = globals()[global_var_name]
    error = globals()[error_var_name]
    if model is not None:
        return model
    if error is not None:
        return None
    with lock:
        model = globals()[global_var_name]
        error = globals()[error_var_name]
        if model is not None:
            return model
        if error is not None:
            return None
        try:
            import torch
            import timm

            m = timm.create_model(model_name, pretrained=True, num_classes=0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            m.eval()
            try:
                m.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                m.to(device)
            globals()[global_var_name] = m
            return m
        except Exception as exc:
            globals()[error_var_name] = exc
            return None


def _load_convnext_model():
    return _load_timm_model("convnext_tiny.fb_in1k", "_CONVNEXT_MODEL", "_CONVNEXT_MODEL_ERROR", _CONVNEXT_MODEL_LOCK)


def _load_efficientnetv2_model():
    return _load_timm_model("efficientnetv2_rw_s.ra2_in1k", "_EFFICIENTNETV2_MODEL", "_EFFICIENTNETV2_MODEL_ERROR", _EFFICIENTNETV2_MODEL_LOCK)


def _load_deit_model():
    return _load_timm_model("deit_small_patch16_224.fb_in1k", "_DEIT_MODEL", "_DEIT_MODEL_ERROR", _DEIT_MODEL_LOCK)


def _load_fastreid_model():
    global _FASTREID_MODEL, _FASTREID_MODEL_ERROR
    if _FASTREID_MODEL is not None:
        return _FASTREID_MODEL
    if _FASTREID_MODEL_ERROR is not None:
        try:
            import fastreid
        except Exception:
            return None
        _FASTREID_MODEL_ERROR = None
    with _FASTREID_MODEL_LOCK:
        if _FASTREID_MODEL is not None:
            return _FASTREID_MODEL
        if _FASTREID_MODEL_ERROR is not None:
            try:
                import fastreid
            except Exception:
                return None
            _FASTREID_MODEL_ERROR = None
        try:
            import torch
            import fastreid
            from fastreid.config import get_cfg
            from fastreid.modeling.meta_arch import build_model

            cfg = get_cfg()
            cfg.MODEL.BACKBONE.NAME = "build_resnet_backbone"
            cfg.MODEL.BACKBONE.DEPTH = "50x"
            cfg.MODEL.BACKBONE.WITH_IBN = False
            cfg.MODEL.HEADS.NUM_CLASSES = 1
            cfg.MODEL.HEADS.POOL_LAYER = "GeneralizedMeanPooling"
            cfg.MODEL.HEADS.CLS_LAYER = "Linear"
            cfg.MODEL.HEADS.SCALE = 1
            cfg.MODEL.HEADS.MARGIN = 0.0
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cfg.MODEL.DEVICE = device
            try:
                model = build_model(cfg)
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError):
                device = "cpu"
                cfg.MODEL.DEVICE = device
                model = build_model(cfg)
            model.eval()
            _FASTREID_MODEL = model
            return model
        except Exception as exc:
            _FASTREID_MODEL_ERROR = exc
            return None


def _load_swin_model():
    global _SWIN_MODEL, _SWIN_PROCESSOR, _SWIN_MODEL_ERROR
    if _SWIN_MODEL is not None:
        return _SWIN_MODEL, _SWIN_PROCESSOR
    if _SWIN_MODEL_ERROR is not None:
        return None, None
    with _SWIN_MODEL_LOCK:
        if _SWIN_MODEL is not None:
            return _SWIN_MODEL, _SWIN_PROCESSOR
        if _SWIN_MODEL_ERROR is not None:
            return None, None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_name = "microsoft/swin-tiny-patch4-window7-224"
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _SWIN_MODEL = model
            _SWIN_PROCESSOR = processor
            return model, processor
        except Exception as exc:
            _SWIN_MODEL_ERROR = exc
            return None, None


def _load_hrnet_model():
    return _load_timm_model("hrnet_w18_small_model_v2.ms_in1k", "_HRNET_MODEL", "_HRNET_MODEL_ERROR", _HRNET_MODEL_LOCK)


def _load_clip_model():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _CLIP_MODEL_ERROR
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR
    if _CLIP_MODEL_ERROR is not None:
        return None, None
    with _CLIP_MODEL_LOCK:
        if _CLIP_MODEL is not None:
            return _CLIP_MODEL, _CLIP_PROCESSOR
        if _CLIP_MODEL_ERROR is not None:
            return None, None
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel

            model_name = "openai/clip-vit-base-patch16"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _CLIP_MODEL = model
            _CLIP_PROCESSOR = processor
            return model, processor
        except Exception as exc:
            _CLIP_MODEL_ERROR = exc
            return None, None


def _load_sam2_model():
    global _SAM2_MODEL, _SAM2_PROCESSOR, _SAM2_MODEL_ERROR
    if _SAM2_MODEL is not None:
        return _SAM2_MODEL, _SAM2_PROCESSOR
    if _SAM2_MODEL_ERROR is not None:
        return None, None
    with _SAM2_MODEL_LOCK:
        if _SAM2_MODEL is not None:
            return _SAM2_MODEL, _SAM2_PROCESSOR
        if _SAM2_MODEL_ERROR is not None:
            return None, None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            model_name = "facebook/sam2-hiera-tiny"
            processor = AutoImageProcessor.from_pretrained(model_name)
            from transformers import Sam2Model
            model = Sam2Model.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.eval()
            try:
                model.to(device)
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                device = torch.device("cpu")
                model.to(device)
            _SAM2_MODEL = model
            _SAM2_PROCESSOR = processor
            return model, processor
        except Exception as exc:
            _SAM2_MODEL_ERROR = exc
            return None, None


def _load_realesrgan_model():
    global _REALESRGAN_MODEL, _REALESRGAN_ERROR
    if _REALESRGAN_MODEL is not None:
        return _REALESRGAN_MODEL
    if _REALESRGAN_ERROR is not None:
        return None
    with _REALESRGAN_LOCK:
        if _REALESRGAN_MODEL is not None:
            return _REALESRGAN_MODEL
        if _REALESRGAN_ERROR is not None:
            return None
        try:
            import torch
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )
            netscale = 4
            model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            cache_dir = Path.home() / ".cache" / "torch" / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached_file = cache_dir / "RealESRGAN_x4plus.pth"
            if not cached_file.exists():
                urllib.request.urlretrieve(model_url, str(cached_file))
            upsampler = RealESRGANer(
                scale=netscale,
                model_path=str(cached_file),
                model=model,
                tile=0,
                pre_pad=0,
                half=False,
            )
            _REALESRGAN_MODEL = upsampler
            return upsampler
        except Exception as exc:
            _REALESRGAN_ERROR = exc
            return None


def _maybe_upscale_realesrgan(image: PILImage.Image) -> PILImage.Image:
    upsampler = _load_realesrgan_model()
    if upsampler is None:
        return _maybe_upscale_lanczos(image)
    try:
        import numpy as np

        np_img = np.array(image)
        output, _ = upsampler.enhance(np_img, outscale=4)
        return PILImage.fromarray(output)
    except Exception:
        return _maybe_upscale_lanczos(image)


def _maybe_upscale_lanczos(image: PILImage.Image) -> PILImage.Image:
    w, h = image.size
    if w < 224 or h < 224:
        scale = max(224 / w, 224 / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        return image.resize((new_w, new_h), PIL_RESAMPLE_LANCZOS)
    return image


def _fallback_image_vector(image: PILImage.Image) -> list[float]:
    import numpy as np

    rgb = image.convert("RGB").resize((16, 16), PIL_RESAMPLE_BILINEAR)
    arr = np.array(rgb).astype(np.float32) / 255.0
    features = np.concatenate(
        [
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            np.histogram(arr[:, :, 0], bins=8, range=(0.0, 1.0))[0].astype(np.float32),
            np.histogram(arr[:, :, 1], bins=8, range=(0.0, 1.0))[0].astype(np.float32),
            np.histogram(arr[:, :, 2], bins=8, range=(0.0, 1.0))[0].astype(np.float32),
        ]
    )
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return [float(v) for v in features.tolist()]


def _run_inference_with_cpu_fallback(model, inputs, inference_fn):
    """Run inference_fn(model, inputs), falling back to CPU on CUDA errors."""
    try:
        return inference_fn(model, inputs)
    except RuntimeError as exc:
        err = str(exc).lower()
        if "cuda" in err or "cublas" in err:
            import torch
            device = torch.device("cpu")
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            return inference_fn(model, inputs)
        raise


def _image_to_vector(image: PILImage.Image, model_type: str = "dinov3") -> list[float]:
    try:
        import torch
        import numpy as np
    except ImportError:
        if model_type in ("dinov3", "vit", "swin", "clip", "sam2"):
            return _fallback_image_vector(image)
        raise

    rgb = image.convert("RGB")

    def _preprocess_imagenet_224(img):
        img = img.resize((224, 224), PIL_RESAMPLE_BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr).unsqueeze(0)

    def _extract_timm_features(model, tensor):
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.inference_mode():
            features = model.forward_features(tensor)
        if features.dim() == 4:
            features = features.mean(dim=[2, 3])
        elif features.dim() == 3:
            features = features[:, 0]
        emb = features[0].detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type == "osnet":
        model = _load_osnet_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"OSNet model not available: {_OSNET_MODEL_ERROR}")
        rgb = rgb.resize((128, 256), PIL_RESAMPLE_BILINEAR)
        arr = np.array(rgb).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.inference_mode():
            emb = model(tensor)[0]
        emb = emb.detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type in ("osnet_finetuned", "osnet_finetuned_improved"):
        if model_type == "osnet_finetuned_improved":
            model = _load_osnet_finetuned_improved_model()
            model_error = _OSNET_FINETUNED_IMPROVED_MODEL_ERROR
        else:
            model = _load_osnet_finetuned_model()
            model_error = _OSNET_FINETUNED_MODEL_ERROR
        if model is None:
            raise HTTPException(status_code=503, detail=f"OSNet fine-tuned model not available: {model_error}")
        rgb = rgb.resize((128, 256), PIL_RESAMPLE_BILINEAR)
        arr = np.array(rgb).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr).unsqueeze(0)
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.inference_mode():
            emb = model(tensor)[0]
        emb = emb.detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type in ("convnext", "efficientnetv2", "deit"):
        loader_map = {
            "convnext": (_load_convnext_model, "_CONVNEXT_MODEL_ERROR", "ConvNeXt-Tiny"),
            "efficientnetv2": (_load_efficientnetv2_model, "_EFFICIENTNETV2_MODEL_ERROR", "EfficientNetV2"),
            "deit": (_load_deit_model, "_DEIT_MODEL_ERROR", "DeiT"),
        }
        loader, error_var, label = loader_map[model_type]
        model = loader()
        if model is None:
            error = globals()[error_var]
            raise HTTPException(status_code=503, detail=f"{label} model not available: {error}")
        tensor = _preprocess_imagenet_224(rgb)
        return _extract_timm_features(model, tensor)

    if model_type == "fastreid":
        model = _load_fastreid_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"FastReID model not available: {_FASTREID_MODEL_ERROR}")
        tensor = _preprocess_imagenet_224(rgb)
        device = next(model.parameters()).device
        tensor = tensor.to(device)
        with torch.inference_mode():
            emb = model(tensor)[0]
        emb = emb.detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type == "hrnet":
        model = _load_hrnet_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"HRNet model not available: {_HRNET_MODEL_ERROR}")
        tensor = _preprocess_imagenet_224(rgb)
        return _extract_timm_features(model, tensor)

    if model_type == "swin":
        model, processor = _load_swin_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"Swin Transformer model not available: {_SWIN_MODEL_ERROR}")
        try:
            inputs = processor(images=np.array(rgb), return_tensors="pt")
        except Exception:
            inputs = _preprocess_imagenet_224(rgb)
            inputs = {"pixel_values": inputs}
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        def _infer_swin(m, inp):
            with torch.inference_mode():
                return m(**inp)
        outputs = _run_inference_with_cpu_fallback(model, inputs, _infer_swin)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        else:
            emb = outputs.last_hidden_state[:, 0, :][0]
        emb = emb.detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type == "clip":
        model, processor = _load_clip_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"CLIP model not available: {_CLIP_MODEL_ERROR}")
        try:
            inputs = processor(images=np.array(rgb), return_tensors="pt")
        except Exception:
            inputs = _preprocess_imagenet_224(rgb)
            inputs = {"pixel_values": inputs}
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        def _infer_clip(m, inp):
            with torch.inference_mode():
                out = m.get_image_features(**inp)
                if hasattr(out, "pooler_output"):
                    return out.pooler_output
                return out
        image_features = _run_inference_with_cpu_fallback(model, inputs, _infer_clip)
        emb = image_features[0].detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type == "sam2":
        model, processor = _load_sam2_model()
        if model is None:
            raise HTTPException(status_code=503, detail=f"SAM 2 model not available: {_SAM2_MODEL_ERROR}")
        try:
            inputs = processor(images=np.array(rgb), return_tensors="pt")
        except Exception:
            inputs = _preprocess_imagenet_224(rgb)
            inputs = {"pixel_values": inputs}
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        def _infer_sam2(m, inp):
            with torch.inference_mode():
                # Use vision_encoder directly for clean feature extraction
                pix = inp.get("pixel_values")
                if pix is not None:
                    return m.vision_encoder(pix)
                return m.vision_encoder(**inp)
        outputs = _run_inference_with_cpu_fallback(model, inputs, _infer_sam2)
        if hasattr(outputs, "last_hidden_state"):
            # SAM2 vision encoder outputs spatial features; pool to 1D
            hidden = outputs.last_hidden_state[0]  # shape: (H, W, C) or (C, H, W)
            if hidden.dim() == 3:
                if hidden.shape[-1] > hidden.shape[0]:
                    # (H, W, C) channels-last format
                    emb = hidden.mean(dim=[0, 1])
                else:
                    # (C, H, W) channels-first format
                    emb = hidden.mean(dim=[1, 2])
            elif hidden.dim() == 2:
                emb = hidden[0]
            else:
                emb = hidden
        else:
            # Fallback: use first output tensor directly
            first_out = outputs[0][0] if hasattr(outputs, "__getitem__") else outputs[0]
            if first_out.dim() == 3:
                if first_out.shape[-1] > first_out.shape[0]:
                    emb = first_out.mean(dim=[0, 1])
                else:
                    emb = first_out.mean(dim=[1, 2])
            elif first_out.dim() == 2:
                emb = first_out[0]
            else:
                emb = first_out
        emb = emb.detach().float().cpu().numpy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return [float(v) for v in emb.tolist()]

    if model_type == "vit":
        model, processor = _load_vit_model()
        error = _VIT_MODEL_ERROR
        label = "ViT"
    else:
        model, processor = _load_dinov3_model()
        error = _DINOV3_MODEL_ERROR
        label = "DINOv3"
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"{label} model not available: {error}",
        )
    try:
        inputs = processor(images=np.array(rgb), return_tensors="pt")
    except Exception:
        size = 224
        w, h = rgb.size
        scale = size / min(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        rgb = rgb.resize((new_w, new_h), PIL_RESAMPLE_BILINEAR)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        rgb = rgb.crop((left, top, left + size, top + size))
        arr = np.array(rgb).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = arr.transpose(2, 0, 1)
        inputs = {"pixel_values": torch.from_numpy(arr).unsqueeze(0)}
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    def _infer_dinov3(m, inp):
        with torch.inference_mode():
            return m(**inp)
    outputs = _run_inference_with_cpu_fallback(model, inputs, _infer_dinov3)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        emb = outputs.pooler_output[0]
    else:
        emb = outputs.last_hidden_state[:, 0, :][0]
    emb = emb.detach().float().cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return [float(v) for v in emb.tolist()]


def _crop_ratio(image: PILImage.Image, box: tuple[float, float, float, float]) -> PILImage.Image:
    width, height = image.size
    left = int(round(width * box[0]))
    top = int(round(height * box[1]))
    right = int(round(width * box[2]))
    bottom = int(round(height * box[3]))
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    return image.crop((left, top, right, bottom))


def _white_balance(image: PILImage.Image) -> PILImage.Image:
    import numpy as np

    arr = np.array(image).astype(np.float32)
    mean_r = float(arr[:, :, 0].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 2].mean())
    if mean_g > 0 and mean_r > 0 and mean_b > 0:
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (mean_g / mean_r), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (mean_g / mean_b), 0, 255)
    return PILImage.fromarray(arr.astype(np.uint8))


def _auto_brightness(image: PILImage.Image) -> PILImage.Image:
    from PIL import ImageStat, ImageEnhance
    stat = ImageStat.Stat(image)
    mean_brightness = sum(stat.mean) / len(stat.mean) if stat.mean else 128.0
    if mean_brightness < 100 and mean_brightness > 0:
        scale = min(2.5, 130.0 / mean_brightness)
        return ImageEnhance.Brightness(image).enhance(scale)
    return image


def _parse_variant(variant: str) -> tuple[str, str, bool, bool]:
    """Parse variant string into (base_variant, model_type, white_balance, auto_brightness)."""
    model_type = "dinov3"
    wb = False
    bright = False
    v = variant
    if v.endswith("_wb_bright"):
        wb = True
        bright = True
        v = v[:-10]
    elif v.endswith("_bright"):
        bright = True
        v = v[:-7]
    elif v.endswith("_wb"):
        wb = True
        v = v[:-3]
    if v.startswith("pose2id_nfc_"):
        v = v[12:]
    if v.startswith("vit_"):
        model_type = "vit"
        v = v[4:]
    elif v.startswith("efficientnetv2_"):
        model_type = "efficientnetv2"
        v = v[15:]
    elif v.startswith("fastreid_"):
        model_type = "fastreid"
        v = v[9:]
    elif v.startswith("convnext_"):
        model_type = "convnext"
        v = v[9:]
    elif v.startswith("deit_"):
        model_type = "deit"
        v = v[5:]
    elif v.startswith("osnet_finetuned_improved_"):
        model_type = "osnet_finetuned_improved"
        v = v[25:]
    elif v.startswith("osnet_finetuned_"):
        model_type = "osnet_finetuned"
        v = v[16:]
    elif v.startswith("osnet_"):
        model_type = "osnet"
        v = v[6:]
    elif v.startswith("swin_"):
        model_type = "swin"
        v = v[5:]
    elif v.startswith("clip_"):
        model_type = "clip"
        v = v[5:]
    elif v.startswith("sam2_"):
        model_type = "sam2"
        v = v[5:]
    # Always apply WB + brightness to combined variants
    if v in ("combined", "combined_upscaled", "combined_upscaled_esrgan"):
        wb = True
        bright = True
    return v, model_type, wb, bright


def _variant_model_family(variant: str) -> str | None:
    """Extract model family from variant ID."""
    _, model_type, _, _ = _parse_variant(variant)
    return model_type


def _split_cluster_result_label(variant: str, full_label: str) -> tuple[str, str]:
    """Split a result label into (model_name, strategy_label)."""
    model_family = _variant_model_family(variant) or "dinov3"
    model_spec = _CLUSTER_MODEL_FAMILY_SPECS.get(model_family, {})
    prefix = model_spec.get("label", "")
    model_name = prefix.strip() or "DINOv3"
    strategy = full_label
    if prefix and strategy.startswith(prefix):
        strategy = strategy[len(prefix):].strip()
    return model_name, strategy


def _variant_has_nfc(variant: str) -> bool:
    """Return True if the variant applies Pose2ID NFC post-processing."""
    v = variant
    if v.endswith("_wb_bright"):
        v = v[:-10]
    elif v.endswith("_bright"):
        v = v[:-7]
    elif v.endswith("_wb"):
        v = v[:-3]
    return v.startswith("pose2id_nfc_")


def _best_threshold_for_variant(variant_id: str) -> float | None:
    """Return the threshold with the highest pairwise_f1 from saved experiments."""
    best_threshold = None
    best_f1 = -1.0
    for path in _experiment_store_dir().glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for result in data.get("results", []):
                if result.get("variant") == variant_id:
                    f1 = (result.get("metrics") or {}).get("pairwise_f1")
                    threshold = result.get("threshold")
                    if isinstance(f1, (int, float)) and isinstance(threshold, (int, float)):
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = threshold
        except Exception:
            continue
    return best_threshold


def _auto_threshold_plan(variants: list[str]) -> dict[str, dict]:
    plan: dict[str, dict] = {}
    for variant in variants:
        best = _best_threshold_for_variant(variant)
        family_id = _variant_model_family(variant)
        if best is not None:
            thresholds = [best]
            source = "saved_best"
        else:
            thresholds = _AUTO_THRESHOLD_FALLBACKS.get(family_id or "dinov3", [0.50, 0.55, 0.60])
            source = "fallback_sweep"
        plan[variant] = {
            "model_family": family_id,
            "thresholds": thresholds,
            "source": source,
            "best": best,
        }
    return plan


def _cluster_run_setup(req: ClusterExperimentRunRequest) -> tuple[list[str], list[str], dict]:
    variants = req.variants or list(CLUSTER_VARIANTS.keys())
    invalid_variants = [variant for variant in variants if variant not in CLUSTER_VARIANTS]
    if invalid_variants:
        raise HTTPException(status_code=400, detail=f"Unknown variants: {invalid_variants}")
    cluster_methods = req.cluster_methods or [req.cluster_method]
    valid_methods = {"greedy", "hdbscan"}
    invalid = [m for m in cluster_methods if m not in valid_methods]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid cluster methods: {invalid}")
    if req.auto_thresholds:
        threshold_plan = _auto_threshold_plan(variants)
    else:
        threshold_plan = {
            variant: {
                "model_family": _variant_model_family(variant),
                "thresholds": [req.threshold],
                "source": "request_default" if req.threshold is None else "request",
                "best": None,
            }
            for variant in variants
        }
    return variants, cluster_methods, threshold_plan


def _cluster_run_total(variants: list[str], cluster_methods: list[str], threshold_plan: dict) -> int:
    total = 0
    for method in cluster_methods:
        for variant in variants:
            plan = threshold_plan.get(variant) or {}
            thresholds = [None] if method == "hdbscan" else (plan.get("thresholds") or [None])
            total += len(thresholds)
    return total


def _iter_cluster_run_results(testset: dict, req: ClusterExperimentRunRequest, variants: list[str], cluster_methods: list[str], threshold_plan: dict):
    for method in cluster_methods:
        for variant in variants:
            plan = threshold_plan.get(variant) or {"thresholds": [req.threshold], "source": "request"}
            thresholds = [None] if method == "hdbscan" else (plan.get("thresholds") or [None])
            for threshold in thresholds:
                res = _run_cluster_variant(
                    testset,
                    variant,
                    threshold=threshold,
                    cluster_method=method,
                    post_merge_threshold=req.post_merge_threshold,
                )
                res["cluster_method"] = method
                res["threshold_source"] = plan.get("source")
                res["model_family"] = plan.get("model_family")
                if plan.get("best"):
                    res["threshold_best"] = plan.get("best")
                yield res


def _cluster_job_public(job: dict) -> dict:
    return {key: value for key, value in job.items() if key not in {"request"}}


def _run_cluster_experiment_job(job_id: str, testset_id: str, req: ClusterExperimentRunRequest) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    started_time = time.time()
    try:
        testset = _load_testset(testset_id)
        variants, cluster_methods, threshold_plan = _cluster_run_setup(req)
        total = _cluster_run_total(variants, cluster_methods, threshold_plan)
        with _CLUSTER_EXPERIMENT_JOBS_LOCK:
            job = _CLUSTER_EXPERIMENT_JOBS.get(job_id)
            if job:
                job.update({
                    "status": "running",
                    "started_at": started_at,
                    "total_runs": total,
                    "completed_runs": 0,
                    "threshold_plan": threshold_plan,
                })
        results = []
        for result in _iter_cluster_run_results(testset, req, variants, cluster_methods, threshold_plan):
            results.append(result)
            with _CLUSTER_EXPERIMENT_JOBS_LOCK:
                job = _CLUSTER_EXPERIMENT_JOBS.get(job_id)
                if job:
                    job["completed_runs"] = len(results)
                    job["latest_result"] = {
                        "variant": result.get("variant"),
                        "threshold": result.get("threshold"),
                        "cluster_method": result.get("cluster_method"),
                        "pairwise_f1": (result.get("metrics") or {}).get("pairwise_f1"),
                    }
        saved = _save_cluster_experiment_payload(
            name=req.name or f"background-{testset.get('name') or testset_id}",
            testset_id=testset_id,
            results=results,
            thresholds=[r.get("threshold") for r in results],
            metadata={
                "job_id": job_id,
                "background": True,
                "request": req.model_dump(),
                "cluster_methods": cluster_methods,
                "post_merge_threshold": req.post_merge_threshold,
                "auto_thresholds": req.auto_thresholds,
                "threshold_plan": threshold_plan,
                "elapsed_sec": round(time.time() - started_time, 3),
            },
        )
        with _CLUSTER_EXPERIMENT_JOBS_LOCK:
            job = _CLUSTER_EXPERIMENT_JOBS.get(job_id)
            if job:
                job.update({
                    "status": "completed",
                    "completed_runs": len(results),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - started_time, 3),
                    "experiment_id": saved["id"],
                    "experiment_name": saved["name"],
                    "error": None,
                })
    except Exception as exc:
        with _CLUSTER_EXPERIMENT_JOBS_LOCK:
            job = _CLUSTER_EXPERIMENT_JOBS.get(job_id)
            if job:
                job.update({
                    "status": "failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_sec": round(time.time() - started_time, 3),
                    "error": str(exc),
                })


_FEATURE_CACHE: dict[str, list[float]] = {}


def _apply_mask_to_image(image: PILImage.Image, mask: dict) -> PILImage.Image:
    import base64
    import numpy as np

    w, h = mask["width"], mask["height"]
    mask_bytes = base64.b64decode(mask["data"])
    mask_arr = np.frombuffer(mask_bytes, dtype=np.uint8).reshape(h, w)
    mask_img = PILImage.fromarray((mask_arr * 255).astype(np.uint8))
    # Resize mask to image size
    mask_img = mask_img.resize(image.size, PIL_RESAMPLE_BILINEAR)
    # Apply as alpha channel over black background
    rgba = image.convert("RGBA")
    rgba.putalpha(mask_img)
    bg = PILImage.new("RGB", image.size, (0, 0, 0))
    bg.paste(rgba, mask=mask_img)
    return bg


def _feature_for_image(
    path: str,
    variant: str,
    bbox: list[float] | None = None,
    mask: dict | None = None,
) -> list[float]:
    cache_key = f"{path}#{variant}#{json.dumps(bbox)}#{'mask' if mask else ''}"
    if cache_key in _FEATURE_CACHE:
        return _FEATURE_CACHE[cache_key]
    base_variant, model_type, wb, bright = _parse_variant(variant)
    image = PILImage.open(path).convert("RGB")
    if bbox is not None and len(bbox) == 4:
        width, height = image.size
        left, top, right, bottom = bbox
        left = max(0, min(width - 1, int(round(left))))
        top = max(0, min(height - 1, int(round(top))))
        right = max(left + 1, min(width, int(round(right))))
        bottom = max(top + 1, min(height, int(round(bottom))))
        image = image.crop((left, top, right, bottom))
    if mask:
        image = _apply_mask_to_image(image, mask)
    if wb:
        image = _white_balance(image)
    if bright:
        image = _auto_brightness(image)

    def _maybe_upscale_lanczos(img):
        width, height = img.size
        shortest = max(1, min(width, height))
        if shortest < 128:
            scale = 128.0 / float(shortest)
            return img.resize(
                (int(round(width * scale)), int(round(height * scale))),
                PIL_RESAMPLE_LANCZOS,
            )
        return img

    result = None
    if base_variant == "upscaled_lanczos":
        result = _image_to_vector(_maybe_upscale_lanczos(image), model_type=model_type)
    elif base_variant == "upscaled_esrgan":
        result = _image_to_vector(_maybe_upscale_realesrgan(image), model_type=model_type)
    elif base_variant == "face_region":
        result = _image_to_vector(_crop_ratio(image, (0.22, 0.00, 0.78, 0.28)), model_type=model_type)
    elif base_variant == "person_parts":
        upper = _image_to_vector(_crop_ratio(image, (0.00, 0.18, 1.00, 0.56)), model_type=model_type)
        lower = _image_to_vector(_crop_ratio(image, (0.00, 0.48, 1.00, 0.88)), model_type=model_type)
        result = [(0.55 * a + 0.45 * b) for a, b in zip(upper, lower)]
    elif base_variant == "whole_plus_face":
        whole = _image_to_vector(image, model_type=model_type)
        face = _image_to_vector(_crop_ratio(image, (0.22, 0.00, 0.78, 0.28)), model_type=model_type)
        result = [
            0.75 * whole_v + 0.25 * face_v
            for whole_v, face_v in zip(whole, face)
        ]
    elif base_variant == "combined":
        whole = _image_to_vector(image, model_type=model_type)
        face = _image_to_vector(_crop_ratio(image, (0.22, 0.00, 0.78, 0.28)), model_type=model_type)
        upper = _image_to_vector(_crop_ratio(image, (0.00, 0.18, 1.00, 0.56)), model_type=model_type)
        lower = _image_to_vector(_crop_ratio(image, (0.00, 0.48, 1.00, 0.88)), model_type=model_type)
        result = [
            0.45 * whole_v + 0.15 * face_v + 0.25 * upper_v + 0.15 * lower_v
            for whole_v, face_v, upper_v, lower_v in zip(whole, face, upper, lower)
        ]
    elif base_variant == "combined_upscaled":
        upscaled = _maybe_upscale_lanczos(image)
        whole = _image_to_vector(upscaled, model_type=model_type)
        face = _image_to_vector(_crop_ratio(upscaled, (0.22, 0.00, 0.78, 0.28)), model_type=model_type)
        upper = _image_to_vector(_crop_ratio(upscaled, (0.00, 0.18, 1.00, 0.56)), model_type=model_type)
        lower = _image_to_vector(_crop_ratio(upscaled, (0.00, 0.48, 1.00, 0.88)), model_type=model_type)
        result = [
            0.45 * whole_v + 0.15 * face_v + 0.25 * upper_v + 0.15 * lower_v
            for whole_v, face_v, upper_v, lower_v in zip(whole, face, upper, lower)
        ]
    elif base_variant == "combined_upscaled_esrgan":
        upscaled = _maybe_upscale_realesrgan(image)
        whole = _image_to_vector(upscaled, model_type=model_type)
        face = _image_to_vector(_crop_ratio(upscaled, (0.22, 0.00, 0.78, 0.28)), model_type=model_type)
        upper = _image_to_vector(_crop_ratio(upscaled, (0.00, 0.18, 1.00, 0.56)), model_type=model_type)
        lower = _image_to_vector(_crop_ratio(upscaled, (0.00, 0.48, 1.00, 0.88)), model_type=model_type)
        result = [
            0.45 * whole_v + 0.15 * face_v + 0.25 * upper_v + 0.15 * lower_v
            for whole_v, face_v, upper_v, lower_v in zip(whole, face, upper, lower)
        ]
    else:
        result = _image_to_vector(image, model_type=model_type)
    _FEATURE_CACHE[cache_key] = result
    return result


def _pairwise_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - _cosine_similarity(a, b)


def _safe_div(num: float, den: float) -> float:
    if den == 0.0:
        return 0.0
    return num / den


def _apply_nfc(features: list[list[float]], k1: int = 2, k2: int = 2) -> list[list[float]]:
    """Apply Pose2ID Neighbor Feature Centralization (NFC) to a batch of features.

    Reference: https://github.com/yuanc3/Pose2ID
    NFC finds mutual k-nearest neighbors in the feature space and adds their
    features to each sample, then re-normalizes. This is training-free and can
    improve clustering by centralizing identity features.
    """
    try:
        import torch
    except ImportError:
        return features

    if len(features) < max(k1, k2) + 1:
        return features

    feat = torch.tensor(features, dtype=torch.float32)
    # Normalize features in-place (original Pose2ID normalizes before k-NN)
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dev = feat.to(device)

    # Pairwise squared Euclidean distance for L2-normalized vectors:
    # ||x - y||^2 = 2 - 2 * (x @ y.T)
    dist = 2.0 - 2.0 * torch.mm(feat_dev, feat_dev.t())
    dist = dist.to("cpu")

    # Mask self-distance
    eye = torch.eye(dist.size(0))
    dist[eye == 1] = 1000.0

    # Find k1 nearest neighbors
    _, rank = dist.topk(k1, largest=False)

    # Mutual nearest neighbors (k2 reciprocity check)
    mutual_topk_list = []
    for i in range(rank.size(0)):
        mutual_list = []
        for j in rank[i]:
            if i in rank[j][:k2]:
                mutual_list.append(j.item())
        mutual_topk_list.append(mutual_list)

    # Centralize features by adding mutual neighbor features
    feat_copy = feat.clone()
    for i in range(rank.size(0)):
        if mutual_topk_list[i]:
            feat[i] += feat_copy[mutual_topk_list[i]].sum(dim=0)

    # Re-normalize
    feat = torch.nn.functional.normalize(feat, dim=1, p=2)
    return feat.detach().cpu().numpy().tolist()


def _split_cluster_by_position(
    indices: list[int],
    positions: list[tuple[float, float, float] | None],
    threshold_m: float,
) -> list[list[int]]:
    """Split a set of observation indices into position-coherent sub-clusters.

    Greedy online centroid clustering over 3D positions: an observation joins an
    existing sub-cluster iff its position is within ``threshold_m`` of that
    sub-cluster's running centroid; otherwise it starts a new sub-cluster.
    Observations without a position (None) are attached to the first sub-cluster
    (or their own if none exists yet) so they are never dropped.

    Returns a list of index-lists (the sub-clusters), in creation order.
    """
    thr_sq = threshold_m * threshold_m
    centroids: list[list[float]] = []
    counts: list[int] = []
    groups: list[list[int]] = []
    for i in indices:
        pos = positions[i]
        best = None
        best_d = None
        if pos is not None:
            for gi, cent in enumerate(centroids):
                d = ((pos[0] - cent[0]) ** 2 + (pos[1] - cent[1]) ** 2 + (pos[2] - cent[2]) ** 2)
                if best_d is None or d < best_d:
                    best_d = d
                    best = gi
        if pos is None or best is None or best_d is None or best_d > thr_sq:
            # start a new sub-cluster (positionless obs only start one if none exist)
            if pos is None and groups:
                groups[0].append(i)
                continue
            groups.append([i])
            if pos is not None:
                centroids.append([pos[0], pos[1], pos[2]])
                counts.append(1)
            else:
                centroids.append([0.0, 0.0, 0.0])
                counts.append(0)
            continue
        groups[best].append(i)
        c = counts[best]
        centroids[best] = [
            (old * c + new) / float(c + 1)
            for old, new in zip(centroids[best], pos)
        ]
        counts[best] = c + 1
    return groups


def _merge_clusters_by_proximity(
    label_to_indices: dict[int, list[int]],
    positions: list[tuple[float, float, float] | None],
    merge_distance_m: float,
) -> dict[int, list[int]]:
    """Merge clusters whose position centroids are within ``merge_distance_m``.

    Used for STATIC classes: two clusters that ended up at nearly the same spot
    (same physical object seen from divergent viewpoints) are merged even though
    the embedding step kept them apart. Clusters without any positioned
    observation are left untouched (never merged). Transitive merges are handled
    via union-find over cluster centroids.

    Returns a new ``label_to_indices`` dict (re-labelled 0..N-1).
    """
    labels = sorted(label_to_indices.keys())
    # position centroid per label (None if the cluster has no positioned obs)
    centroids: dict[int, list[float] | None] = {}
    for lab in labels:
        pts = [positions[i] for i in label_to_indices[lab] if positions[i] is not None]
        if pts:
            centroids[lab] = [
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
                sum(p[2] for p in pts) / len(pts),
            ]
        else:
            centroids[lab] = None

    parent = {lab: lab for lab in labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    thr_sq = merge_distance_m * merge_distance_m
    for ai in range(len(labels)):
        ca = centroids[labels[ai]]
        if ca is None:
            continue
        for bi in range(ai + 1, len(labels)):
            cb = centroids[labels[bi]]
            if cb is None:
                continue
            d = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2 + (ca[2] - cb[2]) ** 2
            if d <= thr_sq:
                union(labels[ai], labels[bi])

    merged: dict[int, list[int]] = {}
    root_to_new: dict[int, int] = {}
    for lab in labels:
        root = find(lab)
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        merged.setdefault(root_to_new[root], []).extend(label_to_indices[lab])
    return merged


def _cluster_features(features: list[list[float]], threshold: float) -> list[int]:
    centroids: list[list[float]] = []
    counts: list[int] = []
    labels: list[int] = []
    for feature in features:
        best_idx = None
        best_sim = -1.0
        for idx, centroid in enumerate(centroids):
            sim = _cosine_similarity(feature, centroid)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx
        if best_idx is None or best_sim < threshold:
            labels.append(len(centroids))
            centroids.append(list(feature))
            counts.append(1)
            continue
        labels.append(best_idx)
        count = counts[best_idx]
        centroids[best_idx] = [
            (old * count + new) / float(count + 1)
            for old, new in zip(centroids[best_idx], feature)
        ]
        counts[best_idx] = count + 1
    return labels


def _cluster_features_hdbscan(features: list[list[float]]) -> list[int]:
    try:
        import hdbscan
        import numpy as np
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"hdbscan not installed: {exc}")
    if len(features) < 2:
        return [0] * len(features)
    X = np.array(features, dtype=np.float32)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean", min_samples=1)
    raw_labels = clusterer.fit_predict(X)
    # HDBSCAN returns -1 for noise; treat each noise point as its own singleton cluster
    next_label = int(max(raw_labels.max() + 1, 0)) if len(raw_labels) > 0 else 0
    labels = []
    for lab in raw_labels.tolist():
        if lab == -1:
            labels.append(int(next_label))
            next_label += 1
        else:
            labels.append(int(lab))
    return labels


def _post_merge_clusters(features: list[list[float]], labels: list[int], merge_threshold: float) -> list[int]:
    """Greedy post-merge: merge clusters whose centroids have cosine similarity >= merge_threshold."""
    if merge_threshold <= 0 or len(features) == 0:
        return labels
    # Build cluster -> feature indices map
    cluster_to_indices: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        cluster_to_indices.setdefault(lab, []).append(idx)
    unique_labels = sorted(cluster_to_indices.keys())
    if len(unique_labels) <= 1:
        return labels
    # Compute centroids
    centroids: dict[int, list[float]] = {}
    for lab in unique_labels:
        indices = cluster_to_indices[lab]
        feats = [features[i] for i in indices]
        centroid = [sum(dim) / len(dim) for dim in zip(*feats)]
        # Normalize centroid
        norm = math.sqrt(sum(v * v for v in centroid))
        if norm > 0:
            centroid = [v / norm for v in centroid]
        centroids[lab] = centroid
    # Union-Find for transitive merging
    parent = {lab: lab for lab in unique_labels}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx
    # Sort pairs by similarity descending and merge if above threshold
    pairs = []
    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            a, b = unique_labels[i], unique_labels[j]
            sim = _cosine_similarity(centroids[a], centroids[b])
            if sim >= merge_threshold:
                pairs.append((sim, a, b))
    pairs.sort(key=lambda x: x[0], reverse=True)
    for _sim, a, b in pairs:
        union(a, b)
    # Remap to compact labels
    root_to_new: dict[int, int] = {}
    new_labels = []
    for lab in labels:
        root = find(lab)
        if root not in root_to_new:
            root_to_new[root] = len(root_to_new)
        new_labels.append(root_to_new[root])
    return new_labels


def _partition_metrics(true_labels: list[int | str | None], pred_labels: list[int]) -> dict:
    valid = [(truth, pred) for truth, pred in zip(true_labels, pred_labels) if truth is not None]
    if not valid:
        return {"available": False}
    truth_values = [truth for truth, _pred in valid]
    pred_values = [pred for _truth, pred in valid]
    n = len(valid)
    gt_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}
    contingency: dict[tuple[int, int], int] = {}
    for truth, pred in valid:
        gt_counts[truth] = gt_counts.get(truth, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        contingency[(truth, pred)] = contingency.get((truth, pred), 0) + 1

    def comb2(value: int) -> int:
        return value * (value - 1) // 2 if value >= 2 else 0

    tp = sum(comb2(count) for count in contingency.values())
    pred_pairs = sum(comb2(count) for count in pred_counts.values())
    gt_pairs = sum(comb2(count) for count in gt_counts.values())
    pair_precision = _safe_div(float(tp), float(pred_pairs))
    pair_recall = _safe_div(float(tp), float(gt_pairs))
    pair_f1 = _safe_div(2.0 * pair_precision * pair_recall, pair_precision + pair_recall)
    pred_to_gt: dict[int, dict[int, int]] = {}
    for truth, pred in valid:
        pred_to_gt.setdefault(pred, {})
        pred_to_gt[pred][truth] = pred_to_gt[pred].get(truth, 0) + 1
    purity = _safe_div(float(sum(max(values.values()) for values in pred_to_gt.values())), float(n))
    return {
        "available": True,
        "num_samples": n,
        "num_gt_ids": len(set(truth_values)),
        "num_pred_clusters": len(set(pred_values)),
        "pairwise_precision": pair_precision,
        "pairwise_recall": pair_recall,
        "pairwise_f1": pair_f1,
        "cluster_purity": purity,
    }


def _ground_truth_coverage(testset: dict) -> dict:
    records = _sample_records_for_testset(testset) if any((image.get("detections") or []) for image in testset.get("images") or []) else (testset.get("images") or [])
    labeled = 0
    identities = set()
    for record in records:
        identity = (record.get("label") or {}).get("identity")
        if identity is None or identity == "":
            continue
        labeled += 1
        identities.add(str(identity))
    return {
        "image_count": len(records),
        "labeled_count": labeled,
        "unlabeled_count": max(0, len(records) - labeled),
        "identity_count": len(identities),
    }


def _normalize_ground_truth_identity(value) -> int | str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except Exception:
        return text


def _cluster_preview(images: list[dict], labels: list[int], max_per_cluster: int = 1000) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for idx, (image, label) in enumerate(zip(images, labels)):
        grouped.setdefault(label, [])
        if len(grouped[label]) < max_per_cluster:
            grouped[label].append(
                {
                    "index": idx,
                    "filename": image.get("filename"),
                    "label": image.get("label"),
                    "image_url": (
                        f"/api/cluster-testsets/detections/{urllib.parse.quote(image.get('detection_id') or '')}/crop"
                        if image.get("detection_id")
                        else f"/api/cluster-testsets/image?path={urllib.parse.quote(image.get('path') or '')}"
                    ),
                }
            )
    result = []
    for cluster_id, examples in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        identity_counts: dict[str, int] = {}
        for ex in examples:
            ident = (ex.get("label") or {}).get("identity")
            if ident:
                identity_counts[ident] = identity_counts.get(ident, 0) + 1
        dominant = max(identity_counts, key=identity_counts.get) if identity_counts else None
        is_noise = cluster_id == -1
        for ex in examples:
            ident = (ex.get("label") or {}).get("identity")
            ex["is_wrong"] = is_noise or (dominant is not None and ident is not None and ident != dominant)
        result.append(
            {"cluster_id": cluster_id, "size": sum(1 for label in labels if label == cluster_id), "examples": examples}
        )
    return result


def _run_cluster_variant(testset: dict, variant: str, threshold: float | None = None, cluster_method: str = "greedy", post_merge_threshold: float | None = None) -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    if variant not in CLUSTER_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Unknown variant: {variant}")
    records = _sample_records_for_testset(testset, labeled_only=True)
    if not records:
        raise HTTPException(
            status_code=400,
            detail="No manually labeled samples found. Label detections/images in the Testset Builder first.",
        )
    features = []
    failures = []
    for idx, record in enumerate(records):
        try:
            features.append(_feature_for_image(record["path"], variant, bbox=record.get("bbox"), mask=None))
        except Exception as exc:
            failures.append({"index": idx, "path": record.get("path"), "error": str(exc)})
            features.append([])
    usable = [feature for feature in features if feature]
    if len(usable) != len(features):
        raise HTTPException(status_code=400, detail={"message": "Some images failed feature extraction", "failures": failures})
    if _variant_has_nfc(variant):
        features = _apply_nfc(features)
    selected_threshold = float(threshold) if threshold is not None else float(CLUSTER_VARIANTS[variant]["threshold"])
    if cluster_method == "hdbscan":
        labels = _cluster_features_hdbscan(features)
    else:
        labels = _cluster_features(features, selected_threshold)
    if post_merge_threshold is not None and post_merge_threshold > 0:
        labels = _post_merge_clusters(features, labels, post_merge_threshold)
    true_labels = [(record.get("label") or {}).get("identity") for record in records]
    metrics = _partition_metrics(true_labels, labels)
    return {
        "variant": variant,
        "label": CLUSTER_VARIANTS[variant]["label"],
        "description": CLUSTER_VARIANTS[variant].get("description", ""),
        "threshold": selected_threshold,
        "metrics": metrics,
        "assignments": [
            {
                "index": idx,
                "image_id": record.get("image_index"),
                "detection_id": record.get("detection_id"),
                "sample_kind": record.get("kind"),
                "filename": record.get("filename"),
                "true_identity": (record.get("label") or {}).get("identity"),
                "cluster_id": label,
                "image_url": (
                    f"/api/cluster-testsets/detections/{urllib.parse.quote(record.get('detection_id') or '')}/crop"
                    if record.get("detection_id")
                    else f"/api/cluster-testsets/image?path={urllib.parse.quote(record.get('path') or '')}"
                ),
            }
            for idx, (record, label) in enumerate(zip(records, labels))
        ],
        "clusters": _cluster_preview(records, labels),
    }


def _known_testset_image_paths() -> set[str]:
    paths: set[str] = set()
    for file_path in _cluster_store_dir().glob("*.json"):
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for image in payload.get("images") or []:
            image_path = image.get("path")
            if image_path:
                paths.add(str(Path(image_path).resolve()))
    return paths


_YOLO_SESSION_CACHE: dict[str, object] = {}
_YOLO_PROBE_ALLOWED_FOLDERS: set[str] = set()


def _load_yolo_session(model_path: str):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="onnxruntime is not installed in the web container. Rebuild web after installing requirements.",
        ) from exc
    resolved = str(Path(model_path).expanduser().resolve())
    if not Path(resolved).exists():
        raise HTTPException(status_code=404, detail=f"YOLO model not found: {resolved}")
    session = _YOLO_SESSION_CACHE.get(resolved)
    if session is None:
        session = ort.InferenceSession(resolved, providers=["CPUExecutionProvider"])
        _YOLO_SESSION_CACHE[resolved] = session
    return session


def _letterbox_image(image: PILImage.Image, size: int = 640) -> tuple[PILImage.Image, float, int, int]:
    width, height = image.size
    scale = min(size / float(width), size / float(height))
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), PIL_RESAMPLE_BILINEAR)
    canvas = PILImage.new("RGB", (size, size), (114, 114, 114))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    return canvas, scale, pad_x, pad_y


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return _safe_div(inter, area_a + area_b - inter)


def _nms_detections(detections: list[dict], iou_threshold: float, limit: int) -> list[dict]:
    kept: list[dict] = []
    for det in sorted(detections, key=lambda item: item["score"], reverse=True):
        if any(det["class_id"] == old["class_id"] and _bbox_iou(det["bbox"], old["bbox"]) >= iou_threshold for old in kept):
            continue
        kept.append(det)
        if len(kept) >= limit:
            break
    return kept


def _run_yolo_on_image(
    image_path: str,
    *,
    session,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
    classes: list[int] | None,
    segmentation: bool = False,
) -> list[dict]:
    image = PILImage.open(image_path).convert("RGB")
    return _run_yolo_on_pil(
        image,
        session=session,
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        classes=classes,
        segmentation=segmentation,
    )


def _run_yolo_on_pil(
    image: "PILImage.Image",
    *,
    session,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
    classes: list[int] | None,
    segmentation: bool = False,
) -> list[dict]:
    import numpy as np

    width, height = image.size
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    size = 640
    if len(input_shape) >= 4 and isinstance(input_shape[2], int):
        size = int(input_shape[2])
    letterboxed, scale, pad_x, pad_y = _letterbox_image(image, size=size)
    array = np.asarray(letterboxed).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))[None, ...]
    outputs = session.run(None, {input_name: array})
    pred = outputs[0]
    pred = np.squeeze(pred)
    if pred.ndim != 2:
        return []
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    num_classes = min(len(COCO_CLASSES), max(1, pred.shape[1] - 4))
    num_masks = 0
    proto = None
    if segmentation and len(outputs) >= 2:
        proto_out = outputs[1]
        if proto_out.ndim == 4:
            num_masks = int(proto_out.shape[1])
            num_classes = max(1, pred.shape[1] - 4 - num_masks)
            proto = proto_out[0]
    raw = []
    allowed = set(classes or [])
    for row in pred:
        class_scores = row[4 : 4 + num_classes]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < confidence:
            continue
        if allowed and class_id not in allowed:
            continue
        cx, cy, bw, bh = [float(value) for value in row[:4]]
        x1 = (cx - bw / 2.0 - pad_x) / scale
        y1 = (cy - bh / 2.0 - pad_y) / scale
        x2 = (cx + bw / 2.0 - pad_x) / scale
        y2 = (cy + bh / 2.0 - pad_y) / scale
        bbox = [
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
            max(0.0, min(float(width), x2)),
            max(0.0, min(float(height), y2)),
        ]
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            continue
        det = {
            "bbox": bbox,
            "score": score,
            "class_id": class_id,
            "class_name": COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else str(class_id),
        }
        if proto is not None and num_masks > 0:
            try:
                mask_coeffs = row[4 + num_classes : 4 + num_classes + num_masks]
                det["mask"] = _compute_detection_mask(proto, mask_coeffs, bbox, scale, pad_x, pad_y, size, width, height)
            except Exception:
                pass
        raw.append(det)
    return _nms_detections(raw, iou_threshold=iou_threshold, limit=max_detections)


def _resolve_yolo_detection_classes(class_preset: str, classes: list[int] | None) -> list[int] | None:
    if classes is not None:
        return sorted({int(class_id) for class_id in classes})
    if class_preset == "persons":
        return [YOLO_PERSON_CLASS_ID]
    if class_preset == "objects":
        return YOLO_OBJECT_CLASS_IDS
    return None


def _yolo_probe_preprocess(
    image: "PILImage.Image",
    *,
    clahe: bool,
    gamma: float,
    sharpen: bool,
    denoise: bool,
    auto_brighten: bool = False,
) -> "PILImage.Image":
    """Apply optional preprocessing transforms before YOLO inference."""
    import numpy as np
    from PIL import ImageFilter

    if auto_brighten:
        image = _auto_brightness(image)

    if denoise:
        image = image.filter(ImageFilter.MedianFilter(size=3))

    if clahe:
        try:
            import cv2
            arr = np.array(image)
            lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
            clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe_obj.apply(lab[:, :, 0])
            image = PILImage.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
        except ImportError:
            # cv2 not available: fall back to PIL histogram equalisation on L channel
            from PIL import ImageOps
            r, g, b = image.split()
            image = PILImage.merge("RGB", [ImageOps.equalize(c) for c in (r, g, b)])

    if abs(gamma - 1.0) > 0.01:
        arr = np.array(image).astype(np.float32) / 255.0
        arr = np.clip(np.power(arr, 1.0 / gamma), 0.0, 1.0)
        image = PILImage.fromarray((arr * 255.0).astype(np.uint8))

    if sharpen:
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=3))

    return image


def _run_yolo_sahi(
    image: "PILImage.Image",
    *,
    session,
    confidence: float,
    iou_threshold: float,
    max_detections: int,
    classes: list[int] | None,
    slice_size: int,
    overlap_ratio: float,
) -> list[dict]:
    """Sliced Inference (SAHI-style): tile the image, run detection per tile, merge with NMS."""
    W, H = image.size
    stride = max(1, int(slice_size * (1.0 - overlap_ratio)))
    xs = list(range(0, max(1, W - slice_size + 1), stride))
    if not xs or (xs[-1] + slice_size) < W:
        xs.append(max(0, W - slice_size))
    ys = list(range(0, max(1, H - slice_size + 1), stride))
    if not ys or (ys[-1] + slice_size) < H:
        ys.append(max(0, H - slice_size))

    all_detections: list[dict] = []
    for x0 in xs:
        for y0 in ys:
            x1 = min(W, x0 + slice_size)
            y1 = min(H, y0 + slice_size)
            tile = image.crop((x0, y0, x1, y1))
            if tile.size != (slice_size, slice_size):
                padded = PILImage.new("RGB", (slice_size, slice_size), (114, 114, 114))
                padded.paste(tile, (0, 0))
                tile = padded
            dets = _run_yolo_on_pil(
                tile,
                session=session,
                confidence=confidence,
                iou_threshold=iou_threshold,
                max_detections=max_detections,
                classes=classes,
                segmentation=False,
            )
            for det in dets:
                bx1, by1, bx2, by2 = det["bbox"]
                all_detections.append({
                    **det,
                    "bbox": [
                        max(0.0, min(float(W), bx1 + x0)),
                        max(0.0, min(float(H), by1 + y0)),
                        max(0.0, min(float(W), bx2 + x0)),
                        max(0.0, min(float(H), by2 + y0)),
                    ],
                    "source": "sahi",
                })
    return _nms_detections(all_detections, iou_threshold=iou_threshold, limit=max_detections)


def _run_yolo_hand_crop_objects(
    image: "PILImage.Image",
    *,
    person_session,
    hand_session,
    obj_session,
    person_confidence: float,
    hand_confidence: float,
    obj_confidence: float,
    iou_threshold: float,
    max_obj_detections: int,
    obj_classes: list[int] | None,
    person_pad_px: int,
    hand_pad_px: int,
    hand_class_id: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    1. Detect persons in the full image.
    2. For each person crop, detect hands (using hand_class_id or any class if hand_class_id<0).
    3. For each hand crop (with padding), run object detection.
    4. Return (person_dets, hand_dets, object_dets) all in original image coordinates.
    """
    W, H = image.size
    person_dets = _run_yolo_on_pil(
        image,
        session=person_session,
        confidence=person_confidence,
        iou_threshold=iou_threshold,
        max_detections=50,
        classes=[YOLO_PERSON_CLASS_ID],
        segmentation=False,
    )
    for det in person_dets:
        det["source"] = "person"

    hand_dets_all: list[dict] = []
    object_dets: list[dict] = []
    hand_classes = [hand_class_id] if hand_class_id >= 0 else None

    for p_det in person_dets:
        px1, py1, px2, py2 = p_det["bbox"]
        cpx1 = max(0, int(px1) - person_pad_px)
        cpy1 = max(0, int(py1) - person_pad_px)
        cpx2 = min(W, int(px2) + person_pad_px)
        cpy2 = min(H, int(py2) + person_pad_px)
        if cpx2 - cpx1 < 4 or cpy2 - cpy1 < 4:
            continue
        person_crop = image.crop((cpx1, cpy1, cpx2, cpy2))

        # Detect hands inside person crop
        hand_dets_in_crop = _run_yolo_on_pil(
            person_crop,
            session=hand_session,
            confidence=hand_confidence,
            iou_threshold=iou_threshold,
            max_detections=10,
            classes=hand_classes,
            segmentation=False,
        )

        for h_det in hand_dets_in_crop:
            # Map hand bbox back to original image coords
            hx1, hy1, hx2, hy2 = h_det["bbox"]
            orig_hx1 = max(0.0, min(float(W), hx1 + cpx1))
            orig_hy1 = max(0.0, min(float(H), hy1 + cpy1))
            orig_hx2 = max(0.0, min(float(W), hx2 + cpx1))
            orig_hy2 = max(0.0, min(float(H), hy2 + cpy1))
            hand_dets_all.append({
                **h_det,
                "bbox": [orig_hx1, orig_hy1, orig_hx2, orig_hy2],
                "class_name": "hand",  # override COCO lookup — hand model class 0 is always "hand"
                "source": "hand",
                "person_bbox": p_det["bbox"],
            })

            # Crop around the hand and run object detection
            chx1 = max(0, int(orig_hx1) - hand_pad_px)
            chy1 = max(0, int(orig_hy1) - hand_pad_px)
            chx2 = min(W, int(orig_hx2) + hand_pad_px)
            chy2 = min(H, int(orig_hy2) + hand_pad_px)
            if chx2 - chx1 < 4 or chy2 - chy1 < 4:
                continue
            hand_crop = image.crop((chx1, chy1, chx2, chy2))
            obj_dets_in_crop = _run_yolo_on_pil(
                hand_crop,
                session=obj_session,
                confidence=obj_confidence,
                iou_threshold=iou_threshold,
                max_detections=max_obj_detections,
                classes=obj_classes,
                segmentation=False,
            )
            for det in obj_dets_in_crop:
                bx1, by1, bx2, by2 = det["bbox"]
                object_dets.append({
                    **det,
                    "bbox": [
                        max(0.0, min(float(W), bx1 + chx1)),
                        max(0.0, min(float(H), by1 + chy1)),
                        max(0.0, min(float(W), bx2 + chx1)),
                        max(0.0, min(float(H), by2 + chy1)),
                    ],
                    "source": "hand_object",
                    "hand_bbox": [orig_hx1, orig_hy1, orig_hx2, orig_hy2],
                    "person_bbox": p_det["bbox"],
                })

    return person_dets, hand_dets_all, object_dets


def _run_yolo_person_crop_objects(
    image: "PILImage.Image",
    *,
    person_session,
    obj_session,
    person_confidence: float,
    obj_confidence: float,
    iou_threshold: float,
    max_obj_detections: int,
    obj_classes: list[int] | None,
    pad_px: int,
) -> tuple[list[dict], list[dict]]:
    """
    1. Detect persons in the full image.
    2. For each person crop (with padding), run object detection.
    3. Return (person_detections, object_detections_in_original_coords).
    """
    W, H = image.size
    person_dets = _run_yolo_on_pil(
        image,
        session=person_session,
        confidence=person_confidence,
        iou_threshold=iou_threshold,
        max_detections=50,
        classes=[YOLO_PERSON_CLASS_ID],
        segmentation=False,
    )
    # tag persons
    for det in person_dets:
        det["source"] = "person"

    object_dets: list[dict] = []
    for p_det in person_dets:
        x1, y1, x2, y2 = p_det["bbox"]
        cx1 = max(0, int(x1) - pad_px)
        cy1 = max(0, int(y1) - pad_px)
        cx2 = min(W, int(x2) + pad_px)
        cy2 = min(H, int(y2) + pad_px)
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            continue
        crop = image.crop((cx1, cy1, cx2, cy2))
        dets = _run_yolo_on_pil(
            crop,
            session=obj_session,
            confidence=obj_confidence,
            iou_threshold=iou_threshold,
            max_detections=max_obj_detections,
            classes=obj_classes,
            segmentation=False,
        )
        for det in dets:
            bx1, by1, bx2, by2 = det["bbox"]
            object_dets.append({
                **det,
                "bbox": [
                    max(0.0, min(float(W), bx1 + cx1)),
                    max(0.0, min(float(H), by1 + cy1)),
                    max(0.0, min(float(W), bx2 + cx1)),
                    max(0.0, min(float(H), by2 + cy1)),
                ],
                "source": "crop_object",
                "person_bbox": p_det["bbox"],
            })
    return person_dets, object_dets


def _compute_detection_mask(proto, mask_coeffs, bbox, scale, pad_x, pad_y, size, width, height):
    import numpy as np
    import base64

    num_masks, proto_h, proto_w = proto.shape
    mask = 1.0 / (1.0 + np.exp(-(mask_coeffs.astype(np.float32) @ proto.reshape(num_masks, -1))))
    mask = mask.reshape(proto_h, proto_w)
    mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
    mask_letter = mask_img.resize((size, size), PIL_RESAMPLE_BILINEAR)
    mask_arr = np.array(mask_letter) / 255.0
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    x1 = max(0, min(size - 1, pad_x))
    y1 = max(0, min(size - 1, pad_y))
    x2 = max(x1 + 1, min(size, pad_x + new_w))
    y2 = max(y1 + 1, min(size, pad_y + new_h))
    image_region = mask_arr[y1:y2, x1:x2]
    region_img = PILImage.fromarray((image_region * 255).astype(np.uint8))
    mask_orig = region_img.resize((width, height), PIL_RESAMPLE_BILINEAR)
    mask_crop = mask_orig.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
    mask_final = (np.array(mask_crop) / 255.0 > 0.5).astype(np.uint8)
    h, w = mask_final.shape
    return {"width": w, "height": h, "data": base64.b64encode(mask_final.tobytes()).decode("ascii")}


@app.get("/api/objects/pending-consolidation")
def get_pending_consolidation(
    similarity_threshold: float = Query(default=0.643, ge=0.0, le=1.0),
    min_observations: int = Query(default=1, ge=1),
    building: str | None = Query(default=None),
):
    """Return object IDs that would be merged if consolidation ran now.

    Computes centroid-to-centroid cosine similarity for all same-class cluster pairs
    and returns pairs above the threshold, along with the set of affected object IDs.
    """
    map_id = _resolve_map_id(building)
    map_filter = ""
    map_params: list = []
    if map_id is not None:
        map_filter = "AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
        map_params = [map_id, map_id]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH centroids AS (
                    SELECT
                        o.id AS object_id,
                        o.class_id,
                        AVG(oo.embedding) AS centroid,
                        COUNT(*) AS n_obs
                    FROM objects o
                    JOIN object_observations oo ON oo.object_id = o.id
                    {map_filter.replace("AND", "WHERE", 1) if not map_filter.startswith("WHERE") else map_filter}
                    GROUP BY o.id, o.class_id
                    HAVING COUNT(*) >= %s
                      AND AVG(oo.embedding) IS NOT NULL
                )
                SELECT
                    c1.object_id AS obj_a,
                    c2.object_id AS obj_b,
                    c1.n_obs AS n_a,
                    c2.n_obs AS n_b,
                    c1.class_id,
                    round((1.0 - (c1.centroid <=> c2.centroid))::numeric, 4) AS similarity
                FROM centroids c1
                JOIN centroids c2
                  ON c1.object_id < c2.object_id
                 AND c1.class_id = c2.class_id
                WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= %s
                ORDER BY similarity DESC
                """,
                tuple(map_params + [min_observations, similarity_threshold]),
            )
            pairs = cur.fetchall()

    pair_list = [
        {
            "obj_a": row[0],
            "obj_b": row[1],
            "n_a": row[2],
            "n_b": row[3],
            "class_id": row[4],
            "similarity": float(row[5]),
        }
        for row in pairs
    ]
    pending_ids = set()
    for p in pair_list:
        pending_ids.add(p["obj_a"])
        pending_ids.add(p["obj_b"])

    return {
        "similarity_threshold": similarity_threshold,
        "pair_count": len(pair_list),
        "pending_object_ids": list(pending_ids),
        "pairs": pair_list,
    }


def _run_single_merge_pass(
    similarity_threshold: float,
    min_observations: int,
) -> tuple[list[dict], list[dict]]:
    """Run one merge pass. Returns (merged, skipped)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH centroids AS (
                    SELECT
                        o.id AS object_id,
                        o.class_id,
                        AVG(oo.embedding) AS centroid,
                        COUNT(*) AS n_obs,
                        MIN(oo.created_at) AS first_seen
                    FROM objects o
                    JOIN object_observations oo ON oo.object_id = o.id
                    GROUP BY o.id, o.class_id
                    HAVING COUNT(*) >= %s
                      AND AVG(oo.embedding) IS NOT NULL
                )
                SELECT
                    c1.object_id AS obj_a,
                    c2.object_id AS obj_b,
                    c1.n_obs AS n_a,
                    c2.n_obs AS n_b,
                    c1.first_seen AS first_a,
                    c2.first_seen AS first_b,
                    round((1.0 - (c1.centroid <=> c2.centroid))::numeric, 4) AS similarity
                FROM centroids c1
                JOIN centroids c2
                  ON c1.object_id < c2.object_id
                 AND c1.class_id = c2.class_id
                WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= %s
                ORDER BY similarity DESC
                """,
                (min_observations, similarity_threshold),
            )
            pairs = cur.fetchall()

    merged: list[dict] = []
    skipped: list[dict] = []
    already_gone: set[int] = set()

    for row in pairs:
        obj_a, obj_b, n_a, n_b, first_a, first_b, similarity = row
        if obj_a in already_gone or obj_b in already_gone:
            skipped.append({"obj_a": obj_a, "obj_b": obj_b, "reason": "already merged this pass"})
            continue

        if n_a > n_b:
            keep_id, drop_id = obj_a, obj_b
        elif n_b > n_a:
            keep_id, drop_id = obj_b, obj_a
        elif (first_a or "") <= (first_b or ""):
            keep_id, drop_id = obj_a, obj_b
        else:
            keep_id, drop_id = obj_b, obj_a

        with get_conn() as conn2:
            with conn2.cursor() as cur2:
                cur2.execute(
                    "UPDATE object_observations SET object_id = %s WHERE object_id = %s",
                    (keep_id, drop_id),
                )
                moved = cur2.rowcount
                cur2.execute("DELETE FROM objects WHERE id = %s", (drop_id,))
            conn2.commit()

        already_gone.add(drop_id)
        merged.append({
            "keep": keep_id,
            "drop": drop_id,
            "observations_moved": moved,
            "similarity": float(similarity),
        })

    return merged, skipped


def _run_convergent_merge(
    similarity_threshold: float,
    min_observations: int,
    max_passes: int = 20,
) -> dict:
    """Iterate merge passes until no new merges occur (convergence) or max_passes is reached."""
    all_merged: list[dict] = []
    all_skipped: list[dict] = []
    for pass_num in range(1, max_passes + 1):
        merged, skipped = _run_single_merge_pass(similarity_threshold, min_observations)
        all_merged.extend(merged)
        all_skipped.extend(skipped)
        if not merged:
            break
    return {
        "merged_count": len(all_merged),
        "skipped_count": len(all_skipped),
        "merged": all_merged,
        "skipped": all_skipped,
    }


# ---------------------------------------------------------------------------
# Embedding recomputation helpers (match ROS service preprocessing)
# ---------------------------------------------------------------------------

def _embed_osnet_finetuned_from_pil(image: PILImage.Image) -> list[float]:
    """Compute OSNet finetuned improved embedding from a PIL image.
    Replicates the white-balance + auto-brightness + resize preprocessing
    used by the runtime osnet_embedding_service."""
    import cv2
    import numpy as np
    import torch

    model = _load_osnet_finetuned_improved_model()
    if model is None:
        raise RuntimeError(f"OSNet finetuned improved model not available: {_OSNET_FINETUNED_IMPROVED_MODEL_ERROR}")

    rgb = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # White balance (same as OSNet service)
    arr = rgb.astype(np.float32)
    mean_r = float(arr[:, :, 2].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 0].mean())
    if mean_g > 0 and mean_r > 0 and mean_b > 0:
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (mean_g / mean_r), 0, 255)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (mean_g / mean_b), 0, 255)
    rgb = arr.astype(np.uint8)

    # Auto brightness (same as OSNet service)
    mean_brightness = float(rgb.mean())
    if 0.0 < mean_brightness < 80.0:
        scale = min(2.0, 128.0 / mean_brightness)
        rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # Resize and normalize
    rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.inference_mode():
        emb = model(tensor)[0].detach().float().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def _embed_dinov3_from_pil(image: PILImage.Image) -> list[float]:
    """Compute DINOv3 embedding from a PIL image with multi-view augmentation.
    Replicates the center-crop + border-suppression logic used by the
    runtime dinov3_embedding_service."""
    import cv2
    import numpy as np
    import torch

    model, processor = _load_dinov3_model()
    if model is None:
        raise RuntimeError(f"DINOv3 model not available: {_DINOV3_MODEL_ERROR}")

    rgb = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Multi-view augmentation (same defaults as DINOv3 service)
    views = [rgb]
    height, width = rgb.shape[:2]
    if min(height, width) >= 72:
        crop_h = int(height * 0.82)
        crop_w = int(width * 0.82)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        y1 = min(height, y0 + crop_h)
        x1 = min(width, x0 + crop_w)
        center_crop = rgb[y0:y1, x0:x1]
        if center_crop.size > 0:
            views.append(center_crop)

        border_h = int(round(height * 0.12))
        border_w = int(round(width * 0.12))
        if border_h > 0 or border_w > 0:
            blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=6.0, sigmaY=6.0)
            focused = rgb.copy()
            if border_h > 0:
                focused[:border_h, :, :] = blurred[:border_h, :, :]
                focused[height - border_h:, :, :] = blurred[height - border_h:, :, :]
            if border_w > 0:
                focused[:, :border_w, :] = blurred[:, :border_w, :]
                focused[:, width - border_w:, :] = blurred[:, width - border_w:, :]
            views.append(focused)

    device = next(model.parameters()).device
    vectors = []
    for view in views:
        inputs = processor(images=view, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        else:
            emb = outputs.last_hidden_state[:, 0, :][0]
        vectors.append(emb.detach().float().cpu().numpy().astype(np.float32))

    emb = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def _run_recompute_embeddings_job(job_id: str, class_id_filter: int | None) -> None:
    """Background runner that recomputes embeddings for all observations."""
    import io, time as _time

    with _RECOMPUTE_JOBS_LOCK:
        _RECOMPUTE_JOBS[job_id]["status"] = "running"
        _RECOMPUTE_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    total_updated = 0
    total_skipped = 0
    total_errors = 0
    first_error: str | None = None

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if class_id_filter is not None:
                    cur.execute(
                        "SELECT COUNT(*) FROM object_observations WHERE class_id = %s AND LENGTH(cropped_image) > 100",
                        (class_id_filter,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM object_observations WHERE LENGTH(cropped_image) > 100")
                total_rows = cur.fetchone()[0]

        with _RECOMPUTE_JOBS_LOCK:
            _RECOMPUTE_JOBS[job_id]["total_observations"] = total_rows
            _RECOMPUTE_JOBS[job_id]["observations_done"] = 0

        batch_size = 50
        offset = 0
        while True:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if class_id_filter is not None:
                        cur.execute(
                            """
                            SELECT id, class_id, cropped_image
                            FROM object_observations
                            WHERE class_id = %s AND LENGTH(cropped_image) > 100
                            ORDER BY id
                            LIMIT %s OFFSET %s
                            """,
                            (class_id_filter, batch_size, offset),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, class_id, cropped_image
                            FROM object_observations
                            WHERE LENGTH(cropped_image) > 100
                            ORDER BY id
                            LIMIT %s OFFSET %s
                            """,
                            (batch_size, offset),
                        )
                    rows = cur.fetchall()

            if not rows:
                break

            for obs_id, class_id, cropped_bytes in rows:
                try:
                    image = PILImage.open(io.BytesIO(bytes(cropped_bytes)))
                    # Person class (0) uses OSNet finetuned improved; everything else uses ConvNeXt-Tiny
                    if class_id == 0:
                        embedding = _embed_osnet_finetuned_from_pil(image)
                    else:
                        embedding = _image_to_vector(image, model_type="convnext")

                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE object_observations SET embedding = %s::vector WHERE id = %s",
                                (json.dumps(embedding), obs_id),
                            )
                        conn.commit()
                    total_updated += 1
                except Exception as exc:
                    total_errors += 1
                    if first_error is None:
                        first_error = str(exc)

            offset += len(rows)
            with _RECOMPUTE_JOBS_LOCK:
                job = _RECOMPUTE_JOBS.get(job_id)
                if job is not None:
                    job["observations_done"] = offset

        with _RECOMPUTE_JOBS_LOCK:
            job = _RECOMPUTE_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["result"] = {
                    "observations_updated": total_updated,
                    "observations_skipped": total_skipped,
                    "observations_errored": total_errors,
                    "first_error": first_error,
                }
    except Exception as exc:
        with _RECOMPUTE_JOBS_LOCK:
            job = _RECOMPUTE_JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = str(exc)


@app.post("/api/objects/recompute-embeddings")
def recompute_embeddings(
    class_id: int | None = Query(default=None),
):
    """Recompute embeddings for all observations using the live models.
    Person observations use OSNet finetuned improved; all others use ConvNeXt-Tiny.
    Returns immediately with a job_id; poll GET /api/objects/recompute-embeddings/jobs/{job_id} for progress."""
    # Pre-check model availability so we fail fast instead of silently erroring every observation
    person_ok = _load_osnet_finetuned_improved_model() is not None
    convnext_model = _load_convnext_model()
    object_ok = convnext_model is not None
    errors: list[str] = []
    if not person_ok:
        errors.append("OSNet finetuned improved model not available")
    if not object_ok:
        errors.append("ConvNeXt-Tiny model not available")
    if errors:
        raise HTTPException(status_code=503, detail="; ".join(errors))

    job_id = f"rec-{uuid.uuid4().hex[:8]}"
    with _RECOMPUTE_JOBS_LOCK:
        _RECOMPUTE_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "class_id": class_id,
            "observations_done": 0,
            "total_observations": 0,
        }
    threading.Thread(
        target=_run_recompute_embeddings_job,
        args=(job_id, class_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/objects/recompute-embeddings/jobs/{job_id}")
def get_recompute_job(job_id: str):
    with _RECOMPUTE_JOBS_LOCK:
        job = _RECOMPUTE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/objects/recompute-embeddings/jobs")
def list_recompute_jobs():
    with _RECOMPUTE_JOBS_LOCK:
        jobs = list(_RECOMPUTE_JOBS.values())
    jobs_sorted = sorted(jobs, key=lambda j: j.get("started_at", ""), reverse=True)[:20]
    return {"jobs": jobs_sorted}


# ---------------------------------------------------------------------------
# Recluster from scratch
# ---------------------------------------------------------------------------

def _run_full_recluster(
    similarity_threshold: float = 0.643,
    class_id_filter: int | None = None,
    progress_callback: Callable | None = None,
    object_similarity_threshold: float | None = None,
    exclude_persons: bool = False,
) -> dict:
    """Re-cluster ALL observations from scratch using greedy cosine thresholding.

    Unlike merge-pending (merge-only), this can also SPLIT clusters:
    observations previously grouped into one object may be reassigned to separate
    objects if their embeddings diverge.

    Algorithm per class:
    1. Fetch all observations (with embeddings) sorted by created_at.
    2. Run greedy cosine clustering to get new cluster labels.
    3. Match each new cluster to an existing object via majority vote.
    4. Largest cluster gets first pick of its majority object.
    5. Clusters whose majority is already claimed get a new object created.
    6. Reassign observations and delete orphaned objects.

    When object_similarity_threshold is set, non-person classes use that
    (typically lower) threshold while persons keep similarity_threshold —
    object embeddings are less viewpoint-robust and over-fragment otherwise.
    """
    import json as _json

    with get_conn() as conn:
        with conn.cursor() as cur:
            if class_id_filter is not None:
                cur.execute(
                    "SELECT DISTINCT class_id FROM objects WHERE class_id = %s ORDER BY class_id",
                    (class_id_filter,),
                )
            elif exclude_persons:
                cur.execute(
                    "SELECT DISTINCT class_id FROM objects WHERE class_id != %s ORDER BY class_id",
                    (YOLO_PERSON_CLASS_ID,),
                )
            else:
                cur.execute("SELECT DISTINCT class_id FROM objects ORDER BY class_id")
            class_ids = [row[0] for row in cur.fetchall()]

    total_objects_created = 0
    total_objects_deleted = 0
    total_observations_reassigned = 0

    for idx, cid in enumerate(class_ids):
        is_static_class = cid != YOLO_PERSON_CLASS_ID and cid not in YOLO_PORTABLE_CLASS_IDS
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT oo.id, oo.object_id, oo.embedding::text, oo.map_id, oo.x, oo.y, oo.z
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE o.class_id = %s AND oo.embedding IS NOT NULL
                    ORDER BY oo.created_at ASC
                    """,
                    (cid,),
                )
                rows = cur.fetchall()

        if not rows:
            if progress_callback:
                progress_callback(idx + 1, len(class_ids))
            continue

        obs_ids = [r[0] for r in rows]
        orig_obj_ids = [r[1] for r in rows]
        embeddings = [_json.loads(r[2]) for r in rows]
        # 3D position with map separation baked into x (different maps are always
        # far apart, so cross-map observations never share a position sub-cluster).
        positions: list[tuple[float, float, float] | None] = [
            (
                float(r[4]) + (float(r[3]) * 1.0e6 if r[3] is not None else 0.0),
                float(r[5]),
                float(r[6]),
            )
            if r[4] is not None and r[5] is not None and r[6] is not None
            else None
            for r in rows
        ]

        cluster_threshold = similarity_threshold
        if cid != YOLO_PERSON_CLASS_ID and object_similarity_threshold is not None:
            cluster_threshold = object_similarity_threshold
        labels = _cluster_features(embeddings, cluster_threshold)

        # Group observation indices by new cluster label
        label_to_indices: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(i)

        # For STATIC classes (chairs, tables, ...), split each embedding-cluster by
        # position so two look-alike objects at different locations are NOT merged
        # into one. Portable classes and persons keep pure-embedding clustering.
        # The map_id is folded into the position key (scaled up) so observations
        # from different maps can never share a position sub-cluster.
        if is_static_class:
            split_label_to_indices: dict[int, list[int]] = {}
            next_label = 0
            for lab in sorted(label_to_indices.keys()):
                sub_groups = _split_cluster_by_position(
                    label_to_indices[lab], positions, RECLUSTER_STATIC_POSITION_THRESHOLD_M
                )
                for grp in sub_groups:
                    split_label_to_indices[next_label] = grp
                    next_label += 1
            label_to_indices = split_label_to_indices
            # Merge clusters whose position centroids are nearly identical (same
            # physical object seen from divergent viewpoints), even if embeddings
            # fell below the similarity threshold.
            label_to_indices = _merge_clusters_by_proximity(
                label_to_indices, positions, RECLUSTER_STATIC_MERGE_DISTANCE_M
            )

        # Majority vote: which existing object_id dominates each new cluster?
        label_to_majority: dict[int, int] = {}
        for lab, indices in label_to_indices.items():
            counts: dict[int, int] = {}
            for i in indices:
                oid = orig_obj_ids[i]
                counts[oid] = counts.get(oid, 0) + 1
            label_to_majority[lab] = max(counts, key=counts.get)

        # Assign existing objects: largest clusters claim first
        sorted_labels = sorted(label_to_indices.keys(), key=lambda l: -len(label_to_indices[l]))
        claimed: set[int] = set()
        label_to_assigned_obj: dict[int, int | None] = {}
        for lab in sorted_labels:
            maj = label_to_majority[lab]
            if maj not in claimed:
                label_to_assigned_obj[lab] = maj
                claimed.add(maj)
            else:
                label_to_assigned_obj[lab] = None  # needs a new object

        # Create new objects for unclaimed clusters
        for lab in sorted_labels:
            if label_to_assigned_obj[lab] is not None:
                continue
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (cid,))
                    new_id = cur.fetchone()[0]
                conn.commit()
            label_to_assigned_obj[lab] = new_id
            total_objects_created += 1

        # Collect reassignments (skip obs already assigned to the right object)
        new_obj_to_obs_ids: dict[int, list[int]] = {}
        for lab, indices in label_to_indices.items():
            target = label_to_assigned_obj[lab]
            for i in indices:
                if orig_obj_ids[i] != target:
                    new_obj_to_obs_ids.setdefault(target, []).append(obs_ids[i])

        if new_obj_to_obs_ids:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for target_obj, obs_id_list in new_obj_to_obs_ids.items():
                        cur.execute(
                            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
                            (target_obj, obs_id_list),
                        )
                        total_observations_reassigned += cur.rowcount
                conn.commit()

        # Delete objects with no remaining observations
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM objects
                    WHERE class_id = %s
                      AND id NOT IN (
                          SELECT DISTINCT object_id FROM object_observations
                          WHERE object_id IS NOT NULL
                      )
                    """,
                    (cid,),
                )
                total_objects_deleted += cur.rowcount
            conn.commit()

        if progress_callback:
            progress_callback(idx + 1, len(class_ids))

    return {
        "objects_created": total_objects_created,
        "objects_deleted": total_objects_deleted,
        "observations_reassigned": total_observations_reassigned,
        "classes_processed": len(class_ids),
    }


# ---------------------------------------------------------------------------
# Face-based person rebuild (ported from scripts/recluster_persons_by_face_embedding.py)
# ---------------------------------------------------------------------------

def _face_norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _face_parse_emb(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        return np.asarray(t, dtype=float)
    try:
        import json as _json
        return np.asarray(_json.loads(t), dtype=float)
    except Exception:
        return None


def _face_cluster(oids, embs, thr):
    """Greedy average-linkage clustering of unit face embeddings. Returns oid->cluster."""
    order = sorted(range(len(oids)), key=lambda i: -len(embs[i]))  # deterministic
    centroids = []   # running sum of unit vectors
    counts = []
    label = {}
    for i in order:
        v = embs[i]
        best, bs = -1, -1.0
        for ci in range(len(centroids)):
            s = float(np.dot(v, centroids[ci] / counts[ci]))
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= thr:
            label[oids[i]] = best
            centroids[best] += v
            counts[best] += 1
        else:
            label[oids[i]] = len(centroids)
            centroids.append(v.copy())
            counts.append(1)
    return label, centroids, counts


def _run_full_face_rebuild(face_threshold: float, min_identity_faces: int, body_floor: float, body_assign: bool = False, body_recluster: bool = True, body_merge_sim: float = 0.70, body_merge_margin: float = 0.02, progress_callback=None, map_id: int | None = None):
    """Rebuild PERSON clusters from offline ArcFace embeddings (person_face_embeddings).

    OSNet body embeddings cannot separate the cast (cross-person sim ~0.95), so faces
    (ArcFace, cross-person <=~0.17) are the ground-truth identity signal. Steps:
      1. Cluster all face embeddings into identities (greedy avg-linkage @ face_threshold).
      2. Assign every person observation to an identity: (a) its own face, (b) same
         (track, scene) as a faced obs. (c) OPTIONAL (body_assign=True): nearest identity
         body-embedding centroid (>= body_floor). Body assignment is OFF by default
         because OSNet cross-person similarity (~0.95) exceeds any usable floor, so it
         pulled unfaced obs of OTHER cast members into the large mains (contamination).
         With body_assign off, unlinked faceless obs become provisional per-(scene,track)
         clusters instead of being forced into a main — keeping the mains pure.
      3. Rebuild one objects row per identity, move observations + parts, delete empties.
      4. Re-populate face_observations from the staging table (the rebuild's object
         deletions cascade-wipe it via face_observations_person_id_fkey ON DELETE CASCADE).
    """
    from collections import defaultdict

    conn = get_conn()
    cur = conn.cursor()

    # Map scoping: when map_id is given, only rebuild person clusters for observations
    # in that map (and only reuse/delete person objects that belong to that map). When
    # None, the rebuild is global (previous behaviour).
    oo_map_sql = ""
    oo_map_params: list = []
    if map_id is not None:
        oo_map_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
        oo_map_params = [map_id, map_id]

    # --- Load all person observations ---
    cur.execute(
        "SELECT id, object_id, yolo_track_id, scene_id, embedding::text "
        "FROM object_observations oo WHERE oo.class_id = %s" + oo_map_sql + " ORDER BY id",
        [YOLO_PERSON_CLASS_ID, *oo_map_params],
    )
    obs_rows = cur.fetchall()
    obs_ids = [r[0] for r in obs_rows]
    body_emb = {r[0]: _face_parse_emb(r[4]) for r in obs_rows}
    obs_track = {r[0]: r[2] for r in obs_rows}
    obs_scene = {r[0]: r[3] for r in obs_rows}

    # --- Load face embeddings (scoped to this map's observations when map_id is set) ---
    cur.execute(
        "SELECT pfe.observation_id, pfe.embedding::text FROM person_face_embeddings pfe "
        "JOIN object_observations oo ON oo.id = pfe.observation_id "
        "WHERE pfe.embedding IS NOT NULL" + oo_map_sql,
        oo_map_params,
    )
    face_emb = {}
    for oid, etxt in cur.fetchall():
        e = _face_norm(_face_parse_emb(etxt))
        if e is not None:
            face_emb[oid] = e
    if len(face_emb) < 2:
        raise RuntimeError("Not enough face embeddings in person_face_embeddings — run face embedding generation first.")

    if progress_callback:
        progress_callback(0, 4)

    # --- 1. Cluster faces into identities ---
    f_oids = list(face_emb.keys())
    f_embs = [face_emb[o] for o in f_oids]
    oid_to_cluster, _cents, _cnts = _face_cluster(f_oids, f_embs, face_threshold)
    cluster_members = defaultdict(list)
    for o, c in oid_to_cluster.items():
        cluster_members[c].append(o)
    identities = {c: m for c, m in cluster_members.items() if len(m) >= min_identity_faces}

    # --- 2. Assign observations ---
    obs_ident = {}
    for o in obs_ids:
        if o in oid_to_cluster and oid_to_cluster[o] in identities:
            obs_ident[o] = oid_to_cluster[o]

    # (b) faceless obs sharing (track, scene) with a faced, identified obs
    trackscene_ident = {}
    for o in obs_ids:
        if o in obs_ident:
            key = (obs_track[o], obs_scene[o])
            trackscene_ident.setdefault(key, defaultdict(int))
            trackscene_ident[key][obs_ident[o]] += 1
    for o in obs_ids:
        if o in obs_ident:
            continue
        key = (obs_track[o], obs_scene[o])
        if key in trackscene_ident:
            obs_ident[o] = max(trackscene_ident[key].items(), key=lambda kv: kv[1])[0]

    # (c) remaining faceless obs -> nearest identity BODY centroid. OPTIONAL and OFF by
    # default: OSNet cannot separate the cast, so this path contaminates the mains.
    leftover = []
    if body_assign:
        ident_body_sum = {}
        ident_body_cnt = defaultdict(int)
        for o, c in obs_ident.items():
            be = _face_norm(body_emb.get(o))
            if be is None:
                continue
            ident_body_sum[c] = be.copy() if c not in ident_body_sum else ident_body_sum[c] + be
            ident_body_cnt[c] += 1
        ident_body_centroid = {c: _face_norm(s / ident_body_cnt[c]) for c, s in ident_body_sum.items() if ident_body_cnt[c] > 0}

        for o in obs_ids:
            if o in obs_ident:
                continue
            be = _face_norm(body_emb.get(o))
            if be is None:
                leftover.append(o)
                continue
            best, bs = None, -1.0
            for c, cen in ident_body_centroid.items():
                if cen is None:
                    continue
                s = float(np.dot(be, cen))
                if s > bs:
                    bs, best = s, c
            if best is not None and bs >= body_floor:
                obs_ident[o] = best
            else:
                leftover.append(o)
    else:
        # No body assignment: every obs not already linked by face/track/scene is leftover.
        for o in obs_ids:
            if o not in obs_ident:
                leftover.append(o)

    ident_sizes = defaultdict(int)
    for o, c in obs_ident.items():
        ident_sizes[c] += 1
    leftover_groups = defaultdict(list)
    for o in leftover:
        leftover_groups[(obs_scene[o], obs_track[o])].append(o)

    if progress_callback:
        progress_callback(1, 4)

    # --- 3. Rebuild objects ---
    # Reuse pool: person objects that belong to this map (have observations in it). When
    # global (map_id None), all person objects are candidates. Scoping prevents an object
    # from ANOTHER map being chosen as the reuse target for one of this map's identities.
    if map_id is not None:
        cur.execute(
            "SELECT DISTINCT o.id FROM objects o "
            "JOIN object_observations oo ON oo.object_id = o.id "
            "WHERE o.class_id = %s" + oo_map_sql,
            [YOLO_PERSON_CLASS_ID, *oo_map_params],
        )
    else:
        cur.execute("SELECT id FROM objects WHERE class_id = %s", (YOLO_PERSON_CLASS_ID,))
    existing_obj_ids = {r[0] for r in cur.fetchall()}
    obs_current_obj = {r[0]: r[1] for r in obs_rows}

    ident_to_obj = {}
    used_objs = set()
    for c in sorted(identities.keys(), key=lambda c: -ident_sizes[c]):
        member_objs = [obs_current_obj[o] for o in obs_ids if obs_ident.get(o) == c]
        cand = defaultdict(int)
        for ob in member_objs:
            cand[ob] += 1
        chosen = None
        for ob, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
            if ob in existing_obj_ids and ob not in used_objs:
                chosen = ob
                break
        if chosen is None:
            cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (YOLO_PERSON_CLASS_ID,))
            chosen = cur.fetchone()[0]
        ident_to_obj[c] = chosen
        used_objs.add(chosen)

    ident_to_obslist = defaultdict(list)
    for o, c in obs_ident.items():
        ident_to_obslist[c].append(o)
    moved = 0
    for c, obj in ident_to_obj.items():
        ol = ident_to_obslist[c]
        if not ol:
            continue
        cur.execute("UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)", (obj, ol))
        cur.execute("UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)", (obj, ol))
        moved += len(ol)

    provisional = 0
    provisional_objs = []  # (provisional_object_id, [obs_ids])
    for (sc, tr), ol in leftover_groups.items():
        cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (YOLO_PERSON_CLASS_ID,))
        obj = cur.fetchone()[0]
        provisional += 1
        provisional_objs.append((obj, ol))
        cur.execute("UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)", (obj, ol))
        cur.execute("UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)", (obj, ol))

    # --- 3b. OPTIONAL body recluster: merge confident provisional clusters into mains.
    # After the face rebuild the mains are pure (face/track-scene confirmed) and the
    # provisional clusters are small, mostly single-person fragments. A provisional
    # cluster is merged into a main only when its OSNet body centroid is both close to
    # that main (>= body_merge_sim) AND clearly closer than to any other main (scaled
    # margin: best-second >= body_merge_margin * body_merge_sim). The scaled margin makes
    # body_merge_sim the visible dial (lower = more merges); the margin still rejects the
    # most MIXED provisional clusters whose centroid sits between two
    # mains, but proportionally to how strict body_merge_sim is.
    body_merged = 0
    if body_recluster and ident_to_obj:
        # Body centroid per main identity object (from its face/track-scene obs).
        main_obs = defaultdict(list)
        for c, obj in ident_to_obj.items():
            main_obs[obj] = ident_to_obslist.get(c, [])
        main_centroid = {}
        for obj, ol in main_obs.items():
            vecs = [_face_norm(body_emb.get(o)) for o in ol]
            vecs = [v for v in vecs if v is not None]
            if vecs:
                main_centroid[obj] = _face_norm(np.mean(np.stack(vecs), axis=0))

        for prov_obj, ol in provisional_objs:
            vecs = [_face_norm(body_emb.get(o)) for o in ol]
            vecs = [v for v in vecs if v is not None]
            if not vecs:
                continue
            pc = _face_norm(np.mean(np.stack(vecs), axis=0))
            if pc is None:
                continue
            scored = []
            for obj, mc in main_centroid.items():
                if mc is None:
                    continue
                scored.append((float(np.dot(pc, mc)), obj))
            if len(scored) < 1:
                continue
            scored.sort(key=lambda x: -x[0])
            best_sim, best_obj = scored[0]
            second_sim = scored[1][0] if len(scored) > 1 else -1.0
            # Scaled margin: the required winner's margin grows with body_merge_sim. This
            # makes body_merge_sim the *visible* dial — lowering it both admits more
            # candidates AND relaxes the disambiguation requirement proportionally, so the
            # merge count changes smoothly across the whole range. With a FIXED margin the
            # margin gate (not body_merge_sim) is the binding constraint on this dataset
            # (OSNet cross-person sim overlaps same-person), so body_merge_sim had no
            # visible effect between ~0.3-0.6. Set body_merge_margin=0 to merge purely on
            # best_sim >= body_merge_sim.
            required_margin = body_merge_margin * body_merge_sim
            if best_sim >= body_merge_sim and (best_sim - second_sim) >= required_margin:
                cur.execute("UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)", (best_obj, ol))
                cur.execute("UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)", (best_obj, ol))
                cur.execute("DELETE FROM objects WHERE id = %s", (prov_obj,))
                body_merged += 1

    # Delete person objects left empty by the rebuild. When map-scoped, only delete
    # empties that belonged to THIS map (existing_obj_ids = objects that had observations
    # here before the rebuild) — never touch other maps' person objects.
    if map_id is not None:
        if existing_obj_ids:
            cur.execute(
                "DELETE FROM objects o WHERE o.class_id = %s AND o.id = ANY(%s) AND NOT EXISTS "
                "(SELECT 1 FROM object_observations oo WHERE oo.object_id = o.id)",
                (YOLO_PERSON_CLASS_ID, list(existing_obj_ids)),
            )
            deleted = cur.rowcount
        else:
            deleted = 0
    else:
        cur.execute(
            "DELETE FROM objects o WHERE o.class_id = %s AND NOT EXISTS "
            "(SELECT 1 FROM object_observations oo WHERE oo.object_id = o.id)",
            (YOLO_PERSON_CLASS_ID,),
        )
        deleted = cur.rowcount
    conn.commit()

    if progress_callback:
        progress_callback(2, 4)

    # --- 4. Re-populate face_observations from staging (object deletions wiped it) ---
    # Scoped to this map's observations when map_id is set.
    cur.execute(
        """
        SELECT pfe.observation_id, oo.object_id, oo.scene_id, oo.yolo_track_id,
               pfe.embedding::text, pfe.det_score
        FROM person_face_embeddings pfe
        JOIN object_observations oo ON oo.id = pfe.observation_id
        LEFT JOIN face_observations fo ON fo.observation_id = pfe.observation_id
        WHERE pfe.embedding IS NOT NULL AND fo.id IS NULL
        """ + oo_map_sql + """
        ORDER BY oo.object_id, pfe.observation_id
        """,
        oo_map_params,
    )
    face_rows = cur.fetchall()
    cluster_person = {}
    faces_inserted = 0
    for oid, object_id, scene_id, track_id, emb_text, det_score in face_rows:
        if object_id not in cluster_person:
            cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (YOLO_PERSON_CLASS_ID,))
            cluster_person[object_id] = cur.fetchone()[0]
        person_id = cluster_person[object_id]
        cur.execute(
            """
            INSERT INTO face_observations
                (person_id, scene_id, object_id, yolo_track_id, embedding, score, observation_id)
            VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
            """,
            (person_id, scene_id, object_id, track_id, emb_text, det_score, oid),
        )
        faces_inserted += 1
        if faces_inserted % 500 == 0:
            conn.commit()
    conn.commit()

    if progress_callback:
        progress_callback(4, 4)

    return {
        "identities": len(ident_to_obj),
        "observations_moved": moved,
        "provisional_clusters": provisional,
        "provisional_body_merged": body_merged,
        "empty_objects_deleted": deleted,
        "face_observations_rebuilt": faces_inserted,
        "faced_observations": len(face_emb),
    }


def _generate_face_embeddings_runtime(progress_callback=None, max_wait_sec: int = 1800, refind_faces: bool = False, map_id: int | None = None) -> dict:
    """Run InsightFace face-embedding generation over person crops in the runtime
    (bordsupr) container, so the face rebuild has embeddings for the newest observations.
    The script (/workspace/src/gen_face_embeddings.py, mounted from the runtime tree) is
    resumable: by default it processes only observations with no staging row. When
    refind_faces is True it ALSO re-processes observations that have a NULL-embedding
    staging row (previously 'no face found'), so a face detected this time updates the row.
    Started in the background and polled to completion. Returns a small status dict."""
    import time as _time

    # Count observations to process first (for progress + early-exit). Scoped to the
    # selected map when map_id is set.
    _map_sql = ""
    _map_params: list = []
    if map_id is not None:
        _map_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
        _map_params = [map_id, map_id]
    try:
        conn = get_conn()
        cur = conn.cursor()
        if refind_faces:
            cur.execute(
                """
                SELECT COUNT(*) FROM object_observations oo
                LEFT JOIN person_face_embeddings pfe ON pfe.observation_id = oo.id
                WHERE oo.class_id = %s AND oo.cropped_image IS NOT NULL
                  AND (pfe.observation_id IS NULL OR pfe.embedding IS NULL)
                """ + _map_sql,
                [YOLO_PERSON_CLASS_ID, *_map_params],
            )
        else:
            cur.execute(
                """
                SELECT COUNT(*) FROM object_observations oo
                LEFT JOIN person_face_embeddings pfe ON pfe.observation_id = oo.id
                WHERE oo.class_id = %s AND pfe.observation_id IS NULL
                """ + _map_sql,
                [YOLO_PERSON_CLASS_ID, *_map_params],
            )
        pending = int(cur.fetchone()[0])
        conn.close()
    except Exception:
        pending = -1
    if pending == 0:
        return {"ran": False, "pending": 0, "reason": "no unprocessed observations"}

    # Start generation in the background inside the runtime container.
    # LD_LIBRARY_PATH points at the pip-installed CUDA-12 nvidia libs so that
    # onnxruntime-gpu's CUDAExecutionProvider can load (gen_face_embeddings.py
    # prefers CUDA and falls back to CPU if unavailable).
    refind_flag = " --refind-null" if refind_faces else ""
    map_flag = f" --map-id {int(map_id)}" if map_id is not None else ""
    _nvidia_lib_path = (
        "/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_runtime/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/curand/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/cufft/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/cuda_nvrtc/lib:"
        "/usr/local/lib/python3.10/dist-packages/nvidia/nvjitlink/lib"
    )
    start = _docker_exec(
        "rm -f /tmp/gen_face_run_web.log; "
        f"nohup env PGHOST=db PGPORT=5432 LD_LIBRARY_PATH={_nvidia_lib_path} "
        f"python3 /workspace/src/gen_face_embeddings.py{refind_flag}{map_flag} "
        "> /tmp/gen_face_run_web.log 2>&1 & echo started"
    )
    if start.returncode != 0:
        raise RuntimeError(f"Failed to start face embedding generation: {start.stderr or start.stdout}")

    # Poll until the process exits or we time out. NOTE: the pgrep pattern uses a
    # bracket ([.] instead of .) so that pgrep does NOT match the `bash -lc "pgrep -f
    # gen_face_embeddings[.]py ..."` wrapper process that _docker_exec spawns to run the
    # poll itself (which would otherwise look like a running generator and loop forever).
    waited = 0
    interval = 5
    while waited < max_wait_sec:
        proc = _docker_exec("pgrep -f 'gen_face_embeddings[.]py' | head -1")
        running = bool((proc.stdout or "").strip())
        if progress_callback:
            progress_callback(waited, max_wait_sec, pending)
        if not running:
            break
        _time.sleep(interval)
        waited += interval

    tail = _docker_exec("grep -vE 'FutureWarning|rcond|lstsq|Affine' /tmp/gen_face_run_web.log | tail -2")
    return {"ran": True, "pending": pending, "waited_sec": waited, "log_tail": (tail.stdout or "").strip()}


def _run_face_recluster_job(job_id: str, face_threshold: float, min_identity_faces: int, body_floor: float, body_assign: bool, body_recluster: bool, body_merge_sim: float, body_merge_margin: float, generate_faces: bool, refind_faces: bool, map_id: int | None = None) -> None:
    import time as _time
    with _FACE_RECLUSTER_JOBS_LOCK:
        _FACE_RECLUSTER_JOBS[job_id]["status"] = "running"
        _FACE_RECLUSTER_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    def progress_callback(done: int, total: int):
        with _FACE_RECLUSTER_JOBS_LOCK:
            job = _FACE_RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["steps_done"] = done
                job["total_steps"] = total
    try:
        gen_info = None
        if generate_faces:
            with _FACE_RECLUSTER_JOBS_LOCK:
                _FACE_RECLUSTER_JOBS[job_id]["phase"] = "generating face embeddings"
            gen_info = _generate_face_embeddings_runtime(refind_faces=refind_faces, map_id=map_id)
        with _FACE_RECLUSTER_JOBS_LOCK:
            _FACE_RECLUSTER_JOBS[job_id]["phase"] = "reclustering"
        result = _run_full_face_rebuild(face_threshold, min_identity_faces, body_floor, body_assign, body_recluster, body_merge_sim, body_merge_margin, progress_callback, map_id=map_id)
        if gen_info is not None:
            result["face_generation"] = gen_info
        with _FACE_RECLUSTER_JOBS_LOCK:
            job = _FACE_RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["result"] = result
    except Exception as exc:
        with _FACE_RECLUSTER_JOBS_LOCK:
            job = _FACE_RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = str(exc)


@app.post("/api/objects/recluster-faces")
def recluster_faces(
    face_threshold: float = Query(default=0.30, ge=0.0, le=1.0),
    min_identity_faces: int = Query(default=3, ge=1),
    body_floor: float = Query(default=0.70, ge=0.0, le=1.0),
    body_assign: bool = Query(default=False),
    body_recluster: bool = Query(default=True),
    body_merge_sim: float = Query(default=0.70, ge=0.0, le=1.0),
    body_merge_margin: float = Query(default=0.02, ge=0.0, le=1.0),
    generate_faces: bool = Query(default=True),
    refind_faces: bool = Query(default=False),
    building: str | None = Query(default=None),
):
    """Rebuild all PERSON clusters from offline ArcFace face embeddings, then (optionally)
    body-recluster the leftover provisional clusters into the mains.

    Stage 1 (generate_faces, default True): run InsightFace in the runtime container over
    any person observations not yet in person_face_embeddings, so the newest observations
    get face embeddings before the rebuild (otherwise they stay faceless). When
    refind_faces is also True (default False), Stage 1 additionally re-runs detection on
    observations that already have a NULL-embedding staging row (previously 'no face
    found'), so any face detected this time is recovered. Note: most NULL rows are
    genuinely back-of-head/occluded, so refind usually recovers few faces and is slower.
    Stage 2: the face rebuild. Faces are far more discriminative than OSNet body
    embeddings, so the face rebuild produces pure mains. body_assign (default False)
    force-assigns unlinked faceless obs by raw body similarity (kept off: it contaminates).
    body_recluster (default True) is the safer second pass: it merges a provisional cluster
    into a main only when its body centroid is close (>= body_merge_sim) AND clearly closer
    than to any other main (margin >= body_merge_margin), which rejects mixed/ambiguous
    fragments. Returns a job_id; poll GET /api/objects/recluster-faces/jobs/{job_id} for progress."""
    import uuid, threading
    map_id = _resolve_map_id(building)
    job_id = f"frcl-{uuid.uuid4().hex[:8]}"
    with _FACE_RECLUSTER_JOBS_LOCK:
        _FACE_RECLUSTER_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "face_threshold": face_threshold,
            "min_identity_faces": min_identity_faces,
            "body_floor": body_floor,
            "body_assign": body_assign,
            "body_recluster": body_recluster,
            "body_merge_sim": body_merge_sim,
            "body_merge_margin": body_merge_margin,
            "generate_faces": generate_faces,
            "refind_faces": refind_faces,
            "building": building,
            "map_id": map_id,
            "steps_done": 0,
            "total_steps": 4,
            "phase": "queued",
        }
    threading.Thread(
        target=_run_face_recluster_job,
        args=(job_id, face_threshold, min_identity_faces, body_floor, body_assign, body_recluster, body_merge_sim, body_merge_margin, generate_faces, refind_faces, map_id),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running", "building": building, "map_id": map_id}


@app.get("/api/objects/recluster-faces/jobs/{job_id}")
def get_face_recluster_job(job_id: str):
    with _FACE_RECLUSTER_JOBS_LOCK:
        job = _FACE_RECLUSTER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _run_recluster_job(job_id: str, similarity_threshold: float, class_id: int | None, object_similarity_threshold: float | None = None, exclude_persons: bool = False) -> None:
    """Background runner that updates _RECLUSTER_JOBS with progress."""
    import time as _time

    with _RECLUSTER_JOBS_LOCK:
        _RECLUSTER_JOBS[job_id]["status"] = "running"
        _RECLUSTER_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    def progress_callback(done: int, total: int):
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["classes_done"] = done
                job["total_classes"] = total

    try:
        result = _run_full_recluster(similarity_threshold, class_id, progress_callback, object_similarity_threshold, exclude_persons)
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["result"] = result
                job["classes_done"] = result["classes_processed"]
                job["total_classes"] = result["classes_processed"]
    except Exception as exc:
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = str(exc)


@app.get("/api/objects/recluster/jobs/{job_id}")
def get_recluster_job(job_id: str):
    with _RECLUSTER_JOBS_LOCK:
        job = _RECLUSTER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/objects/recluster/jobs")
def list_recluster_jobs():
    with _RECLUSTER_JOBS_LOCK:
        jobs = list(_RECLUSTER_JOBS.values())
    # Return newest first, limit to last 20
    jobs_sorted = sorted(jobs, key=lambda j: j.get("started_at", ""), reverse=True)[:20]
    return {"jobs": jobs_sorted}


@app.get("/api/objects/pending-consolidation")
def get_pending_consolidation(
    similarity_threshold: float = Query(default=0.643, ge=0.0, le=1.0),
    min_observations: int = Query(default=1, ge=1),
    building: str | None = Query(default=None),
):
    """Return object IDs that would be merged if consolidation ran now.

    Computes centroid-to-centroid cosine similarity for all same-class cluster pairs
    and returns pairs above the threshold, along with the set of affected object IDs.
    """
    map_id = _resolve_map_id(building)
    map_filter = ""
    map_params: list = []
    if map_id is not None:
        map_filter = "AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
        map_params = [map_id, map_id]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH centroids AS (
                    SELECT
                        o.id AS object_id,
                        o.class_id,
                        AVG(oo.embedding) AS centroid,
                        COUNT(*) AS n_obs
                    FROM objects o
                    JOIN object_observations oo ON oo.object_id = o.id
                    {map_filter.replace("AND", "WHERE", 1) if not map_filter.startswith("WHERE") else map_filter}
                    GROUP BY o.id, o.class_id
                    HAVING COUNT(*) >= %s
                      AND AVG(oo.embedding) IS NOT NULL
                )
                SELECT
                    c1.object_id AS obj_a,
                    c2.object_id AS obj_b,
                    c1.n_obs AS n_a,
                    c2.n_obs AS n_b,
                    c1.class_id,
                    round((1.0 - (c1.centroid <=> c2.centroid))::numeric, 4) AS similarity
                FROM centroids c1
                JOIN centroids c2
                  ON c1.object_id < c2.object_id
                 AND c1.class_id = c2.class_id
                WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= %s
                ORDER BY similarity DESC
                """,
                tuple(map_params + [min_observations, similarity_threshold]),
            )
            pairs = cur.fetchall()

    pair_list = [
        {
            "obj_a": row[0],
            "obj_b": row[1],
            "n_a": row[2],
            "n_b": row[3],
            "class_id": row[4],
            "similarity": float(row[5]),
        }
        for row in pairs
    ]
    pending_ids = set()
    for p in pair_list:
        pending_ids.add(p["obj_a"])
        pending_ids.add(p["obj_b"])

    return {
        "similarity_threshold": similarity_threshold,
        "pair_count": len(pair_list),
        "pending_object_ids": list(pending_ids),
        "pairs": pair_list,
    }


def _run_single_merge_pass(
    similarity_threshold: float,
    min_observations: int,
) -> tuple[list[dict], list[dict]]:
    """Run one merge pass. Returns (merged, skipped)."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH centroids AS (
                    SELECT
                        o.id AS object_id,
                        o.class_id,
                        AVG(oo.embedding) AS centroid,
                        COUNT(*) AS n_obs,
                        MIN(oo.created_at) AS first_seen
                    FROM objects o
                    JOIN object_observations oo ON oo.object_id = o.id
                    GROUP BY o.id, o.class_id
                    HAVING COUNT(*) >= %s
                      AND AVG(oo.embedding) IS NOT NULL
                )
                SELECT
                    c1.object_id AS obj_a,
                    c2.object_id AS obj_b,
                    c1.n_obs AS n_a,
                    c2.n_obs AS n_b,
                    c1.first_seen AS first_a,
                    c2.first_seen AS first_b,
                    round((1.0 - (c1.centroid <=> c2.centroid))::numeric, 4) AS similarity
                FROM centroids c1
                JOIN centroids c2
                  ON c1.object_id < c2.object_id
                 AND c1.class_id = c2.class_id
                WHERE (1.0 - (c1.centroid <=> c2.centroid)) >= %s
                ORDER BY similarity DESC
                """,
                (min_observations, similarity_threshold),
            )
            pairs = cur.fetchall()

    merged: list[dict] = []
    skipped: list[dict] = []
    already_gone: set[int] = set()

    for row in pairs:
        obj_a, obj_b, n_a, n_b, first_a, first_b, similarity = row
        if obj_a in already_gone or obj_b in already_gone:
            skipped.append({"obj_a": obj_a, "obj_b": obj_b, "reason": "already merged this pass"})
            continue

        if n_a > n_b:
            keep_id, drop_id = obj_a, obj_b
        elif n_b > n_a:
            keep_id, drop_id = obj_b, obj_a
        elif (first_a or "") <= (first_b or ""):
            keep_id, drop_id = obj_a, obj_b
        else:
            keep_id, drop_id = obj_b, obj_a

        with get_conn() as conn2:
            with conn2.cursor() as cur2:
                cur2.execute(
                    "UPDATE object_observations SET object_id = %s WHERE object_id = %s",
                    (keep_id, drop_id),
                )
                moved = cur2.rowcount
                cur2.execute("DELETE FROM objects WHERE id = %s", (drop_id,))
            conn2.commit()

        already_gone.add(drop_id)
        merged.append({
            "keep": keep_id,
            "drop": drop_id,
            "observations_moved": moved,
            "similarity": float(similarity),
        })

    return merged, skipped


def _run_convergent_merge(
    similarity_threshold: float,
    min_observations: int,
    max_passes: int = 20,
) -> dict:
    """Iterate merge passes until no new merges occur (convergence) or max_passes is reached."""
    all_merged: list[dict] = []
    all_skipped: list[dict] = []
    for pass_num in range(1, max_passes + 1):
        merged, skipped = _run_single_merge_pass(similarity_threshold, min_observations)
        all_merged.extend(merged)
        all_skipped.extend(skipped)
        if not merged:
            break
    return {
        "merged_count": len(all_merged),
        "skipped_count": len(all_skipped),
        "merged": all_merged,
        "skipped": all_skipped,
    }


@app.post("/api/objects/merge-pending")
def merge_pending_consolidation(
    similarity_threshold: float = Query(default=0.643, ge=0.0, le=1.0),
    min_observations: int = Query(default=1, ge=1),
    max_passes: int = Query(default=20, ge=1, le=100),
):
    """Merge same-class cluster pairs above the similarity threshold, iterating until convergence.

    Uses the same keep/drop rule as the ROS consolidation job:
    keep the cluster with more observations; break ties by older first_seen.
    Repeats passes until no new merges occur or max_passes is reached.
    """
    return _run_convergent_merge(similarity_threshold, min_observations, max_passes)


# ---------------------------------------------------------------------------
# Embedding recomputation helpers (match ROS service preprocessing)
# ---------------------------------------------------------------------------

def _embed_osnet_finetuned_from_pil(image: PILImage.Image) -> list[float]:
    """Compute OSNet finetuned improved embedding from a PIL image.
    Replicates the white-balance + auto-brightness + resize preprocessing
    used by the runtime osnet_embedding_service."""
    import cv2
    import numpy as np
    import torch

    model = _load_osnet_finetuned_improved_model()
    if model is None:
        raise RuntimeError(f"OSNet finetuned improved model not available: {_OSNET_FINETUNED_IMPROVED_MODEL_ERROR}")

    rgb = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    # White balance (same as OSNet service)
    arr = rgb.astype(np.float32)
    mean_r = float(arr[:, :, 2].mean())
    mean_g = float(arr[:, :, 1].mean())
    mean_b = float(arr[:, :, 0].mean())
    if mean_g > 0 and mean_r > 0 and mean_b > 0:
        arr[:, :, 2] = np.clip(arr[:, :, 2] * (mean_g / mean_r), 0, 255)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * (mean_g / mean_b), 0, 255)
    rgb = arr.astype(np.uint8)

    # Auto brightness (same as OSNet service)
    mean_brightness = float(rgb.mean())
    if 0.0 < mean_brightness < 80.0:
        scale = min(2.0, 128.0 / mean_brightness)
        rgb = np.clip(rgb.astype(np.float32) * scale, 0, 255).astype(np.uint8)

    # Resize and normalize
    rgb = cv2.resize(rgb, (128, 256), interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
    device = next(model.parameters()).device
    tensor = tensor.to(device)
    with torch.inference_mode():
        emb = model(tensor)[0].detach().float().cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def _embed_dinov3_from_pil(image: PILImage.Image) -> list[float]:
    """Compute DINOv3 embedding from a PIL image with multi-view augmentation.
    Replicates the center-crop + border-suppression logic used by the
    runtime dinov3_embedding_service."""
    import cv2
    import numpy as np
    import torch

    model, processor = _load_dinov3_model()
    if model is None:
        raise RuntimeError(f"DINOv3 model not available: {_DINOV3_MODEL_ERROR}")

    rgb = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    # Multi-view augmentation (same defaults as DINOv3 service)
    views = [rgb]
    height, width = rgb.shape[:2]
    if min(height, width) >= 72:
        crop_h = int(height * 0.82)
        crop_w = int(width * 0.82)
        y0 = max(0, (height - crop_h) // 2)
        x0 = max(0, (width - crop_w) // 2)
        y1 = min(height, y0 + crop_h)
        x1 = min(width, x0 + crop_w)
        center_crop = rgb[y0:y1, x0:x1]
        if center_crop.size > 0:
            views.append(center_crop)

        border_h = int(round(height * 0.12))
        border_w = int(round(width * 0.12))
        if border_h > 0 or border_w > 0:
            blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=6.0, sigmaY=6.0)
            focused = rgb.copy()
            if border_h > 0:
                focused[:border_h, :, :] = blurred[:border_h, :, :]
                focused[height - border_h:, :, :] = blurred[height - border_h:, :, :]
            if border_w > 0:
                focused[:, :border_w, :] = blurred[:, :border_w, :]
                focused[:, width - border_w:, :] = blurred[:, width - border_w:, :]
            views.append(focused)

    device = next(model.parameters()).device
    vectors = []
    for view in views:
        inputs = processor(images=view, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            emb = outputs.pooler_output[0]
        else:
            emb = outputs.last_hidden_state[:, 0, :][0]
        vectors.append(emb.detach().float().cpu().numpy().astype(np.float32))

    emb = np.mean(np.stack(vectors, axis=0), axis=0)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb.tolist()


def _run_recompute_embeddings_job(job_id: str, class_id_filter: int | None) -> None:
    """Background runner that recomputes embeddings for all observations."""
    import io, time as _time

    with _RECOMPUTE_JOBS_LOCK:
        _RECOMPUTE_JOBS[job_id]["status"] = "running"
        _RECOMPUTE_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    total_updated = 0
    total_skipped = 0
    total_errors = 0
    first_error: str | None = None

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                if class_id_filter is not None:
                    cur.execute(
                        "SELECT COUNT(*) FROM object_observations WHERE class_id = %s AND LENGTH(cropped_image) > 100",
                        (class_id_filter,),
                    )
                else:
                    cur.execute("SELECT COUNT(*) FROM object_observations WHERE LENGTH(cropped_image) > 100")
                total_rows = cur.fetchone()[0]

        with _RECOMPUTE_JOBS_LOCK:
            _RECOMPUTE_JOBS[job_id]["total_observations"] = total_rows
            _RECOMPUTE_JOBS[job_id]["observations_done"] = 0

        batch_size = 50
        offset = 0
        while True:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    if class_id_filter is not None:
                        cur.execute(
                            """
                            SELECT id, class_id, cropped_image
                            FROM object_observations
                            WHERE class_id = %s AND LENGTH(cropped_image) > 100
                            ORDER BY id
                            LIMIT %s OFFSET %s
                            """,
                            (class_id_filter, batch_size, offset),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT id, class_id, cropped_image
                            FROM object_observations
                            WHERE LENGTH(cropped_image) > 100
                            ORDER BY id
                            LIMIT %s OFFSET %s
                            """,
                            (batch_size, offset),
                        )
                    rows = cur.fetchall()

            if not rows:
                break

            for obs_id, class_id, cropped_bytes in rows:
                try:
                    image = PILImage.open(io.BytesIO(bytes(cropped_bytes)))
                    # Person class (0) uses OSNet finetuned improved; everything else uses DINOv3
                    if class_id == 0:
                        embedding = _embed_osnet_finetuned_from_pil(image)
                    else:
                        embedding = _embed_dinov3_from_pil(image)

                    with get_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE object_observations SET embedding = %s::vector WHERE id = %s",
                                (json.dumps(embedding), obs_id),
                            )
                        conn.commit()
                    total_updated += 1
                except Exception as exc:
                    total_errors += 1
                    if first_error is None:
                        first_error = str(exc)

            offset += len(rows)
            with _RECOMPUTE_JOBS_LOCK:
                job = _RECOMPUTE_JOBS.get(job_id)
                if job is not None:
                    job["observations_done"] = offset

        with _RECOMPUTE_JOBS_LOCK:
            job = _RECOMPUTE_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["result"] = {
                    "observations_updated": total_updated,
                    "observations_skipped": total_skipped,
                    "observations_errored": total_errors,
                    "first_error": first_error,
                }
    except Exception as exc:
        with _RECOMPUTE_JOBS_LOCK:
            job = _RECOMPUTE_JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = str(exc)


@app.get("/api/objects/recompute-embeddings/jobs/{job_id}")
def get_recompute_job(job_id: str):
    with _RECOMPUTE_JOBS_LOCK:
        job = _RECOMPUTE_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/objects/recompute-embeddings/jobs")
def list_recompute_jobs():
    with _RECOMPUTE_JOBS_LOCK:
        jobs = list(_RECOMPUTE_JOBS.values())
    jobs_sorted = sorted(jobs, key=lambda j: j.get("started_at", ""), reverse=True)[:20]
    return {"jobs": jobs_sorted}


# ---------------------------------------------------------------------------
# Recluster from scratch
# ---------------------------------------------------------------------------

def _run_full_recluster(
    similarity_threshold: float = 0.643,
    class_id_filter: int | None = None,
    progress_callback: Callable | None = None,
    object_similarity_threshold: float | None = None,
    exclude_persons: bool = False,
) -> dict:
    """Re-cluster ALL observations from scratch using greedy cosine thresholding.

    Unlike merge-pending (merge-only), this can also SPLIT clusters:
    observations previously grouped into one object may be reassigned to separate
    objects if their embeddings diverge.

    Algorithm per class:
    1. Fetch all observations (with embeddings) sorted by created_at.
    2. Run greedy cosine clustering to get new cluster labels.
    3. Match each new cluster to an existing object via majority vote.
    4. Largest cluster gets first pick of its majority object.
    5. Clusters whose majority is already claimed get a new object created.
    6. Reassign observations and delete orphaned objects.

    When object_similarity_threshold is set, non-person classes use that
    (typically lower) threshold while persons keep similarity_threshold —
    object embeddings are less viewpoint-robust and over-fragment otherwise.
    """
    import json as _json

    with get_conn() as conn:
        with conn.cursor() as cur:
            if class_id_filter is not None:
                cur.execute(
                    "SELECT DISTINCT class_id FROM objects WHERE class_id = %s ORDER BY class_id",
                    (class_id_filter,),
                )
            elif exclude_persons:
                cur.execute(
                    "SELECT DISTINCT class_id FROM objects WHERE class_id != %s ORDER BY class_id",
                    (YOLO_PERSON_CLASS_ID,),
                )
            else:
                cur.execute("SELECT DISTINCT class_id FROM objects ORDER BY class_id")
            class_ids = [row[0] for row in cur.fetchall()]

    total_objects_created = 0
    total_objects_deleted = 0
    total_observations_reassigned = 0

    for idx, cid in enumerate(class_ids):
        is_static_class = cid != YOLO_PERSON_CLASS_ID and cid not in YOLO_PORTABLE_CLASS_IDS
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT oo.id, oo.object_id, oo.embedding::text, oo.map_id, oo.x, oo.y, oo.z
                    FROM object_observations oo
                    JOIN objects o ON o.id = oo.object_id
                    WHERE o.class_id = %s AND oo.embedding IS NOT NULL
                    ORDER BY oo.created_at ASC
                    """,
                    (cid,),
                )
                rows = cur.fetchall()

        if not rows:
            if progress_callback:
                progress_callback(idx + 1, len(class_ids))
            continue

        obs_ids = [r[0] for r in rows]
        orig_obj_ids = [r[1] for r in rows]
        embeddings = [_json.loads(r[2]) for r in rows]
        # 3D position with map separation baked into x (different maps are always
        # far apart, so cross-map observations never share a position sub-cluster).
        positions: list[tuple[float, float, float] | None] = [
            (
                float(r[4]) + (float(r[3]) * 1.0e6 if r[3] is not None else 0.0),
                float(r[5]),
                float(r[6]),
            )
            if r[4] is not None and r[5] is not None and r[6] is not None
            else None
            for r in rows
        ]

        cluster_threshold = similarity_threshold
        if cid != YOLO_PERSON_CLASS_ID and object_similarity_threshold is not None:
            cluster_threshold = object_similarity_threshold
        labels = _cluster_features(embeddings, cluster_threshold)

        # Group observation indices by new cluster label
        label_to_indices: dict[int, list[int]] = {}
        for i, lab in enumerate(labels):
            label_to_indices.setdefault(lab, []).append(i)

        # For STATIC classes (chairs, tables, ...), split each embedding-cluster by
        # position so two look-alike objects at different locations are NOT merged
        # into one. Portable classes and persons keep pure-embedding clustering.
        # The map_id is folded into the position key (scaled up) so observations
        # from different maps can never share a position sub-cluster.
        if is_static_class:
            split_label_to_indices: dict[int, list[int]] = {}
            next_label = 0
            for lab in sorted(label_to_indices.keys()):
                sub_groups = _split_cluster_by_position(
                    label_to_indices[lab], positions, RECLUSTER_STATIC_POSITION_THRESHOLD_M
                )
                for grp in sub_groups:
                    split_label_to_indices[next_label] = grp
                    next_label += 1
            label_to_indices = split_label_to_indices
            # Merge clusters whose position centroids are nearly identical (same
            # physical object seen from divergent viewpoints), even if embeddings
            # fell below the similarity threshold.
            label_to_indices = _merge_clusters_by_proximity(
                label_to_indices, positions, RECLUSTER_STATIC_MERGE_DISTANCE_M
            )

        # Majority vote: which existing object_id dominates each new cluster?
        label_to_majority: dict[int, int] = {}
        for lab, indices in label_to_indices.items():
            counts: dict[int, int] = {}
            for i in indices:
                oid = orig_obj_ids[i]
                counts[oid] = counts.get(oid, 0) + 1
            label_to_majority[lab] = max(counts, key=counts.get)

        # Assign existing objects: largest clusters claim first
        sorted_labels = sorted(label_to_indices.keys(), key=lambda l: -len(label_to_indices[l]))
        claimed: set[int] = set()
        label_to_assigned_obj: dict[int, int | None] = {}
        for lab in sorted_labels:
            maj = label_to_majority[lab]
            if maj not in claimed:
                label_to_assigned_obj[lab] = maj
                claimed.add(maj)
            else:
                label_to_assigned_obj[lab] = None  # needs a new object

        # Create new objects for unclaimed clusters
        for lab in sorted_labels:
            if label_to_assigned_obj[lab] is not None:
                continue
            with get_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (cid,))
                    new_id = cur.fetchone()[0]
                conn.commit()
            label_to_assigned_obj[lab] = new_id
            total_objects_created += 1

        # Collect reassignments (skip obs already assigned to the right object)
        new_obj_to_obs_ids: dict[int, list[int]] = {}
        for lab, indices in label_to_indices.items():
            target = label_to_assigned_obj[lab]
            for i in indices:
                if orig_obj_ids[i] != target:
                    new_obj_to_obs_ids.setdefault(target, []).append(obs_ids[i])

        if new_obj_to_obs_ids:
            with get_conn() as conn:
                with conn.cursor() as cur:
                    for target_obj, obs_id_list in new_obj_to_obs_ids.items():
                        cur.execute(
                            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
                            (target_obj, obs_id_list),
                        )
                        total_observations_reassigned += cur.rowcount
                conn.commit()

        # Delete objects with no remaining observations
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM objects
                    WHERE class_id = %s
                      AND id NOT IN (
                          SELECT DISTINCT object_id FROM object_observations
                          WHERE object_id IS NOT NULL
                      )
                    """,
                    (cid,),
                )
                total_objects_deleted += cur.rowcount
            conn.commit()

        if progress_callback:
            progress_callback(idx + 1, len(class_ids))

    return {
        "objects_created": total_objects_created,
        "objects_deleted": total_objects_deleted,
        "observations_reassigned": total_observations_reassigned,
        "classes_processed": len(class_ids),
    }


def _run_recluster_job(job_id: str, similarity_threshold: float, class_id: int | None, object_similarity_threshold: float | None = None, exclude_persons: bool = False) -> None:
    """Background runner that updates _RECLUSTER_JOBS with progress."""
    import time as _time

    with _RECLUSTER_JOBS_LOCK:
        _RECLUSTER_JOBS[job_id]["status"] = "running"
        _RECLUSTER_JOBS[job_id]["started_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")

    def progress_callback(done: int, total: int):
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["classes_done"] = done
                job["total_classes"] = total

    try:
        result = _run_full_recluster(similarity_threshold, class_id, progress_callback, object_similarity_threshold, exclude_persons)
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "done"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["result"] = result
                job["classes_done"] = result["classes_processed"]
                job["total_classes"] = result["classes_processed"]
    except Exception as exc:
        with _RECLUSTER_JOBS_LOCK:
            job = _RECLUSTER_JOBS.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
                job["error"] = str(exc)


@app.post("/api/objects/recluster")
def recluster_objects(
    similarity_threshold: float = Query(default=0.643, ge=0.0, le=1.0),
    class_id: int | None = Query(default=None),
    object_similarity_threshold: float | None = Query(default=None, ge=0.0, le=1.0),
    exclude_persons: bool = Query(default=False),
):
    """Re-cluster all observations from scratch. Can both merge AND split clusters.
    When object_similarity_threshold is set, non-person classes are clustered with
    that threshold while persons keep similarity_threshold. When exclude_persons is
    true, person (class_id 0) clusters are left untouched (use recluster-faces for those).
    Returns immediately with a job_id; poll GET /api/objects/recluster/jobs/{job_id} for progress."""
    import uuid, threading
    job_id = f"rcl-{uuid.uuid4().hex[:8]}"
    with _RECLUSTER_JOBS_LOCK:
        _RECLUSTER_JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "similarity_threshold": similarity_threshold,
            "object_similarity_threshold": object_similarity_threshold,
            "class_id": class_id,
            "exclude_persons": exclude_persons,
            "classes_done": 0,
            "total_classes": 0,
        }
    threading.Thread(
        target=_run_recluster_job,
        args=(job_id, similarity_threshold, class_id, object_similarity_threshold, exclude_persons),
        daemon=True,
    ).start()
    return {"job_id": job_id, "status": "running"}


@app.get("/api/objects/recluster/jobs/{job_id}")
def get_recluster_job(job_id: str):
    with _RECLUSTER_JOBS_LOCK:
        job = _RECLUSTER_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/clusters/{object_id}/similar")
def get_similar_clusters(
    object_id: str,
    limit: int | None = Query(default=20, ge=1, le=5000),
    min_observations: int = Query(default=1, ge=1),
    building: str | None = Query(default=None),
):
    safe_limit = None if limit is None else max(1, min(limit, 5000))
    map_id = _resolve_map_id(building)
    building_where = ""
    building_and = ""
    params: list = []
    if map_id is not None:
        building_where = "WHERE map_id = %s"
        building_and = "AND map_id = %s"
        params.append(map_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH cluster_centroids AS (
                    SELECT object_id, AVG(embedding) AS centroid
                    FROM object_observations
                    {building_where}
                    GROUP BY object_id
                    HAVING COUNT(*) >= %s
                ),
                selected AS (
                    SELECT centroid
                    FROM cluster_centroids
                    WHERE object_id::text = %s
                ),
                ranked AS (
                    SELECT
                        c.object_id,
                        (1.0 - (c.centroid <=> s.centroid)) AS similarity
                    FROM cluster_centroids c
                    CROSS JOIN selected s
                    WHERE c.object_id::text <> %s
                    ORDER BY similarity DESC
                    LIMIT %s
                )
                SELECT
                    r.object_id,
                    r.similarity,
                    o.class_id,
                    oo.yolo_track_id,
                    o.created_at,
                    oo.id AS observation_id,
                    oo.created_at AS observation_created_at,
                    obs_person.person_id
                FROM ranked r
                JOIN objects o ON o.id = r.object_id
                LEFT JOIN LATERAL (
                    SELECT id, object_id, scene_id, created_at, yolo_track_id
                    FROM object_observations oo
                    WHERE oo.object_id = r.object_id
                      {building_and}
                    ORDER BY COALESCE(oo.quality_score, 0) DESC, oo.created_at DESC
                    LIMIT 1
                ) oo ON TRUE
                LEFT JOIN LATERAL (
                    SELECT fo.person_id
                    FROM face_observations fo
                    WHERE fo.object_id = oo.object_id
                       OR (
                            fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                            AND fo.yolo_track_id IS NOT NULL
                            AND oo.yolo_track_id IS NOT NULL
                            AND fo.yolo_track_id = oo.yolo_track_id
                       )
                    ORDER BY fo.created_at DESC, fo.id DESC
                    LIMIT 1
                ) obs_person ON TRUE
                ORDER BY r.similarity DESC
                """,
                tuple(params + [min_observations, object_id, object_id, safe_limit] + (params if map_id is not None else [])),
            )
            rows = cur.fetchall()

            cur.execute(
                f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM object_observations
                    WHERE object_id::text = %s
                      {building_and}
                    GROUP BY object_id
                    HAVING COUNT(*) >= %s
                )
                """,
                tuple([object_id] + (params if map_id is not None else []) + [min_observations]),
            )
            exists_row = cur.fetchone()

            cluster_ids = [str(r[0]) for r in rows if str(r[0]).strip().lower() != object_id.lower()]
            observation_rows = []
            if cluster_ids:
                obs_map_filter = " AND oo.map_id = %s" if map_id is not None else ""
                cur.execute(
                    f"""
                    SELECT
                        oo.object_id::text,
                        oo.id,
                        oo.created_at,
                        oo.yolo_track_id,
                        oo.detection_backend,
                        oo.embedding_backend,
                        obs_person.person_id
                    FROM object_observations oo
                    LEFT JOIN LATERAL (
                        SELECT fo.person_id
                        FROM face_observations fo
                        WHERE fo.object_id = oo.object_id
                           OR (
                                fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                                AND fo.yolo_track_id IS NOT NULL
                                AND oo.yolo_track_id IS NOT NULL
                                AND fo.yolo_track_id = oo.yolo_track_id
                           )
                        ORDER BY fo.created_at DESC, fo.id DESC
                        LIMIT 1
                    ) obs_person ON TRUE
                    WHERE oo.object_id::text = ANY(%s)
                      {obs_map_filter}
                    ORDER BY oo.object_id::text, COALESCE(oo.quality_score, 0) DESC, oo.created_at DESC, oo.id DESC
                    """,
                    tuple([cluster_ids] + ([map_id] if map_id is not None else [])),
                )
                observation_rows = cur.fetchall()

    selected_exists = bool(exists_row[0]) if exists_row else False
    if not selected_exists:
        raise HTTPException(status_code=404, detail="Cluster not found or has no observations")

    selected_key = object_id.lower()
    filtered_rows = [r for r in rows if str(r[0]).strip().lower() != selected_key]
    observations_by_cluster = {}
    for r in observation_rows:
        cluster_key = str(r[0])
        observations_by_cluster.setdefault(cluster_key, []).append(
            {
                "observation_id": r[1],
                "created_at": r[2],
                "yolo_track_id": r[3],
                "detection_backend": r[4],
                "embedding_backend": r[5],
                "image_url": f"/api/observations/{r[1]}/image",
            }
        )

    return {
        "selected_cluster_id": object_id,
        "min_observations": min_observations,
        "count": len(filtered_rows),
        "results": [
            {
                "cluster_id": r[0],
                "similarity": float(r[1]) if r[1] is not None else 0.0,
                "class_id": r[2],
                "class_name": class_name_from_id(r[2]),
                "yolo_track_id": r[3],
                "created_at": r[4],
                "observation_id": r[5],
                "observation_created_at": r[6],
                "person_id": r[7],
                "image_url": f"/api/observations/{r[5]}/image" if r[5] is not None else None,
                "observation_count": len(observations_by_cluster.get(str(r[0]), [])),
                "observations": observations_by_cluster.get(str(r[0]), []),
            }
            for r in filtered_rows
        ],
    }


@app.get("/api/observations/{observation_id}/attributes")
def get_observation_attributes(observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    oo.id,
                    oo.object_id::text,
                    oo.class_id,
                    oo.confidence,
                    oo.quality_score,
                    oo.attributes_json,
                    oo.created_at
                FROM object_observations oo
                WHERE oo.id = %s
                """,
                (observation_id,),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="Observation not found")
            cur.execute(
                """
                SELECT
                    part_name,
                    quality_score,
                    colors_json,
                    bbox_x_min,
                    bbox_y_min,
                    bbox_x_max,
                    bbox_y_max,
                    embedding IS NOT NULL AS has_embedding,
                    preprocessing_json
                FROM object_observation_parts
                WHERE observation_id = %s
                ORDER BY part_name ASC
                """,
                (observation_id,),
            )
            part_rows = cur.fetchall()

    return {
        "observation_id": row[0],
        "object_id": row[1],
        "class_id": row[2],
        "class_name": class_name_from_id(row[2]),
        "confidence": row[3],
        "quality_score": row[4],
        "attributes": row[5] or {},
        "created_at": row[6],
        "image_url": f"/api/observations/{observation_id}/image",
        "parts": [
            {
                "part_name": part[0],
                "quality_score": part[1],
                "colors": part[2] or {},
                "bbox": [part[3], part[4], part[5], part[6]],
                "has_embedding": bool(part[7]),
                "preprocessing": part[8] or {},
            }
            for part in part_rows
        ],
    }


@app.get("/api/clusters/{object_id}/attributes")
def get_cluster_attributes(
    object_id: str,
    limit: int = Query(default=20, ge=1, le=200),
    building: str | None = Query(default=None),
    auto_backfill: bool = Query(default=True),
):
    map_id = _resolve_map_id(building)
    map_filter = " AND map_id = %s" if map_id is not None else ""
    params = [object_id] + ([map_id] if map_id is not None else [])
    with get_conn() as conn:
        with conn.cursor() as cur:
            backfilled = 0
            if auto_backfill:
                backfilled = _backfill_cluster_attributes(
                    cur,
                    object_id,
                    map_id=map_id,
                    limit=max(limit, 200),
                )
                conn.commit()
            cur.execute(
                f"""
                SELECT
                    id,
                    class_id,
                    confidence,
                    quality_score,
                    attributes_json,
                    created_at
                FROM object_observations
                WHERE object_id::text = %s
                  {map_filter}
                ORDER BY COALESCE(quality_score, 0) DESC, created_at DESC
                LIMIT %s
                """,
                tuple(params + [limit]),
            )
            rows = cur.fetchall()
            cur.execute(
                f"""
                SELECT
                    COUNT(*)::bigint,
                    COUNT(*) FILTER (WHERE quality_score IS NOT NULL)::bigint,
                    COUNT(*) FILTER (WHERE attributes_json IS NOT NULL)::bigint,
                    AVG(quality_score),
                    MIN(quality_score),
                    MAX(quality_score)
                FROM object_observations
                WHERE object_id::text = %s
                  {map_filter}
                """,
                tuple(params),
            )
            stats = cur.fetchone()

    if not rows and (not stats or int(stats[0] or 0) == 0):
        raise HTTPException(status_code=404, detail="Cluster not found")

    return {
        "cluster_id": object_id,
        "building_filter": building,
        "auto_backfill": auto_backfill,
        "backfilled_observations": backfilled,
        "stats": {
            "observation_count": int(stats[0] or 0) if stats else 0,
            "observations_with_quality": int(stats[1] or 0) if stats else 0,
            "observations_with_attributes": int(stats[2] or 0) if stats else 0,
            "avg_quality_score": float(stats[3]) if stats and stats[3] is not None else None,
            "min_quality_score": float(stats[4]) if stats and stats[4] is not None else None,
            "max_quality_score": float(stats[5]) if stats and stats[5] is not None else None,
        },
        "observations": [
            {
                "observation_id": row[0],
                "class_id": row[1],
                "class_name": class_name_from_id(row[1]),
                "confidence": row[2],
                "quality_score": row[3],
                "attributes": row[4] or {},
                "created_at": row[5],
                "image_url": f"/api/observations/{row[0]}/image",
                "attributes_url": f"/api/observations/{row[0]}/attributes",
            }
            for row in rows
        ],
    }


@app.post("/api/clusters/{object_id}/attributes/backfill")
def backfill_cluster_attributes(
    object_id: str,
    limit: int = Query(default=500, ge=1, le=5000),
    building: str | None = Query(default=None),
):
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            updated = _backfill_cluster_attributes(cur, object_id, map_id=map_id, limit=limit)
        conn.commit()
    return {
        "cluster_id": object_id,
        "building_filter": building,
        "updated_observations": updated,
    }


@app.get("/api/cluster-testsets/defaults")
def get_cluster_testset_defaults():
    dataset_root = _resolve_dataset_root(CLUSTER_TESTSET_DEFAULT_DATASET_ROOT)
    return {
        "dataset_root": str(dataset_root),
        "store_dir": str(_cluster_store_dir()),
        "yolo_class_presets": {
            "all": None,
            "persons": [YOLO_PERSON_CLASS_ID],
            "objects": YOLO_OBJECT_CLASS_IDS,
        },
        "variants": [
            {"id": variant_id, **spec}
            for variant_id, spec in CLUSTER_VARIANTS.items()
        ],
    }


@app.get("/api/cluster-testsets")
def list_cluster_testsets():
    results = []
    for path in sorted(_cluster_store_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        results.append(
            {
                "id": payload.get("id"),
                "name": payload.get("name"),
                "dataset_root": payload.get("dataset_root"),
                "image_count": len(payload.get("images") or []),
                "detection_count": sum(len(image.get("detections") or []) for image in payload.get("images") or []),
                "ground_truth": _ground_truth_coverage(payload),
                "created_at": payload.get("created_at"),
                "seed": payload.get("seed"),
            }
        )
    return {"testsets": results}


@app.post("/api/cluster-testsets")
def build_cluster_testset(req: ClusterTestsetBuildRequest):
    dataset_root = _resolve_dataset_root(req.dataset_root)
    if not dataset_root.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Dataset root not found: {dataset_root}",
                "tried": _dataset_root_candidates(req.dataset_root),
                "hint": "The path must exist inside the web container. Host paths under bordsupr/runtime are mapped to /opt/bordsupr_runtime when the web container has the runtime volume mounted.",
            },
        )
    images = _discover_dataset_images(dataset_root, split=req.split, recursive=req.recursive)
    if not images:
        raise HTTPException(status_code=404, detail="No images found in dataset root")
    if req.image_count > len(images):
        raise HTTPException(status_code=400, detail=f"Requested {req.image_count} images, but only found {len(images)}")
    seed = req.seed if req.seed is not None else random.randint(1, 2_000_000_000)
    rng = random.Random(seed)
    selected = rng.sample(images, req.image_count)
    created_at = datetime.now(timezone.utc).isoformat()
    name = (req.name or f"testset-{req.image_count}-{seed}").strip()
    base_id = _safe_testset_id(f"{name}-{int(time.time())}-{uuid.uuid4().hex[:8]}")
    payload = {
        "id": base_id,
        "name": name,
        "created_at": created_at,
        "dataset_root": str(dataset_root),
        "split": req.split,
        "seed": seed,
        "image_count": len(selected),
        "ground_truth_updated_at": None,
        "images": [_testset_item_payload(path, dataset_root, idx) for idx, path in enumerate(selected)],
    }
    _write_testset(payload)
    return payload


@app.get("/api/cluster-testsets/{testset_id}")
def get_cluster_testset(testset_id: str):
    return _load_testset(testset_id)


@app.post("/api/cluster-testsets/{testset_id}/detect")
def run_cluster_testset_yolo_detection(testset_id: str, req: ClusterYoloDetectionRequest):
    testset = _load_testset(testset_id)
    model_path = str(CLUSTER_TESTSET_YOLO_MODEL or "").strip()
    if not model_path:
        raise HTTPException(
            status_code=400,
            detail="CLUSTER_TESTSET_YOLO_MODEL is not configured for the web container.",
        )
    session = _load_yolo_session(model_path)
    images = testset.get("images") or []
    total_detections = 0
    processed_images = 0
    failed_images: list[dict] = []
    now = datetime.now(timezone.utc).isoformat()
    allowed_classes = _resolve_yolo_detection_classes(req.class_preset, req.classes)

    for image_index, image in enumerate(images):
        image_path = Path(image.get("path") or "")
        if not image_path.exists() or not image_path.is_file():
            image["detections"] = []
            failed_images.append(
                {
                    "index": image_index,
                    "path": str(image_path),
                    "error": "image_not_found",
                }
            )
            continue
        try:
            detections = _run_yolo_on_image(
                str(image_path),
                session=session,
                confidence=req.confidence,
                iou_threshold=req.iou_threshold,
                max_detections=req.max_detections_per_image,
                classes=allowed_classes,
                segmentation=req.segmentation,
            )
        except Exception as exc:
            image["detections"] = []
            failed_images.append(
                {
                    "index": image_index,
                    "path": str(image_path),
                    "error": str(exc),
                }
            )
            continue

        normalized_detections = []
        existing_labels = {
            str(detection.get("id")): detection.get("label")
            for detection in image.get("detections") or []
            if detection.get("label")
        }
        for detection_index, detection in enumerate(detections):
            detection_id = f"{image.get('id') or f'img-{image_index:05d}'}-det-{detection_index:03d}"
            normalized_detections.append(
                {
                    **detection,
                    "id": detection_id,
                    "label": existing_labels.get(detection_id, {"identity": None, "camera": None, "source": "yolo_unlabeled"}),
                }
            )
        image["detections"] = normalized_detections
        total_detections += len(normalized_detections)
        processed_images += 1

    testset["images"] = images
    testset["detection_updated_at"] = now
    testset["detection_config"] = {
        "model_path": model_path,
        "confidence": req.confidence,
        "iou_threshold": req.iou_threshold,
        "max_detections_per_image": req.max_detections_per_image,
        "class_preset": req.class_preset,
        "classes": allowed_classes,
        "segmentation": req.segmentation,
    }
    _write_testset(testset)
    return {
        "id": testset.get("id"),
        "name": testset.get("name"),
        "processed_images": processed_images,
        "failed_images": failed_images,
        "detection_count": total_detections,
        "ground_truth": _ground_truth_coverage(testset),
    }


@app.get("/api/cluster-testsets/{testset_id}/preview")
def get_cluster_testset_preview(
    testset_id: str,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=500),
    labeled_only: bool = Query(default=False),
    identity: str | None = Query(default=None),
):
    testset = _load_testset(testset_id)
    images = testset.get("images") or []
    identity_filter = identity.strip() if identity else None
    identity_groups: dict[str, list[dict]] = {}
    identity_counts: dict[str, int] = {}
    all_preview_images = []
    for idx, image in enumerate(images):
        has_detections = "detections" in image
        detections = image.get("detections") or []
        if has_detections:
            filtered_dets = []
            for detection in detections:
                det_identity = (detection.get("label") or {}).get("identity")
                det_identity_str = str(det_identity).strip() if det_identity is not None else ""
                if det_identity_str:
                    identity_counts[det_identity_str] = identity_counts.get(det_identity_str, 0) + 1
                    identity_groups.setdefault(det_identity_str, [])
                    if len(identity_groups[det_identity_str]) < 6:
                        identity_groups[det_identity_str].append({
                            "id": detection.get("id"),
                            "crop_url": f"/api/cluster-testsets/{testset_id}/detections/{urllib.parse.quote(str(detection.get('id')))}",
                            "filename": image.get("filename"),
                            "class_name": detection.get("class_name"),
                        })
                include = True
                if labeled_only and not det_identity_str:
                    include = False
                if identity_filter is not None and det_identity_str != identity_filter:
                    include = False
                if include:
                    filtered_dets.append(detection)
            if not filtered_dets:
                continue
            all_preview_images.append({
                **image,
                "index": idx,
                "image_url": f"/api/cluster-testsets/{testset_id}/images/{idx}",
                "detections": [
                    {
                        **detection,
                        "crop_url": f"/api/cluster-testsets/{testset_id}/detections/{urllib.parse.quote(str(detection.get('id')))}"
                    }
                    for detection in filtered_dets
                ],
            })
            continue
        # No detections run yet — treat as single image record
        img_identity = (image.get("label") or {}).get("identity")
        img_identity_str = str(img_identity).strip() if img_identity is not None else ""
        if img_identity_str:
            identity_counts[img_identity_str] = identity_counts.get(img_identity_str, 0) + 1
            identity_groups.setdefault(img_identity_str, [])
            if len(identity_groups[img_identity_str]) < 6:
                identity_groups[img_identity_str].append({
                    "index": idx,
                    "image_url": f"/api/cluster-testsets/{testset_id}/images/{idx}",
                    "filename": image.get("filename"),
                })
        if labeled_only and not img_identity_str:
            continue
        if identity_filter is not None and img_identity_str != identity_filter:
            continue
        all_preview_images.append({
            **image,
            "index": idx,
            "image_url": f"/api/cluster-testsets/{testset_id}/images/{idx}",
            "detections": None,
        })
    total = len(all_preview_images)
    preview_images = all_preview_images[offset:offset + limit]
    return {
        "id": testset.get("id"),
        "name": testset.get("name"),
        "image_count": len(images),
        "ground_truth": _ground_truth_coverage(testset),
        "identity_groups": {k: v for k, v in sorted(identity_groups.items(), key=lambda x: x[0])},
        "identity_counts": {k: v for k, v in sorted(identity_counts.items(), key=lambda x: x[0])},
        "active_identity": identity_filter,
        "pagination": {"offset": offset, "limit": limit, "total": total, "has_next": (offset + limit) < total, "has_prev": offset > 0},
        "images": preview_images,
    }


@app.get("/api/cluster-testsets/{testset_id}/images/{image_index}")
def get_cluster_testset_image(testset_id: str, image_index: int):
    testset = _load_testset(testset_id)
    image_bytes = _image_bytes_for_testset_item(testset, image_index)
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=image_bytes, media_type=media_type)


@app.get("/api/cluster-testsets/{testset_id}/detections/{detection_id}")
def get_cluster_testset_detection_crop(testset_id: str, detection_id: str):
    testset = _load_testset(testset_id)
    image, detection = _detection_by_id(testset, detection_id)
    source = PILImage.open(image["path"]).convert("RGB")
    bbox = detection.get("bbox") or [0, 0, source.size[0], source.size[1]]
    left, top, right, bottom = [int(round(float(value))) for value in bbox]
    left = max(0, min(source.size[0] - 1, left))
    top = max(0, min(source.size[1] - 1, top))
    right = max(left + 1, min(source.size[0], right))
    bottom = max(top + 1, min(source.size[1], bottom))
    crop = source.crop((left, top, right, bottom))
    buf = io.BytesIO()
    crop.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")


@app.get("/api/cluster-testsets/detections/{detection_id}/crop")
def get_cluster_testset_detection_crop_global(detection_id: str):
    for path in _cluster_store_dir().glob("*.json"):
        try:
            testset = json.loads(path.read_text(encoding="utf-8"))
            _image, _detection = _detection_by_id(testset, detection_id)
            return get_cluster_testset_detection_crop(str(testset.get("id")), detection_id)
        except HTTPException:
            continue
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="Detection not found")


@app.post("/api/cluster-testsets/{testset_id}/ground-truth")
def update_cluster_testset_ground_truth(testset_id: str, req: ClusterGroundTruthUpdateRequest):
    testset = _load_testset(testset_id)
    images = testset.get("images") or []
    updated = 0
    for label in req.labels:
        if label.detection_id:
            _image, detection = _detection_by_id(testset, label.detection_id)
            existing = detection.get("label") or {}
            identity = _normalize_ground_truth_identity(label.identity)
            detection["label"] = {
                **existing,
                "identity": identity,
                "camera": label.camera if label.camera is not None else existing.get("camera"),
                "source": "manual" if identity is not None else "manual_empty",
                "note": (label.note or "").strip() or existing.get("note"),
            }
            updated += 1
            continue
        if label.index >= len(images):
            raise HTTPException(status_code=400, detail=f"Image index out of range: {label.index}")
        image = images[label.index]
        existing = image.get("label") or {}
        identity = _normalize_ground_truth_identity(label.identity)
        camera = label.camera if label.camera is not None else existing.get("camera")
        image["label"] = {
            **existing,
            "identity": identity,
            "camera": camera,
            "source": "manual" if identity is not None else "manual_empty",
            "note": (label.note or "").strip() or existing.get("note"),
        }
        updated += 1
    testset["ground_truth_updated_at"] = datetime.now(timezone.utc).isoformat()
    testset["images"] = images
    _write_testset(testset)
    return {
        "id": testset.get("id"),
        "name": testset.get("name"),
        "updated": updated,
        "ground_truth": _ground_truth_coverage(testset),
    }


@app.delete("/api/cluster-testsets/{testset_id}")
def delete_cluster_testset(testset_id: str):
    path = _testset_path(testset_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Testset not found")
    path.unlink()
    return {"deleted": True, "testset_id": testset_id}


@app.delete("/api/cluster-testsets/{testset_id}/ground-truth")
def clear_cluster_testset_ground_truth(testset_id: str):
    testset = _load_testset(testset_id)
    images = testset.get("images") or []
    for image in images:
        image["label"] = {"identity": None, "camera": None, "source": "manual_empty"}
        for detection in image.get("detections") or []:
            detection["label"] = {"identity": None, "camera": None, "source": "manual_empty"}
    testset["ground_truth_updated_at"] = datetime.now(timezone.utc).isoformat()
    testset["images"] = images
    _write_testset(testset)
    return {
        "id": testset.get("id"),
        "cleared": len(images),
        "ground_truth": _ground_truth_coverage(testset),
    }


@app.post("/api/cluster-testsets/{testset_id}/upscale")
def upscale_cluster_testset_images(testset_id: str, req: ClusterUpscaleRequest):
    _PIL_RESAMPLE = {
        "lanczos": PILImage.LANCZOS,
        "bicubic": PILImage.BICUBIC,
        "bilinear": PIL_RESAMPLE_BILINEAR,
        "nearest": PILImage.NEAREST,
    }
    resample = _PIL_RESAMPLE.get(req.method, PILImage.LANCZOS)
    testset = _load_testset(testset_id)
    images = testset.get("images") or []
    upscale_dir = _cluster_store_dir() / "upscaled" / testset_id
    upscale_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    failed: list[dict] = []
    for idx, image in enumerate(images):
        src_path = Path(image.get("path") or "")
        if not src_path.exists() or not src_path.is_file():
            failed.append({"index": idx, "path": str(src_path), "error": "file_not_found"})
            continue
        dest_path = upscale_dir / src_path.name
        try:
            img = PILImage.open(src_path).convert("RGB")
            new_w = img.width * req.scale
            new_h = img.height * req.scale
            img_up = img.resize((new_w, new_h), resample)
            suffix = src_path.suffix.lower()
            fmt = "PNG" if suffix == ".png" else "JPEG"
            save_kwargs = {} if fmt == "PNG" else {"quality": 95}
            img_up.save(dest_path, format=fmt, **save_kwargs)
            image["path"] = str(dest_path)
            processed += 1
        except Exception as exc:
            failed.append({"index": idx, "path": str(src_path), "error": str(exc)})
            skipped += 1
    testset["images"] = images
    testset["upscale_config"] = {
        "scale": req.scale,
        "method": req.method,
        "upscale_dir": str(upscale_dir),
    }
    _write_testset(testset)
    return {
        "id": testset_id,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "scale": req.scale,
        "method": req.method,
    }


@app.get("/api/cluster-testsets/image")
def get_cluster_testset_image_by_path(path: str = Query(...)):
    resolved = str(Path(path).resolve())
    if resolved not in _known_testset_image_paths():
        raise HTTPException(status_code=403, detail="Image path is not part of a saved testset")
    image_bytes = Path(resolved).read_bytes()
    media_type = "image/png" if image_bytes.startswith(b"\x89PNG") else "image/jpeg"
    return Response(content=image_bytes, media_type=media_type)


@app.post("/api/cluster-testsets/{testset_id}/run")
def run_cluster_testset(testset_id: str, req: ClusterExperimentRunRequest):
    testset = _load_testset(testset_id)
    variants, cluster_methods, threshold_plan = _cluster_run_setup(req)

    if req.stream:
        def generate():
            try:
                for res in _iter_cluster_run_results(testset, req, variants, cluster_methods, threshold_plan):
                    yield json.dumps({"result": res}) + "\n"
            except Exception as exc:
                yield json.dumps({"error": str(exc)}) + "\n"
        return StreamingResponse(generate(), media_type="application/x-ndjson")

    started = time.time()
    all_results = list(_iter_cluster_run_results(testset, req, variants, cluster_methods, threshold_plan))
    return {
        "testset_id": testset_id,
        "name": testset.get("name"),
        "image_count": len(testset.get("images") or []),
        "elapsed_sec": round(time.time() - started, 3),
        "cluster_method": req.cluster_method,
        "cluster_methods": cluster_methods,
        "post_merge_threshold": req.post_merge_threshold,
        "auto_thresholds": req.auto_thresholds,
        "threshold_plan": threshold_plan,
        "results": all_results,
    }


@app.post("/api/cluster-testsets/{testset_id}/experiment-jobs")
def start_cluster_experiment_job(testset_id: str, req: ClusterExperimentRunRequest):
    testset = _load_testset(testset_id)
    variants, cluster_methods, threshold_plan = _cluster_run_setup(req)
    job_id = f"job-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    created_at = datetime.now(timezone.utc).isoformat()
    job = {
        "id": job_id,
        "name": (req.name or f"background-{testset.get('name') or testset_id}").strip(),
        "status": "queued",
        "testset_id": testset_id,
        "testset_name": testset.get("name"),
        "created_at": created_at,
        "started_at": None,
        "finished_at": None,
        "elapsed_sec": None,
        "total_runs": _cluster_run_total(variants, cluster_methods, threshold_plan),
        "completed_runs": 0,
        "cluster_methods": cluster_methods,
        "auto_thresholds": req.auto_thresholds,
        "threshold_plan": threshold_plan,
        "experiment_id": None,
        "experiment_name": None,
        "latest_result": None,
        "error": None,
        "request": req.model_dump(),
    }
    with _CLUSTER_EXPERIMENT_JOBS_LOCK:
        _CLUSTER_EXPERIMENT_JOBS[job_id] = job
    worker = threading.Thread(
        target=_run_cluster_experiment_job,
        args=(job_id, testset_id, req),
        daemon=True,
    )
    worker.start()
    return _cluster_job_public(job)


@app.get("/api/cluster-experiments/jobs")
def list_cluster_experiment_jobs():
    with _CLUSTER_EXPERIMENT_JOBS_LOCK:
        jobs = [_cluster_job_public(job.copy()) for job in _CLUSTER_EXPERIMENT_JOBS.values()]
    jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"jobs": jobs}


@app.get("/api/cluster-experiments/jobs/{job_id}")
def get_cluster_experiment_job(job_id: str):
    with _CLUSTER_EXPERIMENT_JOBS_LOCK:
        job = _CLUSTER_EXPERIMENT_JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Experiment job not found")
        return _cluster_job_public(job.copy())


def _save_cluster_experiment_csv(payload: dict, path: Path) -> None:
    import csv as _csv
    results = payload.get("results") or []
    columns = [
        "model", "strategy", "clustering_method", "threshold",
        "f1", "purity", "precision", "recall", "gt_ids", "pred_clusters",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in results:
            m = r.get("metrics") or {}
            model_name, strategy = _split_cluster_result_label(
                r.get("variant", ""), r.get("label", "")
            )
            writer.writerow({
                "model": model_name,
                "strategy": strategy,
                "clustering_method": r.get("cluster_method", ""),
                "threshold": r.get("threshold", ""),
                "f1": m.get("pairwise_f1", ""),
                "purity": m.get("cluster_purity", ""),
                "precision": m.get("pairwise_precision", ""),
                "recall": m.get("pairwise_recall", ""),
                "gt_ids": m.get("num_gt_ids", ""),
                "pred_clusters": m.get("num_pred_clusters", ""),
            })


def _save_cluster_experiment_payload(
    *,
    name: str | None,
    testset_id: str,
    results: list[dict],
    thresholds: list[float | None] | None = None,
    metadata: dict | None = None,
) -> dict:
    exp_id = f"exp-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    payload = {
        "id": exp_id,
        "name": (name or f"Experiment {exp_id}").strip(),
        "testset_id": testset_id,
        "thresholds": thresholds or [],
        "results": results,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    if metadata:
        payload["metadata"] = metadata
    path = _experiment_path(exp_id)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _save_cluster_experiment_csv(payload, path.with_suffix(".csv"))
    return payload


@app.post("/api/cluster-experiments")
def save_cluster_experiment(req: ClusterExperimentSaveRequest):
    payload = _save_cluster_experiment_payload(
        name=req.name,
        testset_id=req.testset_id,
        results=req.results,
        thresholds=req.thresholds,
        metadata=req.metadata,
    )
    return {"id": payload["id"], "name": payload["name"]}


@app.get("/api/cluster-experiments")
def list_cluster_experiments():
    experiments = []
    for path in sorted(_experiment_store_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            experiments.append({
                "id": data.get("id"),
                "name": data.get("name"),
                "testset_id": data.get("testset_id"),
                "saved_at": data.get("saved_at"),
                "variant_count": len(data.get("results") or []),
            })
        except Exception:
            continue
    return {"experiments": experiments}


@app.get("/api/cluster-experiments/{exp_id}")
def get_cluster_experiment(exp_id: str):
    path = _experiment_path(exp_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/cluster-experiments/{exp_id}/csv")
def get_cluster_experiment_csv(exp_id: str):
    path = _experiment_path(exp_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    csv_path = path.with_suffix(".csv")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _save_cluster_experiment_csv(payload, csv_path)
    return StreamingResponse(
        csv_path.open("rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="cluster-exp-{exp_id}.csv"'},
    )


@app.post("/api/cluster-experiments/download-csv")
def download_cluster_experiment_csv(req: ClusterExperimentSaveRequest):
    """Generate CSV from results without saving the experiment."""
    import csv as _csv
    import io

    results = req.results or []
    columns = [
        "model", "strategy", "clustering_method", "threshold",
        "f1", "purity", "precision", "recall", "gt_ids", "pred_clusters",
    ]
    buf = io.StringIO()
    writer = _csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for r in results:
        m = r.get("metrics") or {}
        model_name, strategy = _split_cluster_result_label(
            r.get("variant", ""), r.get("label", "")
        )
        writer.writerow({
            "model": model_name,
            "strategy": strategy,
            "clustering_method": r.get("cluster_method", ""),
            "threshold": r.get("threshold", ""),
            "f1": m.get("pairwise_f1", ""),
            "purity": m.get("cluster_purity", ""),
            "precision": m.get("pairwise_precision", ""),
            "recall": m.get("pairwise_recall", ""),
            "gt_ids": m.get("num_gt_ids", ""),
            "pred_clusters": m.get("num_pred_clusters", ""),
        })
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="cluster-results.csv"'},
    )


@app.delete("/api/cluster-experiments/{exp_id}")
def delete_cluster_experiment(exp_id: str):
    path = _experiment_path(exp_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    path.unlink()
    return {"deleted": True, "exp_id": exp_id}


@app.get("/api/clusters/compare")
def compare_clusters(
    cluster_a: str = Query(...),
    cluster_b: str = Query(...),
    building: str | None = Query(default=None),
    quality_weighted: bool = Query(default=False),
    include_parts: bool = Query(default=True),
):
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            part_similarity = {}
            if quality_weighted:
                centroid_a, count_a = _cluster_centroid(
                    cur, cluster_a, map_id=map_id, quality_weighted=True
                )
                centroid_b, count_b = _cluster_centroid(
                    cur, cluster_b, map_id=map_id, quality_weighted=True
                )
                if not centroid_a or not centroid_b:
                    raise HTTPException(status_code=404, detail="One or both clusters have no observations")
                similarity = _cosine_similarity(centroid_a, centroid_b)
                class_a = class_b = None
                map_where = " AND map_id = %s" if map_id is not None else ""
                cur.execute(
                    f"SELECT MIN(class_id) FROM object_observations WHERE object_id::text = %s {map_where}",
                    tuple([cluster_a] + ([map_id] if map_id is not None else [])),
                )
                row_a = cur.fetchone()
                cur.execute(
                    f"SELECT MIN(class_id) FROM object_observations WHERE object_id::text = %s {map_where}",
                    tuple([cluster_b] + ([map_id] if map_id is not None else [])),
                )
                row_b = cur.fetchone()
                class_a = row_a[0] if row_a else None
                class_b = row_b[0] if row_b else None
                if include_parts:
                    part_similarity = _part_similarities(
                        cur, cluster_a, cluster_b, map_id=map_id, quality_weighted=True
                    )
                return {
                    "cluster_a": cluster_a,
                    "cluster_b": cluster_b,
                    "similarity": float(similarity),
                    "count_a": int(count_a),
                    "count_b": int(count_b),
                    "class_a": class_a,
                    "class_b": class_b,
                    "class_name_a": class_name_from_id(class_a),
                    "class_name_b": class_name_from_id(class_b),
                    "method": "quality-weighted cosine similarity between cluster centroids",
                    "quality_weighted": True,
                    "part_similarities": part_similarity,
                    "building_filter": building,
                }

            map_where_a = ""
            map_where_b = ""
            params = [cluster_a, cluster_b]
            if map_id is not None:
                map_where_a = " AND map_id = %s"
                map_where_b = " AND map_id = %s"
                params = [cluster_a, map_id, cluster_b, map_id]
            cur.execute(
                f"""
                WITH c1 AS (
                    SELECT AVG(embedding) AS centroid, COUNT(*) AS count, MIN(class_id) AS class_id
                    FROM object_observations
                    WHERE object_id::text = %s
                      {map_where_a}
                ),
                c2 AS (
                    SELECT AVG(embedding) AS centroid, COUNT(*) AS count, MIN(class_id) AS class_id
                    FROM object_observations
                    WHERE object_id::text = %s
                      {map_where_b}
                )
                SELECT
                    c1.count AS count_a,
                    c1.class_id AS class_a,
                    c2.count AS count_b,
                    c2.class_id AS class_b,
                    (1.0 - (c1.centroid <=> c2.centroid)) AS similarity
                FROM c1, c2
                """,
                tuple(params),
            )
            row = cur.fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail="One or both clusters not found")
            count_a, class_a, count_b, class_b, similarity = row
            if count_a == 0 or count_b == 0:
                raise HTTPException(status_code=404, detail="One or both clusters have no observations")
            if include_parts:
                part_similarity = _part_similarities(
                    cur, cluster_a, cluster_b, map_id=map_id, quality_weighted=False
                )

    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "similarity": float(similarity) if similarity is not None else 0.0,
        "count_a": int(count_a) if count_a is not None else 0,
        "count_b": int(count_b) if count_b is not None else 0,
        "class_a": class_a,
        "class_b": class_b,
        "class_name_a": class_name_from_id(class_a),
        "class_name_b": class_name_from_id(class_b),
        "method": "cosine similarity between cluster centroids (average embeddings of images in each cluster)",
        "quality_weighted": False,
        "part_similarities": part_similarity,
        "building_filter": building,
    }


@app.get("/api/clusters/{object_id}/intra-similar")
def get_intra_cluster_similarity(object_id: str, limit: int | None = Query(default=None, ge=1, le=5000), building: str | None = Query(default=None)):
    safe_limit = None if limit is None else max(1, min(limit, 5000))
    map_id = _resolve_map_id(building)
    building_where = ""
    params = [object_id]
    if map_id is not None:
        building_where = " AND map_id = %s"
        params.append(map_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT COUNT(*)
                FROM object_observations
                WHERE object_id = %s
                  {building_where}
                """,
                tuple(params),
            )
            count_row = cur.fetchone()
            total_obs = int(count_row[0]) if count_row is not None else 0
            if total_obs == 0:
                raise HTTPException(status_code=404, detail="Cluster not found or has no observations")

            query = f"""
                WITH centroid AS (
                    SELECT AVG(embedding) AS c
                    FROM object_observations
                    WHERE object_id = %s
                      {building_where}
                )
                SELECT
                    oo.id,
                    oo.created_at,
                    (1.0 - (oo.embedding <=> centroid.c)) AS similarity_to_centroid,
                    oo.detection_backend,
                    oo.embedding_backend
                FROM object_observations oo
                CROSS JOIN centroid
                WHERE oo.object_id = %s
                  {building_where}
                ORDER BY similarity_to_centroid DESC, oo.created_at DESC
            """
            params = [object_id] + ([map_id] if map_id is not None else []) + [object_id] + ([map_id] if map_id is not None else [])
            if safe_limit is not None:
                query += " LIMIT %s"
                params.append(safe_limit)
            cur.execute(query, params)
            rows = cur.fetchall()

    return {
        "selected_cluster_id": object_id,
        "total_observations": total_obs,
        "method": "cosine similarity to cluster centroid (average embedding of images in this cluster)",
        "results": [
            {
                "observation_id": r[0],
                "created_at": r[1],
                "similarity": float(r[2]) if r[2] is not None else 0.0,
                "detection_backend": r[3],
                "embedding_backend": r[4],
                "image_url": f"/api/observations/{r[0]}/image",
            }
            for r in rows
        ],
    }


@app.get("/api/clusters/{object_id}/faces")
def get_cluster_faces(object_id: str, limit: int = 120, building: str | None = Query(default=None)):
    clean_object_id = str(object_id).strip()
    safe_limit = max(1, min(limit, 500))
    map_id = _resolve_map_id(building)
    building_where = ""
    exists_params = [clean_object_id]
    if map_id is not None:
        building_where = " AND map_id = %s"
        exists_params.append(map_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT EXISTS (
                    SELECT 1
                    FROM object_observations
                    WHERE object_id::text = %s
                      {building_where}
                )
                """,
                tuple(exists_params),
            )
            exists_row = cur.fetchone()

            if not exists_row or not bool(exists_row[0]):
                raise HTTPException(status_code=404, detail="Cluster not found")

            # Face detections can only belong to person clusters. A non-person
            # cluster (e.g. a refrigerator) must never report faces, even if stale
            # rows or track-id reuse would otherwise link some.
            cur.execute(
                "SELECT class_id FROM objects WHERE id::text = %s",
                (clean_object_id,),
            )
            class_row = cur.fetchone()
            cluster_class_id = int(class_row[0]) if class_row and class_row[0] is not None else None
            if cluster_class_id != YOLO_PERSON_CLASS_ID:
                return {
                    "selected_cluster_id": clean_object_id,
                    "count": 0,
                    "results": [],
                }

            building_oo_where = ""
            building_fo_where = ""
            query_params = [clean_object_id]
            if map_id is not None:
                building_oo_where = " AND oo.map_id = %s"
                building_fo_where = " AND fo.map_id = %s"
                query_params.append(map_id)
            query_params.extend([clean_object_id])
            query_params.extend([clean_object_id])
            if map_id is not None:
                query_params.append(map_id)
            query_params.append(safe_limit)
            cur.execute(
                f"""
                WITH cluster_people AS (
                    SELECT DISTINCT fo.person_id
                    FROM object_observations oo
                    JOIN face_observations fo
                      ON fo.object_id = oo.object_id
                      OR (
                            fo.scene_id IS NOT DISTINCT FROM oo.scene_id
                            AND fo.yolo_track_id IS NOT NULL
                            AND oo.yolo_track_id IS NOT NULL
                            AND fo.yolo_track_id = oo.yolo_track_id
                      )
                    WHERE oo.object_id::text = %s
                      AND fo.person_id IS NOT NULL
                      {building_oo_where}
                ),
                ranked_faces AS (
                    SELECT
                        fo.id,
                        fo.person_id,
                        fo.scene_id,
                        fo.object_id,
                        fo.yolo_track_id,
                        fo.person_x_min,
                        fo.person_y_min,
                        fo.person_x_max,
                        fo.person_y_max,
                        fo.face_x_min,
                        fo.face_y_min,
                        fo.face_x_max,
                        fo.face_y_max,
                        fo.score,
                        fo.created_at,
                        CASE
                            WHEN fo.object_id::text = %s THEN 2
                            WHEN fo.person_id IN (SELECT person_id FROM cluster_people) THEN 1
                            ELSE 0
                        END AS match_rank
                    FROM face_observations fo
                    WHERE (fo.object_id::text = %s
                       OR fo.person_id IN (SELECT person_id FROM cluster_people))
                       {building_fo_where}
                )
                SELECT
                    rf.id,
                    rf.person_id,
                    rf.scene_id,
                    rf.object_id,
                    rf.yolo_track_id,
                    rf.person_x_min,
                    rf.person_y_min,
                    rf.person_x_max,
                    rf.person_y_max,
                    rf.face_x_min,
                    rf.face_y_min,
                    rf.face_x_max,
                    rf.face_y_max,
                    rf.score,
                    rf.created_at,
                    rf.match_rank,
                    s.timestamp AS scene_timestamp
                FROM ranked_faces rf
                LEFT JOIN scenes s ON s.id = rf.scene_id
                ORDER BY rf.match_rank DESC, rf.created_at DESC
                LIMIT %s
                """,
                tuple(query_params),
            )
            rows = cur.fetchall()

    return {
        "selected_cluster_id": clean_object_id,
        "count": len(rows),
        "results": [
            {
                "face_observation_id": r[0],
                "person_id": r[1],
                "scene_id": r[2],
                "object_id": r[3],
                "yolo_track_id": r[4],
                "person_bbox": [r[5], r[6], r[7], r[8]],
                "face_bbox": [r[9], r[10], r[11], r[12]],
                "score": float(r[13]) if r[13] is not None else None,
                "created_at": r[14],
                "direct_cluster_match": bool(int(r[15]) >= 2),
                "scene_timestamp": r[16],
                "image_url": f"/api/faces/{r[0]}/image",
                "scene_image_url": f"/api/scenes/{r[2]}/original_image" if r[2] is not None else None,
            }
            for r in rows
        ],
    }

@app.get("/api/debug/embedding-stats")
def get_embedding_stats():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_observations,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) AS observations_with_embedding,
                    COALESCE(MIN(vector_dims(embedding)), 0) AS min_embedding_dim,
                    COALESCE(MAX(vector_dims(embedding)), 0) AS max_embedding_dim
                FROM object_observations
                """
            )
            stats = cur.fetchone()

            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_scenes,
                    COUNT(*) FILTER (WHERE caption IS NOT NULL AND btrim(caption) <> '') AS scenes_with_caption,
                    COUNT(*) FILTER (WHERE caption IS NULL OR btrim(caption) = '') AS scenes_without_caption
                FROM scenes
                """
            )
            scene_stats = cur.fetchone()

            cur.execute(
                """
                SELECT
                    COUNT(*) AS total_face_observations,
                    COUNT(DISTINCT person_id) AS distinct_persons,
                    COALESCE(MIN(vector_dims(embedding)), 0) AS min_face_embedding_dim,
                    COALESCE(MAX(vector_dims(embedding)), 0) AS max_face_embedding_dim
                FROM face_observations
                """
            )
            face_stats = cur.fetchone()

    return {
        "observations": {
            "total": stats[0],
            "with_embedding": stats[1],
            "min_dim": stats[2],
            "max_dim": stats[3],
        },
        "scenes": {
            "total": scene_stats[0],
            "with_caption": scene_stats[1],
            "without_caption": scene_stats[2],
        },
        "faces": {
            "total": face_stats[0],
            "persons": face_stats[1],
            "min_dim": face_stats[2],
            "max_dim": face_stats[3],
        },
    }

@app.get("/api/debug/recent-scenes")
def get_recent_scenes(limit: int = 20):
    safe_limit = max(1, min(limit, 200))
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, timestamp, caption,
                       CASE WHEN caption IS NULL OR btrim(caption) = '' THEN false ELSE true END AS has_caption
                FROM scenes
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (safe_limit,),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "caption": r[2],
            "has_caption": r[3],
        }
        for r in rows
    ]

@app.get("/api/observations/{observation_id}/image")
def get_observation_image(observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cropped_image
                FROM object_observations
                WHERE id = %s
            """, (observation_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")

    image_bytes = bytes(row[0]) if row[0] is not None else b""
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/observations/{observation_id}/original_image")
def get_observation_original_image(observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT original_cropped_image
                FROM object_observations
                WHERE id = %s
            """, (observation_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")

    image_bytes = bytes(row[0]) if row[0] is not None else b""
    if not image_bytes:
        raise HTTPException(status_code=404, detail="Original image not available")
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/observations/{observation_id}/mask")
def get_observation_mask(observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT mask_image
                FROM object_observations
                WHERE id = %s
            """, (observation_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")

    mask_bytes = bytes(row[0]) if row[0] is not None else b""
    if mask_bytes.startswith(b"\x89PNG"):
        media_type = "image/png"
    elif mask_bytes.startswith(b"\xff\xd8"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"
    return Response(content=mask_bytes, media_type=media_type)


@app.get("/api/observations/{observation_id}/masked_image")
def get_observation_masked_image(observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT cropped_image, mask_image
                FROM object_observations
                WHERE id = %s
            """, (observation_id,))
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Observation not found")

    crop_bytes = bytes(row[0]) if row[0] is not None else b""
    mask_bytes = bytes(row[1]) if row[1] is not None else b""

    if not mask_bytes:
        # No mask available; return original crop
        if crop_bytes.startswith(b"\x89PNG"):
            media_type = "image/png"
        elif crop_bytes.startswith(b"\xff\xd8"):
            media_type = "image/jpeg"
        else:
            media_type = "application/octet-stream"
        return Response(content=crop_bytes, media_type=media_type)

    try:
        crop = PILImage.open(io.BytesIO(crop_bytes)).convert("RGBA")
        mask = PILImage.open(io.BytesIO(mask_bytes)).convert("L")

        if mask.size != crop.size:
            mask = mask.resize(crop.size, PIL_RESAMPLE_NEAREST)

        r, g, b, _a = crop.split()
        out = PILImage.merge("RGBA", (r, g, b, mask))

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to composite masked image: {exc}")


def iterate_agent_duet(
    agent_a_system_prompt: str,
    agent_b_system_prompt: str,
    opening_message: str,
    turns: int = 2,
):
    from agent.agent import run_custom_agent
    from agent.tools import get_room_navigation_target, move_to_position

    message = opening_message
    history: list[dict] = []
    prompts = [agent_a_system_prompt, agent_b_system_prompt]
    speakers = ["agent_a", "agent_b"]
    for turn_index in range(max(0, turns)):
        speaker = speakers[turn_index % 2]
        answer, tool_log = run_custom_agent(
            message,
            system_prompt=prompts[turn_index % 2],
            history=history,
        )
        tool_log = list(tool_log or [])
        if speaker == "agent_a" and not tool_log:
            match = re.search(r"\bRoom\s+\d+\b|room\s+[A-Za-z0-9_.-]+", f"{message} {answer}", re.IGNORECASE)
            if match and "explor" in f"{message} {answer}".lower():
                room_name = match.group(0)
                room_name = room_name[:1].upper() + room_name[1:]
                target = get_room_navigation_target(room_name)
                tool_log.append({"tool": "get_room_navigation_target", "args": {"room_name": room_name}, "result": target})
                if isinstance(target, dict) and target.get("found"):
                    move = move_to_position(
                        x=target.get("x"),
                        y=target.get("y"),
                        reason=f"fallback_room_navigation:{room_name}",
                    )
                    tool_log.append({
                        "tool": "move_to_position",
                        "args": {
                            "x": target.get("x"),
                            "y": target.get("y"),
                            "reason": f"fallback_room_navigation:{room_name}",
                        },
                        "result": move,
                    })
        item = {"speaker": speaker, "content": answer, "tool_log": tool_log}
        yield item
        history.append({"role": "assistant", "content": answer, "tool_log": tool_log})
        message = answer


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return render_page(request, "chat.html", "chat")


@app.get("/model-chat", response_class=HTMLResponse)
def model_chat_page(request: Request):
    return render_page(request, "model_chat.html", "model_chat")

class ChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None
    building: str | None = None
    mode: str | None = None

class ObjectNameUpdateRequest(BaseModel):
    name: str | None = None


class ModelChatRequest(BaseModel):
    question: str
    history: list[dict] | None = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    from agent.agent import run_agent
    from agent.tools import set_active_map_override
    import logging

    def _extract_object_target(tool_log):
        """If the agent resolved an object location and navigated to it, surface
        the object identity so the UI can verify arrival visually afterwards."""
        if not isinstance(tool_log, list):
            return None
        has_nav = any(
            isinstance(e, dict) and e.get("tool") == "move_to_position" for e in tool_log
        )
        if not has_nav:
            return None
        for entry in reversed(tool_log):
            if not isinstance(entry, dict) or entry.get("tool") != "get_object_last_location":
                continue
            result = entry.get("result") or {}
            if result.get("found") and result.get("x") is not None:
                return {
                    "object_id": str(entry.get("args", {}).get("object_id") or result.get("object_id") or ""),
                    "object_name": result.get("name"),
                    "x": result.get("x"),
                    "y": result.get("y"),
                    "room": result.get("room"),
                }
        return None

    def _build_response(answer, tool_log, metadata, note: str | None = None):
        return {
            "answer": answer,
            "tool_log": tool_log,
            "model": metadata.get("model"),
            "used_model": metadata.get("used_model", False),
            "iterations": metadata.get("iterations", 0),
            "usage": metadata.get("usage", {}),
            "trace": metadata.get("trace", []),
            "system_prompt": metadata.get("system_prompt"),
            "note": note,
            "object_target": _extract_object_target(tool_log),
        }

    try:
        set_active_map_override(req.building)
        if req.mode == "navigation":
            from agent.navigation_direct import resolve_navigation_command

            navigation_result = resolve_navigation_command(req.question)
            if navigation_result is not None:
                answer, tool_log = navigation_result
                return _build_response(
                    answer,
                    tool_log,
                    {
                        "model": "navigation_direct",
                        "used_model": False,
                        "iterations": 0,
                        "usage": {},
                        "trace": [],
                        "system_prompt": None,
                    },
                    note="navigation_direct",
                )

        allowed_tools = None
        if req.mode == "no_interaction":
            allowed_tools = {
                "search_objects_by_class_id",
                "get_object_last_location",
                "get_object_first_location",
                "list_objects",
                "get_room_navigation_target",
                "move_to_position",
                "get_robot_location",
                "get_object_image",
                "get_object_summary",
                "get_object_observations",
                "get_latest_scene",
                "get_room_exploration_status",
            }
        answer, tool_log, metadata = run_agent(
            req.question,
            history=req.history,
            return_metadata=True,
            allowed_tool_names=allowed_tools,
        )
        return _build_response(answer, tool_log, metadata)
    except Exception as exc:
        logging.exception("Error in /api/chat")
        detail = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "AuthenticationError" and hasattr(exc, "response"):
            try:
                err_dict = exc.response.json()
                if "error" in err_dict and "message" in err_dict["error"]:
                    detail = f"Authentication API Error: {err_dict['error']['message']} (Check your API Key / USE_KIMI / USE_GEMINI settings)"
            except Exception:
                pass

        raise HTTPException(
            status_code=502,
            detail=detail,
        )


_OBJECT_SEARCH_PREFIXES = (
    "go to the", "go to", "find the", "find", "search for the", "search for",
    "search", "navigate to the", "navigate to", "locate the", "locate",
    "where is the", "where is", "fetch the", "fetch", "bring the", "bring",
    "show me the", "show me",
)


def _object_search_phrase(question: str) -> str:
    """Reduce a command like 'Go to sven's backpack' to a searchable
    instance phrase ('backpack'). Possessives are dropped on purpose — this
    baseline deliberately ignores ownership."""
    text = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.!")
    for prefix in _OBJECT_SEARCH_PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix):].strip()
            break
    tokens = [t for t in text.split(" ") if not t.endswith("'s")]
    text = " ".join(tokens).strip()
    for article in ("the ", "a ", "an ", "my "):
        if text.startswith(article):
            text = text[len(article):]
    return text or question.strip()


# Relational adjectives that, when leading the target noun, are really ownership
# descriptors the baseline cannot ground ('shared table' -> target 'table').
_OBJECT_SEARCH_SHARED_WORDS = {"shared", "communal", "public", "common"}


def _object_search_parse(question: str) -> tuple[str, str | None]:
    """Reduce a referring expression to (searchable_target, dropped_descriptor).

    This baseline does NOT do ownership/relational inference — it is the lower-bound
    control. But it must at least *tokenize* the query correctly: strip the relative
    clause / possessive / prepositional modifier so the name search matches the head
    noun ('the table that is shared' -> 'table') instead of failing on the literal full
    string. The dropped descriptor ('that is shared', "sven's") is returned so the
    answer can state plainly that the baseline could not resolve it.
    """
    text = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?.!")
    for prefix in _OBJECT_SEARCH_PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix):].strip()
            break

    dropped: str | None = None

    # Possessive: "sven's backpack" -> target 'backpack', dropped "sven's".
    m = re.match(r"^([a-z]+)'s\s+(.+)$", text)
    if m:
        dropped = f"{m.group(1)}'s"
        text = m.group(2).strip()
    else:
        # Relative clause: '<noun> that is shared' / '<noun> that is <name>'s' etc.
        rel = re.search(r"\b(?:that|which|who)\s+(?:is|are|was|were)?\s*(.*)$", text)
        if rel:
            dropped = text[rel.start():].strip()
            text = text[: rel.start()].strip()
        else:
            # Prepositional modifier we cannot ground by name: 'next to the desk' etc.
            prep = re.search(r"\b(?:next to|beside|near|behind|in front of|on top of|under|owned by|belonging to)\b", text)
            if prep:
                dropped = text[prep.start():].strip()
                text = text[: prep.start()].strip()

    # Leading adjective that is really a relational descriptor: 'shared table' /
    # 'communal laptop' -> target 'table'/'laptop', dropped 'shared'/'communal'.
    first = text.split(" ", 1)
    if len(first) == 2 and first[0] in _OBJECT_SEARCH_SHARED_WORDS:
        dropped = first[0] if dropped is None else dropped
        text = first[1].strip()

    # Strip leading article from the (now reduced) target noun phrase.
    for article in ("the ", "a ", "an ", "my "):
        if text.startswith(article):
            text = text[len(article):]
    target = text.strip() or question.strip()
    return target, dropped



@app.post("/api/object_search")
def object_search(req: ChatRequest):
    """Object-search baseline (no ownership inference).

    COOL pipeline : perception -> ReID -> spatial memory -> interaction evidence
                    -> ownership inference -> select instance -> navigate
    This baseline : perception -> ReID -> spatial memory
                    -> conventional instance selection (most recently seen
                    instance of the requested class/name) -> navigate

    It tokenizes the referring expression correctly (strips the relative clause /
    possessive so 'the table that is shared' searches 'table'), but deliberately does
    NOT resolve relational/ownership descriptors — it is the lower-bound control that
    the COOL pipeline's ownership inference is measured against. When a descriptor was
    dropped, the answer says so plainly. Deterministic (no LLM). Returns the same shape
    as /api/chat so the SLAM UI can plan/execute and run the arrival-verification loop
    unchanged.
    """
    from agent.tools import (
        get_object_last_location,
        move_to_position,
        search_objects_by_class_id,
        set_active_map_override,
    )
    import logging

    try:
        set_active_map_override(req.building)
        phrase, dropped = _object_search_parse(req.question)
        tool_log: list[dict] = []

        search_args = {"object_name": phrase}
        search_result = search_objects_by_class_id(**search_args)
        tool_log.append({"tool": "search_objects_by_class_id", "args": search_args, "result": search_result})

        candidates = search_result.get("results") or []
        if not candidates:
            return {
                "answer": (
                    f"Object-search baseline: no tracked instance matching "
                    f"'{phrase}' exists in spatial memory."
                ),
                "tool_log": tool_log,
                "model": "object_search_baseline",
                "used_model": False,
                "iterations": 0,
                "usage": {},
                "trace": [],
                "object_target": None,
            }

        # Conventional selection: candidates arrive ordered by last_seen DESC. Pick the
        # most recently seen instance that has a navigable position. This baseline does
        # NOT resolve relational/ownership descriptors — it is the lower-bound control.
        selected = None
        selected_id = selected_name = None
        loc = None
        skipped: list[str] = []
        for candidate in candidates[:5]:
            cand_id = str(candidate.get("object_id") or candidate.get("id"))
            cand_name = candidate.get("name") or f"object {cand_id}"
            loc_args = {"object_id": cand_id, "require_navigation_usable": True}
            cand_loc = get_object_last_location(**loc_args)
            if not cand_loc.get("found"):
                cand_loc = get_object_last_location(object_id=cand_id)
            tool_log.append({"tool": "get_object_last_location", "args": loc_args, "result": cand_loc})
            if cand_loc.get("found"):
                selected, selected_id, selected_name, loc = candidate, cand_id, cand_name, cand_loc
                break
            skipped.append(cand_name)

        if selected is None:
            return {
                "answer": (
                    f"Object-search baseline: {len(skipped) or len(candidates)} instance(s) "
                    f"matching '{phrase}' exist in spatial memory "
                    f"({', '.join(skipped) if skipped else 'most recent first'}) but none "
                    f"has a navigable position."
                ),
                "tool_log": tool_log,
                "model": "object_search_baseline",
                "used_model": False,
                "iterations": 0,
                "usage": {},
                "trace": [],
                "object_target": None,
            }

        move_args = {
            "x": loc["x"],
            "y": loc["y"],
            "reason": f"Object-search baseline: most recently seen instance '{selected_name}'",
            "standoff_m": 1.0,
        }
        move_result = move_to_position(**move_args)
        tool_log.append({"tool": "move_to_position", "args": move_args, "result": move_result})

        last_seen = str(selected.get("last_seen_at") or loc.get("last_seen_at") or "")[:16]
        skip_note = (
            f" Skipped {len(skipped)} more recently seen instance(s) without a "
            f"navigable position ({', '.join(skipped)})." if skipped else ""
        )
        # Honest limitation: the baseline cannot resolve relational/ownership
        # descriptors, so it picks the most recently seen instance of the head noun.
        dropped_note = (
            f" Note: this baseline cannot resolve the descriptor '{dropped}' — it selected "
            f"the most recently seen '{phrase}' regardless." if dropped else ""
        )
        answer = (
            f"Object-search baseline (conventional instance selection, no ownership "
            f"inference): navigating to '{selected_name}' — the most recently seen "
            f"instance matching '{phrase}' (last seen {last_seen}, "
            f"{selected.get('observation_count', '?')} observations), located at "
            f"({loc['x']:.2f}, {loc['y']:.2f}) in {loc.get('room', 'unknown')}."
            f"{dropped_note}{skip_note} "
            f"Arrival will be verified visually; if not found, the next known "
            f"location is tried automatically."
        )
        return {
            "answer": answer,
            "tool_log": tool_log,
            "model": "object_search_baseline",
            "used_model": False,
            "iterations": 0,
            "usage": {},
            "trace": [],
            "object_target": {
                "object_id": selected_id,
                "object_name": selected_name,
                "x": loc["x"],
                "y": loc["y"],
                "room": loc.get("room"),
            },
        }
    except Exception as exc:
        logging.exception("Error in /api/object_search")
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")


@app.post("/api/model-chat")
def model_chat(req: ModelChatRequest):
    from agent.agent import run_model_chat
    import logging

    try:
        answer, _, metadata = run_model_chat(
            req.question,
            history=req.history,
            return_metadata=True,
        )
        return {
            "answer": answer,
            "model": metadata.get("model"),
            "used_model": metadata.get("used_model", False),
            "iterations": metadata.get("iterations", 0),
            "usage": metadata.get("usage", {}),
            "trace": metadata.get("trace", []),
            "system_prompt": metadata.get("system_prompt"),
        }
    except Exception as exc:
        logging.exception("Error in /api/model-chat")
        detail = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "AuthenticationError" and hasattr(exc, "response"):
            try:
                err_dict = exc.response.json()
                if "error" in err_dict and "message" in err_dict["error"]:
                    detail = f"Authentication API Error: {err_dict['error']['message']}"
            except Exception:
                pass

        raise HTTPException(status_code=502, detail=detail)


@app.post("/api/objects/{object_id}/name")
def update_object_name(object_id: str, request: ObjectNameUpdateRequest):
    clean_object_id = str(object_id).strip()
    if not clean_object_id:
        raise HTTPException(status_code=400, detail="object_id is required")

    normalized_name = None
    if request.name is not None:
        text = str(request.name).strip()
        normalized_name = text or None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE objects SET name = %s WHERE id = %s RETURNING id, name",
                (normalized_name, clean_object_id),
            )
            row = cur.fetchone()
        conn.commit()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Object {clean_object_id} not found")

    return {"id": row[0], "name": row[1]}


@app.get("/api/persons")
def get_persons(building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    face_building_sql = " AND (fo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = fo.scene_id AND s2.map_id = %s))" if map_id is not None else ""
    object_building_sql = " AND (oo.map_id = %s OR EXISTS (SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))" if map_id is not None else ""
    map_filter_sql = "" if map_id is None else " AND (face_summary.last_seen_at IS NOT NULL OR object_summary.last_seen_at IS NOT NULL)"
    params = []
    if map_id is not None:
        params.extend([map_id, map_id, map_id, map_id])
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    o.id,
                    o.name,
                    o.created_at,
                    COALESCE(
                        GREATEST(face_summary.last_seen_at, object_summary.last_seen_at),
                        face_summary.last_seen_at,
                        object_summary.last_seen_at,
                        o.created_at
                    ) AS last_seen_at,
                    COALESCE(face_summary.face_observation_count, 0) AS face_observation_count,
                    face_summary.latest_face_observation_id,
                    COALESCE(object_summary.latest_scene_id, face_summary.latest_scene_id) AS latest_scene_id,
                    COALESCE(object_summary.latest_yolo_track_id, face_summary.latest_yolo_track_id) AS latest_yolo_track_id
                FROM objects o
                LEFT JOIN LATERAL (
                    SELECT
                        COUNT(fo.id) AS face_observation_count,
                        MAX(fo.created_at) AS last_seen_at,
                        (
                            ARRAY_AGG(fo.id ORDER BY fo.created_at DESC)
                            FILTER (WHERE fo.id IS NOT NULL)
                        )[1] AS latest_face_observation_id,
                        (
                            ARRAY_AGG(fo.scene_id ORDER BY fo.created_at DESC)
                            FILTER (WHERE fo.scene_id IS NOT NULL)
                        )[1] AS latest_scene_id,
                        (
                            ARRAY_AGG(fo.yolo_track_id ORDER BY fo.created_at DESC)
                            FILTER (WHERE fo.yolo_track_id IS NOT NULL)
                        )[1] AS latest_yolo_track_id
                    FROM face_observations fo
                    WHERE fo.person_id::text = o.id::text
                      {face_building_sql}
                ) AS face_summary ON TRUE
                LEFT JOIN LATERAL (
                    SELECT
                        MAX(oo.created_at) AS last_seen_at,
                        (
                            ARRAY_AGG(oo.scene_id ORDER BY oo.created_at DESC)
                            FILTER (WHERE oo.scene_id IS NOT NULL)
                        )[1] AS latest_scene_id,
                        (
                            ARRAY_AGG(oo.yolo_track_id ORDER BY oo.created_at DESC)
                            FILTER (WHERE oo.yolo_track_id IS NOT NULL)
                        )[1] AS latest_yolo_track_id
                    FROM object_observations oo
                    WHERE oo.object_id::text = o.id::text
                      {object_building_sql}
                ) AS object_summary ON TRUE
                WHERE o.class_id = 0{map_filter_sql}
                ORDER BY last_seen_at DESC NULLS LAST, o.created_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "name": r[1],
            "created_at": r[2],
            "last_seen_at": r[3],
            "face_observation_count": r[4],
            "latest_face_observation_id": r[5],
            "latest_scene_id": r[6],
            "latest_yolo_track_id": r[7],
            "image_url": f"/api/faces/{r[5]}/image" if r[5] is not None else None,
        }
        for r in rows
    ]


@app.get("/api/persons/{person_id}/faces")
def get_person_faces(person_id: str, building: str | None = Query(default=None)):
    map_id = _resolve_map_id(building)
    with get_conn() as conn:
        with conn.cursor() as cur:
            if map_id is not None:
                cur.execute(
                    """
                    SELECT
                        fo.id,
                        fo.person_id,
                        fo.scene_id,
                        fo.object_id,
                        fo.yolo_track_id,
                        fo.person_x_min,
                        fo.person_y_min,
                        fo.person_x_max,
                        fo.person_y_max,
                        fo.face_x_min,
                        fo.face_y_min,
                        fo.face_x_max,
                        fo.face_y_max,
                        fo.score,
                        fo.created_at
                    FROM face_observations fo
                    WHERE fo.person_id::text = %s
                      AND fo.map_id = %s
                    ORDER BY fo.created_at DESC
                    """,
                    (str(person_id).strip(), map_id),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        fo.id,
                        fo.person_id,
                        fo.scene_id,
                        fo.object_id,
                        fo.yolo_track_id,
                        fo.person_x_min,
                        fo.person_y_min,
                        fo.person_x_max,
                        fo.person_y_max,
                        fo.face_x_min,
                        fo.face_y_min,
                        fo.face_x_max,
                        fo.face_y_max,
                        fo.score,
                        fo.created_at
                    FROM face_observations fo
                    WHERE fo.person_id::text = %s
                    ORDER BY fo.created_at DESC
                    """,
                    (str(person_id).strip(),),
                )
            rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "person_id": r[1],
            "scene_id": r[2],
            "object_id": r[3],
            "yolo_track_id": r[4],
            "person_bbox": [r[5], r[6], r[7], r[8]],
            "face_bbox": [r[9], r[10], r[11], r[12]],
            "score": r[13],
            "created_at": r[14],
            "image_url": f"/api/faces/{r[0]}/image",
        }
        for r in rows
    ]


@app.get("/api/persons/{person_id}/clusters")
def get_person_clusters(person_id: str, min_observations: int = 2, building: str | None = Query(default=None)):
    clean_person_id = str(person_id).strip()
    map_id = _resolve_map_id(building)
    building_sql = ""
    building_sql_oo_join = ""
    building_sql_oo_outer = ""
    building_sql_oo_self = ""
    params = [clean_person_id]
    if map_id is not None:
        building_sql = " AND fo.map_id = %s"
        params.append(map_id)
    params.append(clean_person_id)
    if map_id is not None:
        building_sql_oo_self = " AND oo.map_id = %s"
        params.append(map_id)
        building_sql_oo_outer = " AND oo.map_id = %s"
        params.append(map_id)
    if map_id is not None:
        building_sql_oo_join = " AND oo.map_id = %s"
        params.append(map_id)
    params.append(min_observations)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH person_faces AS (
                    SELECT
                        fo.person_id,
                        fo.scene_id,
                        fo.object_id,
                        fo.yolo_track_id,
                        fo.created_at
                    FROM face_observations fo
                    WHERE fo.person_id::text = %s
                      {building_sql}
                ),
                candidate_objects AS (
                    SELECT pf.object_id
                    FROM person_faces pf
                    WHERE pf.object_id IS NOT NULL AND btrim(pf.object_id::text) <> ''

                    UNION

                    SELECT oo.object_id
                    FROM object_observations oo
                    JOIN objects o2 ON o2.id = oo.object_id
                    WHERE oo.object_id::text = %s
                      AND o2.class_id = 0
                      {building_sql_oo_self}

                    UNION

                    SELECT oo.object_id
                    FROM person_faces pf
                    JOIN object_observations oo
                      ON pf.yolo_track_id IS NOT NULL
                     AND btrim(pf.yolo_track_id) <> ''
                     AND oo.yolo_track_id = pf.yolo_track_id
                     AND (
                        pf.scene_id IS NULL
                        OR oo.scene_id = pf.scene_id
                     )
                     {building_sql_oo_join}
                )
                SELECT
                    oo.object_id,
                    o.name,
                    MAX(oo.class_id) AS class_id,
                    COUNT(oo.id) AS observation_count,
                    MAX(oo.created_at) AS last_seen_at,
                    (
                        ARRAY_AGG(oo.yolo_track_id ORDER BY oo.created_at DESC)
                        FILTER (WHERE oo.yolo_track_id IS NOT NULL)
                    )[1] AS latest_yolo_track_id,
                    (
                        ARRAY_AGG(oo.scene_id ORDER BY oo.created_at DESC)
                        FILTER (WHERE oo.scene_id IS NOT NULL)
                    )[1] AS latest_scene_id,
                    (
                        ARRAY_AGG(oo.id ORDER BY oo.created_at DESC)
                        FILTER (WHERE oo.id IS NOT NULL)
                    )[1] AS latest_observation_id
                FROM object_observations oo
                JOIN objects o ON o.id::text = oo.object_id::text
                JOIN candidate_objects co ON co.object_id::text = oo.object_id::text
                {building_sql_oo_outer}
                GROUP BY oo.object_id, o.name
                HAVING COUNT(oo.id) >= %s
                ORDER BY observation_count DESC, last_seen_at DESC
                """,
                tuple(params),
            )
            rows = cur.fetchall()

    return [
        {
            "object_id": r[0],
            "name": r[1],
            "class_id": r[2],
            "class_name": class_name_from_id(r[2]),
            "observation_count": r[3],
            "last_seen_at": r[4],
            "latest_yolo_track_id": r[5],
            "latest_scene_id": r[6],
            "latest_observation_id": r[7],
            "image_url": f"/api/observations/{r[7]}/image" if r[7] is not None else None,
        }
        for r in rows
    ]


@app.get("/api/faces/{face_observation_id}/image")
def get_face_image(face_observation_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT face_image
                FROM face_observations
                WHERE id = %s
                """,
                (face_observation_id,),
            )
            row = cur.fetchone()

    if row and row[0] is not None:
        return Response(content=bytes(row[0]), media_type="image/jpeg")

    # Fallback: face_image was not stored (e.g. rows re-populated from the offline
    # staging table, which keeps only the embedding + crop-relative face_bbox). Crop the
    # face on the fly from the observation's stored person crop using the staging bbox.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT oo.cropped_image, pfe.face_bbox
                FROM face_observations fo
                JOIN object_observations oo ON oo.id = fo.observation_id
                JOIN person_face_embeddings pfe ON pfe.observation_id = fo.observation_id
                WHERE fo.id = %s
                """,
                (face_observation_id,),
            )
            row = cur.fetchone()

    if not row or row[0] is None or row[1] is None:
        raise HTTPException(status_code=404, detail="Face image not found")

    try:
        import io as _io
        from PIL import Image as _Image

        crop = _Image.open(_io.BytesIO(bytes(row[0]))).convert("RGB")
        bbox = row[1]  # jsonb -> list [x1, y1, x2, y2], crop-relative
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        w, h = crop.size
        x1 = max(0, min(int(round(x1)), w - 1))
        y1 = max(0, min(int(round(y1)), h - 1))
        x2 = max(x1 + 1, min(int(round(x2)), w))
        y2 = max(y1 + 1, min(int(round(y2)), h))
        face = crop.crop((x1, y1, x2, y2))
        buf = _io.BytesIO()
        face.save(buf, format="JPEG", quality=90)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception:
        raise HTTPException(status_code=404, detail="Face image not found")


@app.get("/api/scenes/{scene_id}/stitched_image")
def get_scene_stitched_image(scene_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT stitched_scene_image
                FROM scenes
                WHERE id = %s
            """, (scene_id,))
            row = cur.fetchone()

    if not row or row[0] is None:
        raise HTTPException(status_code=404, detail="Stitched scene image not found")

    image_bytes = bytes(row[0])
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/api/scenes/{scene_id}/original_image")
def get_scene_original_image(scene_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT original_scene_image
                FROM scenes
                WHERE id = %s
            """, (scene_id,))
            row = cur.fetchone()

    if not row or row[0] is None:
        raise HTTPException(status_code=404, detail="Original scene image not found")

    image_bytes = bytes(row[0])
    return Response(content=image_bytes, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# SLAM tab endpoints (independent from existing map/toolbox code)
# ---------------------------------------------------------------------------

class SlamCommandRequest(BaseModel):
    action: str
    name: str | None = None
    x: float | None = None
    y: float | None = None
    theta: float | None = None


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def _load_json_file(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _slam_tab_pose_payload(x: float, y: float, theta: float) -> dict:
    return {
        "position": {
            "x": float(x),
            "y": float(y),
            "z": 0.0,
        },
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": math.sin(float(theta) / 2.0),
            "w": math.cos(float(theta) / 2.0),
        },
    }


def _touch_slam_tab_claim() -> None:
    _atomic_write_json(
        SLAM_TAB_CLAIM_PATH,
        {
            "owner": "web_slam_tab",
            "updated_at": time.time(),
        },
    )


def _infer_robot_connection_from_slam_pose_file() -> dict:
    try:
        stat = SLAM_TAB_ROBOT_POSE_PATH.stat()
    except OSError:
        return {
            "connected": False,
            "message": "No /spot/odometry received yet.",
            "last_seen_at": None,
            "age_sec": None,
            "topic": "/spot/odometry",
        }

    last_seen_at = stat.st_mtime
    age_sec = max(0.0, time.time() - last_seen_at)
    connected = age_sec <= ROBOT_ODOM_STALE_SEC
    return {
        "connected": connected,
        "message": "Robot connected." if connected else "Robot connection stale.",
        "last_seen_at": last_seen_at,
        "age_sec": age_sec,
        "topic": "/spot/odometry",
    }


def _refresh_robot_connection_payload(robot_connection: dict | None) -> dict:
    if not isinstance(robot_connection, dict):
        return _infer_robot_connection_from_slam_pose_file()

    raw_connected = robot_connection.get("connected")
    raw_last_seen_at = robot_connection.get("last_seen_at")
    topic = robot_connection.get("topic") or "/spot/odometry"
    message = str(robot_connection.get("message") or "").strip()

    if raw_connected is False:
        age_sec = None
        try:
            last_seen_at = float(raw_last_seen_at)
        except (TypeError, ValueError):
            return _infer_robot_connection_from_slam_pose_file()
        else:
            age_sec = max(0.0, time.time() - last_seen_at)
        return {
            "connected": False,
            "message": message or "Robot connection stale.",
            "last_seen_at": last_seen_at,
            "age_sec": age_sec,
            "topic": topic,
        }

    try:
        last_seen_at = float(raw_last_seen_at)
    except (TypeError, ValueError):
        return _infer_robot_connection_from_slam_pose_file()

    age_sec = max(0.0, time.time() - last_seen_at)
    connected = age_sec <= ROBOT_ODOM_STALE_SEC
    return {
        "connected": connected,
        "message": "Robot connected." if connected else "Robot connection stale.",
        "last_seen_at": last_seen_at,
        "age_sec": age_sec,
        "topic": topic,
    }


@app.get("/api/slam/status")
def get_slam_status():
    _touch_slam_tab_claim()
    payload = _load_json_file(SLAM_TAB_STATUS_PATH)
    if not isinstance(payload, dict):
        return {
            "state": "idle",
            "message": "SLAM manager not running yet.",
            "saved_maps": [],
            "robot_connection": _infer_robot_connection_from_slam_pose_file(),
        }
    payload["robot_connection"] = _refresh_robot_connection_payload(payload.get("robot_connection"))
    return payload


@app.post("/api/slam/command")
def post_slam_command(req: SlamCommandRequest):
    if req.action == "set_pose" and (req.x is None or req.y is None or req.theta is None):
        raise HTTPException(status_code=400, detail="set_pose requires x, y, and theta")

    request_id = str(uuid.uuid4())
    payload = {
        "request_id": request_id,
        "created_at": time.time(),
        "action": req.action,
        "name": req.name,
    }
    for key in ("x", "y", "theta"):
        value = getattr(req, key)
        if value is not None:
            payload[key] = value

    _atomic_write_json(
        SLAM_TAB_REQUEST_PATH,
        payload,
    )
    return {"ok": True, "request_id": request_id, "action": req.action}


@app.get("/api/slam/map")
def get_slam_map():
    _touch_slam_tab_claim()
    active_payload = load_active_toolbox_map_payload()
    normalized_active_map = _slam_tab_response_from_normalized_map_payload(active_payload)
    if isinstance(normalized_active_map, dict):
        return normalized_active_map

    payload = _load_json_file(SLAM_TAB_MAP_PATH)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="SLAM map not available yet.")
    return payload


@app.get("/api/slam/local_costmap")
def get_slam_local_costmap():
    _touch_slam_tab_claim()
    payload = _load_json_file(SLAM_TAB_LOCAL_COSTMAP_PATH)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="SLAM local costmap not available yet.")
    return payload


@app.get("/api/slam/robot_pose")
def get_slam_robot_pose():
    _touch_slam_tab_claim()
    payload = _load_json_file(SLAM_TAB_ROBOT_POSE_PATH)
    if not isinstance(payload, dict):
        raise HTTPException(status_code=404, detail="Robot pose not available yet.")
    return payload


@app.get("/api/slam/robot_path")
def get_slam_robot_path(since: float = 0.0, last: int = 0):
    _touch_slam_tab_claim()
    if not ROBOT_PATH_FILE.exists():
        return {"points": [], "count": 0, "latest_t": 0.0}
    points = []
    latest_t = 0.0
    total_count = 0
    try:
        with ROBOT_PATH_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    total_count += 1
                    t = float(rec.get("t") or 0.0)
                    if t > latest_t:
                        latest_t = t
                    if not rec.get("map"):
                        continue
                    if since and t <= since:
                        continue
                    points.append({
                        "t": t,
                        "x": rec.get("x"),
                        "y": rec.get("y"),
                        "yaw": rec.get("yaw"),
                    })
                except Exception:
                    continue
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read path: {exc}")
    if last > 0 and len(points) > last:
        points = points[-last:]
    return {"points": points, "count": total_count, "latest_t": latest_t}


ROBOT_PATH_RECORDING_SIGNAL = Path("/shared/robot_path_recording.signal")


@app.get("/api/slam/robot_path/status")
def get_slam_robot_path_status():
    _touch_slam_tab_claim()
    is_recording = ROBOT_PATH_RECORDING_SIGNAL.exists()
    count = 0
    latest_t = 0.0
    if ROBOT_PATH_FILE.exists():
        try:
            with ROBOT_PATH_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        count += 1
                        t = float(rec.get("t") or 0.0)
                        if t > latest_t:
                            latest_t = t
                    except Exception:
                        continue
        except Exception:
            pass
    return {"recording": is_recording, "count": count, "latest_t": latest_t}


@app.post("/api/slam/robot_path/start_recording")
def post_slam_robot_path_start_recording():
    _touch_slam_tab_claim()
    # Archive current file if it exists and has content
    if ROBOT_PATH_FILE.exists() and ROBOT_PATH_FILE.stat().st_size > 0:
        archive_name = f"/shared/robot_path_{int(time.time())}.jsonl"
        try:
            ROBOT_PATH_FILE.rename(archive_name)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to archive existing path: {exc}")
    # Create signal file
    ROBOT_PATH_RECORDING_SIGNAL.touch()
    return {"recording": True, "message": "Recording started. Previous path archived."}


@app.post("/api/slam/robot_path/stop_recording")
def post_slam_robot_path_stop_recording():
    _touch_slam_tab_claim()
    ROBOT_PATH_RECORDING_SIGNAL.unlink(missing_ok=True)
    count = 0
    if ROBOT_PATH_FILE.exists():
        try:
            with ROBOT_PATH_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        count += 1
        except Exception:
            pass
    return {"recording": False, "count": count, "message": "Recording stopped."}


@app.post("/api/slam/robot_path/save")
def post_slam_robot_path_save(payload: dict):
    _touch_slam_tab_claim()
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    building_name = str(payload.get("building_name") or "").strip() or None
    if not ROBOT_PATH_FILE.exists():
        raise HTTPException(status_code=404, detail="No path data to save. Start recording first.")
    points = []
    try:
        with ROBOT_PATH_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("map"):
                        points.append({
                            "t": rec.get("t"),
                            "x": rec.get("x"),
                            "y": rec.get("y"),
                            "yaw": rec.get("yaw"),
                        })
                except Exception:
                    continue
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read path: {exc}")
    if not points:
        raise HTTPException(status_code=400, detail="No valid map-frame points to save.")
    # Also write a pretty-printed JSON file to /shared/ with the same name.
    shared_path = Path(f"/shared/{name}.json")
    try:
        with shared_path.open("w", encoding="utf-8") as f:
            json.dump(points, f, indent=2)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write shared file: {exc}")

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO robot_paths (name, building_name, point_count, path_data)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (name, building_name, len(points), json.dumps(points)),
            )
            row = cur.fetchone()
            conn.commit()
            return {"ok": True, "id": row[0], "name": name, "point_count": len(points)}
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()


@app.get("/api/slam/robot_path/saved")
def get_slam_robot_path_saved():
    _touch_slam_tab_claim()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, building_name, created_at, updated_at, point_count
                FROM robot_paths
                ORDER BY created_at DESC
                LIMIT 200
                """
            )
            rows = cur.fetchall()
            paths = []
            for row in rows:
                paths.append({
                    "id": row[0],
                    "name": row[1],
                    "building_name": row[2],
                    "created_at": row[3].isoformat() if row[3] else None,
                    "updated_at": row[4].isoformat() if row[4] else None,
                    "point_count": row[5],
                })
            return {"paths": paths}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()


@app.get("/api/slam/robot_path/get")
def get_slam_robot_path_by_id(path_id: int):
    _touch_slam_tab_claim()
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, point_count, path_data FROM robot_paths WHERE id = %s",
                (path_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Path not found.")
            name, point_count, path_data = row
            points = path_data if isinstance(path_data, list) else json.loads(path_data)
            return {"id": path_id, "name": name, "point_count": len(points), "points": points}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()


@app.post("/api/slam/robot_path/load")
def post_slam_robot_path_load(payload: dict):
    _touch_slam_tab_claim()
    path_id = payload.get("id")
    if not path_id:
        raise HTTPException(status_code=400, detail="id is required")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name, point_count, path_data FROM robot_paths WHERE id = %s",
                (path_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Path not found.")
            name, point_count, path_data = row
            points = path_data if isinstance(path_data, list) else json.loads(path_data)
            # Write loaded path to the JSONL file so the UI can display it
            ROBOT_PATH_FILE.parent.mkdir(parents=True, exist_ok=True)
            with ROBOT_PATH_FILE.open("w", encoding="utf-8") as f:
                for p in points:
                    f.write(json.dumps({**p, "map": True, "iso": ""}, separators=(",", ":")) + "\n")
            return {"ok": True, "name": name, "point_count": len(points)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()


@app.post("/api/slam/robot_path/delete")
def post_slam_robot_path_delete(payload: dict):
    _touch_slam_tab_claim()
    path_id = payload.get("id")
    if not path_id:
        raise HTTPException(status_code=400, detail="id is required")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM robot_paths WHERE id = %s RETURNING name", (path_id,))
            row = cur.fetchone()
            conn.commit()
            if not row:
                raise HTTPException(status_code=404, detail="Path not found.")
            return {"ok": True, "deleted": row[0]}
    except HTTPException:
        raise
    except Exception as exc:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        conn.close()


@app.get("/api/slam/saved_maps")
def get_slam_saved_maps():
    SLAM_TAB_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    maps = []
    for path in sorted(SLAM_TAB_SAVE_DIR.glob("*.posegraph")):
        maps.append({"name": path.stem, "updated_at": path.stat().st_mtime})
    return {"maps": maps}


@app.post("/api/slam/exploration/step")
def post_slam_exploration_step(payload: dict):
    from agent.slam_exploration import run_slam_exploration_step

    map_name = str(payload.get("map_name") or "").strip()
    robot_pose = payload.get("robot_pose")
    strategy_type = str(payload.get("strategy_type") or "v4").strip().lower()
    disabled_rooms = payload.get("disabled_rooms") or []
    if isinstance(disabled_rooms, str):
        disabled_rooms = [r.strip() for r in disabled_rooms.split(",") if r.strip()]
    min_dwell_minutes = payload.get("min_dwell_minutes")
    try:
        min_dwell_minutes = int(min_dwell_minutes) if min_dwell_minutes is not None else 10
    except (ValueError, TypeError):
        min_dwell_minutes = 10
    min_dwell_minutes = max(1, min(60, min_dwell_minutes))
    logging.info("[SLAM_STEP] map=%s strategy=%s disabled_rooms=%s min_dwell=%s", map_name, strategy_type, disabled_rooms, min_dwell_minutes)
    if not map_name:
        raise HTTPException(status_code=400, detail="map_name is required")
    result = run_slam_exploration_step(
        map_name=map_name,
        robot_pose=robot_pose,
        strategy_type=strategy_type,
        disabled_rooms=disabled_rooms,
        min_dwell_minutes=min_dwell_minutes,
    )
    return result


@app.post("/api/slam/exploration/clear")
def post_slam_exploration_clear(payload: dict):
    from agent.slam_exploration import clear_exploration_history

    map_name = str(payload.get("map_name") or "").strip()
    if not map_name:
        raise HTTPException(status_code=400, detail="map_name is required")
    result = clear_exploration_history(map_name)
    return result
