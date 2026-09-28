import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
from bosdyn.client import create_standard_sdk
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.point_cloud import build_pc_request


CONFIG_PATH = Path(os.getenv("SPOT_CONFIG_PATH", "/ros_ws/src/robot.yaml"))
OUTPUT_PATH = Path(os.getenv("VELODYNE_SCAN_PATH", "/shared/velodyne_scan_snapshot.json"))
FILTER_SETTINGS_PATH = Path(
    os.getenv("VELODYNE_FILTER_SETTINGS_PATH", "/shared/velodyne_filter_settings.json")
)
SERVICE_NAME = os.getenv("VELODYNE_SERVICE_NAME", "velodyne-point-cloud")
WORLD_FRAME_NAME = os.getenv("VELODYNE_WORLD_FRAME", ODOM_FRAME_NAME)
WORLD_FRAME_LABEL = os.getenv("VELODYNE_WORLD_FRAME_LABEL", "world")
SCAN_PERIOD_SEC = float(os.getenv("VELODYNE_SCAN_PERIOD_SEC", "1.0"))
MAX_POINTS = int(os.getenv("VELODYNE_SCAN_MAX_POINTS", "5000"))
MIN_RANGE_M = float(os.getenv("VELODYNE_MIN_RANGE_M", "1.0"))
MAX_RANGE_M = float(os.getenv("VELODYNE_MAX_RANGE_M", "12.0"))
MIN_Z_M = float(os.getenv("VELODYNE_MIN_Z_M", "-0.35"))
MAX_Z_M = float(os.getenv("VELODYNE_MAX_Z_M", "0.75"))
CONNECT_RETRY_SEC = float(os.getenv("VELODYNE_CONNECT_RETRY_SEC", "5.0"))


DEFAULT_FILTERS = {
    "min_range_m": MIN_RANGE_M,
    "max_range_m": MAX_RANGE_M,
    "min_z_m": MIN_Z_M,
    "max_z_m": MAX_Z_M,
}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["/**"]["ros__parameters"]


def point_cloud_to_xyz(proto) -> np.ndarray:
    raw = np.frombuffer(proto.data, dtype=np.float32)
    if raw.size < 3:
        return np.empty((0, 3), dtype=np.float32)
    raw = raw[: (raw.size // 3) * 3]
    points = raw.reshape((-1, 3))
    mask = np.isfinite(points).all(axis=1)
    return points[mask]


def load_runtime_filters() -> dict:
    settings = dict(DEFAULT_FILTERS)
    if not FILTER_SETTINGS_PATH.exists():
        return settings

    try:
        with FILTER_SETTINGS_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return settings

    if not isinstance(payload, dict):
        return settings

    for key in settings:
        value = payload.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            settings[key] = value

    if settings["max_range_m"] <= settings["min_range_m"]:
        settings["max_range_m"] = settings["min_range_m"] + 0.1
    if settings["max_z_m"] <= settings["min_z_m"]:
        settings["max_z_m"] = settings["min_z_m"] + 0.05
    return settings


def filter_sensor_frame_points(points: np.ndarray, filters: dict) -> np.ndarray:
    if points.size == 0:
        return points

    xy_range = np.hypot(points[:, 0], points[:, 1])
    mask = np.isfinite(xy_range)
    min_range_m = float(filters["min_range_m"])
    max_range_m = float(filters["max_range_m"])
    min_z_m = float(filters["min_z_m"])
    max_z_m = float(filters["max_z_m"])
    if min_range_m > 0.0:
        mask &= xy_range >= min_range_m
    if max_range_m > 0.0:
        mask &= xy_range <= max_range_m
    z_values = points[:, 2]
    if min_z_m < max_z_m:
        mask &= z_values >= min_z_m
        mask &= z_values <= max_z_m
    return points[mask]


def select_points_for_export(points: np.ndarray, filters: dict) -> tuple[np.ndarray, dict]:
    if points.size == 0:
        return points, {
            "applied": False,
            "fallback_to_unfiltered": False,
            "reason": "raw_cloud_empty",
        }

    filtered_points = filter_sensor_frame_points(points, filters)
    if filtered_points.size > 0:
        return filtered_points, {
            "applied": True,
            "fallback_to_unfiltered": False,
            "reason": "filtered_points_available",
        }

    return points, {
        "applied": False,
        "fallback_to_unfiltered": True,
        "reason": "filter_rejected_all_points",
    }


def quaternion_to_rotation_matrix(rotation) -> np.ndarray:
    x = float(rotation.x)
    y = float(rotation.y)
    z = float(rotation.z)
    w = float(rotation.w)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (yy + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def transform_points_to_world_frame(point_cloud, points: np.ndarray) -> tuple[np.ndarray, str]:
    sensor_frame = point_cloud.source.frame_name_sensor
    transform = get_a_tform_b(point_cloud.source.transforms_snapshot, WORLD_FRAME_NAME, sensor_frame)
    if transform is None:
        return points, sensor_frame

    rotation = quaternion_to_rotation_matrix(transform.rot)
    translation = np.array(
        [
            float(transform.x),
            float(transform.y),
            float(transform.z),
        ],
        dtype=np.float32,
    )
    transformed = points @ rotation.T + translation
    return transformed, WORLD_FRAME_LABEL


def filter_and_downsample_world_points(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points

    mask = np.isfinite(points).all(axis=1)
    points = points[mask]
    if points.shape[0] > MAX_POINTS:
        stride = max(1, math.ceil(points.shape[0] / MAX_POINTS))
        points = points[::stride]
    return points


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = Path(tmp.name)
    os.replace(temp_path, path)


def write_error_snapshot(message: str) -> None:
    atomic_write_json(
        OUTPUT_PATH,
        {
            "available": False,
            "generated_at": time.time(),
            "error": message,
        },
    )


def main() -> None:
    cfg = load_config()
    while True:
        try:
            sdk = create_standard_sdk("velodyne_scan_exporter")
            robot = sdk.create_robot(cfg["hostname"])
            robot.authenticate(cfg["username"], cfg["password"])
            robot.time_sync.wait_for_sync()
            client = robot.ensure_client(SERVICE_NAME)

            while True:
                try:
                    response = client.get_point_cloud([build_pc_request(SERVICE_NAME)])
                    point_cloud = response[0].point_cloud
                    points = point_cloud_to_xyz(point_cloud)
                    filters = load_runtime_filters()
                    selected_sensor_points, filter_result = select_points_for_export(points, filters)
                    world_points, world_frame = transform_points_to_world_frame(point_cloud, selected_sensor_points)
                    world_points = filter_and_downsample_world_points(world_points)
                    atomic_write_json(
                        OUTPUT_PATH,
                        {
                            "available": True,
                            "generated_at": time.time(),
                            "frame_id": world_frame,
                            "sensor_frame_id": point_cloud.source.frame_name_sensor,
                            "raw_num_points": int(points.shape[0]),
                            "filtered_sensor_num_points": int(filter_sensor_frame_points(points, filters).shape[0]),
                            "selected_sensor_num_points": int(selected_sensor_points.shape[0]),
                            "num_points": int(world_points.shape[0]),
                            "points": world_points[:, :2].astype(float).tolist(),
                            "filters": filters,
                            "filter_result": filter_result,
                        },
                    )
                except Exception as exc:
                    write_error_snapshot(str(exc))
                    break
                time.sleep(SCAN_PERIOD_SEC)
        except Exception as exc:
            write_error_snapshot(str(exc))
        time.sleep(CONNECT_RETRY_SEC)


if __name__ == "__main__":
    main()
