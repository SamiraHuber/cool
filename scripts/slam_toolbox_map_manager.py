import json
import os
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path


REQUEST_PATH = Path(os.getenv("TOOLBOX_MAP_REQUEST_PATH", "/shared/toolbox_map_request.json"))
STATUS_PATH = Path(os.getenv("TOOLBOX_MAP_STATUS_PATH", "/shared/toolbox_map_status.json"))
SAVE_DIR = Path(os.getenv("TOOLBOX_MAP_SAVE_DIR", "/shared/maps/toolbox_saved"))
LEGACY_SAVE_DIR = Path(os.getenv("SLAM_TAB_SAVE_DIR", "/shared/slam_tab/saved"))
AUTOLOAD_PATH = Path(os.getenv("TOOLBOX_MAP_AUTOLOAD_PATH", str(SAVE_DIR / "autoload.json")))
METADATA_PATH = Path(os.getenv("TOOLBOX_MAP_METADATA_PATH", str(SAVE_DIR / "metadata.json")))
RESET_REQUEST_PATH = Path(os.getenv("MAP_RESET_REQUEST_PATH", "/shared/map_reset_request.json"))
RESET_STATUS_PATH = Path(os.getenv("MAP_RESET_STATUS_PATH", "/shared/map_reset_state.json"))
ACTIVE_OVERRIDE_PATH = Path(os.getenv("TOOLBOX_MAP_ACTIVE_PATH", str(SAVE_DIR / "active.json")))
FROZEN_SNAPSHOT_PATH = Path(
    os.getenv("TOOLBOX_MAP_FROZEN_SNAPSHOT_PATH", "/shared/maps/toolbox_saved/active_snapshot.json")
)
SLAM_TAB_MAP_PATH = Path(os.getenv("SLAM_TAB_MAP_PATH", "/shared/slam_tab/map.json"))
DEFAULT_NAME = os.getenv("TOOLBOX_MAP_DEFAULT_NAME", "default_map")
POLL_PERIOD_SEC = float(os.getenv("TOOLBOX_MAP_MANAGER_PERIOD_SEC", "1.0"))
AUTOLOAD_ENABLED = os.getenv("TOOLBOX_MAP_AUTOLOAD", "1").strip().lower() not in {"0", "false", "no"}
SAVE_FILE_WAIT_TIMEOUT_SEC = float(os.getenv("TOOLBOX_MAP_SAVE_FILE_WAIT_TIMEOUT_SEC", "5.0"))
SAVE_FILE_WAIT_INTERVAL_SEC = float(os.getenv("TOOLBOX_MAP_SAVE_FILE_WAIT_INTERVAL_SEC", "0.1"))
SLAM_TOOLBOX_PARAMS_FILE = os.getenv("SLAM_TOOLBOX_PARAMS_FILE", "/opt/local/slam_toolbox_params.yaml")
SLAM_TOOLBOX_WRAPPER_PIDFILE = Path(os.getenv("SLAM_TOOLBOX_WRAPPER_PIDFILE", "/tmp/slam_toolbox_wrapper.pid"))
SLAM_TOOLBOX_MODE_FILE = Path(os.getenv("SLAM_TOOLBOX_MODE_FILE", "/tmp/slam_toolbox_mode.txt"))
MAP_SEARCH_DIRS = tuple(dict.fromkeys((SAVE_DIR, LEGACY_SAVE_DIR)).keys())


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def load_json_file(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_name(value: str | None) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or DEFAULT_NAME


def normalize_building_name(value: str | None, fallback_name: str | None = None) -> str:
    text = str(value or "").strip()
    if text:
        return text
    if fallback_name:
        return str(fallback_name).strip() or DEFAULT_NAME
    return DEFAULT_NAME


def map_base_path(name: str) -> Path:
    return SAVE_DIR / sanitize_name(name)


def _build_map_record_for_dir(name: str, storage_dir: Path) -> dict:
    sanitized_name = sanitize_name(name)
    base = storage_dir / sanitized_name
    yaml_path = base.with_suffix(".yaml")
    pgm_path = base.with_suffix(".pgm")
    posegraph_path = base.with_suffix(".posegraph")
    data_path = base.with_suffix(".data")
    metadata = get_map_metadata(sanitized_name)
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
        "building_name": normalize_building_name(metadata.get("building_name"), fallback_name=sanitized_name),
        "storage_dir": str(storage_dir),
        "yaml_path": str(yaml_path),
        "pgm_path": str(pgm_path),
        "posegraph_path": str(posegraph_path),
        "data_path": str(data_path),
        "yaml_exists": yaml_path.exists(),
        "pgm_exists": pgm_path.exists(),
        "posegraph_exists": posegraph_path.exists(),
        "data_exists": data_path.exists(),
        "ready_to_load": posegraph_path.exists() and data_path.exists(),
        "updated_at": updated_at,
    }


def load_metadata() -> dict:
    payload = load_json_file(METADATA_PATH)
    if not isinstance(payload, dict):
        return {"maps": {}}
    maps = payload.get("maps")
    if not isinstance(maps, dict):
        payload["maps"] = {}
    return payload


def save_metadata(payload: dict) -> None:
    atomic_write_json(METADATA_PATH, payload)


def get_map_metadata(name: str) -> dict:
    payload = load_metadata()
    maps = payload.get("maps") or {}
    entry = maps.get(name)
    return entry if isinstance(entry, dict) else {}


def update_map_metadata(name: str, building_name: str | None = None) -> dict:
    sanitized_name = sanitize_name(name)
    metadata = load_metadata()
    maps = metadata.setdefault("maps", {})
    entry = maps.get(sanitized_name)
    if not isinstance(entry, dict):
        entry = {}
        maps[sanitized_name] = entry

    entry["name"] = sanitized_name
    entry["building_name"] = normalize_building_name(building_name, fallback_name=sanitized_name)
    entry["updated_at"] = time.time()
    save_metadata(metadata)
    return entry


def build_map_record(name: str) -> dict:
    best_record = None
    best_score = None
    for storage_dir in MAP_SEARCH_DIRS:
        record = _build_map_record_for_dir(name, storage_dir)
        score = (
            1 if record["ready_to_load"] else 0,
            1 if any((record["yaml_exists"], record["pgm_exists"], record["posegraph_exists"], record["data_exists"])) else 0,
            record["updated_at"],
        )
        if best_score is None or score > best_score:
            best_record = record
            best_score = score
    return best_record


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def wait_for_saved_map_files(name: str, timeout_sec: float = SAVE_FILE_WAIT_TIMEOUT_SEC) -> dict:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        record = build_map_record(name)
        if record["yaml_exists"] and record["pgm_exists"] and record["posegraph_exists"] and record["data_exists"]:
            return record
        if time.monotonic() >= deadline:
            return record
        time.sleep(max(0.05, SAVE_FILE_WAIT_INTERVAL_SEC))


def list_saved_maps() -> list[dict]:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    names = set()
    for storage_dir in MAP_SEARCH_DIRS:
        if not storage_dir.exists():
            continue
        for pattern in ("*.yaml", "*.pgm", "*.posegraph", "*.data"):
            for path in storage_dir.glob(pattern):
                names.add(path.stem)
    return sorted(
        (build_map_record(name) for name in names),
        key=lambda item: (item["updated_at"], item["name"]),
        reverse=True,
    )


def write_status(state: str, message: str, **extra) -> None:
    payload = {
        "available": True,
        "state": state,
        "message": message,
        "updated_at": time.time(),
        "default_name": DEFAULT_NAME,
        "autoload_enabled": AUTOLOAD_ENABLED,
        "maps": list_saved_maps(),
    }
    payload.update(extra)
    atomic_write_json(STATUS_PATH, payload)


def run_service_call(service_name: str, service_type: str, request: str, timeout_sec: float = 90.0) -> str:
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source /ros_ws/install/setup.bash && "
        f"ros2 service call {service_name} {service_type} '{request}'"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if result.returncode != 0:
        raise RuntimeError(output or f"{service_name} failed with exit code {result.returncode}")
    return output


def _ros_shell_prefix() -> str:
    return "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash && "


def _ros_command(command: str) -> str:
    return _ros_shell_prefix() + command


def _read_wrapper_pid() -> int | None:
    try:
        return int(SLAM_TOOLBOX_WRAPPER_PIDFILE.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def restart_slam_toolbox() -> None:
    pid = _read_wrapper_pid()
    if pid is None:
        raise RuntimeError(f"slam_toolbox wrapper pid file is unavailable: {SLAM_TOOLBOX_WRAPPER_PIDFILE}")
    os.kill(pid, signal.SIGUSR1)
    time.sleep(1.0)


def set_slam_toolbox_mode(mode: str) -> None:
    normalized = str(mode or "").strip().lower()
    if normalized not in {"mapping", "localization"}:
        raise RuntimeError(f"Unsupported slam_toolbox mode '{mode}'")
    SLAM_TOOLBOX_MODE_FILE.write_text(normalized, encoding="utf-8")


def get_slam_toolbox_mode() -> str:
    try:
        return SLAM_TOOLBOX_MODE_FILE.read_text(encoding="utf-8").strip().lower() or "mapping"
    except Exception:
        return "mapping"


def wait_for_service(service_name: str, timeout_sec: float = 20.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["bash", "-lc", _ros_command(f"ros2 service list | grep -Fx {json.dumps(service_name)}")],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return
        time.sleep(0.2)
    raise RuntimeError(f"Timed out waiting for service '{service_name}'")


def load_start_pose(name: str) -> tuple[float, float, float]:
    record = build_map_record(name)
    start_path = Path(record["storage_dir"]) / f"{record['name']}.start.json"
    if not start_path.exists():
        return 0.0, 0.0, 0.0
    try:
        payload = load_json_file(start_path)
        return (
            float(payload.get("x", 0.0)),
            float(payload.get("y", 0.0)),
            float(payload.get("theta", 0.0)),
        )
    except Exception:
        return 0.0, 0.0, 0.0


def persist_autoload_name(name: str) -> None:
    record = build_map_record(name)
    atomic_write_json(
        AUTOLOAD_PATH,
        {
            "name": record["name"],
            "building_name": record["building_name"],
            "updated_at": time.time(),
        },
    )


def write_reset_status(state: str, message: str, **extra) -> None:
    payload = {
        "available": True,
        "state": state,
        "message": message,
        "updated_at": time.time(),
    }
    payload.update(extra)
    atomic_write_json(RESET_STATUS_PATH, payload)


def clear_current_map() -> None:
    try:
        ACTIVE_OVERRIDE_PATH.unlink(missing_ok=True)
    except Exception:
        pass
    set_slam_toolbox_mode("mapping")
    restart_slam_toolbox()


def save_map(name: str, building_name: str | None = None) -> dict:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    base = map_base_path(name)
    # save_map just triggers map_saver; we verify the files at the end.
    run_service_call(
        "/slam_toolbox/save_map",
        "slam_toolbox/srv/SaveMap",
        f'{{name: {{data: "{base}"}}}}',
    )

    # Posegraph serialization is only possible in mapping mode.
    if get_slam_toolbox_mode() != "localization":
        posegraph_path = base.with_suffix(".posegraph")
        data_path = base.with_suffix(".data")
        try:
            run_service_call(
                "/slam_toolbox/serialize_map",
                "slam_toolbox/srv/SerializePoseGraph",
                f'{{filename: "{base}"}}',
                timeout_sec=300.0,
            )
        except subprocess.TimeoutExpired:
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                if posegraph_path.exists() and data_path.exists():
                    if posegraph_path.stat().st_size > 0 and data_path.stat().st_size > 0:
                        break
                time.sleep(1.0)
            if not (posegraph_path.exists() and data_path.exists()):
                raise RuntimeError(
                    "serialize_map timed out and output files were not created."
                )

    update_map_metadata(base.name, building_name=building_name)
    record = wait_for_saved_map_files(base.name)
    # In localization mode only PGM/YAML are required; posegraph is optional.
    if not (record["yaml_exists"] and record["pgm_exists"]):
        raise RuntimeError("slam_toolbox reported success but the expected saved map files were not all written")
    if get_slam_toolbox_mode() != "localization" and not (record.get("posegraph_exists") and record.get("data_exists")):
        raise RuntimeError("slam_toolbox reported success but the expected saved map files were not all written")

    persist_autoload_name(base.name)
    return record


def load_map(name: str) -> dict:
    record = build_map_record(name)
    base = Path(record["storage_dir"]) / record["name"]
    if not record["ready_to_load"]:
        raise RuntimeError(f"Saved posegraph for '{record['name']}' is missing")

    initial_x, initial_y, initial_theta = load_start_pose(record["name"])
    set_slam_toolbox_mode("localization")
    restart_slam_toolbox()
    wait_for_service("/slam_toolbox/deserialize_map", timeout_sec=20.0)

    # Set the active override early so snapshot publishers know we're loading.
    try:
        ACTIVE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            ACTIVE_OVERRIDE_PATH,
            {
                "mode": "frozen",
                "name": record["name"],
                "building_name": record["building_name"],
                "updated_at": time.time(),
            },
        )
    except Exception:
        pass

    # Record file mtimes before deserialization so we can detect updates
    # if the DDS response is dropped by the middleware.
    pre_snapshot_mtime = _safe_mtime(FROZEN_SNAPSHOT_PATH)
    pre_map_mtime = _safe_mtime(SLAM_TAB_MAP_PATH)

    deserialize_timed_out = False
    try:
        run_service_call(
            "/slam_toolbox/deserialize_map",
            "slam_toolbox/srv/DeserializePoseGraph",
            (
                f'{{filename: "{base}", match_type: 3, '
                f'initial_pose: {{x: {initial_x}, y: {initial_y}, theta: {initial_theta}}}}}'
            ),
            timeout_sec=300.0,
        )
    except subprocess.TimeoutExpired:
        deserialize_timed_out = True

    if deserialize_timed_out:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            if _safe_mtime(FROZEN_SNAPSHOT_PATH) > pre_snapshot_mtime:
                break
            if _safe_mtime(SLAM_TAB_MAP_PATH) > pre_map_mtime:
                break
            time.sleep(1.0)
        else:
            raise RuntimeError(
                "deserialize_map timed out and no map update was detected after 60 s."
            )

    persist_autoload_name(record["name"])
    return record


def maybe_autoload_saved_map() -> None:
    if not AUTOLOAD_ENABLED:
        return
    payload = load_json_file(AUTOLOAD_PATH)
    if not isinstance(payload, dict):
        return
    name = sanitize_name(payload.get("name"))
    record = build_map_record(name)
    if not record["ready_to_load"]:
        return
    write_status("loading", f"Auto-loading saved map '{name}'.", action="autoload", name=name)
    try:
        record = load_map(name)
        write_status("loaded", f"Auto-loaded saved map '{name}'.", action="autoload", name=name, map_record=record)
    except Exception as exc:
        write_status("error", f"Failed to auto-load saved map '{name}': {exc}", action="autoload", name=name)


def main() -> None:
    last_request_id = None
    last_reset_request_id = None
    write_status("idle", "slam_toolbox map save/load manager is running.")
    maybe_autoload_saved_map()

    while True:
        reset_payload = load_json_file(RESET_REQUEST_PATH)
        if isinstance(reset_payload, dict):
            reset_request_id = str(reset_payload.get("request_id") or "")
            if reset_request_id and reset_request_id != last_reset_request_id:
                last_reset_request_id = reset_request_id
                try:
                    write_reset_status("resetting", "Clearing the current slam_toolbox map.", request_id=reset_request_id)
                    clear_current_map()
                    write_reset_status("reset", "Cleared the current slam_toolbox map.", request_id=reset_request_id)
                    write_status("idle", "slam_toolbox map save/load manager is running.")
                except Exception as exc:
                    write_reset_status("error", f"Failed to clear the current slam_toolbox map: {exc}", request_id=reset_request_id)
                finally:
                    try:
                        RESET_REQUEST_PATH.unlink(missing_ok=True)
                    except Exception:
                        pass

        request_payload = load_json_file(REQUEST_PATH)
        if not isinstance(request_payload, dict):
            current_status = load_json_file(STATUS_PATH)
            if not isinstance(current_status, dict) or current_status.get("state") in {"queued", "saving", "loading"}:
                write_status("idle", "slam_toolbox map save/load manager is running.")
            # Do NOT reset terminal statuses (saved, loaded, error) here.
            # The frontend's finalize_saved_toolbox_map_if_ready() consumes
            # "saved" by deleting the status file. Resetting it prematurely
            # causes the frontend to miss the finalization and never switch
            # out of recording mode after a save.
            time.sleep(max(0.2, POLL_PERIOD_SEC))
            continue

        request_id = str(request_payload.get("request_id") or "")
        if request_id and request_id == last_request_id:
            time.sleep(max(0.2, POLL_PERIOD_SEC))
            continue

        last_request_id = request_id
        action = str(request_payload.get("action") or "").strip().lower()
        name = sanitize_name(request_payload.get("name"))
        building_name = normalize_building_name(request_payload.get("building_name"), fallback_name=name)

        try:
            if action == "save":
                write_status(
                    "saving",
                    f"Saving current slam_toolbox map for building '{building_name}'.",
                    action=action,
                    name=name,
                    building_name=building_name,
                )
                record = save_map(name, building_name=building_name)
                write_status(
                    "saved",
                    f"Saved current map for building '{record['building_name']}'.",
                    action=action,
                    name=name,
                    building_name=record["building_name"],
                    map_record=record,
                )
            elif action == "load":
                record = build_map_record(name)
                write_status(
                    "loading",
                    f"Loading saved slam_toolbox map for building '{record['building_name']}'.",
                    action=action,
                    name=name,
                    building_name=record["building_name"],
                )
                record = load_map(name)
                write_status(
                    "loaded",
                    f"Loaded saved map for building '{record['building_name']}'.",
                    action=action,
                    name=name,
                    building_name=record["building_name"],
                    map_record=record,
                )
            else:
                raise RuntimeError(f"Unsupported toolbox map action '{action}'")
        except Exception as exc:
            write_status("error", str(exc), action=action or "unknown", name=name)
        finally:
            try:
                REQUEST_PATH.unlink(missing_ok=True)
            except Exception:
                pass

        time.sleep(max(0.2, POLL_PERIOD_SEC))


if __name__ == "__main__":
    main()
