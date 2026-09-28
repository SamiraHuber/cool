import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path


REQUEST_PATH = Path(os.getenv("NAV_STACK_REQUEST_PATH", "/shared/nav_stack_request.json"))
STATUS_PATH = Path(os.getenv("NAV_STACK_STATUS_PATH", "/shared/nav_stack_status.json"))
POLL_PERIOD_SEC = float(os.getenv("NAV_STACK_MANAGER_PERIOD_SEC", "2.0"))
ROS2_CLI_TIMEOUT_SEC = float(os.getenv("NAV_STACK_ROS2_TIMEOUT_SEC", "8.0"))
MUTATE_LIFECYCLE = os.getenv("NAV_STACK_MANAGER_MUTATE_LIFECYCLE", "0").strip().lower() in {"1", "true", "yes"}

DESIRED_ACTIVE_NODES = (
    "/planner_server",
    "/smoother_server",
    "/controller_server",
    "/behavior_server",
    "/bt_navigator",
    "/velocity_smoother",
)

LIFECYCLE_STATE_PATTERN = re.compile(
    r"\b(unconfigured|inactive|active|finalized|unknown|errorprocessing)\b",
    re.IGNORECASE,
)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


def _run_ros2_command(*args: str) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=ROS2_CLI_TIMEOUT_SEC,
        )
    except FileNotFoundError:
        return 127, "ros2 CLI not found"
    except subprocess.TimeoutExpired:
        return 124, f"command timed out after {ROS2_CLI_TIMEOUT_SEC:.1f}s"

    output = "\n".join(part for part in (result.stdout.strip(), result.stderr.strip()) if part).strip()
    return result.returncode, output


def _get_lifecycle_state(node_name: str) -> tuple[str, str]:
    code, output = _run_ros2_command("ros2", "lifecycle", "get", node_name)
    if code != 0 and not output:
        return "missing", f"ros2 lifecycle get failed with exit code {code}"
    if "Node not found" in output:
        return "missing", output

    match = LIFECYCLE_STATE_PATTERN.search(output)
    if match:
        return match.group(1).lower(), output
    return "unknown", output or f"ros2 lifecycle get returned exit code {code}"


def _set_lifecycle_state(node_name: str, transition: str) -> tuple[bool, str]:
    code, output = _run_ros2_command("ros2", "lifecycle", "set", node_name, transition)
    success = code == 0 and "Transitioning successful" in output
    return success, output or f"ros2 lifecycle set returned exit code {code}"


def _ensure_node_active(node_name: str) -> dict:
    state, detail = _get_lifecycle_state(node_name)
    result = {
        "state": state,
        "detail": detail,
        "actions": [],
        "active": state == "active",
    }

    if state == "missing":
        return result

    if not MUTATE_LIFECYCLE:
        return result

    if state == "unconfigured":
        success, output = _set_lifecycle_state(node_name, "configure")
        result["actions"].append({"transition": "configure", "ok": success, "output": output})
        if not success:
            result["detail"] = output
            return result
        state, detail = _get_lifecycle_state(node_name)
        result["state"] = state
        result["detail"] = detail

    if state == "inactive":
        success, output = _set_lifecycle_state(node_name, "activate")
        result["actions"].append({"transition": "activate", "ok": success, "output": output})
        if not success:
            result["detail"] = output
            return result
        state, detail = _get_lifecycle_state(node_name)
        result["state"] = state
        result["detail"] = detail

    result["active"] = result["state"] == "active"
    return result


def build_status_payload() -> dict:
    request_exists = REQUEST_PATH.exists()
    node_results = {node_name: _ensure_node_active(node_name) for node_name in DESIRED_ACTIVE_NODES}
    missing_nodes = [name for name, result in node_results.items() if result["state"] == "missing"]
    inactive_nodes = [name for name, result in node_results.items() if not result["active"] and result["state"] != "missing"]

    if all(result["active"] for result in node_results.values()):
        state = "active"
        message = "Nav2 lifecycle nodes are active and ready for navigation requests."
    elif missing_nodes:
        state = "starting"
        message = f"Waiting for Nav2 nodes: {', '.join(missing_nodes)}"
    elif inactive_nodes:
        state = "starting"
        message = f"Activating Nav2 nodes: {', '.join(inactive_nodes)}"
    else:
        state = "degraded"
        message = "Nav2 lifecycle bringup is incomplete."

    if request_exists and state != "active":
        message = f"{message} A navigation request is queued."

    return {
        "available": True,
        "state": state,
        "updated_at": time.time(),
        "request_pending": request_exists,
        "message": message,
        "nodes": node_results,
    }


def main() -> None:
    while True:
        status_payload = build_status_payload()
        atomic_write_json(STATUS_PATH, status_payload)
        time.sleep(max(0.5, POLL_PERIOD_SEC))


if __name__ == "__main__":
    main()
