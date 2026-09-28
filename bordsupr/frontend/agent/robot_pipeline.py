"""Robot pipeline simulator runner.

Orchestrates a pipeline simulation run:
- Creates a map + rooms for the run
- Starts the video publisher in the bordsupr container
- Polls the DB for new scenes
- Runs the v4 strategy on each scene
- Saves decisions to navigation_decisions and robot_visits
- Streams progress events via a thread-safe queue
"""

from __future__ import annotations

import datetime
import json
import os
import queue
import shlex
import threading
import time
from pathlib import Path
from datetime import timezone
from typing import Any

import psycopg2

from .pipeline_strategy import run_v4_decision, run_v5_decision

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")
VIDEO_PUBLISHER_CONTAINER = os.getenv("VIDEO_PUBLISHER_CONTAINER", "bordsupr")
VIDEO_PUBLISHER_PID_FILE = os.getenv("VIDEO_PUBLISHER_PID_FILE", "/tmp/bordsupr_video_publisher.pid")
VIDEO_PUBLISHER_LOG_FILE = os.getenv("VIDEO_PUBLISHER_LOG_FILE", "/tmp/bordsupr_video_publisher.log")
VIDEO_PUBLISHER_SCRIPT_PATH = os.getenv(
    "VIDEO_PUBLISHER_SCRIPT_PATH", "/workspace/src/bordsupr/bordsupr/video_publisher.py")
BORDSUPR_HOST_RUNTIME_ROOT = os.getenv(
    "BORDSUPR_HOST_RUNTIME_ROOT", str(Path(__file__).resolve().parents[3] / "bordsupr/runtime")
)
BORDSUPR_CONTAINER_RUNTIME_ROOT = os.getenv(
    "BORDSUPR_CONTAINER_RUNTIME_ROOT", "/workspace/src")


def _get_conn():
    return psycopg2.connect(DATABASE_URL)


from app import _docker_exec


def _to_container_frame_dir(frame_dir: str) -> str:
    """Translate a host frame dir path to the container equivalent."""
    p = os.path.normpath(os.path.expanduser(frame_dir))

    # Try extra runtime mounts first (e.g. /host/path=/container/path)
    extra_mounts = os.getenv("VIDEO_PUBLISHER_EXTRA_RUNTIME_MOUNTS", "")
    for pair in extra_mounts.split(";"):
        if "=" not in pair:
            continue
        host_part, container_part = pair.split("=", 1)
        host_part = os.path.normpath(os.path.expanduser(host_part.strip()))
        container_part = os.path.normpath(container_part.strip())
        try:
            rel = os.path.relpath(p, host_part)
            if not rel.startswith(".."):
                return os.path.normpath(os.path.join(container_part, rel))
        except ValueError:
            pass

    # Fall back to the primary runtime root mapping
    host_root = os.path.normpath(os.path.expanduser(BORDSUPR_HOST_RUNTIME_ROOT))
    container_root = os.path.normpath(BORDSUPR_CONTAINER_RUNTIME_ROOT)
    try:
        rel = os.path.relpath(p, host_root)
        if not rel.startswith(".."):
            return os.path.normpath(os.path.join(container_root, rel))
    except ValueError:
        pass
    return p


def _create_map_and_rooms(room_list: list[str]) -> tuple[int, str]:
    """Create a new map and rooms for the pipeline run.

    Returns:
        (map_id, map_name)
    """
    map_name = f"pipeline_run_{datetime.datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO maps (name) VALUES (%s) RETURNING id", (map_name,))
            map_id = cur.fetchone()[0]

            for i, room_name in enumerate(room_list):
                x1 = float(i * 2.0)
                y1 = 0.0
                x2 = x1 + 1.0
                y2 = 1.0
                cur.execute(
                    """
                    INSERT INTO rooms (name, x1, y1, x2, y2, map_id, map_name)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (room_name, x1, y1, x2, y2, map_id, map_name),
                )
        conn.commit()
    return map_id, map_name


def _ensure_database_node() -> None:
    """Start database_node in the bordsupr container if it is not running."""
    check_script = """
for pid in $(pgrep -f "database_node" 2>/dev/null); do
  if [ -r "/proc/$pid/stat" ]; then
    state=$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null || true)
    if [ "$state" != "Z" ]; then
      echo "running"
      exit 0
    fi
  fi
done
echo "not_running"
"""
    result = _docker_exec(check_script)
    status = (result.stdout or "").strip()
    if status == "running":
        return

    start_script = """
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
nohup ros2 run bordsupr database_node >/tmp/database_node_manual.log 2>&1 &
echo $! > /tmp/database_node_manual.pid
sleep 2
if pgrep -f "database_node" >/dev/null 2>&1; then
  echo "started"
else
  echo "failed"
fi
"""
    result = _docker_exec(start_script)
    if "started" not in (result.stdout or ""):
        import logging
        logging.getLogger(__name__).warning(
            "database_node does not appear to be running in %s: %s",
            VIDEO_PUBLISHER_CONTAINER,
            result.stderr or result.stdout,
        )


def _start_video_publisher(
    frame_dir: str,
    stride: int = 5,
    hz: float = 1.0,
    generate_captions: bool = True,
) -> dict:
    """Start the video publisher in the bordsupr container.

    Returns the status payload or raises on error.
    """
    container_frame_dir = _to_container_frame_dir(frame_dir)
    loop_value = "false"
    captions_value = "true" if generate_captions else "false"

    script = f"""set -e
PID_FILE={shlex.quote(VIDEO_PUBLISHER_PID_FILE)}
LOG_FILE={shlex.quote(VIDEO_PUBLISHER_LOG_FILE)}
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
if [ ! -d {shlex.quote(container_frame_dir)} ]; then
  echo "missing_frame_dir:{container_frame_dir}"
  exit 12
fi
source /opt/ros/humble/setup.bash
if [ -f /workspace/install/setup.bash ]; then
  source /workspace/install/setup.bash
fi
if ! timeout 15s ros2 param set /scene_description_node captions_enabled {captions_value} >/tmp/bordsupr_caption_param_set.log 2>&1; then
  cat /tmp/bordsupr_caption_param_set.log >&2 || true
fi
nohup python3.10 {shlex.quote(VIDEO_PUBLISHER_SCRIPT_PATH)} --ros-args \\
  -p rgb_topic:=/spot/camera/frontleft/image_rotated \\
  -p image_folder:={shlex.quote(container_frame_dir)} \\
  -p publish_hz:={hz} \\
  -p image_stride:={stride} \\
  -p loop:={loop_value} \\
  -p recursive:=false \\
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
"""
    result = _docker_exec(script)
    if result.returncode == 20:
        raise RuntimeError("Video publisher is already running.")
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to start video publisher: {result.stderr or result.stdout}"
        )
    return {"status": "started", "frame_dir": frame_dir}


def _stop_video_publisher() -> dict:
    """Stop the video publisher in the bordsupr container."""
    script = f"""
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
    _docker_exec(script)
    return {"status": "stopped"}


def _publisher_status() -> dict:
    """Check if the video publisher is running."""
    script = f"""
PID_FILE={shlex.quote(VIDEO_PUBLISHER_PID_FILE)}
status="stopped"
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
printf '%s\\n' "$status"
"""
    result = _docker_exec(script)
    status = (result.stdout or "").strip().splitlines()[0] if result.stdout else "unknown"
    return {"status": status}


def _count_frames(frame_dir: str) -> int:
    """Count image files in the frame directory."""
    try:
        import pathlib
        path = pathlib.Path(frame_dir)
        if not path.exists():
            return 0
        count = 0
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            count += len(list(path.glob(ext)))
        return count
    except Exception:
        return 0


class RobotPipelineRunner:
    """Orchestrates a robot pipeline simulation run."""

    def __init__(self) -> None:
        self.run_id: str | None = None
        self.map_id: int | None = None
        self.map_name: str | None = None
        self.room_list: list[str] = []
        self.frames_per_room: int = 5
        self.poll_interval_seconds: float = 5.0
        self.pause_on_stay: bool = True
        self.room_frame_counter: int = 0
        self.current_room: str | None = None
        self.last_processed_scene_id: int | None = None
        self.step_number: int = 0
        self.total_frame_count: int = 0
        self.processed_frame_count: int = 0
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._sse_queue: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._status: str = "idle"  # idle, running, completed, error
        self._error: str | None = None
        self._latest_decision: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        frame_dir: str,
        room_list: list[str],
        frames_per_room: int = 5,
        poll_interval_seconds: float = 5.0,
        pause_on_stay: bool = True,
        stride: int = 5,
        hz: float = 1.0,
        generate_captions: bool = True,
        strategy_type: str = "v4",
    ) -> str:
        """Start a new pipeline simulation run."""
        with self._lock:
            if self._status == "running":
                self.stop()

            self._reset_state()
            self.room_list = [r.strip() for r in room_list if r.strip()]
            if not self.room_list:
                raise ValueError("room_list must not be empty")
            self.frames_per_room = max(1, frames_per_room)
            self.poll_interval_seconds = max(1.0, poll_interval_seconds)
            self.pause_on_stay = pause_on_stay
            self.total_frame_count = _count_frames(frame_dir)

            # Create map + rooms
            self.map_id, self.map_name = _create_map_and_rooms(self.room_list)
            self.run_id = self.map_name
            self.strategy_type = strategy_type
            self._status = "running"

        # Ensure database_node is running so scenes get saved
        _ensure_database_node()

        # Start video publisher
        _start_video_publisher(frame_dir, stride, hz, generate_captions)

        # Activate map so the ROS pipeline saves scenes under it
        _activate_toolbox_map(self.map_name)

        # Emit started event
        self._put_event({
            "type": "started",
            "run_id": self.run_id,
            "map_name": self.map_name,
            "room_list": self.room_list,
            "frame_dir": frame_dir,
            "total_frames": self.total_frame_count,
        })

        # Start polling thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(frame_dir,),
            daemon=True,
        )
        self._thread.start()

        return self.run_id

    def stop(self) -> dict:
        """Stop the current run."""
        with self._lock:
            if self._status != "running":
                return {"status": self._status, "run_id": self.run_id}

        self._stop_event.set()
        _stop_video_publisher()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)

        with self._lock:
            self._close_open_visit()
            if self._status == "running":
                self._status = "completed"

        self._put_event({"type": "done", "run_id": self.run_id, "status": self._status})
        return {"status": self._status, "run_id": self.run_id}

    def get_status(self) -> dict:
        """Return current run status."""
        with self._lock:
            return {
                "run_id": self.run_id,
                "status": self._status,
                "current_room": self.current_room,
                "step_number": self.step_number,
                "total_frames": self.total_frame_count,
                "processed_frames": self.processed_frame_count,
                "room_frame_counter": self.room_frame_counter,
                "map_id": self.map_id,
                "map_name": self.map_name,
                "latest_decision": self._latest_decision,
                "error": self._error,
            }

    def get_sse_queue(self) -> queue.Queue:
        """Return the SSE event queue."""
        return self._sse_queue

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.run_id = None
        self.map_id = None
        self.map_name = None
        self.room_list = []
        self.frames_per_room = 5
        self.poll_interval_seconds = 5.0
        self.pause_on_stay = True
        self.room_frame_counter = 0
        self.current_room = None
        self.last_processed_scene_id = None
        self.step_number = 0
        self.total_frame_count = 0
        self.processed_frame_count = 0
        self._status = "idle"
        self._error = None
        self._latest_decision = None
        # Clear queue
        while not self._sse_queue.empty():
            try:
                self._sse_queue.get_nowait()
            except queue.Empty:
                break

    def _run_loop(self, frame_dir: str) -> None:
        """Background polling loop."""
        heartbeat_counter = 0
        try:
            while not self._stop_event.is_set():
                # Check publisher status
                pub_status = _publisher_status()
                if pub_status["status"] == "stopped":
                    # Publisher finished — check if we should auto-stop
                    if self._should_auto_stop():
                        break

                # Query new scenes
                new_scenes = self._fetch_new_scenes()
                if new_scenes:
                    for scene in new_scenes:
                        if self._stop_event.is_set():
                            break
                        # Emit scene-discovered event before processing
                        self._put_event({
                            "type": "scene",
                            "run_id": self.run_id,
                            "scene_id": scene["id"],
                            "caption": scene.get("caption") or "",
                        })
                        self._process_scene(scene)
                else:
                    heartbeat_counter += 1
                    if heartbeat_counter >= 3:
                        heartbeat_counter = 0
                        self._put_event({
                            "type": "heartbeat",
                            "run_id": self.run_id,
                            "status": "polling",
                            "processed_frames": self.processed_frame_count,
                            "current_room": self.current_room,
                        })

                # Sleep until next poll
                time.sleep(self.poll_interval_seconds)

            # Loop ended normally
            with self._lock:
                if self._status == "running":
                    self._status = "completed"
                self._close_open_visit()

        except Exception as exc:
            with self._lock:
                self._status = "error"
                self._error = str(exc)
            self._put_event({"type": "error", "error": str(exc)})
        finally:
            _stop_video_publisher()
            self._put_event({"type": "done", "run_id": self.run_id, "status": self._status})

    def _fetch_new_scenes(self) -> list[dict]:
        """Fetch scenes from DB that we haven't processed yet."""
        with _get_conn() as conn:
            with conn.cursor() as cur:
                last_id = self.last_processed_scene_id or 0
                cur.execute(
                    """
                    SELECT id, caption, timestamp, source_frame
                    FROM scenes
                    WHERE map_id = %s AND id > %s
                    ORDER BY id
                    """,
                    (self.map_id, last_id),
                )
                rows = cur.fetchall()
                return [
                    {
                        "id": row[0],
                        "caption": row[1],
                        "timestamp": row[2],
                        "source_frame": row[3],
                    }
                    for row in rows
                ]

    def _process_scene(self, scene: dict) -> None:
        """Process a single scene: assign room, run v4, save decision."""
        with self._lock:
            self.step_number += 1
            self.processed_frame_count += 1
            self.last_processed_scene_id = scene["id"]

            # Determine current room
            room_index = self.room_frame_counter // self.frames_per_room
            assigned_room = self.room_list[room_index % len(self.room_list)]
            previous_room = self.current_room
            self.current_room = assigned_room

        # Run decision
        if getattr(self, "strategy_type", "v4") == "v5":
            decision = run_v5_decision(
                scene_id=scene["id"],
                room_name=assigned_room,
                scene_caption=scene["caption"] or "",
                map_id=self.map_id,
                room_list=self.room_list,
                step_number=self.step_number,
            )
        else:
            decision = run_v4_decision(
                scene_id=scene["id"],
                room_name=assigned_room,
                scene_caption=scene["caption"] or "",
                map_id=self.map_id,
                room_list=self.room_list,
                step_number=self.step_number,
            )

        with self._lock:
            self._latest_decision = decision

        # Save decision to DB
        self._save_decision(scene["id"], assigned_room, decision)

        # Handle room transitions and round-robin
        action = decision.get("action", "stay")
        if action == "move" or not self.pause_on_stay:
            with self._lock:
                self.room_frame_counter += 1

        # If room changed (either by round-robin or explicit move), update visits
        next_room_index = self.room_frame_counter // self.frames_per_room
        next_room = self.room_list[next_room_index % len(self.room_list)]
        if previous_room != next_room and previous_room is not None:
            self._transition_visit(previous_room, next_room)

        # Emit SSE event
        self._put_event({
            "type": "step",
            "run_id": self.run_id,
            "step_number": self.step_number,
            "scene_id": scene["id"],
            "room": assigned_room,
            "caption": scene["caption"],
            "decision": decision,
        })

    def _save_decision(
        self,
        scene_id: int,
        room_name: str,
        decision: dict,
    ) -> None:
        """Insert into navigation_decisions."""
        tool_calls = decision.get("tool_calls") or []
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO navigation_decisions
                    (decision_type, target_room, target_x, target_y,
                     dwell_time_seconds, reasoning, scene_change_prediction,
                     map_id, scene_changed, change_severity, activities_changed,
                     tool_calls_json, step_number)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        "agent_scene_change_v4",
                        decision.get("target_room"),
                        None,  # target_x — resolved later if needed
                        None,  # target_y
                        None,  # dwell_time_seconds
                        decision.get("reasoning"),
                        None,  # scene_change_prediction
                        self.map_id,
                        decision.get("scene_changed"),
                        decision.get("change") or decision.get("change_severity"),
                        decision.get("activities_changed"),
                        json.dumps(tool_calls) if tool_calls else None,
                        self.step_number,
                    ),
                )
            conn.commit()

    def _transition_visit(self, from_room: str, to_room: str) -> None:
        """Close visit in from_room, open visit in to_room."""
        now = datetime.datetime.now(timezone.utc)
        with _get_conn() as conn:
            with conn.cursor() as cur:
                # Close previous visit
                cur.execute(
                    """
                    UPDATE robot_visits
                    SET departed_at = %s
                    WHERE ctid = (
                        SELECT ctid FROM robot_visits
                        WHERE room_name = %s AND map_id = %s AND departed_at IS NULL
                        ORDER BY arrived_at DESC
                        LIMIT 1
                    )
                    """,
                    (now, from_room, self.map_id),
                )
                # Open new visit
                cur.execute(
                    """
                    INSERT INTO robot_visits
                    (room_name, map_id, arrived_at, scene_count)
                    VALUES (%s, %s, %s, 0)
                    RETURNING id
                    """,
                    (to_room, self.map_id, now),
                )
            conn.commit()

    def _close_open_visit(self) -> None:
        """Close any open visit for the current run."""
        if self.map_id is None:
            return
        now = datetime.datetime.now(timezone.utc)
        with _get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE robot_visits
                    SET departed_at = %s
                    WHERE map_id = %s AND departed_at IS NULL
                    """,
                    (now, self.map_id),
                )
            conn.commit()

    def _should_auto_stop(self) -> bool:
        """Check if we should auto-stop (publisher finished + no new scenes)."""
        # If we've processed at least as many frames as expected, stop
        if self.total_frame_count > 0 and self.processed_frame_count >= self.total_frame_count:
            return True
        # If publisher is stopped and no new scenes for 2 poll intervals
        # (this is handled by the loop: we fetch 0 scenes and sleep)
        # After a few empty polls, stop
        return False

    def _put_event(self, event: dict) -> None:
        """Put an event into the SSE queue."""
        try:
            self._sse_queue.put_nowait(event)
        except queue.Full:
            pass


# ------------------------------------------------------------------------------
# Toolbox map activation helper
# ------------------------------------------------------------------------------

def _activate_toolbox_map(map_name: str) -> None:
    """Write the active toolbox map record so the ROS pipeline uses this map."""
    import pathlib
    active_path = pathlib.Path(os.getenv("TOOLBOX_MAP_ACTIVE_PATH", "/shared/maps/toolbox_saved/active.json"))
    active_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"name": map_name}
    active_path.write_text(json.dumps(payload), encoding="utf-8")


# ------------------------------------------------------------------------------
# Global runner instance
# ------------------------------------------------------------------------------

_runner_lock = threading.Lock()
_global_runner: RobotPipelineRunner | None = None


def get_runner() -> RobotPipelineRunner:
    """Return the global runner instance, creating it if needed."""
    global _global_runner
    with _runner_lock:
        if _global_runner is None:
            _global_runner = RobotPipelineRunner()
        return _global_runner


