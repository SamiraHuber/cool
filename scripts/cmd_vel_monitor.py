import json
import os
import tempfile
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


STATUS_PATH = Path(os.getenv("CMD_VEL_STATUS_PATH", "/shared/cmd_vel_status.json"))
TOPIC_NAME = os.getenv("CMD_VEL_TOPIC", "/spot/cmd_vel")
WRITE_MIN_PERIOD_SEC = float(os.getenv("CMD_VEL_WRITE_MIN_PERIOD_SEC", "0.1"))
WRITE_HEARTBEAT_SEC = float(os.getenv("CMD_VEL_WRITE_HEARTBEAT_SEC", "0.5"))


def twist_to_payload(message: Twist) -> dict:
    return {
        "linear": {
            "x": float(message.linear.x),
            "y": float(message.linear.y),
            "z": float(message.linear.z),
        },
        "angular": {
            "x": float(message.angular.x),
            "y": float(message.angular.y),
            "z": float(message.angular.z),
        },
    }


def is_nonzero_twist(payload: dict) -> bool:
    linear = payload.get("linear") or {}
    angular = payload.get("angular") or {}
    values = (
        float(linear.get("x") or 0.0),
        float(linear.get("y") or 0.0),
        float(linear.get("z") or 0.0),
        float(angular.get("x") or 0.0),
        float(angular.get("y") or 0.0),
        float(angular.get("z") or 0.0),
    )
    return any(abs(value) > 1e-6 for value in values)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8") as tmp:
        json.dump(payload, tmp)
        temp_path = tmp.name
    os.replace(temp_path, path)


class CmdVelMonitor(Node):
    def __init__(self) -> None:
        super().__init__("cmd_vel_monitor")

        self._last_signature: tuple[float, float, float, float, float, float] | None = None
        self._last_write_at = 0.0

        self.create_subscription(Twist, TOPIC_NAME, self._handle_cmd_vel, 10)
        atomic_write_json(
            STATUS_PATH,
            {
                "available": False,
                "topic": TOPIC_NAME,
                "message": "No cmd_vel received yet.",
                "updated_at": time.time(),
                "last_nonzero": None,
            },
        )
        self.get_logger().info(f"cmd_vel monitor: watching {TOPIC_NAME}, status at {STATUS_PATH}")

    def _handle_cmd_vel(self, message: Twist) -> None:
        now = time.time()
        signature = (
            round(float(message.linear.x), 6),
            round(float(message.linear.y), 6),
            round(float(message.linear.z), 6),
            round(float(message.angular.x), 6),
            round(float(message.angular.y), 6),
            round(float(message.angular.z), 6),
        )

        if self._last_signature == signature and (now - self._last_write_at) < WRITE_HEARTBEAT_SEC:
            return
        if (now - self._last_write_at) < WRITE_MIN_PERIOD_SEC:
            return

        twist_payload = twist_to_payload(message)
        previous_payload = None
        try:
            if STATUS_PATH.exists():
                previous_payload = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            previous_payload = None

        status_payload = {
            "available": True,
            "topic": TOPIC_NAME,
            "updated_at": now,
            **twist_payload,
            "last_nonzero": (previous_payload or {}).get("last_nonzero"),
        }
        if is_nonzero_twist(twist_payload):
            status_payload["last_nonzero"] = {
                "updated_at": now,
                **twist_payload,
            }

        atomic_write_json(STATUS_PATH, status_payload)
        self._last_signature = signature
        self._last_write_at = now


def main() -> None:
    rclpy.init()
    node = CmdVelMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()