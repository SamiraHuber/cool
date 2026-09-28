"""Unified tool adapter interface for simulation and real-world back-ends.

All agent code (simulator, pipeline strategy, chat agent) should call tools
through a ToolAdapter instance.  This guarantees identical JSON shapes
whether the back-end is a perfect simulation, a noisy simulation, or the
live PostgreSQL database.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/bordsupr")


class ToolAdapter(ABC):
    """Abstract interface for robot tool queries.

    Every concrete adapter (real DB, perfect simulation, noisy simulation)
    must return the *same* JSON-serializable dict shapes so that VLM prompts
    are back-end agnostic.
    """

    # ------------------------------------------------------------------
    # Navigation / curiosity tools
    # ------------------------------------------------------------------

    @abstractmethod
    def get_room_visit_history(self, room_list: list[str]) -> dict[str, Any]:
        """Return visit and change counts per room.

        Returns:
            {
                "rooms": [
                    {
                        "room": str,
                        "visits": int,
                        "changes": int,
                        "last_arrived": str | None,  # ISO-8601 or None
                    }
                ]
            }
        """
        ...

    @abstractmethod
    def get_room_change_rates(self, room_list: list[str]) -> dict[str, Any]:
        """Return changes-per-visit rate per room.

        Returns:
            {
                "rooms": [
                    {
                        "room": str,
                        "visits": int,
                        "changes": int,
                        "rate": float | None,
                        "status": "never visited" | "visited",
                    }
                ]
            }
        """
        ...

    @abstractmethod
    def get_time_since_last_change(self, room_list: list[str]) -> dict[str, Any]:
        """Return minutes since the last observed change per room.

        Returns:
            {
                "rooms": [
                    {
                        "room": str,
                        "minutes_ago": int | None,
                        "last_change": str | None,  # ISO-8601 or None
                        "status": "never observed a change" | "changed observed",
                    }
                ]
            }
        """
        ...

    @abstractmethod
    def get_stale_rooms(self, room_list: list[str], threshold_minutes: int = 15) -> dict[str, Any]:
        """Return rooms not visited recently, ordered by staleness.

        Returns:
            {
                "stale_rooms": [
                    {
                        "room": str,
                        "minutes_since_visit": int | None,
                        "never_visited": bool,
                    }
                ]
            }
        """
        ...

    # ------------------------------------------------------------------
    # Convenience dispatcher
    # ------------------------------------------------------------------


class RealToolAdapter(ToolAdapter):
    """Production adapter that queries the live PostgreSQL database."""

    def __init__(self, map_id: int | None = None) -> None:
        self.map_id = map_id

    def _get_conn(self):
        return psycopg2.connect(DATABASE_URL)

    def get_room_visit_history(self, room_list: list[str]) -> dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        room_name,
                        COUNT(*) AS visit_count,
                        MAX(arrived_at) AS last_arrived
                    FROM robot_visits
                    WHERE map_id = %s
                    GROUP BY room_name
                    ORDER BY room_name
                    """,
                    (self.map_id,),
                )
                rows = cur.fetchall()

        visit_history: dict[str, dict[str, Any]] = {}
        for room_name, visit_count, last_arrived in rows:
            visit_history[room_name] = {
                "visits": visit_count,
                "last_arrived": last_arrived.isoformat() if last_arrived else None,
            }

        return {
            "rooms": [
                {
                    "room": room,
                    "visits": visit_history.get(room, {}).get("visits", 0),
                    "changes": 0,  # populated by get_room_change_rates if needed
                    "last_arrived": visit_history.get(room, {}).get("last_arrived"),
                }
                for room in room_list
            ]
        }

    def get_room_change_rates(self, room_list: list[str]) -> dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        target_room,
                        COUNT(*) FILTER (WHERE scene_changed = true) AS changes,
                        COUNT(*) AS total_decisions
                    FROM navigation_decisions
                    WHERE map_id = %s AND target_room IS NOT NULL
                    GROUP BY target_room
                    ORDER BY target_room
                    """,
                    (self.map_id,),
                )
                rows = cur.fetchall()

        change_rates: dict[str, dict[str, Any]] = {}
        for target_room, changes, total in rows:
            rate = changes / total if total > 0 else 0.0
            change_rates[target_room] = {
                "visits": total,
                "changes": changes,
                "rate": round(rate, 3),
            }

        return {
            "rooms": [
                {
                    "room": room,
                    "visits": change_rates.get(room, {}).get("visits", 0),
                    "changes": change_rates.get(room, {}).get("changes", 0),
                    "rate": change_rates.get(room, {}).get("rate") if room in change_rates else None,
                    "status": "never visited" if room not in change_rates else "visited",
                }
                for room in room_list
            ]
        }

    def get_time_since_last_change(self, room_list: list[str]) -> dict[str, Any]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        target_room,
                        MAX(created_at) AS last_change_time
                    FROM navigation_decisions
                    WHERE map_id = %s AND scene_changed = true AND target_room IS NOT NULL
                    GROUP BY target_room
                    ORDER BY target_room
                    """,
                    (self.map_id,),
                )
                rows = cur.fetchall()

        now = datetime.now(timezone.utc)
        time_since: dict[str, dict[str, Any]] = {}
        for target_room, last_change_time in rows:
            if last_change_time:
                minutes_ago = int((now - last_change_time).total_seconds() / 60)
                time_since[target_room] = {
                    "minutes_ago": minutes_ago,
                    "last_change": last_change_time.isoformat(),
                }

        return {
            "rooms": [
                {
                    "room": room,
                    "minutes_ago": time_since.get(room, {}).get("minutes_ago"),
                    "last_change": time_since.get(room, {}).get("last_change"),
                    "status": "never observed a change" if room not in time_since else "changed observed",
                }
                for room in room_list
            ]
        }

    def get_stale_rooms(self, room_list: list[str], threshold_minutes: int = 15) -> dict[str, Any]:
        # Re-use visit history to compute staleness
        visit_history = self.get_room_visit_history(room_list)
        now = datetime.now(timezone.utc)
        stale: list[dict[str, Any]] = []

        for room in room_list:
            entry = next((r for r in visit_history.get("rooms", []) if r["room"] == room), None)
            last = entry.get("last_arrived") if entry else None
            if last is None:
                stale.append({
                    "room": room,
                    "minutes_since_visit": None,
                    "never_visited": True,
                })
            else:
                last_dt = datetime.fromisoformat(last)
                minutes_ago = int((now - last_dt).total_seconds() / 60)
                if minutes_ago >= threshold_minutes:
                    stale.append({
                        "room": room,
                        "minutes_since_visit": minutes_ago,
                        "never_visited": False,
                    })

        stale.sort(key=lambda x: x["minutes_since_visit"] or float("inf"), reverse=True)
        return {"stale_rooms": stale[:5]}
