#!/usr/bin/env python3
"""Baseline answer engines for robot-perception QA and navigation evaluation.

Each baseline receives the same questions as the full agent and returns an
answer string (and optionally a tool_log for navigation grading).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:35432/bordsupr")


def _get_db_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _extract_object_name(question: str) -> str | None:
    """Find patterns like bottle_001, coffee_machine_001, laptop_001, etc."""
    m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*_\d+)", question)
    if m:
        return m.group(1)
    # Also try generic names that might be in the DB, but return the generic
    # name itself; the caller can resolve it to a specific DB object if needed.
    for pattern in [r"coffee machine", r"dishwasher", r"headphones", r"laptop", r"mug", r"plate", r"bottle", r"notebook"]:
        if re.search(pattern, question, re.IGNORECASE):
            return pattern.replace(" ", "_")
    return None


def _resolve_generic_name(generic: str) -> str | None:
    """Map a generic name like 'coffee_machine' to the first matching object name."""
    try:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT name FROM objects WHERE name ILIKE %s LIMIT 1",
                (f"%{generic}%",),
            )
            row = cur.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def _extract_person_name(question: str) -> str | None:
    """Find capitalized person names like Simba, Nala, Mufasa, Scar."""
    names = ["Simba", "Nala", "Mufasa", "Scar"]
    for name in names:
        if name.lower() in question.lower():
            return name
    return None


def _extract_time(question: str) -> str | None:
    """Find HH:MM or ISO timestamps. Returns full ISO datetime if available."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2})", question)
    if m:
        return m.group(1)
    m = re.search(r"(\d{2}:\d{2})", question)
    if m:
        return m.group(1)
    return None


def _extract_room_name(question: str) -> str | None:
    """Find room names mentioned in the question."""
    rooms = ["kitchen", "meeting room", "open office", "private office"]
    for room in rooms:
        if room in question.lower():
            return room.title() if room != "meeting room" else "Meeting Room"
    return None


# ---------------------------------------------------------------------------
# Baseline 1 – Direct LLM with flattened context
# ---------------------------------------------------------------------------

class DirectLLMBaseline:
    """Give the LLM the question plus a serialized textual snapshot of the DB."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")
        # Default to localhost:8000 when running outside Docker; vlm_server only resolves inside containers.
        self.vlm_url = os.getenv("VLM_API_URL", "http://localhost:8000/v1")
        self.vlm_model = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-4B-Instruct")

    def _build_context(self, building: str | None = None) -> str:
        conn = _get_db_conn()
        lines: list[str] = []

        # Building / map
        with conn.cursor() as cur:
            cur.execute("SELECT id, name FROM maps ORDER BY id")
            maps = cur.fetchall()
        if maps:
            lines.append("BUILDINGS:")
            for m_id, m_name in maps:
                lines.append(f"  - {m_name} (id={m_id})")

        # Rooms
        with conn.cursor() as cur:
            cur.execute(
                "SELECT r.name, m.name FROM rooms r JOIN maps m ON r.map_id = m.id ORDER BY r.name"
            )
            rooms = cur.fetchall()
        if rooms:
            lines.append("ROOMS:")
            for r_name, m_name in rooms:
                lines.append(f"  - {r_name} (in {m_name})")

        # People
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.name, o.id,
                       oo.x,
                       oo.y,
                       oo.created_at
                FROM objects o
                LEFT JOIN LATERAL (
                    SELECT object_observations.x, object_observations.y, object_observations.created_at
                    FROM object_observations
                    WHERE object_observations.object_id = o.id
                    ORDER BY object_observations.created_at DESC
                    LIMIT 1
                ) oo ON true
                WHERE o.class_id = 0 AND o.name IS NOT NULL
                ORDER BY o.name
                """
            )
            people = cur.fetchall()
        if people:
            lines.append("PEOPLE (latest known location):")
            for p_name, p_id, x, y, ts in people:
                loc = f"({x:.2f}, {y:.2f})" if x is not None else "unknown location"
                lines.append(f"  - {p_name} (id={p_id}) last seen at {loc} at {ts}")

        # Objects (non-person)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT o.name, o.id,
                       oo.x,
                       oo.y
                FROM objects o
                LEFT JOIN LATERAL (
                    SELECT object_observations.x, object_observations.y
                    FROM object_observations
                    WHERE object_observations.object_id = o.id
                    ORDER BY object_observations.created_at DESC
                    LIMIT 1
                ) oo ON true
                WHERE o.class_id != 0 AND o.name IS NOT NULL
                ORDER BY o.name
                LIMIT 30
                """
            )
            objects = cur.fetchall()
        if objects:
            lines.append("OBJECTS (latest known location):")
            for o_name, o_id, x, y in objects:
                loc = f"({x:.2f}, {y:.2f})" if x is not None else "unknown location"
                lines.append(f"  - {o_name} (id={o_id}) last seen at {loc}")

        # Recent interactions
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.action, i.caption, i.created_at,
                       sp.name AS subject_name, op.name AS object_name
                FROM interactions i
                LEFT JOIN object_observations so ON so.id = i.subject_id
                LEFT JOIN objects sp ON sp.id = so.object_id
                LEFT JOIN object_observations oo ON oo.id = i.object_id
                LEFT JOIN objects op ON op.id = oo.object_id
                ORDER BY i.created_at DESC
                LIMIT 40
                """
            )
            interactions = cur.fetchall()
        if interactions:
            lines.append("RECENT INTERACTIONS (newest first):")
            for action, caption, ts, subj, obj in interactions:
                who = subj or "unknown"
                what = obj or ""
                lines.append(f"  - [{ts}] {who} {action or 'interacted with'} {what}: {caption}")

        # Recent scenes
        with conn.cursor() as cur:
            cur.execute(
                "SELECT caption, x, y, timestamp FROM scenes ORDER BY timestamp DESC LIMIT 20"
            )
            scenes = cur.fetchall()
        if scenes:
            lines.append("RECENT SCENES:")
            for caption, x, y, ts in scenes:
                loc = f"({x:.2f}, {y:.2f})" if x is not None else "unknown location"
                lines.append(f"  - [{ts}] {loc}: {caption}")

        conn.close()
        return "\n".join(lines)

    def _call_llm(self, question: str, context: str) -> str:
        payload = {
            "model": self.vlm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a robot perception assistant. Answer the user's question "
                        "using ONLY the provided database context below. Do not invent facts. "
                        "If the answer is not in the context, say you don't know.\n\n"
                        f"{context}"
                    ),
                },
                {"role": "user", "content": question},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        request = urllib.request.Request(
            f"{self.vlm_url.rstrip('/')}/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"] or "")
        except Exception as exc:
            return f"[Direct LLM error: {exc}]"

    def answer(self, question: str, building: str | None = None) -> tuple[str, list[dict]]:
        context = self._build_context(building)
        answer = self._call_llm(question, context)
        return answer, []


# ---------------------------------------------------------------------------
# Baseline 2 – Structured retrieval / Rule-based
# ---------------------------------------------------------------------------

class RuleBasedBaseline:
    """Deterministic SQL-based retrieval without any LLM."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def _room_from_scene(alias: str = "s") -> str:
        return f"""(SELECT r.name FROM rooms r
         WHERE r.map_id = {alias}.map_id
           AND {alias}.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
           AND {alias}.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
         LIMIT 1)"""

    def _resolve_object(self, name: str) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, name, class_id FROM objects WHERE name ILIKE %s LIMIT 1",
                (f"%{name}%",),
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "class_id": row[2]}

    def _resolve_person(self, name: str) -> dict | None:
        return self._resolve_object(name)

    def _get_latest_interaction(self, object_id: int, action: str | None = None) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            if action:
                cur.execute(
                    """
                    SELECT i.action, i.caption, i.created_at,
                           (SELECT r.name FROM rooms r
                            WHERE r.map_id = so.map_id
                              AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                              AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                            LIMIT 1) as room,
                           sp.name AS subject_name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    ) AND i.action = %s
                    ORDER BY i.created_at DESC
                    LIMIT 1
                    """,
                    (object_id, action),
                )
            else:
                cur.execute(
                    """
                    SELECT i.action, i.caption, i.created_at,
                           (SELECT r.name FROM rooms r
                            WHERE r.map_id = so.map_id
                              AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                              AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                            LIMIT 1) as room,
                           sp.name AS subject_name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    )
                    ORDER BY i.created_at DESC
                    LIMIT 1
                    """,
                    (object_id,),
                )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "action": row[0],
            "caption": row[1],
            "created_at": row[2],
            "room": row[3],
            "person": row[4],
        }

    def _get_all_interaction_people(self, object_id: int, action: str | None = None) -> list[str]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            if action:
                cur.execute(
                    """
                    SELECT DISTINCT sp.name
                    FROM interactions i
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    ) AND i.action = %s
                    AND sp.name IS NOT NULL
                    ORDER BY sp.name
                    """,
                    (object_id, action),
                )
            else:
                cur.execute(
                    """
                    SELECT DISTINCT sp.name
                    FROM interactions i
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    )
                    AND sp.name IS NOT NULL
                    ORDER BY sp.name
                    """,
                    (object_id,),
                )
            rows = cur.fetchall()
        conn.close()
        return [r[0] for r in rows if r[0]]

    def _get_interaction_people_stats(self, object_id: int) -> list[dict]:
        """Per-person interaction counts + action types for an object (most active first)."""
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sp.name, COUNT(*) AS cnt, array_agg(DISTINCT i.action) AS actions
                FROM interactions i
                LEFT JOIN object_observations so ON so.id = i.subject_id
                LEFT JOIN objects sp ON sp.id = so.object_id
                WHERE i.object_id IN (
                    SELECT id FROM object_observations WHERE object_id = %s
                ) AND sp.name IS NOT NULL
                GROUP BY sp.name
                ORDER BY cnt DESC, sp.name ASC
                """,
                (object_id,),
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {"name": r[0], "count": int(r[1]), "actions": [a for a in (r[2] or []) if a]}
            for r in rows
        ]

    # Actions that indicate real ownership/use (vs. passive proximity).
    _OWNERSHIP_ACTIONS = {"using", "use", "holding", "hold", "carrying", "carry",
                          "putting", "placeing", "put_down", "loading", "picked up",
                          "picking", "taking", "opening", "closing"}

    def _infer_owner_from_stats(self, people_stats: list[dict]) -> str | None:
        """Return the primary owner's name, or None if the object is public/shared.

        New rule: an object is shared only when 3+ people use it at a SIMILAR
        frequency AND with similar ownership-relevant action types. If one person
        dominates by count or is the only one performing meaningful ownership
        actions, that person is the owner.
        """
        if not people_stats:
            return None

        def ownership_count(p: dict) -> int:
            # Weight ownership-relevant actions; proximity-only people count less.
            acts = {str(a).lower() for a in p.get("actions", [])}
            return p["count"] if (acts & self._OWNERSHIP_ACTIONS) else 0

        # Prefer people who perform ownership-relevant actions.
        owners = [p for p in people_stats if ownership_count(p) > 0]
        candidates = owners if owners else people_stats

        # Fewer than 3 distinct people -> the most active one owns it.
        if len(people_stats) < 3:
            top = max(candidates, key=lambda p: (ownership_count(p), p["count"]))
            return top["name"]

        # 3+ people: shared only if usage is balanced (no dominant person).
        counts = sorted((ownership_count(p) for p in candidates), reverse=True)
        top_count = counts[0]
        if top_count <= 0:
            return None
        # Consider only the people with a meaningful share (>= 25% of top).
        significant = [c for c in counts if c >= 0.25 * top_count]
        # Dominant if the top person has at least 2x the second-best significant user.
        second = significant[1] if len(significant) > 1 else 0
        if top_count >= 2 * max(second, 1):
            top_person = max(candidates, key=lambda p: (ownership_count(p), p["count"]))
            return top_person["name"]
        # Balanced usage among 3+ people -> shared.
        return None

    def _get_most_frequent_interaction_person(self, object_id: int) -> str | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT sp.name, COUNT(*) as cnt
                FROM interactions i
                LEFT JOIN object_observations so ON so.id = i.subject_id
                LEFT JOIN objects sp ON sp.id = so.object_id
                WHERE i.object_id IN (
                    SELECT id FROM object_observations WHERE object_id = %s
                ) AND sp.name IS NOT NULL
                GROUP BY sp.name
                ORDER BY cnt DESC, sp.name ASC
                LIMIT 1
                """,
                (object_id,),
            )
            row = cur.fetchone()
        conn.close()
        return row[0] if row else None

    def _get_object_interactions(self, object_id: int) -> list[dict]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.action, i.caption, i.created_at, sp.name
                FROM interactions i
                LEFT JOIN object_observations so ON so.id = i.subject_id
                LEFT JOIN objects sp ON sp.id = so.object_id
                WHERE i.object_id IN (
                    SELECT id FROM object_observations WHERE object_id = %s
                )
                ORDER BY i.created_at DESC
                """,
                (object_id,),
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {"action": r[0], "caption": r[1], "created_at": r[2], "person": r[3]}
            for r in rows
        ]

    def _get_latest_observation(self, object_id: int) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT r.name FROM rooms r
                     WHERE r.map_id = object_observations.map_id
                       AND object_observations.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                       AND object_observations.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                     LIMIT 1) as room,
                    x, y, created_at
                FROM object_observations
                WHERE object_id = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (object_id,),
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"room": row[0], "x": row[1], "y": row[2], "created_at": row[3]}

    def _get_observation_at_time(self, object_id: int, time_str: str) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            if re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", time_str):
                cur.execute(
                    """
                    SELECT
                        (SELECT r.name FROM rooms r
                         WHERE r.map_id = object_observations.map_id
                           AND object_observations.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                           AND object_observations.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                         LIMIT 1) as room,
                        x, y, created_at
                    FROM object_observations
                    WHERE object_id = %s
                    ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - %s::timestamptz)))
                    LIMIT 1
                    """,
                    (object_id, time_str),
                )
            else:
                cur.execute(
                    """
                    SELECT
                        (SELECT r.name FROM rooms r
                         WHERE r.map_id = object_observations.map_id
                           AND object_observations.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                           AND object_observations.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                         LIMIT 1) as room,
                        x, y, created_at
                    FROM object_observations
                    WHERE object_id = %s
                    ORDER BY ABS(EXTRACT(EPOCH FROM (created_at::time - %s::time)))
                    LIMIT 1
                    """,
                    (object_id, time_str),
                )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"room": row[0], "x": row[1], "y": row[2], "created_at": row[3]}

    def _get_person_interactions(self, person_name: str) -> list[dict]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.action, i.caption, i.created_at,
                       op.name AS other_name
                FROM interactions i
                JOIN object_observations so ON so.id = i.subject_id
                JOIN objects sp ON sp.id = so.object_id
                LEFT JOIN object_observations oo ON oo.id = i.object_id
                LEFT JOIN objects op ON op.id = oo.object_id
                WHERE sp.name = %s
                ORDER BY i.created_at DESC
                """,
                (person_name,),
            )
            rows = cur.fetchall()
        conn.close()
        return [
            {"action": r[0], "caption": r[1], "created_at": r[2], "other": r[3]}
            for r in rows
        ]

    def _get_top_co_participant(self, person_name: str) -> str | None:
        interactions = self._get_person_interactions(person_name)
        # Filter to person-to-person interactions (other is a person)
        person_names = {"Simba", "Nala", "Mufasa", "Scar"}
        counts: Counter[str] = Counter()
        for inter in interactions:
            other = inter.get("other") or ""
            if other in person_names and other != person_name:
                counts[other] += 1
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def _get_interaction_at_time(self, time_str: str, room: str | None = None) -> list[dict]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            is_full_dt = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", time_str)
            room_subq = """(SELECT r.name FROM rooms r
                             WHERE r.map_id = so.map_id
                               AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                               AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                             LIMIT 1)"""
            if room:
                if is_full_dt:
                    cur.execute(
                        f"""
                        SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                        FROM interactions i
                        LEFT JOIN scenes s ON s.id = i.scene_id
                        LEFT JOIN object_observations so ON so.id = i.subject_id
                        LEFT JOIN objects sp ON sp.id = so.object_id
                        WHERE i.created_at BETWEEN %s::timestamptz - INTERVAL '60 minutes' AND %s::timestamptz + INTERVAL '60 minutes'
                          AND {room_subq} = %s
                        ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at - %s::timestamptz)))
                        LIMIT 5
                        """,
                        (time_str, time_str, room, time_str),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                        FROM interactions i
                        LEFT JOIN scenes s ON s.id = i.scene_id
                        LEFT JOIN object_observations so ON so.id = i.subject_id
                        LEFT JOIN objects sp ON sp.id = so.object_id
                        WHERE i.created_at::time BETWEEN %s::time - INTERVAL '60 minutes' AND %s::time + INTERVAL '60 minutes'
                          AND {room_subq} = %s
                        ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at::time - %s::time)))
                        LIMIT 5
                        """,
                        (time_str, time_str, room, time_str),
                    )
            else:
                if is_full_dt:
                    cur.execute(
                        f"""
                        SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                        FROM interactions i
                        LEFT JOIN scenes s ON s.id = i.scene_id
                        LEFT JOIN object_observations so ON so.id = i.subject_id
                        LEFT JOIN objects sp ON sp.id = so.object_id
                        WHERE i.created_at BETWEEN %s::timestamptz - INTERVAL '60 minutes' AND %s::timestamptz + INTERVAL '60 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at - %s::timestamptz)))
                        LIMIT 5
                        """,
                        (time_str, time_str, time_str),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                        FROM interactions i
                        LEFT JOIN scenes s ON s.id = i.scene_id
                        LEFT JOIN object_observations so ON so.id = i.subject_id
                        LEFT JOIN objects sp ON sp.id = so.object_id
                        WHERE i.created_at::time BETWEEN %s::time - INTERVAL '60 minutes' AND %s::time + INTERVAL '60 minutes'
                        ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at::time - %s::time)))
                        LIMIT 5
                        """,
                        (time_str, time_str, time_str),
                    )
            rows = cur.fetchall()
        conn.close()
        return [
            {"action": r[0], "caption": r[1], "created_at": r[2], "room": r[3], "person": r[4]}
            for r in rows
        ]

    def _get_interactions_after_time(self, time_str: str, room: str | None = None, action_filter: str | None = None) -> list[dict]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            room_subq = """(SELECT r.name FROM rooms r
                             WHERE r.map_id = so.map_id
                               AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                               AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                             LIMIT 1)"""
            sql = f"""
                SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                FROM interactions i
                LEFT JOIN scenes s ON s.id = i.scene_id
                LEFT JOIN object_observations so ON so.id = i.subject_id
                LEFT JOIN objects sp ON sp.id = so.object_id
                WHERE i.created_at::time >= %s::time
            """
            params = [time_str]
            if room:
                sql += f" AND {room_subq} = %s"
                params.append(room)
            if action_filter:
                sql += " AND i.action = %s"
                params.append(action_filter)
            sql += " ORDER BY i.created_at ASC LIMIT 20"
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.close()
        return [
            {"action": r[0], "caption": r[1], "created_at": r[2], "room": r[3], "person": r[4]}
            for r in rows
        ]

    def _get_object_interaction_at_time(self, object_id: int, time_str: str, room: str | None = None, action_keywords: list[str] | None = None) -> list[dict]:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            is_full_dt = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", time_str)
            room_subq = """(SELECT r.name FROM rooms r
                             WHERE r.map_id = so.map_id
                               AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                               AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                             LIMIT 1)"""
            if is_full_dt:
                sql = f"""
                    SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    )
                    AND i.created_at BETWEEN %s::timestamptz - INTERVAL '60 minutes' AND %s::timestamptz + INTERVAL '60 minutes'
                """
                params = [object_id, time_str, time_str]
            else:
                sql = f"""
                    SELECT i.action, i.caption, i.created_at, {room_subq}, sp.name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    LEFT JOIN object_observations so ON so.id = i.subject_id
                    LEFT JOIN objects sp ON sp.id = so.object_id
                    WHERE i.object_id IN (
                        SELECT id FROM object_observations WHERE object_id = %s
                    )
                    AND i.created_at::time BETWEEN %s::time - INTERVAL '60 minutes' AND %s::time + INTERVAL '60 minutes'
                """
                params = [object_id, time_str, time_str]
            if room:
                sql += f" AND {room_subq} = %s"
                params.append(room)
            if action_keywords:
                sql += " AND i.action = ANY(%s)"
                params.append(action_keywords)
            sql += " ORDER BY i.created_at DESC LIMIT 5"
            cur.execute(sql, params)
            rows = cur.fetchall()
        conn.close()
        return [
            {"action": r[0], "caption": r[1], "created_at": r[2], "room": r[3], "person": r[4]}
            for r in rows
        ]

    def _get_object_by_person_action_time(self, person_name: str, time_str: str, room: str | None = None, action_keywords: list[str] | None = None) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            is_full_dt = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", time_str)
            room_subq = """(SELECT r.name FROM rooms r
                             WHERE r.map_id = so.map_id
                               AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                               AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                             LIMIT 1)"""
            if is_full_dt:
                sql = f"""
                    SELECT i.action, i.caption, i.created_at, {room_subq}, op.name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    JOIN object_observations so ON so.id = i.subject_id
                    JOIN objects sp ON sp.id = so.object_id
                    LEFT JOIN object_observations oo ON oo.id = i.object_id
                    LEFT JOIN objects op ON op.id = oo.object_id
                    WHERE sp.name = %s
                    AND i.created_at BETWEEN %s::timestamptz - INTERVAL '60 minutes' AND %s::timestamptz + INTERVAL '60 minutes'
                """
                params = [person_name, time_str, time_str]
                order_by = " ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at - %s::timestamptz)))"
            else:
                sql = f"""
                    SELECT i.action, i.caption, i.created_at, {room_subq}, op.name
                    FROM interactions i
                    LEFT JOIN scenes s ON s.id = i.scene_id
                    JOIN object_observations so ON so.id = i.subject_id
                    JOIN objects sp ON sp.id = so.object_id
                    LEFT JOIN object_observations oo ON oo.id = i.object_id
                    LEFT JOIN objects op ON op.id = oo.object_id
                    WHERE sp.name = %s
                    AND i.created_at::time BETWEEN %s::time - INTERVAL '60 minutes' AND %s::time + INTERVAL '60 minutes'
                """
                params = [person_name, time_str, time_str]
                order_by = " ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at::time - %s::time)))"
            if room:
                sql += f" AND {room_subq} = %s"
                params.append(room)
            if action_keywords:
                sql += " AND i.action = ANY(%s)"
                params.append(action_keywords)
            sql += order_by
            params.append(time_str)
            sql += " LIMIT 1"
            cur.execute(sql, params)
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"action": row[0], "caption": row[1], "created_at": row[2], "room": row[3], "object_name": row[4]}

    def _get_person_interaction_object_at_time(self, person_name: str, time_str: str) -> str | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            is_full_dt = re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", time_str)
            if is_full_dt:
                cur.execute(
                    """
                    SELECT op.name
                    FROM interactions i
                    JOIN object_observations so ON so.id = i.subject_id
                    JOIN objects sp ON sp.id = so.object_id
                    LEFT JOIN object_observations oo ON oo.id = i.object_id
                    LEFT JOIN objects op ON op.id = oo.object_id
                    WHERE sp.name = %s
                      AND i.created_at BETWEEN %s::timestamptz - INTERVAL '60 minutes' AND %s::timestamptz + INTERVAL '60 minutes'
                      AND oo.class_id != 0
                    ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at - %s::timestamptz)))
                    LIMIT 1
                    """,
                    (person_name, time_str, time_str, time_str),
                )
            else:
                cur.execute(
                    """
                    SELECT op.name
                    FROM interactions i
                    JOIN object_observations so ON so.id = i.subject_id
                    JOIN objects sp ON sp.id = so.object_id
                    LEFT JOIN object_observations oo ON oo.id = i.object_id
                    LEFT JOIN objects op ON op.id = oo.object_id
                    WHERE sp.name = %s
                      AND i.created_at::time BETWEEN %s::time - INTERVAL '60 minutes' AND %s::time + INTERVAL '60 minutes'
                      AND oo.class_id != 0
                    ORDER BY ABS(EXTRACT(EPOCH FROM (i.created_at::time - %s::time)))
                    LIMIT 1
                    """,
                    (person_name, time_str, time_str, time_str),
                )
            row = cur.fetchone()
        conn.close()
        return row[0] if row and row[0] else None

    def _get_navigation_target(self, question: str) -> dict | None:
        """For navigation questions, find target coordinates."""
        person = _extract_person_name(question)
        obj_name = _extract_object_name(question)
        time_str = _extract_time(question)

        # Person location
        if person and ("navigate to" in question.lower() or "go to" in question.lower()):
            if "most recently used" in question.lower():
                # Need to find who most recently used an object
                obj = self._resolve_object(obj_name) if obj_name else None
                if not obj:
                    return None
                inter = self._get_latest_interaction(obj["id"])
                if inter and inter.get("person"):
                    p = self._resolve_person(inter["person"])
                    if p:
                        obs = self._get_latest_observation(p["id"])
                        if obs:
                            return {
                                "x": obs["x"], "y": obs["y"],
                                "reason": f"{inter['person']} most recently used {obj['name']}; navigating to their last known location."
                            }
            elif "most frequently" in question.lower():
                obj = self._resolve_object(obj_name) if obj_name else None
                if not obj:
                    return None
                top_person = self._get_most_frequent_interaction_person(obj["id"])
                if top_person:
                    p = self._resolve_person(top_person)
                    if p:
                        obs = self._get_latest_observation(p["id"])
                        if obs:
                            return {
                                "x": obs["x"], "y": obs["y"],
                                "reason": f"{top_person} used {obj['name']} most frequently; navigating to their last known location."
                            }
            else:
                p = self._resolve_person(person)
                if p:
                    obs = self._get_latest_observation(p["id"])
                    if obs:
                        return {
                            "x": obs["x"], "y": obs["y"],
                            "reason": f"Last known location of {person}."
                        }

        # Object location
        if obj_name and ("navigate to" in question.lower() or "go to" in question.lower()):
            obj = self._resolve_object(obj_name)
            if obj:
                obs = self._get_latest_observation(obj["id"])
                if obs:
                    return {
                        "x": obs["x"], "y": obs["y"],
                        "reason": f"Last known location of {obj_name}."
                    }

        # Ownership-based navigation: "Simba's bottle"
        for pattern in [r"(\w+)'s\s+(\w+)", r"(\w+)\s+'s\s+(\w+)"]:
            m = re.search(pattern, question, re.IGNORECASE)
            if m:
                owner_name = m.group(1).capitalize()
                obj_guess = m.group(2).lower()
                # Find object with that name
                conn = _get_db_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id, name FROM objects WHERE name ILIKE %s LIMIT 1",
                        (f"%{obj_guess}%",),
                    )
                    row = cur.fetchone()
                conn.close()
                if row:
                    obs = self._get_latest_observation(row[0])
                    if obs:
                        return {
                            "x": obs["x"], "y": obs["y"],
                            "reason": f"Navigating to {row[1]}, associated with {owner_name}."
                        }

        return None

    def answer(self, question: str, building: str | None = None) -> tuple[str, list[dict]]:
        q_lower = question.lower()

        # Navigation mode detection
        if q_lower.startswith("navigate to") or q_lower.startswith("go to"):
            target = self._get_navigation_target(question)
            if target:
                tool_log = [{
                    "tool": "move_to_position",
                    "args": {"x": target["x"], "y": target["y"], "reason": target["reason"]},
                    "result": {"ok": True, "x": target["x"], "y": target["y"]},
                }]
                return f"Navigating to ({target['x']}, {target['y']}). {target['reason']}", tool_log
            return "I don't know where to navigate.", []

        # action_history
        if "used" in q_lower and "most recently" in q_lower:
            obj_name = _extract_object_name(question)
            if not obj_name:
                return "I don't know which object.", []
            obj = self._resolve_object(obj_name)
            if not obj:
                return f"Object {obj_name} not found.", []
            inter = self._get_latest_interaction(obj["id"])
            if inter and inter.get("person"):
                return f"{inter['person']} used {obj_name} most recently.", []
            return "No interactions found.", []

        if "has ever used" in q_lower or ("who" in q_lower and "used" in q_lower and "ever" in q_lower):
            obj_name = _extract_object_name(question)
            if not obj_name:
                return "I don't know which object.", []
            obj = self._resolve_object(obj_name)
            if not obj:
                return f"Object {obj_name} not found.", []
            people = self._get_all_interaction_people(obj["id"])
            if people:
                return f"People who have used {obj_name}: {', '.join(people)}.", []
            return "No one has used it.", []

        if "loaded" in q_lower or "put items into" in q_lower or "put" in q_lower:
            obj_name = _extract_object_name(question)
            if not obj_name:
                return "I don't know which object.", []
            obj = self._resolve_object(obj_name)
            if not obj:
                return f"Object {obj_name} not found.", []
            interactions = self._get_object_interactions(obj["id"])
            people = []
            for inter in interactions:
                action = (inter.get("action") or "").lower()
                if action in {"putting", "placeing", "put_down", "loading"}:
                    person = inter.get("person")
                    if person and person not in people:
                        people.append(person)
            if people:
                return f"People who put items into {obj_name}: {', '.join(people)}.", []
            return "No one loaded it.", []

        # ownership / primary user
        if "owns" in q_lower or "owner" in q_lower:
            obj_name = _extract_object_name(question)
            if not obj_name:
                return "I don't know which object.", []
            obj = self._resolve_object(obj_name)
            if not obj:
                return f"Object {obj_name} not found.", []
            people_stats = self._get_interaction_people_stats(obj["id"])
            owner = self._infer_owner_from_stats(people_stats)
            if owner is None:
                return "The object is public/shared and has no single personal owner.", []
            return f"{owner} owns {obj_name}.", []

        if "primarily uses" in q_lower or "primary user" in q_lower:
            obj_name = _extract_object_name(question)
            if not obj_name:
                return "I don't know which object.", []
            obj = self._resolve_object(obj_name)
            if not obj:
                return f"Object {obj_name} not found.", []
            top_person = self._get_most_frequent_interaction_person(obj["id"])
            if top_person:
                return f"{top_person} primarily uses {obj_name}.", []
            return "No usage data found.", []

        # person_interaction
        if "most often" in q_lower and ("talk" in q_lower or "talked" in q_lower):
            person = _extract_person_name(question)
            if not person:
                return "I don't know which person.", []
            top = self._get_top_co_participant(person)
            if top:
                return f"{top} talked to {person} most often.", []
            return "No interactions found.", []

        if ("talked to" in q_lower or "talk to" in q_lower) and "most often" not in q_lower:
            person = _extract_person_name(question)
            if person:
                conn = _get_db_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT DISTINCT op.name
                        FROM interactions i
                        JOIN object_observations so ON so.id = i.subject_id
                        JOIN objects sp ON sp.id = so.object_id
                        LEFT JOIN object_observations oo ON oo.id = i.object_id
                        LEFT JOIN objects op ON op.id = oo.object_id
                        WHERE sp.name = %s AND i.action = 'talking_to' AND op.name IS NOT NULL
                        ORDER BY op.name
                        """,
                        (person,),
                    )
                    rows = cur.fetchall()
                conn.close()
                people = [r[0] for r in rows if r[0] and r[0] != person]
                if people:
                    return f"People who talked to {person}: {', '.join(people)}.", []
                return "No one talked to them.", []

        if "talked" in q_lower and "after" in q_lower:
            talk_room = _extract_room_name(question)
            talk_time = _extract_time(question)
            if talk_room and talk_time:
                inters = self._get_interactions_after_time(talk_time, talk_room, "talking_to")
                people = sorted({i["person"] for i in inters if i.get("person")})
                if people:
                    return f"People who talked in the {talk_room} after {talk_time}: {', '.join(people)}.", []
                return "No talking interactions found.", []

        if "which people interacted" in q_lower:
            conn = _get_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT sp.name, op.name
                    FROM interactions i
                    JOIN object_observations so ON so.id = i.subject_id
                    JOIN objects sp ON sp.id = so.object_id
                    LEFT JOIN object_observations oo ON oo.id = i.object_id
                    LEFT JOIN objects op ON op.id = oo.object_id
                    WHERE sp.class_id = 0 AND op.class_id = 0
                    """
                )
                pairs = cur.fetchall()
            conn.close()
            unique_pairs = sorted({tuple(sorted((a, b))) for a, b in pairs if a and b and a != b})
            if unique_pairs:
                return f"People who interacted: {', '.join(f'{a} and {b}' for a, b in unique_pairs)}.", []
            return "No interactions found.", []

        if "relationship" in q_lower:
            person = _extract_person_name(question)
            if not person:
                return "I don't know which people.", []
            top = self._get_top_co_participant(person)
            if top:
                return f"{person} and {top} have a frequent workplace conversation or collaboration.", []
            return "No relationship detected.", []

        # time_location / interaction_location_time
        time_str = _extract_time(question)
        room = _extract_room_name(question)
        person = _extract_person_name(question)
        obj_name = _extract_object_name(question)

        if time_str and ("where was" in q_lower or "which room" in q_lower):
            target = person or obj_name
            if target:
                entity = self._resolve_object(target)
                if entity:
                    obs = self._get_observation_at_time(entity["id"], time_str)
                    if obs:
                        what = ""
                        if person and ("interacting with" in q_lower or "what was" in q_lower):
                            obj = self._get_person_interaction_object_at_time(person, time_str)
                            if obj:
                                what = f" and was interacting with {obj}"
                        return f"{target} was in {obs['room']} at {time_str}{what}.", []

        # Specific action + time + object queries (must come before generic time+who)
        if time_str and obj_name and "used" in q_lower:
            obj = self._resolve_object(obj_name)
            if obj:
                inters = self._get_object_interaction_at_time(obj["id"], time_str, room, ["using", "used"])
                if inters:
                    return f"{inters[0]['person']} used {obj_name} at {time_str}.", []

        if ("pick" in q_lower and "up" in q_lower) or "picked up" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    inters = self._get_object_interaction_at_time(obj["id"], time_str, room, ["pick_up"]) if time_str else []
                    if not inters:
                        inters = self._get_object_interactions(obj["id"])
                    for inter in inters:
                        if (inter.get("action") or "").lower() == "pick_up":
                            loc = inter.get("room") or "unknown location"
                            return f"{inter.get('person', 'Someone')} picked up {obj_name} in {loc}.", []
                    return "No pick-up interactions found.", []

        if "carried" in q_lower or "carry" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name and time_str:
                obj = self._resolve_object(obj_name)
                if obj:
                    inters = self._get_object_interaction_at_time(obj["id"], time_str, room, ["carrying", "bring"])
                    if inters:
                        return f"{inters[0]['person']} carried {obj_name} in {inters[0]['room']} at {time_str}.", []
                    return "No carrying interactions found at that time.", []

        if ("place" in q_lower or "put" in q_lower) and person and time_str:
            result = self._get_object_by_person_action_time(
                person, time_str, room, ["placeing", "put_down", "putting", "placing"]
            )
            if result and result.get("object_name"):
                loc = result.get("room") or "unknown location"
                return f"{person} placed {result['object_name']} in the {loc} at {time_str}.", []

        if time_str and "who" in q_lower:
            inters = self._get_interaction_at_time(time_str, room)
            if inters:
                people = [i["person"] for i in inters if i.get("person")]
                if people:
                    return f"At {time_str}: {', '.join(people)}.", []

        # Boolean transport + later usage
        if "transport" in q_lower and "even though" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    interactions = self._get_object_interactions(obj["id"])
                    transporter = None
                    later_user = None
                    for inter in interactions:
                        action = (inter.get("action") or "").lower()
                        if action in {"carrying", "transport", "transported"} and not transporter:
                            transporter = inter.get("person")
                        if action in {"using", "used"} and not later_user:
                            later_user = inter.get("person")
                    if transporter and later_user and transporter != later_user:
                        return f"Yes, {transporter} transported {obj_name} before {later_user} used it afterward.", []
                    return "No, the transport and usage pattern does not match that description.", []

        # object_location
        if "where was" in q_lower and "last seen" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    obs = self._get_latest_observation(obj["id"])
                    if obs:
                        return f"{obj_name} was last seen in {obs['room']}.", []

        if "which room was" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    if time_str:
                        obs = self._get_observation_at_time(obj["id"], time_str)
                    else:
                        obs = self._get_latest_observation(obj["id"])
                    if obs:
                        return f"{obj_name} was in {obs['room']}.", []

        # object_usage
        if "who interacted with" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    people = self._get_all_interaction_people(obj["id"])
                    if people:
                        return f"People who interacted with {obj_name}: {', '.join(people)}.", []

        # ownership_ambiguity
        if "transported" in q_lower and "before" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    interactions = self._get_object_interactions(obj["id"])
                    transporter = None
                    for inter in interactions:
                        action = (inter.get("action") or "").lower()
                        if action in {"carrying", "transport", "transported"}:
                            transporter = inter.get("person")
                            break
                    if transporter:
                        return f"{transporter} transported {obj_name} before.", []

        if "besides" in q_lower and "used" in q_lower:
            obj_name = _extract_object_name(question)
            if obj_name:
                obj = self._resolve_object(obj_name)
                if obj:
                    people = self._get_all_interaction_people(obj["id"])
                    # Filter out the owner
                    owner = _extract_person_name(question)
                    others = [p for p in people if p != owner]
                    if others:
                        return f"{', '.join(others)} also used {obj_name}.", []

        # person
        if "came into" in q_lower and "earliest" in q_lower:
            room = _extract_room_name(question)
            if room:
                conn = _get_db_conn()
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT sp.name, i.created_at
                        FROM interactions i
                        LEFT JOIN scenes s ON s.id = i.scene_id
                        JOIN object_observations so ON so.id = i.subject_id
                        JOIN objects sp ON sp.id = so.object_id
                        WHERE (SELECT r.name FROM rooms r
                               WHERE r.map_id = so.map_id
                                 AND so.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                                 AND so.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                               LIMIT 1) = %s
                          AND sp.class_id = 0
                        ORDER BY i.created_at ASC
                        LIMIT 1
                        """,
                        (room,),
                    )
                    row = cur.fetchone()
                conn.close()
                if row:
                    return f"{row[0]} came into the {room} earliest.", []

        # Fallback
        return "I don't have enough information to answer that.", []


# ---------------------------------------------------------------------------
# Baseline 3 – No-interaction ablation
# ---------------------------------------------------------------------------

class NoInteractionBaseline:
    """Call the web agent with interaction-related tools disabled."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url.rstrip("/")

    def answer(self, question: str, building: str | None = None) -> tuple[str, list[dict]]:
        payload = {
            "question": question,
            "history": [],
            "building": building,
            "mode": "no_interaction",
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120.0) as response:
                body = json.loads(response.read().decode("utf-8"))
                answer = str(body.get("answer") or "")
                tool_log = body.get("tool_log") or []
                return answer, tool_log
        except Exception as exc:
            return f"[No-interaction baseline error: {exc}]", []


# ---------------------------------------------------------------------------
# Baseline 4 – Latest-observation heuristic
# ---------------------------------------------------------------------------

class HeuristicBaseline:
    """Navigate to the latest observed entity matching the most salient name."""

    def __init__(self) -> None:
        pass

    def _find_latest_observation(self, name: str) -> dict | None:
        conn = _get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT r.name FROM rooms r
                     WHERE r.map_id = oo.map_id
                       AND oo.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                       AND oo.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                     LIMIT 1) as room,
                    oo.x, oo.y, oo.created_at, o.name
                FROM objects o
                JOIN object_observations oo ON oo.object_id = o.id
                WHERE o.name ILIKE %s
                ORDER BY oo.created_at DESC
                LIMIT 1
                """,
                (f"%{name}%",),
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"room": row[0], "x": row[1], "y": row[2], "created_at": row[3], "name": row[4]}

    def _find_nearby_object(self, person_name: str, object_guess: str) -> dict | None:
        """Find the latest object near a person."""
        conn = _get_db_conn()
        with conn.cursor() as cur:
            # Get person's latest location
            cur.execute(
                """
                SELECT x, y FROM object_observations oo
                JOIN objects o ON o.id = oo.object_id
                WHERE o.name = %s
                ORDER BY oo.created_at DESC LIMIT 1
                """,
                (person_name,),
            )
            prow = cur.fetchone()
            if not prow or prow[0] is None:
                conn.close()
                return None
            px, py = prow[0], prow[1]
            # Find nearby objects
            cur.execute(
                """
                SELECT
                    (SELECT r.name FROM rooms r
                     WHERE r.map_id = oo.map_id
                       AND oo.x BETWEEN LEAST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1)) AND GREATEST(r.x1, r.x2, r.x3, COALESCE(r.x4, r.x1))
                       AND oo.y BETWEEN LEAST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1)) AND GREATEST(r.y1, r.y2, r.y3, COALESCE(r.y4, r.y1))
                     LIMIT 1) as room,
                    oo.x, oo.y, oo.created_at, o.name
                FROM objects o
                JOIN object_observations oo ON oo.object_id = o.id
                WHERE o.name ILIKE %s AND o.class_id != 0
                ORDER BY ((oo.x - %s)^2 + (oo.y - %s)^2) ASC, oo.created_at DESC
                LIMIT 1
                """,
                (f"%{object_guess}%", px, py),
            )
            row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {"room": row[0], "x": row[1], "y": row[2], "created_at": row[3], "name": row[4]}

    def answer(self, question: str, building: str | None = None) -> tuple[str, list[dict]]:
        q_lower = question.lower()
        is_nav = q_lower.startswith("navigate to") or q_lower.startswith("go to")

        # Extract salient entities
        person = _extract_person_name(question)
        obj_name = _extract_object_name(question)
        room = _extract_room_name(question)

        # Ownership pattern: "Simba's bottle"
        for pattern in [r"(\w+)'s\s+(\w+)", r"(\w+)\s+'s\s+(\w+)"]:
            m = re.search(pattern, question, re.IGNORECASE)
            if m:
                owner_name = m.group(1).capitalize()
                obj_guess = m.group(2).lower()
                obs = self._find_nearby_object(owner_name, obj_guess)
                if not obs:
                    obs = self._find_latest_observation(obj_guess)
                if obs and is_nav:
                    tool_log = [{
                        "tool": "move_to_position",
                        "args": {"x": obs["x"], "y": obs["y"], "reason": f"Latest observation of {obs['name']} near {owner_name}."},
                        "result": {"ok": True, "x": obs["x"], "y": obs["y"]},
                    }]
                    return f"Navigating to {obs['name']} at ({obs['x']}, {obs['y']}).", tool_log
                if obs:
                    return f"{obs['name']} was last seen in {obs['room']}.", []

        # Navigate to person
        if is_nav and person:
            obs = self._find_latest_observation(person)
            if obs:
                tool_log = [{
                    "tool": "move_to_position",
                    "args": {"x": obs["x"], "y": obs["y"], "reason": f"Latest observation of {person}."},
                    "result": {"ok": True, "x": obs["x"], "y": obs["y"]},
                }]
                return f"Navigating to {person} at ({obs['x']}, {obs['y']}).", tool_log

        # Navigate to object
        if is_nav and obj_name:
            obs = self._find_latest_observation(obj_name)
            if obs:
                tool_log = [{
                    "tool": "move_to_position",
                    "args": {"x": obs["x"], "y": obs["y"], "reason": f"Latest observation of {obj_name}."},
                    "result": {"ok": True, "x": obs["x"], "y": obs["y"]},
                }]
                return f"Navigating to {obj_name} at ({obs['x']}, {obs['y']}).", tool_log

        # Navigate to room
        if is_nav and room:
            conn = _get_db_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT x1, y1, x2, y2 FROM rooms WHERE name = %s LIMIT 1",
                    (room,),
                )
                row = cur.fetchone()
            conn.close()
            if row:
                cx = (row[0] + row[2]) / 2
                cy = (row[1] + row[3]) / 2
                tool_log = [{
                    "tool": "move_to_position",
                    "args": {"x": cx, "y": cy, "reason": f"Center of {room}."},
                    "result": {"ok": True, "x": cx, "y": cy},
                }]
                return f"Navigating to the center of {room}.", tool_log

        # QA fallback: just return latest observation of the most salient entity
        target = person or obj_name
        if target:
            obs = self._find_latest_observation(target)
            if obs:
                return f"{target} was last seen in {obs['room']} at ({obs['x']}, {obs['y']}).", []

        return "I don't know.", []
