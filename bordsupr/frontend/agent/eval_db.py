"""Database persistence for scene-change strategy evaluation results."""

import json
import logging
import os
from typing import Any

import psycopg2

logger = logging.getLogger(__name__)

DEFAULT_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/bordsupr",
)


def get_conn():
    """Return a new psycopg2 connection."""
    return psycopg2.connect(DEFAULT_DB_URL)


# ---------------------------------------------------------------------------
# Scene-change strategy results
# ---------------------------------------------------------------------------

CREATE_SCENE_CHANGE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS scene_change_strategy_results (
    id SERIAL PRIMARY KEY,
    run_batch_id TEXT NOT NULL,
    strategy TEXT NOT NULL,
    seed INTEGER,
    prompt_template TEXT,
    system_prompt TEXT,
    total_changes INTEGER,
    total_major_changes INTEGER,
    observed_changes INTEGER,
    total_visits INTEGER,
    visits_with_changes INTEGER,
    change_recall REAL DEFAULT 0.0,
    change_precision REAL DEFAULT 0.0,
    change_f1 REAL DEFAULT 0.0,
    hit_rate REAL DEFAULT 0.0,
    scene_change_tp INTEGER DEFAULT 0,
    scene_change_tn INTEGER DEFAULT 0,
    scene_change_fp INTEGER DEFAULT 0,
    scene_change_fn INTEGER DEFAULT 0,
    scene_change_accuracy REAL DEFAULT 0.0,
    activity_change_tp INTEGER DEFAULT 0,
    activity_change_tn INTEGER DEFAULT 0,
    activity_change_fp INTEGER DEFAULT 0,
    activity_change_fn INTEGER DEFAULT 0,
    activity_change_accuracy REAL DEFAULT 0.0,
    navigation_to_changed_room INTEGER DEFAULT 0,
    total_moves INTEGER DEFAULT 0,
    navigation_precision REAL DEFAULT 0.0,
    exploration_coverage REAL DEFAULT 0.0,
    vlm_calls_made INTEGER DEFAULT 0,
    total_prompt_tokens INTEGER DEFAULT 0,
    total_completion_tokens INTEGER DEFAULT 0,
    elapsed_seconds REAL DEFAULT 0.0,
    minute_trace JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sc_strategy_results_batch ON scene_change_strategy_results(run_batch_id);
CREATE INDEX IF NOT EXISTS idx_sc_strategy_results_created ON scene_change_strategy_results(created_at DESC);
"""


def ensure_scene_change_table():
    """Create the scene_change_strategy_results table if it doesn't exist."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_SCENE_CHANGE_TABLE_SQL)
        conn.commit()


def save_scene_change_result(
    run_batch_id: str,
    result: Any,  # SceneChangeSimulationResult
) -> int:
    """Persist a single SceneChangeSimulationResult to the DB."""
    ensure_scene_change_table()

    def _trace_to_json(trace):
        out = []
        for step in trace:
            d = {
                "timestamp": step.timestamp.isoformat() if step.timestamp else None,
                "time_str": step.time_str,
                "room": step.room,
                "scene_text": step.scene_text,
                "gt_change_type": step.gt_change_type,
                "gt_activity_changed": step.gt_activity_changed,
                "vlm_detected_change": step.vlm_detected_change,
                "vlm_change": step.vlm_change,
                "vlm_detected_activity_change": step.vlm_detected_activity_change,
                "action": step.action,
                "target_room": step.target_room,
                "decision": step.decision,
                "people_present": step.people_present,
                "objects_present": step.objects_present,
            }
            out.append(d)
        return out

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO scene_change_strategy_results (
                    run_batch_id, strategy, seed, prompt_template, system_prompt,
                    total_changes, total_major_changes, observed_changes,
                    total_visits, visits_with_changes,
                    change_recall, change_precision, change_f1, hit_rate,
                    scene_change_tp, scene_change_tn, scene_change_fp, scene_change_fn, scene_change_accuracy,
                    activity_change_tp, activity_change_tn, activity_change_fp, activity_change_fn, activity_change_accuracy,
                    navigation_to_changed_room, total_moves, navigation_precision,
                    exploration_coverage,
                    vlm_calls_made, total_prompt_tokens, total_completion_tokens,
                    elapsed_seconds, minute_trace
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    run_batch_id,
                    result.strategy,
                    getattr(result, "seed", None),
                    getattr(result, "prompt_template", None),
                    result.system_prompt,
                    result.total_changes,
                    result.total_major_changes,
                    result.observed_changes,
                    result.total_visits,
                    result.visits_with_changes,
                    result.change_recall,
                    result.change_precision,
                    result.change_f1,
                    result.hit_rate,
                    result.scene_change_tp,
                    result.scene_change_tn,
                    result.scene_change_fp,
                    result.scene_change_fn,
                    result.scene_change_accuracy,
                    result.activity_change_tp,
                    result.activity_change_tn,
                    result.activity_change_fp,
                    result.activity_change_fn,
                    result.activity_change_accuracy,
                    result.navigation_to_changed_room,
                    result.total_moves,
                    result.navigation_precision,
                    result.exploration_coverage,
                    result.vlm_calls_made,
                    result.total_prompt_tokens,
                    result.total_completion_tokens,
                    result.elapsed_seconds,
                    json.dumps(_trace_to_json(result.minute_trace)) if result.minute_trace else None,
                ),
            )
            row = cur.fetchone()
            conn.commit()
            return row[0] if row else 0
