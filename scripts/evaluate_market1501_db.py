#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

MARKET_NAME_RE = re.compile(r"^(-?\d+)_c(\d+)")


@dataclass
class ObservationRecord:
    observation_id: int
    object_id: str
    class_id: int
    source_frame: str
    caption: str
    timestamp_epoch: float
    pid: int
    cam_id: int


def _sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _run_psql(
    *,
    container: str,
    db_user: str,
    db_name: str,
    sql: str,
) -> list[str]:
    cmd = [
        "docker",
        "exec",
        "-i",
        container,
        "psql",
        "-U",
        db_user,
        "-d",
        db_name,
        "-At",
        "-F",
        "\t",
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        sql,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "psql command failed\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _build_filters(
    *,
    since: str | None,
    until: str | None,
    source_prefixes: Sequence[str],
    person_class_id: int,
) -> list[str]:
    filters = [
        "oo.scene_id IS NOT NULL",
        "s.source_frame IS NOT NULL",
        "btrim(s.source_frame) <> ''",
    ]

    if person_class_id >= 0:
        filters.append(f"COALESCE(oo.class_id, -1) = {int(person_class_id)}")

    if since:
        filters.append(f"s.timestamp >= {_sql_quote(since)}::timestamptz")
    if until:
        filters.append(f"s.timestamp <= {_sql_quote(until)}::timestamptz")

    if source_prefixes:
        escaped = [p.replace("%", "\\%") for p in source_prefixes]
        prefix_clause = " OR ".join(
            f"s.source_frame LIKE {_sql_quote(prefix + '%')}" for prefix in escaped
        )
        filters.append(f"({prefix_clause})")

    return filters


def _parse_market_pid_cam(source_frame: str) -> tuple[int, int] | None:
    name = os.path.basename(source_frame)
    match = MARKET_NAME_RE.match(name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _comb2(value: int) -> int:
    if value < 2:
        return 0
    return value * (value - 1) // 2


def _evaluate_partition(true_labels: Sequence[int], pred_labels: Sequence[str]) -> dict:
    n = len(true_labels)
    if n == 0:
        return {
            "num_samples": 0,
            "num_gt_ids": 0,
            "num_pred_clusters": 0,
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "bcubed_precision": 0.0,
            "bcubed_recall": 0.0,
            "bcubed_f1": 0.0,
            "cluster_purity": 0.0,
            "mean_clusters_per_gt_id": 0.0,
        }

    gt_counts = Counter(true_labels)
    pred_counts = Counter(pred_labels)
    contingency: dict[int, Counter[str]] = defaultdict(Counter)

    for gt, pred in zip(true_labels, pred_labels):
        contingency[gt][pred] += 1

    tp = sum(_comb2(count) for per_gt in contingency.values() for count in per_gt.values())
    pred_pairs = sum(_comb2(count) for count in pred_counts.values())
    gt_pairs = sum(_comb2(count) for count in gt_counts.values())
    fp = pred_pairs - tp
    fn = gt_pairs - tp

    pairwise_precision = _safe_div(tp, tp + fp)
    pairwise_recall = _safe_div(tp, tp + fn)
    pairwise_f1 = _safe_div(2.0 * pairwise_precision * pairwise_recall, pairwise_precision + pairwise_recall)

    bcubed_precision_sum = 0.0
    bcubed_recall_sum = 0.0
    for gt, pred in zip(true_labels, pred_labels):
        n_ij = contingency[gt][pred]
        bcubed_precision_sum += _safe_div(n_ij, pred_counts[pred])
        bcubed_recall_sum += _safe_div(n_ij, gt_counts[gt])

    bcubed_precision = _safe_div(bcubed_precision_sum, n)
    bcubed_recall = _safe_div(bcubed_recall_sum, n)
    bcubed_f1 = _safe_div(2.0 * bcubed_precision * bcubed_recall, bcubed_precision + bcubed_recall)

    pred_to_gt: dict[str, Counter[int]] = defaultdict(Counter)
    for gt, pred in zip(true_labels, pred_labels):
        pred_to_gt[pred][gt] += 1
    purity = _safe_div(sum(max(counter.values()) for counter in pred_to_gt.values()), n)

    gt_to_pred_clusters: dict[int, set[str]] = defaultdict(set)
    for gt, pred in zip(true_labels, pred_labels):
        gt_to_pred_clusters[gt].add(pred)
    mean_clusters_per_gt = _safe_div(
        float(sum(len(cluster_ids) for cluster_ids in gt_to_pred_clusters.values())),
        float(len(gt_to_pred_clusters)),
    )

    return {
        "num_samples": n,
        "num_gt_ids": len(gt_counts),
        "num_pred_clusters": len(pred_counts),
        "pairwise_precision": pairwise_precision,
        "pairwise_recall": pairwise_recall,
        "pairwise_f1": pairwise_f1,
        "bcubed_precision": bcubed_precision,
        "bcubed_recall": bcubed_recall,
        "bcubed_f1": bcubed_f1,
        "cluster_purity": purity,
        "mean_clusters_per_gt_id": mean_clusters_per_gt,
    }


def _collect_source_prefixes(args: argparse.Namespace) -> list[str]:
    prefixes: list[str] = []
    for value in args.source_prefix:
        normalized = value.strip().strip("/")
        if normalized:
            prefixes.append(normalized + "/")
    for split in args.split:
        normalized = split.strip().strip("/")
        if normalized:
            prefixes.append(normalized + "/")

    deduped = []
    seen = set()
    for prefix in prefixes:
        if prefix in seen:
            continue
        deduped.append(prefix)
        seen.add(prefix)
    return deduped


def _query_records(args: argparse.Namespace, source_prefixes: Sequence[str]) -> tuple[list[ObservationRecord], dict, list[str]]:
    filters = _build_filters(
        since=args.since,
        until=args.until,
        source_prefixes=source_prefixes,
        person_class_id=args.person_class_id,
    )
    where_clause = " AND ".join(filters)

    rows_sql = f"""
    SELECT
        oo.id,
        oo.object_id,
        COALESCE(oo.class_id, -1),
        s.source_frame,
        COALESCE(NULLIF(btrim(s.caption), ''), ''),
        EXTRACT(EPOCH FROM s.timestamp)
    FROM object_observations oo
    JOIN scenes s ON s.id = oo.scene_id
    WHERE {where_clause}
    ORDER BY s.timestamp ASC, oo.id ASC
    """
    raw_rows = _run_psql(
        container=args.db_container,
        db_user=args.db_user,
        db_name=args.db_name,
        sql=rows_sql,
    )

    scene_only_filters = [flt for flt in filters if "oo." not in flt]

    scene_stats_sql = f"""
    SELECT
        COUNT(*)::bigint,
        COUNT(*) FILTER (WHERE caption IS NOT NULL AND btrim(caption) <> '')::bigint,
        COUNT(*) FILTER (WHERE caption IS NULL OR btrim(caption) = '')::bigint,
        COUNT(DISTINCT source_frame)::bigint
    FROM scenes s
    WHERE s.source_frame IS NOT NULL
      AND btrim(s.source_frame) <> ''
            AND {" AND ".join(scene_only_filters)}
    """
    scene_stats_lines = _run_psql(
        container=args.db_container,
        db_user=args.db_user,
        db_name=args.db_name,
        sql=scene_stats_sql,
    )

    caption_examples_sql = f"""
    SELECT source_frame, caption
    FROM scenes s
    WHERE s.source_frame IS NOT NULL
      AND btrim(s.source_frame) <> ''
      AND caption IS NOT NULL
      AND btrim(caption) <> ''
            AND {" AND ".join(scene_only_filters)}
    ORDER BY timestamp DESC
    LIMIT {int(args.caption_examples)}
    """
    caption_examples_lines = _run_psql(
        container=args.db_container,
        db_user=args.db_user,
        db_name=args.db_name,
        sql=caption_examples_sql,
    )

    scene_stats = {
        "scene_count": 0,
        "scenes_with_caption": 0,
        "scenes_without_caption": 0,
        "distinct_source_frames": 0,
    }
    if scene_stats_lines:
        parts = scene_stats_lines[0].split("\t")
        if len(parts) >= 4:
            scene_stats = {
                "scene_count": int(parts[0]),
                "scenes_with_caption": int(parts[1]),
                "scenes_without_caption": int(parts[2]),
                "distinct_source_frames": int(parts[3]),
            }

    caption_examples = []
    for line in caption_examples_lines:
        parts = line.split("\t", maxsplit=1)
        if len(parts) != 2:
            continue
        caption_examples.append(f"{parts[0]} => {parts[1]}")

    records: list[ObservationRecord] = []
    skipped_non_market = 0
    skipped_junk = 0

    for line in raw_rows:
        parts = line.split("\t")
        if len(parts) < 6:
            continue

        source_frame = parts[3]
        pid_cam = _parse_market_pid_cam(source_frame)
        if pid_cam is None:
            skipped_non_market += 1
            continue

        pid, cam_id = pid_cam
        if pid <= 0:
            skipped_junk += 1
            continue

        records.append(
            ObservationRecord(
                observation_id=int(parts[0]),
                object_id=parts[1],
                class_id=int(parts[2]),
                source_frame=source_frame,
                caption=parts[4],
                timestamp_epoch=float(parts[5]),
                pid=pid,
                cam_id=cam_id,
            )
        )

    skipped_notes = [
        f"Skipped {skipped_non_market} non-Market formatted source_frame rows.",
        f"Skipped {skipped_junk} junk/distractor rows with pid <= 0.",
    ]

    return records, scene_stats, caption_examples + skipped_notes


def _write_json(output_path: Path, payload: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DB clustering assignments against Market-1501 IDs using scenes.source_frame.",
    )
    parser.add_argument("--dataset-root", default="data/Market-1501-v15.09.15")
    parser.add_argument("--db-container", default="db")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-name", default="bordsupr")
    parser.add_argument("--since", default=None, help="ISO8601 timestamp (inclusive), e.g. 2026-04-12T10:00:00Z")
    parser.add_argument("--until", default=None, help="ISO8601 timestamp (inclusive)")
    parser.add_argument("--source-prefix", action="append", default=[], help="Optional source_frame prefix, e.g. query")
    parser.add_argument("--split", action="append", default=[], help="Convenience alias for --source-prefix")
    parser.add_argument("--person-class-id", type=int, default=0, help="Class id to evaluate. Use -1 to disable filtering.")
    parser.add_argument("--caption-examples", type=int, default=5)
    parser.add_argument("--output", default="reports/market1501_eval.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    source_prefixes = _collect_source_prefixes(args)

    records, scene_stats, caption_notes = _query_records(args, source_prefixes)

    true_labels = [record.pid for record in records]
    pred_labels = [record.object_id for record in records]

    metrics = _evaluate_partition(true_labels, pred_labels)

    payload = {
        "dataset_root": str(dataset_root.resolve()),
        "db_container": args.db_container,
        "db_name": args.db_name,
        "filters": {
            "since": args.since,
            "until": args.until,
            "source_prefixes": source_prefixes,
            "person_class_id": args.person_class_id,
        },
        "scene_stats": {
            **scene_stats,
            "caption_coverage": _safe_div(
                float(scene_stats.get("scenes_with_caption", 0)),
                float(scene_stats.get("scene_count", 0)),
            ),
        },
        "evaluation": metrics,
        "caption_notes": caption_notes,
    }

    output_path = Path(args.output)
    _write_json(output_path, payload)

    print("Market-1501 DB evaluation complete")
    print(f"Output file: {output_path}")
    print(
        "Samples={num_samples}, GT IDs={num_gt_ids}, Clusters={num_pred_clusters}".format(
            **metrics
        )
    )
    print(
        "Pairwise F1={pairwise_f1:.4f}, B-cubed F1={bcubed_f1:.4f}, Purity={cluster_purity:.4f}".format(
            **metrics
        )
    )
    print(
        "Scenes={scene_count}, with_caption={scenes_with_caption}, without_caption={scenes_without_caption}, caption_coverage={caption_coverage:.4f}".format(
            **payload["scene_stats"]
        )
    )

    if payload["caption_notes"]:
        print("Caption samples / notes:")
        for item in payload["caption_notes"]:
            print(f"- {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
