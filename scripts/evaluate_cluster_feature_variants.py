#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from evaluate_market1501_db import (
    _collect_source_prefixes,
    _evaluate_partition,
    _parse_market_pid_cam,
    _run_psql,
    _safe_div,
    _sql_quote,
)


def _parse_vector(value: str) -> list[float]:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(part) for part in text.split(",") if part.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _weighted_centroid(vectors: list[list[float]], weights: list[float]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    totals = [0.0] * dim
    weight_sum = 0.0
    for vector, weight in zip(vectors, weights):
        if len(vector) != dim:
            continue
        weight = max(0.05, min(1.0, float(weight)))
        for idx, value in enumerate(vector):
            totals[idx] += value * weight
        weight_sum += weight
    if weight_sum <= 0.0:
        return []
    return [value / weight_sum for value in totals]


def _separation_metrics(scores_same: list[float], scores_diff: list[float]) -> dict:
    if not scores_same or not scores_diff:
        return {
            "same_pairs": len(scores_same),
            "different_pairs": len(scores_diff),
            "same_mean": None,
            "different_mean": None,
            "mean_gap": None,
            "pair_auc": None,
        }
    rng = random.Random(7)
    comparisons = min(20000, len(scores_same) * len(scores_diff))
    wins = 0.0
    for _ in range(comparisons):
        same = rng.choice(scores_same)
        diff = rng.choice(scores_diff)
        if same > diff:
            wins += 1.0
        elif same == diff:
            wins += 0.5
    same_mean = sum(scores_same) / len(scores_same)
    diff_mean = sum(scores_diff) / len(scores_diff)
    return {
        "same_pairs": len(scores_same),
        "different_pairs": len(scores_diff),
        "same_mean": same_mean,
        "different_mean": diff_mean,
        "mean_gap": same_mean - diff_mean,
        "pair_auc": wins / comparisons if comparisons else None,
    }


def _query_records(args: argparse.Namespace) -> list[dict]:
    filters = [
        "oo.scene_id IS NOT NULL",
        "s.source_frame IS NOT NULL",
        "btrim(s.source_frame) <> ''",
    ]
    if args.person_class_id >= 0:
        filters.append(f"COALESCE(oo.class_id, -1) = {int(args.person_class_id)}")
    if args.since:
        filters.append(f"s.timestamp >= {_sql_quote(args.since)}::timestamptz")
    if args.until:
        filters.append(f"s.timestamp <= {_sql_quote(args.until)}::timestamptz")
    source_prefixes = _collect_source_prefixes(args)
    if source_prefixes:
        filters.append(
            "("
            + " OR ".join(f"s.source_frame LIKE {_sql_quote(prefix + '%')}" for prefix in source_prefixes)
            + ")"
        )

    sql = f"""
    SELECT
        oo.id,
        oo.object_id::text,
        COALESCE(oo.class_id, -1),
        s.source_frame,
        oo.embedding::text,
        COALESCE(oo.quality_score, 0.5),
        COALESCE(oo.attributes_json, '{{}}'::jsonb)::text
    FROM object_observations oo
    JOIN scenes s ON s.id = oo.scene_id
    WHERE {" AND ".join(filters)}
    ORDER BY s.timestamp ASC, oo.id ASC
    """
    rows = _run_psql(container=args.db_container, db_user=args.db_user, db_name=args.db_name, sql=sql)

    records = []
    observation_ids = []
    for line in rows:
        parts = line.split("\t", maxsplit=6)
        if len(parts) < 7:
            continue
        pid_cam = _parse_market_pid_cam(parts[3])
        if pid_cam is None or pid_cam[0] <= 0:
            continue
        records.append(
            {
                "observation_id": int(parts[0]),
                "object_id": parts[1],
                "class_id": int(parts[2]),
                "source_frame": parts[3],
                "embedding": _parse_vector(parts[4]),
                "quality": float(parts[5] or 0.5),
                "attributes": json.loads(parts[6] or "{}"),
                "pid": pid_cam[0],
                "parts": {},
            }
        )
        observation_ids.append(int(parts[0]))

    if observation_ids:
        id_list = ",".join(str(value) for value in observation_ids)
        part_sql = f"""
        SELECT observation_id, part_name, embedding::text, COALESCE(quality_score, 0.5), COALESCE(preprocessing_json, '{{}}'::jsonb)::text
        FROM object_observation_parts
        WHERE observation_id IN ({id_list})
        """
        part_rows = _run_psql(container=args.db_container, db_user=args.db_user, db_name=args.db_name, sql=part_sql)
        by_id = {record["observation_id"]: record for record in records}
        for line in part_rows:
            parts = line.split("\t", maxsplit=4)
            if len(parts) < 5:
                continue
            record = by_id.get(int(parts[0]))
            if record is None:
                continue
            record["parts"][parts[1]] = {
                "embedding": _parse_vector(parts[2]),
                "quality": float(parts[3] or 0.5),
                "preprocessing": json.loads(parts[4] or "{}"),
            }
    return records


def _pair_scores(records: list[dict], *, variant: str, max_pairs: int) -> tuple[list[float], list[float]]:
    rng = random.Random(11)
    pairs = []
    n = len(records)
    if n < 2:
        return [], []
    max_possible = n * (n - 1) // 2
    target = min(max_pairs, max_possible)
    seen = set()
    while len(pairs) < target:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        key = tuple(sorted((i, j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((records[key[0]], records[key[1]]))

    same_scores = []
    diff_scores = []
    for a, b in pairs:
        score = 0.0
        if variant == "whole":
            score = _cosine(a["embedding"], b["embedding"])
        elif variant == "quality_weighted":
            score = _cosine(a["embedding"], b["embedding"]) * math.sqrt(max(0.05, a["quality"]) * max(0.05, b["quality"]))
        elif variant == "upper_lower_parts":
            values = []
            for part_name in ("upper_body", "lower_body"):
                part_a = a["parts"].get(part_name, {})
                part_b = b["parts"].get(part_name, {})
                sim = _cosine(part_a.get("embedding", []), part_b.get("embedding", []))
                if sim:
                    values.append(sim)
            score = sum(values) / len(values) if values else _cosine(a["embedding"], b["embedding"])
        elif variant == "combined":
            whole = _cosine(a["embedding"], b["embedding"])
            part_values = []
            for part_name in ("upper_body", "lower_body"):
                part_a = a["parts"].get(part_name, {})
                part_b = b["parts"].get(part_name, {})
                sim = _cosine(part_a.get("embedding", []), part_b.get("embedding", []))
                if sim:
                    part_values.append(sim)
            part_score = sum(part_values) / len(part_values) if part_values else whole
            quality = math.sqrt(max(0.05, a["quality"]) * max(0.05, b["quality"]))
            score = (0.65 * whole + 0.35 * part_score) * quality

        if a["pid"] == b["pid"]:
            same_scores.append(score)
        else:
            diff_scores.append(score)
    return same_scores, diff_scores


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate clustering feature variants on Market-1501 rows in the DB.")
    parser.add_argument("--dataset-root", default="data/Market-1501-v15.09.15")
    parser.add_argument("--db-container", default="db")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-name", default="bordsupr")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--source-prefix", action="append", default=[])
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--person-class-id", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--output", default="reports/market1501_feature_variants.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        return 2

    records = _query_records(args)
    true_labels = [record["pid"] for record in records]
    pred_labels = [record["object_id"] for record in records]
    variants = {}
    for variant in ("whole", "quality_weighted", "upper_lower_parts", "combined"):
        same, diff = _pair_scores(records, variant=variant, max_pairs=args.max_pairs)
        variants[variant] = _separation_metrics(same, diff)

    preprocessing_counts = defaultdict(int)
    for record in records:
        for part in record["parts"].values():
            method = (part.get("preprocessing") or {}).get("method") or "none"
            preprocessing_counts[method] += 1

    payload = {
        "dataset_root": str(dataset_root.resolve()),
        "num_records": len(records),
        "baseline_db_partition": _evaluate_partition(true_labels, pred_labels),
        "feature_variants": variants,
        "coverage": {
            "with_quality": sum(1 for record in records if record["quality"] is not None),
            "with_upper_body_part": sum(1 for record in records if "upper_body" in record["parts"]),
            "with_lower_body_part": sum(1 for record in records if "lower_body" in record["parts"]),
            "with_part_embeddings": sum(
                1
                for record in records
                if any(part.get("embedding") for part in record["parts"].values())
            ),
            "preprocessing_methods": dict(sorted(preprocessing_counts.items())),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Cluster feature variant evaluation complete")
    print(f"Output file: {output_path}")
    for name, metrics in variants.items():
        print(
            f"{name}: gap={metrics['mean_gap']}, auc={metrics['pair_auc']}, "
            f"same={metrics['same_pairs']}, diff={metrics['different_pairs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
