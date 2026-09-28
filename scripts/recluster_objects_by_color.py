#!/usr/bin/env python3
"""Batch re-cluster over-merged OBJECT clusters by dominant color.

Object re-id (ConvNeXt embedding + position) does not discriminate color well, so
distinctly-colored objects of the same class (e.g. a pink laptop and a black laptop)
get merged into one cluster. Every observation already carries a coarse color
histogram in attributes_json->'colors'->'histogram'. This script finds non-person
clusters that span multiple distinct dominant-color groups and splits each into
per-color clusters, so differently-colored objects are separated.

Neutral colors (black/white/gray) are merged into one group because they co-occur
and are hard to separate reliably. The largest color group keeps the original
object_id; each other sufficiently-large group gets a new objects row.
object_observation_parts and face_observations are re-linked to follow their obs.

The live consolidation color-guard (database_node) keeps the split clusters from
being re-merged by embedding-centroid similarity afterwards.

Usage:
    python3 scripts/recluster_objects_by_color.py --dry-run
    python3 scripts/recluster_objects_by_color.py --apply
    python3 scripts/recluster_objects_by_color.py --class-id 63 --apply
    python3 scripts/recluster_objects_by_color.py --cluster 9812303421 --apply
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import psycopg2

NEUTRAL_COLORS = {"black", "white", "gray"}
PERSON_CLASS_ID = 0


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "35432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def dominant_color(hist):
    """Return the dominant color name from a histogram dict, neutral-merged."""
    if not hist:
        return None
    if isinstance(hist, str):
        try:
            hist = json.loads(hist)
        except Exception:
            return None
    if not isinstance(hist, dict) or not hist:
        return None
    best_name, best_val = None, 0.0
    for name, frac in hist.items():
        try:
            v = float(frac)
        except (TypeError, ValueError):
            continue
        if v > best_val:
            best_val, best_name = v, name
    if best_name is None:
        return None
    return "neutral" if best_name in NEUTRAL_COLORS else best_name


def find_candidate_clusters(cur, min_obs, class_id=None, cluster=None):
    """Non-person clusters with >= min_obs observations."""
    if cluster is not None:
        cur.execute(
            "SELECT o.id, o.class_id, COUNT(oo.id) FROM objects o "
            "JOIN object_observations oo ON oo.object_id=o.id "
            "WHERE o.id=%s GROUP BY o.id, o.class_id",
            (cluster,),
        )
    elif class_id is not None:
        cur.execute(
            "SELECT o.id, o.class_id, COUNT(oo.id) FROM objects o "
            "JOIN object_observations oo ON oo.object_id=o.id "
            "WHERE o.class_id=%s GROUP BY o.id, o.class_id HAVING COUNT(oo.id) >= %s",
            (class_id, min_obs),
        )
    else:
        cur.execute(
            "SELECT o.id, o.class_id, COUNT(oo.id) FROM objects o "
            "JOIN object_observations oo ON oo.object_id=o.id "
            "WHERE o.class_id != %s GROUP BY o.id, o.class_id HAVING COUNT(oo.id) >= %s",
            (PERSON_CLASS_ID, min_obs),
        )
    return cur.fetchall()


def split_one_cluster(cur, cluster_id, class_id, min_group, apply):
    """Split a single cluster by color. Returns (created, moved, detail) or None if
    fewer than 2 splittable groups."""
    cur.execute(
        "SELECT id, attributes_json->'colors'->'histogram' FROM object_observations "
        "WHERE object_id = %s ORDER BY id",
        (cluster_id,),
    )
    group_to_obs = defaultdict(list)
    for obs_id, hist in cur.fetchall():
        grp = dominant_color(hist)
        if grp is not None:
            group_to_obs[grp].append(obs_id)

    splittable = {g: o for g, o in group_to_obs.items() if len(o) >= min_group}
    if len(splittable) < 2:
        return None

    sorted_groups = sorted(splittable.items(), key=lambda kv: -len(kv[1]))
    keep_group, keep_obs = sorted_groups[0]
    split_groups = sorted_groups[1:]

    detail = {
        "cluster": cluster_id,
        "class_id": class_id,
        "keep": (keep_group, len(keep_obs)),
        "splits": [(g, len(o)) for g, o in split_groups],
    }
    if not apply:
        return (0, 0, detail)

    created = 0
    moved = 0
    for grp, obs_ids in split_groups:
        cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (class_id,))
        new_obj = cur.fetchone()[0]
        created += 1
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (new_obj, obs_ids),
        )
        moved += cur.rowcount
        cur.execute(
            "UPDATE object_observation_parts SET object_id = %s "
            "WHERE object_id = %s AND observation_id = ANY(%s)",
            (new_obj, cluster_id, obs_ids),
        )
        cur.execute(
            """
            UPDATE face_observations fo
            SET object_id = %s
            FROM object_observations oo
            WHERE oo.id = ANY(%s)
              AND fo.yolo_track_id = oo.yolo_track_id
              AND fo.scene_id = oo.scene_id
              AND fo.object_id = %s
            """,
            (new_obj, obs_ids, cluster_id),
        )
    return (created, moved, detail)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-id", type=int, default=None, help="limit to one COCO class")
    ap.add_argument("--cluster", type=int, default=None, help="limit to one cluster")
    ap.add_argument("--min-obs", type=int, default=30, help="candidate cluster min obs")
    ap.add_argument("--min-group", type=int, default=5, help="color groups smaller than this stay")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    conn = get_conn()
    cur = conn.cursor()
    candidates = find_candidate_clusters(cur, args.min_obs, args.class_id, args.cluster)
    print(f"Candidate object clusters (>= {args.min_obs} obs): {len(candidates)}")

    total_created = 0
    total_moved = 0
    split_clusters = 0
    for cluster_id, class_id, n_obs in candidates:
        result = split_one_cluster(cur, cluster_id, class_id, args.min_group, args.apply)
        if result is None:
            continue
        created, moved, detail = result
        split_clusters += 1
        total_created += created
        total_moved += moved
        kg, kn = detail["keep"]
        print(f"\ncluster {cluster_id} (class {class_id}, {n_obs} obs):")
        print(f"  keep '{kg}' ({kn} obs) on {cluster_id}")
        for g, n in detail["splits"]:
            print(f"  split '{g}' ({n} obs) -> new cluster")
        if args.apply:
            conn.commit()

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}:")
    print(f"  clusters split:        {split_clusters}")
    print(f"  new clusters created:  {total_created}")
    print(f"  observations moved:    {total_moved}")
    if args.dry_run:
        print("  (no changes made — re-run with --apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
