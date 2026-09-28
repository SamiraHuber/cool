#!/usr/bin/env python3
"""Split an over-merged object cluster by dominant color.

Object re-id (ConvNeXt embedding + position) does not discriminate color well, so
distinctly-colored objects of the same class (e.g. a pink laptop and a black
laptop) get merged into one cluster. Every observation already carries a coarse
color histogram in attributes_json->'colors'->'histogram'. This script groups the
cluster's observations by dominant color (neutral colors black/white/gray are
merged into one group, since they are hard to separate reliably) and reassigns
each color group to its own cluster, so differently-colored objects are separated.

The largest color group keeps the original object_id (so the id stays valid);
each other group gets a new objects row. object_observation_parts and
face_observations are re-linked to follow their parent observation.

Usage:
    python3 scripts/split_cluster_by_color.py --cluster 9812303421 --dry-run
    python3 scripts/split_cluster_by_color.py --cluster 9812303421 --apply
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import psycopg2

NEUTRAL_COLORS = {"black", "white", "gray"}


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True, help="object_id of the cluster to split")
    ap.add_argument("--class-id", type=int, default=None,
                    help="class_id for new objects (default: read from the cluster)")
    ap.add_argument("--min-group", type=int, default=2,
                    help="color groups smaller than this stay on the original cluster")
    ap.add_argument("--dry-run", action="store_true", help="report only, no changes")
    ap.add_argument("--apply", action="store_true", help="apply the reassignment")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    cluster = int(args.cluster)
    conn = get_conn()
    cur = conn.cursor()

    # Cluster class_id.
    if args.class_id is not None:
        class_id = args.class_id
    else:
        cur.execute("SELECT class_id FROM objects WHERE id = %s", (cluster,))
        row = cur.fetchone()
        if row is None:
            print(f"Cluster {cluster} not found in objects table")
            return 1
        class_id = int(row[0])

    # Fetch each observation's dominant color group.
    cur.execute(
        """
        SELECT id, attributes_json->'colors'->'histogram' AS hist
        FROM object_observations
        WHERE object_id = %s
        ORDER BY id
        """,
        (cluster,),
    )
    group_to_obs = defaultdict(list)
    no_color = []
    for obs_id, hist in cur.fetchall():
        grp = dominant_color(hist)
        if grp is None:
            no_color.append(obs_id)
        else:
            group_to_obs[grp].append(obs_id)

    total = sum(len(v) for v in group_to_obs.values()) + len(no_color)
    print(f"Cluster: {cluster} (class_id={class_id})")
    print(f"  total observations: {total}")
    print(f"  without color data: {len(no_color)} (stay on original cluster)")
    print(f"  color groups:")
    for grp, obs in sorted(group_to_obs.items(), key=lambda kv: -len(kv[1])):
        marker = "" if len(obs) >= args.min_group else "  (below --min-group, stays)"
        print(f"    {grp:>10}: {len(obs)} obs{marker}")

    # Decide which groups split off.
    splittable = {g: o for g, o in group_to_obs.items() if len(o) >= args.min_group}
    if len(splittable) < 2:
        print("\nFewer than 2 splittable color groups — nothing to do.")
        return 0

    # Keep the largest group on the original cluster id.
    sorted_groups = sorted(splittable.items(), key=lambda kv: -len(kv[1]))
    keep_group, keep_obs = sorted_groups[0]
    split_groups = sorted_groups[1:]

    print(f"\nPlan: keep '{keep_group}' ({len(keep_obs)} obs) on cluster {cluster};")
    for grp, obs in split_groups:
        print(f"  split '{grp}' ({len(obs)} obs) -> new cluster")
    print(f"  small groups + no-color stay on {cluster}")

    if args.dry_run:
        print("\nDRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    # APPLY.
    created = 0
    reassigned = 0
    for grp, obs_ids in split_groups:
        cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (class_id,))
        new_obj = cur.fetchone()[0]
        created += 1
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (new_obj, obs_ids),
        )
        reassigned += cur.rowcount
        # Re-link parts to follow their observation.
        cur.execute(
            "UPDATE object_observation_parts SET object_id = %s WHERE object_id = %s AND observation_id = ANY(%s)",
            (new_obj, cluster, obs_ids),
        )
        # Re-link face observations that were bridged to these observations (track+scene).
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
            (new_obj, obs_ids, cluster),
        )

    conn.commit()
    print(f"\nAPPLIED:")
    print(f"  new clusters created:        {created}")
    print(f"  observations moved off:      {reassigned}")
    cur.execute("SELECT COUNT(*) FROM object_observations WHERE object_id = %s", (cluster,))
    remaining = cur.fetchone()[0]
    print(f"  cluster {cluster} now holds: {remaining} obs ('{keep_group}' + small/no-color)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
