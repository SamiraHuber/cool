#!/usr/bin/env python3
"""Split the over-merged person "sink" cluster using face identity.

The body-embedding kNN matcher snowballed nearly all person observations into one
giant sink cluster. Faces (ArcFace) are far more discriminative, and each physical
person has a distinct face person_id. This script reassigns the sink cluster's
observations into per-person clusters keyed by face person_id (matched via
yolo_track_id + scene_id), so that each person exists exactly once.

Usage:
    python3 scripts/split_sink_cluster_by_face.py --sink 9812306591 --dry-run
    python3 scripts/split_sink_cluster_by_face.py --sink 9812306591 --apply
"""

import argparse
import os
import sys
from collections import defaultdict

import psycopg2


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "35432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sink", required=True, help="object_id of the sink cluster")
    ap.add_argument("--person-class-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report only, no changes")
    ap.add_argument("--apply", action="store_true", help="apply the reassignment")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    sink = args.sink
    conn = get_conn()
    cur = conn.cursor()

    # 1. Map each sink observation -> face person_id (via track + scene).
    cur.execute(
        """
        SELECT oo.id, fo.person_id
        FROM object_observations oo
        JOIN face_observations fo
          ON fo.yolo_track_id = oo.yolo_track_id
         AND fo.scene_id = oo.scene_id
        WHERE oo.object_id = %s
        ORDER BY oo.id
        """,
        (sink,),
    )
    obs_to_person = {}
    for obs_id, person_id in cur.fetchall():
        # If multiple faces, keep the first (most recent ordering could be added).
        obs_to_person.setdefault(obs_id, str(person_id))

    # 2. All sink observations.
    cur.execute("SELECT id FROM object_observations WHERE object_id = %s", (sink,))
    all_obs = [r[0] for r in cur.fetchall()]
    obs_with_face = set(obs_to_person.keys())
    obs_without_face = [o for o in all_obs if o not in obs_with_face]

    # 3. Group face-tagged observations by person.
    person_to_obs = defaultdict(list)
    for obs_id, person_id in obs_to_person.items():
        person_to_obs[person_id].append(obs_id)

    print(f"Sink cluster: {sink}")
    print(f"  total observations:        {len(all_obs)}")
    print(f"  with face identity:        {len(obs_with_face)}")
    print(f"  without face identity:     {len(obs_without_face)}")
    print(f"  distinct face person_ids:  {len(person_to_obs)}")
    sizes = sorted((len(v) for v in person_to_obs.values()), reverse=True)
    print(f"  largest per-person groups: {sizes[:10]}")

    if args.dry_run:
        print("\nDRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    # APPLY: assign each person group to its own object cluster.
    # Keep the sink object for the largest person group (so the id stays valid),
    # create new objects for the rest.
    sorted_persons = sorted(person_to_obs.items(), key=lambda kv: -len(kv[1]))
    created = 0
    reassigned = 0

    for idx, (person_id, obs_ids) in enumerate(sorted_persons):
        if idx == 0:
            target_obj = int(sink)  # keep sink id for the dominant person
        else:
            cur.execute(
                "INSERT INTO objects (class_id) VALUES (%s) RETURNING id",
                (args.person_class_id,),
            )
            target_obj = cur.fetchone()[0]
            created += 1

        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (target_obj, obs_ids),
        )
        reassigned += cur.rowcount
        # Re-link the face observations for this person to the target cluster.
        cur.execute(
            "UPDATE face_observations SET object_id = %s WHERE person_id = %s",
            (target_obj, person_id),
        )

    # Observations without a face: leave them on the sink object? No — the sink is
    # now the dominant person's cluster. Move faceless observations to their own
    # per-track singleton clusters so they don't pollute the dominant person.
    faceless_new = 0
    if obs_without_face:
        # Group faceless observations by (scene_id, yolo_track_id) to keep each
        # distinct track/scene as its own provisional cluster.
        cur.execute(
            """
            SELECT id, scene_id, yolo_track_id
            FROM object_observations
            WHERE id = ANY(%s)
            ORDER BY scene_id, yolo_track_id
            """,
            (obs_without_face,),
        )
        groups = defaultdict(list)
        for obs_id, scene_id, track_id in cur.fetchall():
            groups[(scene_id, track_id)].append(obs_id)

        for (_scene, _track), obs_ids in groups.items():
            cur.execute(
                "INSERT INTO objects (class_id) VALUES (%s) RETURNING id",
                (args.person_class_id,),
            )
            new_obj = cur.fetchone()[0]
            cur.execute(
                "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
                (new_obj, obs_ids),
            )
            faceless_new += 1

    conn.commit()
    print(f"\nAPPLIED:")
    print(f"  new person clusters (from faces):   {created}")
    print(f"  observations reassigned (faces):    {reassigned}")
    print(f"  provisional clusters (faceless):    {faceless_new}")
    print(f"  sink cluster now holds dominant person group: {len(sorted_persons[0][1]) if sorted_persons else 0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
