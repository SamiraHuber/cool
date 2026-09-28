#!/usr/bin/env python3
"""Enforce 1 face = 1 person observation (class 0) on existing data.

Each face detection belongs to exactly one person observation. This script:
  1. ensures face_observations.observation_id exists (same migration as the node);
  2. backfills observation_id for each face by matching its class-0 person
     observation (track + scene + exact person bbox, else closest in time), and
     sets object_id to that observation's cluster;
  3. DROPS faces that have no matching person observation (user decision);
  4. unlinks (object_id=NULL) any face pointing to a non-person cluster.

Usage:
    python3 scripts/enforce_face_observation_link.py --dry-run
    python3 scripts/enforce_face_observation_link.py --apply
"""

import argparse
import os
import sys

import psycopg2

PERSON_CLASS_ID = 0


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "35432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def ensure_column(cur):
    cur.execute("ALTER TABLE face_observations ADD COLUMN IF NOT EXISTS observation_id BIGINT")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_face_observations_observation_id ON face_observations(observation_id)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--link-window", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    w = args.link_window
    conn = get_conn()
    cur = conn.cursor()
    ensure_column(cur)
    conn.commit()

    # Resolve each face to its person observation (class 0). Exact track+scene+bbox
    # first; else closest-in-time same-track class-0 obs within the window.
    cur.execute(
        """
        SELECT fo.id,
          COALESCE(
            (SELECT oo.id FROM object_observations oo JOIN objects o ON o.id=oo.object_id
              WHERE oo.yolo_track_id=fo.yolo_track_id AND oo.scene_id=fo.scene_id
                AND o.class_id=%s AND oo.bbox_x_min=fo.person_x_min AND oo.bbox_y_min=fo.person_y_min
              ORDER BY oo.id DESC LIMIT 1),
            (SELECT oo.id FROM object_observations oo JOIN objects o ON o.id=oo.object_id
              WHERE oo.yolo_track_id=fo.yolo_track_id AND o.class_id=%s
                AND ABS(EXTRACT(EPOCH FROM (oo.created_at - fo.created_at))) <= %s
              ORDER BY ABS(EXTRACT(EPOCH FROM (oo.created_at - fo.created_at))) ASC LIMIT 1)
          ) AS obs_id
        FROM face_observations fo
        """,
        (PERSON_CLASS_ID, PERSON_CLASS_ID, w),
    )
    face_to_obs = {fid: oid for fid, oid in cur.fetchall()}
    matched = {f: o for f, o in face_to_obs.items() if o is not None}
    unmatched = [f for f, o in face_to_obs.items() if o is None]

    # Faces currently pointing to a non-person cluster (must be unlinked).
    cur.execute(
        "SELECT fo.id FROM face_observations fo JOIN objects o ON o.id=fo.object_id WHERE o.class_id != %s",
        (PERSON_CLASS_ID,),
    )
    non_person_faces = [r[0] for r in cur.fetchall()]

    print(f"Total faces:                 {len(face_to_obs)}")
    print(f"  matched to a person obs:   {len(matched)}")
    print(f"  no person obs (DROP):      {len(unmatched)}")
    print(f"  on non-person cluster:     {len(non_person_faces)} (unlink object_id)")

    if args.dry_run:
        print("\nDRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    # Backfill observation_id + object_id + scene_id from the matched observation.
    for fid, oid in matched.items():
        cur.execute(
            "UPDATE face_observations SET observation_id=%s, "
            "object_id=(SELECT object_id FROM object_observations WHERE id=%s), "
            "scene_id=(SELECT scene_id FROM object_observations WHERE id=%s) WHERE id=%s",
            (oid, oid, oid, fid),
        )

    # Unlink faces on non-person clusters.
    for fid in non_person_faces:
        cur.execute("UPDATE face_observations SET object_id=NULL WHERE id=%s", (fid,))

    # Drop faces with no person observation.
    cur.execute("DELETE FROM face_observations WHERE id = ANY(%s)", (unmatched,))
    dropped = cur.rowcount

    conn.commit()
    print(f"\nAPPLIED:")
    print(f"  backfilled observation_id: {len(matched)}")
    print(f"  unlinked non-person faces: {len(non_person_faces)}")
    print(f"  dropped (no person obs):   {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
