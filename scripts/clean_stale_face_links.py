#!/usr/bin/env python3
"""Clean up stale face->cluster links.

Historically, face_observations.object_id was set by a track-id sweep with no time
bound, so faces were linked to clusters even when the corresponding body was never
observed there (BoTSORT track-id reuse). This leaves clusters with inflated,
wrong face links (e.g. 9 body obs but 55 faces).

For every face observation whose object_id points to a cluster with NO body
observation of the same track within the link window:
  - if the face HAS a time-close body observation of the same track in a DIFFERENT
    cluster, re-link it to that (correct) cluster;
  - otherwise unlink it (object_id = NULL) — the face has no matching body.

Usage:
    python3 scripts/clean_stale_face_links.py --dry-run
    python3 scripts/clean_stale_face_links.py --apply
"""

import argparse
import os
import sys

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

    # Faces with a stale link (object_id set, no time-close body of same track in it).
    cur.execute(
        """
        SELECT fo.id, fo.object_id, fo.yolo_track_id,
          (SELECT oo.object_id FROM object_observations oo
            WHERE oo.yolo_track_id = fo.yolo_track_id
              AND ABS(EXTRACT(EPOCH FROM (oo.created_at - fo.created_at))) <= %s
            ORDER BY ABS(EXTRACT(EPOCH FROM (oo.created_at - fo.created_at))) ASC
            LIMIT 1) AS correct_object_id
        FROM face_observations fo
        WHERE fo.object_id IS NOT NULL
          AND NOT EXISTS (
            SELECT 1 FROM object_observations oo
            WHERE oo.object_id = fo.object_id AND oo.yolo_track_id = fo.yolo_track_id
              AND ABS(EXTRACT(EPOCH FROM (oo.created_at - fo.created_at))) <= %s
          )
        """,
        (w, w),
    )
    rows = cur.fetchall()
    relink = [(fid, correct) for fid, _cur, _tr, correct in rows if correct is not None]
    unlink = [fid for fid, _cur, _tr, correct in rows if correct is None]

    print(f"Stale face links:            {len(rows)}")
    print(f"  re-link to correct cluster: {len(relink)}")
    print(f"  unlink (no matching body):  {len(unlink)}")

    if args.dry_run:
        print("\nDRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    for fid, correct in relink:
        cur.execute("UPDATE face_observations SET object_id = %s WHERE id = %s", (correct, fid))
    for fid in unlink:
        cur.execute("UPDATE face_observations SET object_id = NULL WHERE id = %s", (fid,))
    conn.commit()
    print(f"\nAPPLIED: re-linked {len(relink)}, unlinked {len(unlink)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
