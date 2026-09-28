#!/usr/bin/env python3
"""Re-cluster an over-merged PERSON cluster by face identity + body embedding.

A person "sink" cluster fuses many physical people (body-embedding re-id is weak).
Faces (ArcFace) are far more discriminative, so each distinct face person_id marks
a real person. This script:

  1. SEEDS one cluster per face person_id from the cluster's body observations that
     are linkable to that person (same yolo_track_id, |dt| <= link window). This is
     the "split only if the face is different" arbiter.
  2. ASSIGNS the remaining (faceless) body observations to the nearest seeded
     cluster by body-embedding cosine similarity (>= assign threshold).
  3. Leftover observations (below threshold) are grouped by (scene_id, track_id)
     into provisional per-track clusters so they don't pollute the seeded persons.

The largest seeded cluster keeps the original object_id; others get new objects
rows. face_observations (by person_id) and object_observation_parts (by obs) are
re-linked to follow. Shirt/clothing color is NOT used (unreliable red-cast).

Usage:
    python3 scripts/recluster_persons_by_face.py --cluster 9812307372 --dry-run
    python3 scripts/recluster_persons_by_face.py --cluster 9812307372 --apply
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
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


def parse_emb(text):
    if text is None:
        return None
    if isinstance(text, (list, tuple)):
        return np.asarray(text, dtype=float)
    try:
        return np.asarray(json.loads(text), dtype=float)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True, type=int, help="person object_id to split")
    ap.add_argument("--link-window", type=float, default=5.0,
                    help="max seconds between a face and a body obs of the same track")
    ap.add_argument("--assign-threshold", type=float, default=0.68,
                    help="min cosine similarity to assign a faceless obs to a person")
    ap.add_argument("--min-seed", type=int, default=1,
                    help="face persons with fewer seeded obs than this are ignored")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    cluster = int(args.cluster)
    conn = get_conn()
    cur = conn.cursor()

    # All body observations in the cluster (id, track, scene, embedding).
    cur.execute(
        "SELECT id, yolo_track_id, scene_id, embedding::text FROM object_observations "
        "WHERE object_id = %s ORDER BY id",
        (cluster,),
    )
    obs_rows = cur.fetchall()
    obs_ids = [r[0] for r in obs_rows]
    emb_by_obs = {r[0]: parse_emb(r[3]) for r in obs_rows}
    print(f"Cluster {cluster}: {len(obs_ids)} body observations")

    # 1. SEED: map each face person_id -> seeded body obs (same track, |dt|<=window).
    cur.execute(
        """
        SELECT fo.person_id, oo.id
        FROM face_observations fo
        JOIN object_observations oo
          ON oo.yolo_track_id = fo.yolo_track_id
         AND oo.object_id = %s
         AND ABS(EXTRACT(EPOCH FROM (fo.created_at - oo.created_at))) <= %s
        WHERE fo.object_id = %s AND fo.person_id IS NOT NULL
        ORDER BY fo.person_id, oo.id
        """,
        (cluster, args.link_window, cluster),
    )
    person_to_obs = defaultdict(set)
    for person_id, obs_id in cur.fetchall():
        person_to_obs[int(person_id)].add(obs_id)

    # Drop tiny-seed persons.
    person_to_obs = {p: s for p, s in person_to_obs.items() if len(s) >= args.min_seed}
    seeded_obs = set().union(*person_to_obs.values()) if person_to_obs else set()
    faceless_obs = [o for o in obs_ids if o not in seeded_obs]

    print(f"  face persons (seeds):  {len(person_to_obs)}")
    print(f"  seeded observations:   {len(seeded_obs)}")
    print(f"  faceless observations: {len(faceless_obs)}")

    if len(person_to_obs) < 2:
        print("\nFewer than 2 face-proven persons — nothing to split.")
        return 0

    # 2. ASSIGN faceless obs to nearest seeded cluster by body-embedding cosine sim.
    # Compute each seeded person's mean embedding.
    person_list = sorted(person_to_obs.keys())
    person_centroid = {}
    for p in person_list:
        embs = [emb_by_obs[o] for o in person_to_obs[p] if emb_by_obs.get(o) is not None]
        if embs:
            m = np.mean(np.stack(embs), axis=0)
            n = np.linalg.norm(m)
            person_centroid[p] = m / n if n > 0 else None
        else:
            person_centroid[p] = None

    assign_to_person = {}   # obs_id -> person_id
    leftover_obs = []
    for o in faceless_obs:
        e = emb_by_obs.get(o)
        if e is None:
            leftover_obs.append(o)
            continue
        en = e / np.linalg.norm(e) if np.linalg.norm(e) > 0 else None
        if en is None:
            leftover_obs.append(o)
            continue
        best_p, best_sim = None, -1.0
        for p in person_list:
            c = person_centroid.get(p)
            if c is None:
                continue
            sim = float(np.dot(en, c))
            if sim > best_sim:
                best_sim, best_p = sim, p
        if best_p is not None and best_sim >= args.assign_threshold:
            assign_to_person[o] = best_p
        else:
            leftover_obs.append(o)

    # Final per-person obs sets.
    final_person_obs = {p: set(s) for p, s in person_to_obs.items()}
    for o, p in assign_to_person.items():
        final_person_obs[p].add(o)

    # Leftovers -> provisional per-(scene, track) groups.
    track_scene = {r[0]: (r[1], r[2]) for r in obs_rows}
    leftover_groups = defaultdict(list)
    for o in leftover_obs:
        tr, sc = track_scene.get(o, (None, None))
        leftover_groups[(sc, tr)].append(o)

    print(f"\nPlan:")
    print(f"  assigned faceless obs: {len(assign_to_person)}")
    print(f"  leftover obs:          {len(leftover_obs)} -> {len(leftover_groups)} provisional clusters")
    sizes = sorted((len(v) for v in final_person_obs.values()), reverse=True)
    print(f"  per-person sizes (top 10): {sizes[:10]}")

    if args.dry_run:
        print("\nDRY RUN — no changes made. Re-run with --apply to execute.")
        return 0

    # APPLY. Largest person keeps the original cluster id.
    sorted_persons = sorted(final_person_obs.items(), key=lambda kv: -len(kv[1]))
    person_to_object = {}
    created = 0
    for idx, (p, obs_set) in enumerate(sorted_persons):
        if idx == 0:
            obj = cluster
        else:
            cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,))
            obj = cur.fetchone()[0]
            created += 1
        person_to_object[p] = obj
        obs_list = list(obs_set)
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (obj, obs_list),
        )
        cur.execute(
            "UPDATE object_observation_parts SET object_id = %s "
            "WHERE object_id = %s AND observation_id = ANY(%s)",
            (obj, cluster, obs_list),
        )
        # Re-link this person's face observations to their cluster.
        cur.execute(
            "UPDATE face_observations SET object_id = %s WHERE person_id = %s",
            (obj, p),
        )

    # Leftovers -> provisional per-track clusters.
    provisional = 0
    for (sc, tr), obs_list in leftover_groups.items():
        cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,))
        obj = cur.fetchone()[0]
        provisional += 1
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (obj, obs_list),
        )
        cur.execute(
            "UPDATE object_observation_parts SET object_id = %s "
            "WHERE object_id = %s AND observation_id = ANY(%s)",
            (obj, cluster, obs_list),
        )

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM object_observations WHERE object_id = %s", (cluster,))
    remaining = cur.fetchone()[0]
    print(f"\nAPPLIED:")
    print(f"  person clusters (kept id on largest): {len(sorted_persons)}")
    print(f"  new person clusters created:          {created}")
    print(f"  provisional leftover clusters:        {provisional}")
    print(f"  cluster {cluster} now holds:            {remaining} obs (largest person)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
