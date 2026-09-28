#!/usr/bin/env python3
"""Split a contaminated person cluster into separate people by face identity.

Used for cluster 9812307415, where BoTSORT reused one track id for two people
(a man and a woman). We cluster the track's face embeddings into sub-identities,
then assign each body observation to the sub-identity seen in its scene.

Usage:
    python3 scripts/split_cluster_by_face_subidentity.py --cluster 9812307415 --dry-run
    python3 scripts/split_cluster_by_face_subidentity.py --cluster 9812307415 --apply
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import psycopg2

PERSON_CLASS_ID = 0
FACE_THRESHOLD = 0.50   # ArcFace cosine sim to merge faces into one sub-identity
MIN_SUB_IDENTITY_FACES = 3  # sub-identities with fewer faces are treated as noise


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "35432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def greedy(embs, thr):
    vecs = [norm(e) for e in embs]
    cs, ct, lab = [], [], []
    for v in vecs:
        bl, bs = -1, -1.0
        for ci, s in enumerate(cs):
            sim = float(np.dot(v, s / ct[ci]))
            if sim > bs:
                bs, bl = sim, ci
        if bl >= 0 and bs >= thr:
            lab.append(bl); cs[bl] += v; ct[bl] += 1
        else:
            lab.append(len(cs)); cs.append(v.copy()); ct.append(1)
    return lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    cluster = args.cluster
    conn = get_conn()
    cur = conn.cursor()

    # Body observations in the cluster.
    cur.execute(
        "SELECT id, scene_id, yolo_track_id FROM object_observations WHERE object_id = %s ORDER BY id",
        (cluster,),
    )
    body = cur.fetchall()
    if not body:
        print(f"Cluster {cluster} has no observations")
        return 1
    tracks = {r[2] for r in body if r[2] is not None}

    # Face observations on those tracks.
    cur.execute(
        "SELECT id, scene_id, embedding::text FROM face_observations "
        "WHERE yolo_track_id = ANY(%s) AND embedding IS NOT NULL ORDER BY id",
        (list(tracks),),
    )
    faces = cur.fetchall()
    if len(faces) < 2:
        print(f"Not enough faces ({len(faces)}) to split cluster {cluster}")
        return 1

    face_ids = [r[0] for r in faces]
    face_scenes = [r[1] for r in faces]
    face_embs = [json.loads(r[2]) for r in faces]
    labels = greedy(face_embs, FACE_THRESHOLD)

    # Keep only sub-identities with enough faces (drop noise).
    counts = defaultdict(int)
    for l in labels:
        counts[l] += 1
    valid = {l for l, c in counts.items() if c >= MIN_SUB_IDENTITY_FACES}
    print(f"Cluster {cluster}: {len(body)} body obs, {len(faces)} faces on tracks {sorted(tracks)}")
    print(f"  face sub-identities (label: count): {dict(sorted(counts.items(), key=lambda kv:-kv[1]))}")
    print(f"  valid sub-identities (>= {MIN_SUB_IDENTITY_FACES} faces): {sorted(valid)}")
    if len(valid) < 2:
        print("Fewer than 2 valid sub-identities — nothing to split.")
        return 0

    # scene -> sub-identity (majority face label in that scene)
    scene_to_label = {}
    scene_label_votes = defaultdict(lambda: defaultdict(int))
    for scene, l in zip(face_scenes, labels):
        if l in valid and scene is not None:
            scene_label_votes[scene][l] += 1
    for scene, votes in scene_label_votes.items():
        scene_to_label[scene] = max(votes, key=votes.get)

    # Assign each body observation to a sub-identity via its scene.
    label_to_obs = defaultdict(list)
    unassigned = []
    for oid, scene, _track in body:
        l = scene_to_label.get(scene)
        if l is None:
            unassigned.append(oid)
        else:
            label_to_obs[l].append(oid)

    for l in sorted(label_to_obs):
        print(f"  sub-identity {l}: {len(label_to_obs[l])} body obs")
    print(f"  unassigned (no face in scene): {len(unassigned)}")

    if args.dry_run:
        print("\nDRY RUN — no changes. Re-run with --apply.")
        return 0

    # APPLY: keep the cluster id for the largest sub-identity; new objects for others.
    sorted_labels = sorted(label_to_obs.keys(), key=lambda l: -len(label_to_obs[l]))
    label_to_object = {}
    for i, l in enumerate(sorted_labels):
        if i == 0:
            label_to_object[l] = int(cluster)
        else:
            cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,))
            label_to_object[l] = cur.fetchone()[0]

    for l, oids in label_to_obs.items():
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (label_to_object[l], oids),
        )

    # Re-link face observations to their sub-identity's object, and set person_id.
    for fid, l in zip(face_ids, labels):
        if l in label_to_object:
            cur.execute(
                "UPDATE face_observations SET object_id = %s, person_id = %s WHERE id = %s",
                (label_to_object[l], label_to_object[l], fid),
            )

    # Unassigned body observations stay on the original cluster id (largest group)
    # only if that group is the dominant person; otherwise move to the dominant.
    if unassigned:
        dominant = label_to_object[sorted_labels[0]]
        cur.execute(
            "UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)",
            (dominant, unassigned),
        )

    conn.commit()
    print(f"\nAPPLIED: cluster {cluster} split into {len(label_to_object)} person clusters: "
          f"{ {l: label_to_object[l] for l in sorted_labels} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
