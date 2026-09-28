#!/usr/bin/env python3
"""Rebuild PERSON clusters from offline face embeddings.

Uses `person_face_embeddings` (ArcFace 512-d, from gen_face_embeddings.py) as the
ground-truth identity signal, because OSNet body embeddings cannot separate the
cast. Steps:

  1. Cluster all face embeddings into identities (hierarchical, average linkage,
     cosine threshold). Each identity = one physical person.
  2. Assign every person observation to an identity:
       a. its own face embedding, if it has one;
       b. else the identity shared by a faced observation in the same
          (yolo_track_id, scene_id);
       c. else nearest identity centroid by body embedding (>= body floor);
       d. else a provisional per-(scene, track) cluster.
  3. Rebuild objects: one objects row per identity (largest keeps an existing id),
     move object_observations, re-link object_observation_parts and face rows.

Backups (objects_backup_facerebuild / object_observations_backup_facerebuild) must
exist first. Run with --dry-run to inspect the plan, then --apply.

Usage (in bordsupr):
    python3 /tmp/recluster_persons_by_face_embedding.py --dry-run
    python3 /tmp/recluster_persons_by_face_embedding.py --apply
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
        host=os.environ.get("PGHOST", "db"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def parse_emb(t):
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        return np.asarray(t, dtype=float)
    try:
        return np.asarray(json.loads(t), dtype=float)
    except Exception:
        return None


def norm(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def cluster_faces(oids, embs, thr):
    """Greedy average-linkage clustering of face embeddings. Returns oid->cluster."""
    order = sorted(range(len(oids)), key=lambda i: -len(embs[i]))  # deterministic
    centroids = []   # running sum of unit vectors
    counts = []
    label = {}
    for i in order:
        v = embs[i]
        best, bs = -1, -1.0
        for ci in range(len(centroids)):
            s = float(np.dot(v, centroids[ci] / counts[ci]))
            if s > bs:
                bs, best = s, ci
        if best >= 0 and bs >= thr:
            label[oids[i]] = best
            centroids[best] += v
            counts[best] += 1
        else:
            label[oids[i]] = len(centroids)
            centroids.append(v.copy())
            counts.append(1)
    return label, centroids, counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--face-threshold", type=float, default=0.30,
                    help="cosine threshold to merge faces into one identity")
    ap.add_argument("--min-identity-faces", type=int, default=3,
                    help="identities with fewer faced obs are treated as noise/provisional")
    ap.add_argument("--body-floor", type=float, default=0.70,
                    help="min body-embedding cosine to attach a faceless obs to an identity")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if not args.dry_run and not args.apply:
        print("Specify --dry-run or --apply")
        return 2

    conn = get_conn()
    cur = conn.cursor()

    # --- Load all person observations ---
    cur.execute(
        "SELECT id, object_id, yolo_track_id, scene_id, embedding::text "
        "FROM object_observations WHERE class_id = %s ORDER BY id",
        (PERSON_CLASS_ID,),
    )
    obs_rows = cur.fetchall()
    obs_ids = [r[0] for r in obs_rows]
    body_emb = {r[0]: parse_emb(r[4]) for r in obs_rows}
    obs_track = {r[0]: r[2] for r in obs_rows}
    obs_scene = {r[0]: r[3] for r in obs_rows}
    print(f"person observations: {len(obs_ids)}")

    # --- Load face embeddings ---
    cur.execute(
        "SELECT observation_id, embedding::text FROM person_face_embeddings WHERE embedding IS NOT NULL"
    )
    face_rows = cur.fetchall()
    face_emb = {}
    for oid, etxt in face_rows:
        e = norm(parse_emb(etxt))
        if e is not None:
            face_emb[oid] = e
    print(f"faced observations: {len(face_emb)}")

    if len(face_emb) < 2:
        print("Not enough face embeddings — run gen_face_embeddings.py first.")
        return 1

    # --- 1. Cluster faces into identities ---
    f_oids = list(face_emb.keys())
    f_embs = [face_emb[o] for o in f_oids]
    oid_to_cluster, centroids, counts = cluster_faces(f_oids, f_embs, args.face_threshold)
    cluster_members = defaultdict(list)
    for o, c in oid_to_cluster.items():
        cluster_members[c].append(o)
    # Keep only identities with enough faced obs; others become noise.
    identities = {c: m for c, m in cluster_members.items() if len(m) >= args.min_identity_faces}
    noise_clusters = {c: m for c, m in cluster_members.items() if len(m) < args.min_identity_faces}
    print(f"face clusters: {len(cluster_members)} total, {len(identities)} identities "
          f"(>= {args.min_identity_faces} faces), {len(noise_clusters)} noise")

    # Identity centroids (unit).
    ident_centroid = {}
    for c, members in identities.items():
        m = np.mean(np.stack([face_emb[o] for o in members]), axis=0)
        ident_centroid[c] = norm(m)

    # --- 2. Assign observations ---
    # (a) faced obs -> their identity (or none if noise cluster)
    obs_ident = {}
    for o in obs_ids:
        if o in oid_to_cluster:
            c = oid_to_cluster[o]
            if c in identities:
                obs_ident[o] = c

    # (b) faceless obs sharing (track, scene) with a faced, identified obs
    trackscene_ident = {}
    for o in obs_ids:
        if o in obs_ident:
            key = (obs_track[o], obs_scene[o])
            trackscene_ident.setdefault(key, defaultdict(int))
            trackscene_ident[key][obs_ident[o]] += 1
    n_ts = 0
    for o in obs_ids:
        if o in obs_ident:
            continue
        key = (obs_track[o], obs_scene[o])
        if key in trackscene_ident:
            # majority identity in this track+scene
            best_ident = max(trackscene_ident[key].items(), key=lambda kv: kv[1])[0]
            obs_ident[o] = best_ident
            n_ts += 1

    # (c) remaining faceless obs -> nearest identity by BODY embedding. Build a
    # body-embedding centroid per identity from the obs already assigned to it
    # (faced + track/scene). NOTE: body (OSNet) and face (ArcFace) embeddings live
    # in different spaces, so we must NOT compare body obs to face centroids.
    ident_body_sum = defaultdict(lambda: None)
    ident_body_cnt = defaultdict(int)
    for o, c in obs_ident.items():
        be = norm(body_emb.get(o))
        if be is None:
            continue
        if ident_body_sum[c] is None:
            ident_body_sum[c] = be.copy()
        else:
            ident_body_sum[c] += be
        ident_body_cnt[c] += 1
    ident_body_centroid = {}
    for c, s in ident_body_sum.items():
        if ident_body_cnt[c] > 0:
            ident_body_centroid[c] = norm(s / ident_body_cnt[c])

    n_body = 0
    leftover = []
    for o in obs_ids:
        if o in obs_ident:
            continue
        be = norm(body_emb.get(o))
        if be is None:
            leftover.append(o)
            continue
        best, bs = None, -1.0
        for c, cen in ident_body_centroid.items():
            if cen is None:
                continue
            s = float(np.dot(be, cen))
            if s > bs:
                bs, best = s, c
        if best is not None and bs >= args.body_floor:
            obs_ident[o] = best
            n_body += 1
        else:
            leftover.append(o)

    print(f"assignment: faced={sum(1 for o in obs_ids if o in oid_to_cluster and o in obs_ident)}, "
          f"track/scene={n_ts}, body={n_body}, leftover={len(leftover)}")

    # Per-identity sizes
    ident_sizes = defaultdict(int)
    for o, c in obs_ident.items():
        ident_sizes[c] += 1
    sizes = sorted(ident_sizes.values(), reverse=True)
    print(f"identity sizes (top 20): {sizes[:20]}")

    # Leftovers -> provisional per-(scene, track)
    leftover_groups = defaultdict(list)
    for o in leftover:
        leftover_groups[(obs_scene[o], obs_track[o])].append(o)
    print(f"leftover provisional clusters: {len(leftover_groups)}")

    if args.dry_run:
        print("\nDRY RUN — no changes. Re-run with --apply.")
        return 0

    # --- 3. APPLY: rebuild objects ---
    # Map each identity to an objects row. Reuse existing object ids where possible:
    # for each identity, pick the most common current object_id among its obs.
    cur.execute("SELECT id FROM objects WHERE class_id = %s", (PERSON_CLASS_ID,))
    existing_obj_ids = {r[0] for r in cur.fetchall()}
    obs_current_obj = {r[0]: r[1] for r in obs_rows}

    ident_to_obj = {}
    used_objs = set()
    # Sort identities by size desc so the largest keeps an existing id.
    for c in sorted(identities.keys(), key=lambda c: -ident_sizes[c]):
        member_objs = [obs_current_obj[o] for o in obs_ids if obs_ident.get(o) == c]
        # most common current object among members that is still free
        cand = defaultdict(int)
        for ob in member_objs:
            cand[ob] += 1
        chosen = None
        for ob, _ in sorted(cand.items(), key=lambda kv: -kv[1]):
            if ob in existing_obj_ids and ob not in used_objs:
                chosen = ob
                break
        if chosen is None:
            cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,))
            chosen = cur.fetchone()[0]
        ident_to_obj[c] = chosen
        used_objs.add(chosen)

    # Move observations.
    ident_to_obslist = defaultdict(list)
    for o, c in obs_ident.items():
        ident_to_obslist[c].append(o)
    moved = 0
    for c, obj in ident_to_obj.items():
        ol = ident_to_obslist[c]
        if not ol:
            continue
        cur.execute("UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)", (obj, ol))
        cur.execute("UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)", (obj, ol))
        moved += len(ol)

    # Leftovers -> provisional objects.
    provisional = 0
    for (sc, tr), ol in leftover_groups.items():
        cur.execute("INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,))
        obj = cur.fetchone()[0]
        provisional += 1
        cur.execute("UPDATE object_observations SET object_id = %s WHERE id = ANY(%s)", (obj, ol))
        cur.execute("UPDATE object_observation_parts SET object_id = %s WHERE observation_id = ANY(%s)", (obj, ol))

    # Delete now-empty old person objects (not reused).
    cur.execute(
        "DELETE FROM objects o WHERE o.class_id = %s AND NOT EXISTS "
        "(SELECT 1 FROM object_observations oo WHERE oo.object_id = o.id)",
        (PERSON_CLASS_ID,),
    )
    deleted = cur.rowcount

    conn.commit()
    print(f"\nAPPLIED:")
    print(f"  identities -> objects: {len(ident_to_obj)}")
    print(f"  observations moved:    {moved}")
    print(f"  provisional clusters:  {provisional}")
    print(f"  empty objects deleted: {deleted}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
