#!/usr/bin/env python3
"""Offline parameter sweep for the website's reclustering endpoints.

Replicates (read-only, in-memory) the exact algorithms of:
  - POST /api/objects/recluster-faces  (_run_full_face_rebuild: _face_cluster greedy
    avg-linkage on unit ArcFace embeddings + identity filter + body-merge pass)
  - POST /api/objects/recluster        (_cluster_features greedy running-mean cosine)

Sweeps thresholds and reports metrics so the best settings can be chosen WITHOUT
mutating the live database.

People metrics: #identities, top sizes, sven-labelled cohesion/split, identity
separation, body-merge coverage.
Object metrics: #clusters, singletons, BCubed P/R/F1 against named instances
(backpack_001/002, laptop_001/004, mug_001, ...).

Usage: python scripts/sweep_recluster_settings.py [--dsn postgresql://...]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict

import numpy as np
import psycopg2

DEFAULT_DSN = "postgresql://postgres:postgres@localhost:35432/bordsupr"
sven_OBJECT_ID = 9812345785  # only named person with labelled face observations


# --------------------------------------------------------------------------
# DB loading (read-only)
# --------------------------------------------------------------------------

def _parse_vec(text):
    if text is None:
        return None
    t = str(text).strip()
    if t.startswith("["):
        t = t[1:-1]
    if not t:
        return None
    return np.asarray([float(p) for p in t.split(",") if p.strip()], dtype=np.float64)


def _unit(v):
    if v is None:
        return None
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def load_data(cur):
    print("loading face embeddings ...", flush=True)
    cur.execute(
        "SELECT observation_id, embedding::text FROM person_face_embeddings "
        "WHERE embedding IS NOT NULL ORDER BY observation_id"
    )
    face_ids, face_vecs = [], []
    for oid, etxt in cur.fetchall():
        v = _unit(_parse_vec(etxt))
        if v is not None:
            face_ids.append(oid)
            face_vecs.append(v)
    face_mat = np.stack(face_vecs)  # (n, 512) unit rows

    print("loading labelled (sven) face observations ...", flush=True)
    cur.execute(
        "SELECT observation_id, embedding::text FROM face_observations WHERE object_id = %s",
        (sven_OBJECT_ID,),
    )
    sven_rows = cur.fetchall()
    sven_oids = {r[0] for r in sven_rows if r[0] is not None}
    # face_observations.observation_id often references wiped staging rows; fall back
    # to matching its own ArcFace embedding column against the staging vectors.
    sven_vecs = []
    for _oid, etxt in sven_rows:
        v = _unit(_parse_vec(etxt))
        if v is not None:
            sven_vecs.append(v)
    if sven_vecs:
        sven_mat = np.stack(sven_vecs)
    else:
        sven_mat = None

    print("loading person observations (body embeddings) ...", flush=True)
    cur.execute(
        "SELECT id, yolo_track_id, scene_id, embedding::text FROM object_observations "
        "WHERE class_id = 0 ORDER BY id"
    )
    pobs_ids, pobs_track, pobs_scene, pobs_body = [], {}, {}, {}
    for oid, tr, sc, etxt in cur.fetchall():
        pobs_ids.append(oid)
        pobs_track[oid] = tr
        pobs_scene[oid] = sc
        pobs_body[oid] = _unit(_parse_vec(etxt))

    print("loading named object instances + their observations ...", flush=True)
    cur.execute("SELECT id, name, class_id FROM objects WHERE name IS NOT NULL AND class_id != 0")
    named_objects = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    obj_to_name = {oid: name for oid, (name, _cid) in named_objects.items()}

    # all observations (any class) that belong to named objects, per class
    cur.execute(
        "SELECT oo.id, oo.object_id, o.class_id, oo.embedding::text "
        "FROM object_observations oo JOIN objects o ON o.id = oo.object_id "
        "WHERE oo.embedding IS NOT NULL AND o.class_id != 0 ORDER BY oo.created_at, oo.id"
    )
    class_obs = defaultdict(list)  # class_id -> [(obs_id, name_or_None, vec)]
    for oid, object_id, cid, etxt in cur.fetchall():
        v = _parse_vec(etxt)
        if v is None:
            continue
        class_obs[cid].append((oid, obj_to_name.get(object_id), v))

    # cluster counts per class need ALL obs of the class, not only named
    print("loading all non-person observations per class ...", flush=True)
    cur.execute(
        "SELECT oo.id, o.class_id, oo.embedding::text "
        "FROM object_observations oo JOIN objects o ON o.id = oo.object_id "
        "WHERE oo.embedding IS NOT NULL AND o.class_id != 0 ORDER BY oo.created_at, oo.id"
    )
    class_all_obs = defaultdict(list)  # class_id -> [(obs_id, vec)]
    for oid, cid, etxt in cur.fetchall():
        v = _parse_vec(etxt)
        if v is not None:
            class_all_obs[cid].append((oid, v))

    return {
        "face_ids": face_ids,
        "face_mat": face_mat,
        "sven_oids": sven_oids,
        "sven_mat": sven_mat,
        "pobs_ids": pobs_ids,
        "pobs_track": pobs_track,
        "pobs_scene": pobs_scene,
        "pobs_body": pobs_body,
        "class_obs": class_obs,
        "class_all_obs": class_all_obs,
    }


# --------------------------------------------------------------------------
# Exact algorithm replicas
# --------------------------------------------------------------------------

def face_cluster(vecs: np.ndarray, thr: float) -> np.ndarray:
    """Replica of app.py _face_cluster: greedy avg-linkage on unit vectors.

    order = stable sort by -len(vec) == input order (all dims equal)."""
    n = len(vecs)
    labels = np.empty(n, dtype=int)
    cent_sum = None  # (k, dim) running sums
    counts = []
    for i in range(n):
        v = vecs[i]
        if counts:
            sims = (cent_sum / np.asarray(counts)[:, None]) @ v
            best = int(np.argmax(sims))
            bs = float(sims[best])
            if bs >= thr:
                labels[i] = best
                cent_sum[best] += v
                counts[best] += 1
                continue
        labels[i] = len(counts)
        cent_sum = v[None, :].copy() if cent_sum is None else np.vstack([cent_sum, v])
        counts.append(1)
    return labels


def cluster_features(vecs: list[np.ndarray], thr: float) -> list[int]:
    """Replica of app.py _cluster_features: cosine vs running-MEAN centroid."""
    labels: list[int] = []
    cent_sum = None
    counts: list[int] = []
    for v in vecs:
        vn = v / (np.linalg.norm(v) + 1e-12)
        if counts:
            means = cent_sum / np.asarray(counts)[:, None]
            means = means / (np.linalg.norm(means, axis=1, keepdims=True) + 1e-12)
            sims = means @ vn
            best = int(np.argmax(sims))
            if float(sims[best]) >= thr:
                labels.append(best)
                cent_sum[best] += v
                counts[best] += 1
                continue
        labels.append(len(counts))
        cent_sum = v[None, :].copy() if cent_sum is None else np.vstack([cent_sum, v])
        counts.append(1)
    return labels


# --------------------------------------------------------------------------
# People sweep
# --------------------------------------------------------------------------

def sweep_people(data, face_thresholds, min_faces_options):
    face_ids = data["face_ids"]
    face_mat = data["face_mat"]
    sven_oids = data["sven_oids"]
    sven_mat = data.get("sven_mat")
    sven_idx = [i for i, o in enumerate(face_ids) if o in sven_oids]
    # Fallback: map labelled embeddings onto staging rows by near-exact match (cos > 0.999)
    if not sven_idx and sven_mat is not None and len(face_ids):
        sims = sven_mat @ face_mat.T  # (n_sven, n_faces), both unit
        best = sims.argmax(axis=1)
        sven_idx = sorted({int(best[j]) for j in range(len(sven_mat)) if sims[j, best[j]] > 0.95})
    print(f"\n=== PEOPLE: {len(face_ids)} face embeddings, {len(sven_idx)} sven-labelled ===")

    results = []
    for thr in face_thresholds:
        t0 = time.time()
        labels = face_cluster(face_mat, thr)
        k = labels.max() + 1
        sizes = np.bincount(labels)
        row = {"face_threshold": thr, "n_face_clusters": k}
        for minf in min_faces_options:
            idents = [c for c in range(k) if sizes[c] >= minf]
            ident_set = set(idents)
            # sven cohesion: into how many identities do sven faces fall?
            sven_ident_counts = defaultdict(int)
            for i in sven_idx:
                c = labels[i]
                if c in ident_set:
                    sven_ident_counts[c] += 1
                else:
                    sven_ident_counts[-1] += 1  # discarded small cluster
            n_sven_idents = len([c for c in sven_ident_counts if c != -1])
            top_sizes = sorted((int(sizes[c]) for c in idents), reverse=True)[:10]
            # separation: min cosine between the top-6 identity centroids
            cents = []
            for c in sorted(idents, key=lambda c: -sizes[c])[:6]:
                m = face_mat[labels == c].mean(axis=0)
                n = np.linalg.norm(m)
                cents.append(m / n if n > 0 else m)
            min_sep = None
            for a in range(len(cents)):
                for b in range(a + 1, len(cents)):
                    s = float(np.dot(cents[a], cents[b]))
                    min_sep = s if min_sep is None else min(min_sep, s)
            row[f"minf={minf}"] = {
                "n_identities": len(idents),
                "top_sizes": top_sizes,
                "sven_identities": n_sven_idents,
                "sven_in_largest": max(sven_ident_counts.values()) if sven_ident_counts else 0,
                "min_top6_centroid_cos": round(min_sep, 3) if min_sep is not None else None,
            }
        row["seconds"] = round(time.time() - t0, 1)
        results.append(row)
        print(f"  thr={thr:.2f} done in {row['seconds']}s "
              + " | ".join(
                  f"minf{mf}: {row[f'minf={mf}']['n_identities']} id, sven_split={row[f'minf={mf}']['sven_identities']}, sep={row[f'minf={mf}']['min_top6_centroid_cos']}"
                  for mf in min_faces_options), flush=True)
    return results


def sweep_people_body_merge(data, face_threshold, min_faces, merge_sims, merge_margins):
    """Coverage of the 3b body-merge pass: % of person obs that land in mains."""
    face_ids = data["face_ids"]
    face_mat = data["face_mat"]
    pobs_ids = data["pobs_ids"]
    pobs_track = data["pobs_track"]
    pobs_scene = data["pobs_scene"]
    pobs_body = data["pobs_body"]

    labels = face_cluster(face_mat, face_threshold)
    sizes = np.bincount(labels)
    face_cluster_of = {o: labels[i] for i, o in enumerate(face_ids)}
    identities = {c for c in range(len(sizes)) if sizes[c] >= min_faces}

    obs_ident = {}
    for o in pobs_ids:
        c = face_cluster_of.get(o)
        if c is not None and c in identities:
            obs_ident[o] = c
    # (b) track/scene propagation
    trackscene = defaultdict(lambda: defaultdict(int))
    for o, c in obs_ident.items():
        trackscene[(pobs_track[o], pobs_scene[o])][c] += 1
    for o in pobs_ids:
        if o in obs_ident:
            continue
        key = (pobs_track[o], pobs_scene[o])
        if key in trackscene:
            obs_ident[o] = max(trackscene[key].items(), key=lambda kv: kv[1])[0]

    leftover_groups = defaultdict(list)
    for o in pobs_ids:
        if o not in obs_ident:
            leftover_groups[(pobs_scene[o], pobs_track[o])].append(o)

    ident_obs = defaultdict(list)
    for o, c in obs_ident.items():
        ident_obs[c].append(o)
    main_centroid = {}
    for c, ol in ident_obs.items():
        vecs = [pobs_body[o] for o in ol if pobs_body.get(o) is not None]
        if vecs:
            m = np.mean(np.stack(vecs), axis=0)
            n = np.linalg.norm(m)
            if n > 0:
                main_centroid[c] = m / n

    prov_centroids = []
    prov_sizes = []
    for (_sc, _tr), ol in leftover_groups.items():
        vecs = [pobs_body[o] for o in ol if pobs_body.get(o) is not None]
        if not vecs:
            continue
        m = np.mean(np.stack(vecs), axis=0)
        n = np.linalg.norm(m)
        if n > 0:
            prov_centroids.append(m / n)
            prov_sizes.append(len(ol))

    print(f"\n=== PEOPLE body-merge coverage (face_thr={face_threshold}, min_faces={min_faces}) ===")
    print(f"  baseline: {len(obs_ident)}/{len(pobs_ids)} obs in mains "
          f"({100.0 * len(obs_ident) / max(1, len(pobs_ids)):.1f}%), "
          f"{len(leftover_groups)} provisional groups")
    results = []
    mc_keys = list(main_centroid.keys())
    mc_mat = np.stack([main_centroid[c] for c in mc_keys]) if mc_keys else None
    for msim in merge_sims:
        for mmarg in merge_margins:
            merged_obs = 0
            merged_groups = 0
            if mc_mat is not None:
                for pc, psz in zip(prov_centroids, prov_sizes):
                    sims = mc_mat @ pc
                    order = np.argsort(-sims)
                    best = float(sims[order[0]])
                    second = float(sims[order[1]]) if len(order) > 1 else -1.0
                    if best >= msim and (best - second) >= mmarg * msim:
                        merged_groups += 1
                        merged_obs += psz
            cov = 100.0 * (len(obs_ident) + merged_obs) / max(1, len(pobs_ids))
            results.append({"body_merge_sim": msim, "body_merge_margin": mmarg,
                            "merged_groups": merged_groups, "coverage_pct": round(cov, 1)})
            print(f"  sim>={msim:.2f} margin={mmarg:.2f}: +{merged_groups} groups -> coverage {cov:.1f}%", flush=True)
    return results


# --------------------------------------------------------------------------
# Object sweep
# --------------------------------------------------------------------------

def bcubed(labels_pred, labels_true):
    """BCubed precision/recall/F1 over the labelled subset (labels_true != None)."""
    n = len(labels_pred)
    p_sum = r_sum = cnt = 0
    cluster_members = defaultdict(list)
    label_members = defaultdict(list)
    for i in range(n):
        if labels_true[i] is None:
            continue
        cluster_members[labels_pred[i]].append(i)
        label_members[labels_true[i]].append(i)
        cnt += 1
    if cnt == 0:
        return None, None, None
    for members in cluster_members.values():
        same = defaultdict(int)
        for i in members:
            same[labels_true[i]] += 1
        for i in members:
            p_sum += same[labels_true[i]] / len(members)
    for members in label_members.values():
        ccount = defaultdict(int)
        for i in members:
            ccount[labels_pred[i]] += 1
        for i in members:
            r_sum += ccount[labels_pred[i]] / len(members)
    p = p_sum / cnt
    r = r_sum / cnt
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def sweep_objects(data, thresholds, top_n_classes=8):
    class_all = data["class_all_obs"]
    class_named = data["class_obs"]
    named_by_class = defaultdict(dict)  # cid -> obs_id -> name
    for cid, rows in class_named.items():
        for oid, name, _v in rows:
            if name is not None:
                named_by_class[cid][oid] = name

    top_classes = sorted(class_all.keys(), key=lambda c: -len(class_all[c]))[:top_n_classes]
    print(f"\n=== OBJECTS: sweeping {len(top_classes)} classes "
          f"({', '.join(f'{c}:{len(class_all[c])}' for c in top_classes)}) ===")

    per_threshold = []
    for thr in thresholds:
        tot_f1_num = 0.0
        tot_f1_den = 0
        tot_clusters = 0
        tot_singletons = 0
        class_detail = {}
        for cid in top_classes:
            rows = class_all[cid]
            vecs = [v for _oid, v in rows]
            labels = cluster_features(vecs, thr)
            k = max(labels) + 1 if labels else 0
            sizes = defaultdict(int)
            for lab in labels:
                sizes[lab] += 1
            singletons = sum(1 for s in sizes.values() if s == 1)
            names = [named_by_class.get(cid, {}).get(oid) for oid, _v in rows]
            p, r, f1 = bcubed(labels, names)
            if f1 is not None:
                n_named = sum(1 for x in names if x is not None)
                tot_f1_num += f1 * n_named
                tot_f1_den += n_named
            tot_clusters += k
            tot_singletons += singletons
            class_detail[cid] = {"clusters": k, "singletons": singletons,
                                 "bcubed_f1": round(f1, 3) if f1 is not None else None}
        agg = {"threshold": thr,
               "weighted_named_f1": round(tot_f1_num / tot_f1_den, 3) if tot_f1_den else None,
               "total_clusters": tot_clusters,
               "total_singletons": tot_singletons,
               "classes": class_detail}
        per_threshold.append(agg)
        print(f"  thr={thr:.2f}: named-F1={agg['weighted_named_f1']} clusters={tot_clusters} singletons={tot_singletons}", flush=True)
    return per_threshold


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-people", action="store_true")
    ap.add_argument("--skip-objects", action="store_true")
    args = ap.parse_args()

    conn = psycopg2.connect(args.dsn)
    data = load_data(conn.cursor())
    conn.close()

    out = {}
    if not args.skip_people:
        out["people"] = sweep_people(
            data,
            face_thresholds=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
            min_faces_options=[2, 3, 5],
        )
        out["people_body_merge"] = sweep_people_body_merge(
            data, face_threshold=0.30, min_faces=3,
            merge_sims=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            merge_margins=[0.0, 0.02, 0.05],
        )
    if not args.skip_objects:
        out["objects"] = sweep_objects(
            data,
            thresholds=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
        )

    out_path = args.out or f"evaluation_outputs/recluster_sweep_{int(time.time())}.json"
    def _json_default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(str(type(o)))
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
