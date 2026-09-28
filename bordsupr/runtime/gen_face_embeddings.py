#!/usr/bin/env python3
"""Offline face-embedding generation for all PERSON observations.

Runs InsightFace (buffalo_l / ArcFace) on each person observation's stored
cropped_image and stores the best face's 512-d embedding + detection score into a
staging table `person_face_embeddings`. Resumable: skips obs already present.
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import psycopg2
from PIL import Image

PERSON_CLASS_ID = 0


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "db"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "bordsupr"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max obs to process this run (0=all)")
    ap.add_argument("--det-size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=200, help="commit every N obs")
    ap.add_argument(
        "--refind-null",
        action="store_true",
        help="re-process observations that already have a NULL-embedding staging row "
        "(previously 'no face found'), updating the row if a face is now detected. "
        "Default (off) processes only observations with no staging row at all.",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="force CPU even if a CUDA GPU provider is available.",
    )
    ap.add_argument(
        "--map-id",
        type=int,
        default=0,
        help="only process observations belonging to this map id (0=all maps). "
        "Matches object_observations.map_id or the observation's scene map_id.",
    )
    args = ap.parse_args()

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS person_face_embeddings (
            observation_id BIGINT PRIMARY KEY,
            embedding vector(512),
            det_score DOUBLE PRECISION,
            face_bbox JSONB,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    conn.commit()

    # Observations to process. Default: only those with no staging row at all.
    # --refind-null: also re-process rows that already have a NULL embedding
    # (previously 'no face found') so a face detected this time updates the row.
    # --map-id: restrict to observations belonging to one map.
    map_sql = ""
    map_params: list = []
    if args.map_id:
        map_sql = (
            " AND (oo.map_id = %s OR EXISTS "
            "(SELECT 1 FROM scenes s2 WHERE s2.id = oo.scene_id AND s2.map_id = %s))"
        )
        map_params = [args.map_id, args.map_id]
    if args.refind_null:
        cur.execute(
            """
            SELECT oo.id, oo.cropped_image
            FROM object_observations oo
            LEFT JOIN person_face_embeddings pfe ON pfe.observation_id = oo.id
            WHERE oo.class_id = %s AND oo.cropped_image IS NOT NULL
              AND (pfe.observation_id IS NULL OR pfe.embedding IS NULL)
            """ + map_sql + """
            ORDER BY oo.id
            """,
            [PERSON_CLASS_ID, *map_params],
        )
    else:
        cur.execute(
            """
            SELECT oo.id, oo.cropped_image
            FROM object_observations oo
            LEFT JOIN person_face_embeddings pfe ON pfe.observation_id = oo.id
            WHERE oo.class_id = %s AND oo.cropped_image IS NOT NULL AND pfe.observation_id IS NULL
            """ + map_sql + """
            ORDER BY oo.id
            """,
            [PERSON_CLASS_ID, *map_params],
        )
    rows = cur.fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"observations to process: {total}", flush=True)
    if total == 0:
        return 0

    from insightface.app import FaceAnalysis

    # Prefer GPU (CUDA) if available; fall back to CPU if the CUDA provider
    # fails to load (e.g. missing CUDA libs). ctx_id=-1 forces CPU.
    use_gpu = not getattr(args, "cpu", False)
    providers = ["CPUExecutionProvider"]
    ctx_id = -1
    if use_gpu:
        try:
            import onnxruntime as ort

            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
        except Exception:
            pass
    app = FaceAnalysis(name="buffalo_l", providers=providers)
    app.prepare(ctx_id=ctx_id, det_size=(args.det_size, args.det_size))
    active = set()
    for _m in app.models.values():
        try:
            active.update(_m.session.get_providers())
        except Exception:
            pass
    print(f"face-embed providers requested={providers} active={sorted(active)} ctx_id={ctx_id}", flush=True)

    ins = conn.cursor()
    t0 = time.time()
    faced = 0
    for i, (oid, img) in enumerate(rows):
        emb = None
        score = None
        bbox = None
        if img is not None and len(img) > 100:
            try:
                im = np.array(Image.open(io.BytesIO(bytes(img))).convert("RGB"))[:, :, ::-1]
                faces = app.get(im)
                if faces:
                    f = max(faces, key=lambda x: x.det_score)
                    emb = f.embedding / np.linalg.norm(f.embedding)
                    score = float(f.det_score)
                    bbox = [float(v) for v in f.bbox]
            except Exception as exc:
                print(f"  obs {oid}: error {exc}", flush=True)
        if emb is not None:
            vec = "[" + ",".join(f"{v:.6f}" for v in emb.tolist()) + "]"
            ins.execute(
                "INSERT INTO person_face_embeddings (observation_id, embedding, det_score, face_bbox) "
                "VALUES (%s, %s::vector, %s, %s::jsonb) "
                "ON CONFLICT (observation_id) DO UPDATE SET "
                "embedding = EXCLUDED.embedding, det_score = EXCLUDED.det_score, "
                "face_bbox = EXCLUDED.face_bbox, created_at = now()",
                (oid, vec, score, __import__("json").dumps(bbox)),
            )
            faced += 1
        else:
            # Record a NULL-embedding row so we don't reprocess (unless --refind-null).
            ins.execute(
                "INSERT INTO person_face_embeddings (observation_id, embedding, det_score, face_bbox) "
                "VALUES (%s, NULL, NULL, NULL) ON CONFLICT (observation_id) DO NOTHING",
                (oid,),
            )
        if (i + 1) % args.batch == 0:
            conn.commit()
            dt = time.time() - t0
            rate = (i + 1) / dt if dt > 0 else 0
            eta = (total - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{total} ({rate:.1f}/s, eta {eta:.0f}s) faced={faced}", flush=True)
    conn.commit()
    dt = time.time() - t0
    print(f"DONE: {total} processed in {dt:.0f}s, {faced} with faces ({100.0*faced/max(total,1):.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
