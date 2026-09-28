#!/usr/bin/env python3
"""Offline face-embedding generation for all PERSON observations.

Runs InsightFace (buffalo_l / ArcFace) on each person observation's stored
cropped_image and stores the best face's 512-d embedding + detection score into a
staging table `person_face_embeddings`. Resumable: skips obs already present.

Usage (in samira_bordsupr):
    python3 /tmp/gen_face_embeddings.py [--limit N] [--det-size 320]
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

    # Observations not yet processed.
    cur.execute(
        """
        SELECT oo.id, oo.cropped_image
        FROM object_observations oo
        LEFT JOIN person_face_embeddings pfe ON pfe.observation_id = oo.id
        WHERE oo.class_id = %s AND oo.cropped_image IS NOT NULL AND pfe.observation_id IS NULL
        ORDER BY oo.id
        """,
        (PERSON_CLASS_ID,),
    )
    rows = cur.fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]
    total = len(rows)
    print(f"observations to process: {total}", flush=True)
    if total == 0:
        return 0

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))

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
                "VALUES (%s, %s::vector, %s, %s::jsonb) ON CONFLICT (observation_id) DO NOTHING",
                (oid, vec, score, __import__("json").dumps(bbox)),
            )
            faced += 1
        else:
            # Record a NULL-embedding row so we don't reprocess.
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
