#!/usr/bin/env python3
"""Populate face_observations from the offline person_face_embeddings staging table.

After the face-based person rebuild, each clean person body-cluster needs face-identity
evidence in face_observations so the live node's face-identity consolidation guard
(_person_clusters_face_conflict) can recognise that two clusters are different people.

For every faced observation (person_face_embeddings.embedding IS NOT NULL) we insert a
face_observations row with:
  - observation_id / object_id: the observation's (new) body cluster
  - embedding: the ArcFace vector
  - person_id: ONE face-identity objects row per body cluster (created if needed), so a
    cluster's dominant face person_id is unique to that cluster/person.

Idempotent: skips observations already present in face_observations.
"""
import os
import sys
import psycopg2

PERSON_CLASS_ID = 0


def get_conn():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "db"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname="bordsupr", user="postgres", password="postgres",
    )


def main():
    conn = get_conn()
    cur = conn.cursor()

    # Map each faced observation to its current body cluster.
    cur.execute(
        """
        SELECT pfe.observation_id, oo.object_id, oo.scene_id, oo.yolo_track_id,
               pfe.embedding::text, pfe.det_score
        FROM person_face_embeddings pfe
        JOIN object_observations oo ON oo.id = pfe.observation_id
        LEFT JOIN face_observations fo ON fo.observation_id = pfe.observation_id
        WHERE pfe.embedding IS NOT NULL AND fo.id IS NULL
        ORDER BY oo.object_id, pfe.observation_id
        """
    )
    rows = cur.fetchall()
    print(f"faced observations to link: {len(rows)}")
    if not rows:
        return 0

    # One face-identity person object per body cluster.
    cluster_person = {}
    inserted = 0
    for oid, object_id, scene_id, track_id, emb_text, det_score in rows:
        if object_id not in cluster_person:
            # Create a face-identity object (a person row) to represent this cluster's identity.
            cur.execute(
                "INSERT INTO objects (class_id) VALUES (%s) RETURNING id", (PERSON_CLASS_ID,)
            )
            cluster_person[object_id] = cur.fetchone()[0]
        person_id = cluster_person[object_id]
        cur.execute(
            """
            INSERT INTO face_observations
                (person_id, scene_id, object_id, yolo_track_id, embedding, score, observation_id)
            VALUES (%s, %s, %s, %s, %s::vector, %s, %s)
            """,
            (person_id, scene_id, object_id, track_id, emb_text, det_score, oid),
        )
        inserted += 1
        if inserted % 500 == 0:
            conn.commit()
            print(f"  inserted {inserted}")
    conn.commit()
    print(f"DONE: inserted {inserted} face_observations across {len(cluster_person)} person identities")
    return 0


if __name__ == "__main__":
    sys.exit(main())
