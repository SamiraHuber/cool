#!/usr/bin/env python3

from __future__ import annotations

import argparse

from agent.text_embeddings import embed_text, get_text_embedding_dim
from agent.tools import _get_conn


def _to_vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _ensure_schema(cur) -> None:
    dim = get_text_embedding_dim()
    cur.execute(f"ALTER TABLE scenes ADD COLUMN IF NOT EXISTS caption_embedding VECTOR({dim})")
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_scenes_caption_embedding_hnsw "
        "ON scenes USING hnsw (caption_embedding vector_cosine_ops)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill scene caption embeddings.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of scenes to update (0 = all missing).")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings even when caption_embedding is already set.")
    args = parser.parse_args()

    with _get_conn() as conn:
        with conn.cursor() as cur:
            _ensure_schema(cur)
            where_clause = "WHERE caption IS NOT NULL AND btrim(caption) <> ''"
            if not args.force:
                where_clause += " AND caption_embedding IS NULL"
            limit_clause = ""
            params: list[object] = []
            if args.limit and args.limit > 0:
                limit_clause = " LIMIT %s"
                params.append(int(args.limit))
            cur.execute(
                f"""
                SELECT id, caption
                FROM scenes
                {where_clause}
                ORDER BY id ASC
                {limit_clause}
                """,
                tuple(params),
            )
            rows = cur.fetchall()

            updated = 0
            skipped = 0
            for scene_id, caption in rows:
                embedding = embed_text(str(caption or ""))
                if not embedding:
                    skipped += 1
                    continue
                cur.execute(
                    "UPDATE scenes SET caption_embedding = %s::vector WHERE id = %s",
                    (_to_vector_literal(embedding), int(scene_id)),
                )
                updated += 1
                if updated % 25 == 0:
                    print(f"updated {updated} scenes...")

        conn.commit()

    print(f"updated={updated} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())