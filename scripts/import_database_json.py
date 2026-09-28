#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


DATABASE_URL_DEFAULT = "postgresql://postgres:postgres@localhost:35432/bordsupr"
KNOWN_TABLE_ORDER = (
    "maps",
    "rooms",
    "scenes",
    "objects",
    "object_observations",
    "face_observations",
    "interactions",
)


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    db_type: str
    nullable: bool
    has_default: bool


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import a table-wise JSON dataset export into the PostgreSQL database."
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to the JSON file to import.",
    )
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", DATABASE_URL_DEFAULT),
        help="PostgreSQL connection string.",
    )
    parser.add_argument(
        "--schema",
        default="public",
        help="Destination schema name.",
    )
    parser.add_argument(
        "--truncate-first",
        action="store_true",
        help="Delete existing rows in the imported tables before loading the JSON rows, including previously imported maps/buildings.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the import inside a transaction and roll it back at the end.",
    )
    return parser.parse_args(argv)


def get_db_driver():
    try:
        import psycopg2 as db_driver  # type: ignore

        return db_driver
    except ImportError:
        pass

    try:
        import psycopg as db_driver  # type: ignore

        return db_driver
    except ImportError as exc:
        raise RuntimeError(
            "No PostgreSQL Python driver found. Install psycopg2 or psycopg in the execution environment."
        ) from exc


def load_dataset(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"JSON file not found: {json_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file {json_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Top-level JSON value must be an object keyed by table name.")

    unknown_tables = sorted(set(payload.keys()) - set(KNOWN_TABLE_ORDER))
    if unknown_tables:
        joined = ", ".join(unknown_tables)
        raise RuntimeError(f"Unsupported top-level table keys in JSON: {joined}")

    dataset: Dict[str, List[Dict[str, Any]]] = {}
    for table_name in KNOWN_TABLE_ORDER:
        if table_name not in payload:
            continue
        rows = payload[table_name]
        if not isinstance(rows, list):
            raise RuntimeError(f"JSON value for table '{table_name}' must be a list of row objects.")
        normalized_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise RuntimeError(f"Row {index} for table '{table_name}' is not a JSON object.")
            normalized_rows.append(row)
        dataset[table_name] = normalized_rows

    if not dataset:
        raise RuntimeError("The JSON file does not contain any supported table data.")

    return dataset


def collect_import_notes(dataset: Mapping[str, Sequence[Mapping[str, Any]]], truncate_first: bool) -> List[str]:
    notes: List[str] = []

    map_rows = dataset.get("maps") or []
    map_names = [str(row.get("name") or "").strip() for row in map_rows if str(row.get("name") or "").strip()]
    if len(map_names) != len(set(map_names)):
        notes.append("The JSON contains duplicate map names. Each map must have a distinct name because maps.name is unique.")

    if truncate_first and map_rows:
        notes.append(
            "--truncate-first removes previously imported maps before loading this dataset, so the website will only show buildings from the current import."
        )

    if map_rows:
        notes.append(
            "The website building selector uses maps.name. Changing only a map id will not create a second selectable building; use a distinct map name for each building you want listed."
        )

    return notes


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def qualified_table_name(schema: str, table: str) -> str:
    return f"{quote_ident(schema)}.{quote_ident(table)}"


def fetch_table_columns(conn, schema: str, table: str) -> List[ColumnInfo]:
    query = """
        SELECT
            a.attname,
            pg_catalog.format_type(a.atttypid, a.atttypmod) AS formatted_type,
            NOT a.attnotnull AS is_nullable,
            ad.adbin IS NOT NULL AS has_default
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_catalog.pg_attrdef ad ON ad.adrelid = a.attrelid AND ad.adnum = a.attnum
        WHERE n.nspname = %s
          AND c.relname = %s
          AND a.attnum > 0
          AND NOT a.attisdropped
        ORDER BY a.attnum
    """

    with conn.cursor() as cur:
        cur.execute(query, (schema, table))
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(f"Destination table not found: {schema}.{table}")

    return [
        ColumnInfo(
            name=str(row[0]),
            db_type=str(row[1]),
            nullable=bool(row[2]),
            has_default=bool(row[3]),
        )
        for row in rows
    ]


def ensure_supported_schema(conn, schema: str) -> None:
    maps_table = qualified_table_name(schema, "maps")
    rooms_table = qualified_table_name(schema, "rooms")
    scenes_table = qualified_table_name(schema, "scenes")
    objects_table = qualified_table_name(schema, "objects")
    object_observations_table = qualified_table_name(schema, "object_observations")
    interactions_table = qualified_table_name(schema, "interactions")

    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {maps_table} (
            id BIGSERIAL PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {rooms_table} (
            id BIGSERIAL PRIMARY KEY,
            map_id BIGINT REFERENCES {maps_table}(id) ON DELETE CASCADE,
            name TEXT NOT NULL,
            x1 DOUBLE PRECISION NOT NULL,
            y1 DOUBLE PRECISION NOT NULL,
            x2 DOUBLE PRECISION NOT NULL,
            y2 DOUBLE PRECISION NOT NULL,
            x3 DOUBLE PRECISION,
            y3 DOUBLE PRECISION,
            x4 DOUBLE PRECISION,
            y4 DOUBLE PRECISION
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {scenes_table} (
            id BIGSERIAL PRIMARY KEY,
            x DOUBLE PRECISION,
            y DOUBLE PRECISION,
            caption TEXT NOT NULL,
            scene_image BYTEA,
            source_frame TEXT,
            timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {objects_table} (
            id BIGSERIAL PRIMARY KEY,
            class_id BIGINT NOT NULL,
            name TEXT,
            canonical_embedding VECTOR(512),
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {object_observations_table} (
            id BIGSERIAL PRIMARY KEY,
            object_id BIGINT NOT NULL REFERENCES {objects_table}(id) ON DELETE CASCADE,
            scene_id BIGINT REFERENCES {scenes_table}(id) ON DELETE CASCADE,
            yolo_track_id TEXT,
            person_id BIGINT REFERENCES {objects_table}(id) ON DELETE SET NULL,
            class_id BIGINT,
            cropped_image BYTEA NOT NULL,
            bbox_x_min BIGINT,
            bbox_y_min BIGINT,
            bbox_x_max BIGINT,
            bbox_y_max BIGINT,
            x DOUBLE PRECISION,
            y DOUBLE PRECISION,
            z DOUBLE PRECISION,
            robot_x DOUBLE PRECISION,
            robot_y DOUBLE PRECISION,
            robot_z DOUBLE PRECISION,
            embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {interactions_table} (
            id BIGSERIAL PRIMARY KEY,
            action TEXT,
            caption TEXT NOT NULL,
            model_source TEXT,
            confidence DOUBLE PRECISION,
            subject_bbox JSONB,
            object_bbox JSONB,
            subject_id BIGINT REFERENCES {object_observations_table}(id) ON DELETE SET NULL,
            object_id BIGINT REFERENCES {object_observations_table}(id) ON DELETE SET NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """,
        f"ALTER TABLE {rooms_table} ADD COLUMN IF NOT EXISTS x3 DOUBLE PRECISION",
        f"ALTER TABLE {rooms_table} ADD COLUMN IF NOT EXISTS y3 DOUBLE PRECISION",
        f"ALTER TABLE {rooms_table} ADD COLUMN IF NOT EXISTS x4 DOUBLE PRECISION",
        f"ALTER TABLE {rooms_table} ADD COLUMN IF NOT EXISTS y4 DOUBLE PRECISION",
        f"ALTER TABLE {rooms_table} ADD COLUMN IF NOT EXISTS map_id BIGINT",
        f"ALTER TABLE {rooms_table} DROP CONSTRAINT IF EXISTS rooms_map_id_fkey",
        f"ALTER TABLE {rooms_table} ADD CONSTRAINT rooms_map_id_fkey FOREIGN KEY (map_id) REFERENCES {maps_table}(id) ON DELETE CASCADE",
        f"ALTER TABLE {scenes_table} ADD COLUMN IF NOT EXISTS map_id BIGINT",
        f"ALTER TABLE {scenes_table} DROP CONSTRAINT IF EXISTS scenes_map_id_fkey",
        f"ALTER TABLE {scenes_table} ADD CONSTRAINT scenes_map_id_fkey FOREIGN KEY (map_id) REFERENCES {maps_table}(id) ON DELETE SET NULL",
        f"ALTER TABLE {scenes_table} ADD COLUMN IF NOT EXISTS scene_image BYTEA",
        f"ALTER TABLE {scenes_table} ADD COLUMN IF NOT EXISTS source_frame TEXT",
        f"ALTER TABLE {objects_table} ADD COLUMN IF NOT EXISTS name TEXT",
        f"ALTER TABLE {objects_table} ADD COLUMN IF NOT EXISTS canonical_embedding VECTOR(512)",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS map_id BIGINT",
        f"ALTER TABLE {object_observations_table} DROP CONSTRAINT IF EXISTS object_observations_map_id_fkey",
        f"ALTER TABLE {object_observations_table} ADD CONSTRAINT object_observations_map_id_fkey FOREIGN KEY (map_id) REFERENCES {maps_table}(id) ON DELETE SET NULL",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS yolo_track_id TEXT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS person_id TEXT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS class_id BIGINT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS bbox_x_min BIGINT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS bbox_y_min BIGINT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS bbox_x_max BIGINT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS bbox_y_max BIGINT",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS robot_x DOUBLE PRECISION",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS robot_y DOUBLE PRECISION",
        f"ALTER TABLE {object_observations_table} ADD COLUMN IF NOT EXISTS robot_z DOUBLE PRECISION",
        f"ALTER TABLE {object_observations_table} DROP CONSTRAINT IF EXISTS fk_object_observations_person_id",
        f"ALTER TABLE {object_observations_table} DROP CONSTRAINT IF EXISTS object_observations_person_id_fkey",
        f"ALTER TABLE {object_observations_table} ADD CONSTRAINT fk_object_observations_person_id FOREIGN KEY (person_id) REFERENCES {objects_table}(id) ON DELETE SET NULL NOT VALID",
        f"ALTER TABLE {qualified_table_name(schema, 'face_observations')} ADD COLUMN IF NOT EXISTS map_id BIGINT",
        f"ALTER TABLE {qualified_table_name(schema, 'face_observations')} DROP CONSTRAINT IF EXISTS face_observations_map_id_fkey",
        f"ALTER TABLE {qualified_table_name(schema, 'face_observations')} ADD CONSTRAINT face_observations_map_id_fkey FOREIGN KEY (map_id) REFERENCES {maps_table}(id) ON DELETE SET NULL",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS model_source TEXT",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS confidence DOUBLE PRECISION",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS subject_bbox JSONB",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS object_bbox JSONB",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS subject_id BIGINT",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS object_id BIGINT",
        f"ALTER TABLE {interactions_table} ADD COLUMN IF NOT EXISTS map_id BIGINT",
        f"ALTER TABLE {interactions_table} DROP CONSTRAINT IF EXISTS interactions_map_id_fkey",
        f"ALTER TABLE {interactions_table} ADD CONSTRAINT interactions_map_id_fkey FOREIGN KEY (map_id) REFERENCES {maps_table}(id) ON DELETE SET NULL",
    ]

    with conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)


def parse_bytea(value: Any, table: str, column: str) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, list):
        return bytes(value)
    if isinstance(value, str) and value.startswith("\\x"):
        return bytes.fromhex(value[2:])
    raise RuntimeError(
        f"Unsupported bytea value for {table}.{column}. Expected PostgreSQL hex string like \\xABCD."
    )


def _parse_vector_dim(db_type: str) -> int | None:
    import re

    m = re.search(r"vector\((\d+)\)", db_type, re.IGNORECASE)
    return int(m.group(1)) if m else None


def format_vector(values: Any, table: str, column: str) -> str:
    if not isinstance(values, list):
        raise RuntimeError(f"Expected a JSON array for vector column {table}.{column}.")

    components: List[str] = []
    for item in values:
        if not isinstance(item, (int, float)):
            raise RuntimeError(f"Vector column {table}.{column} contains a non-numeric component: {item!r}")
        components.append(format(float(item), ".17g"))
    return "[" + ",".join(components) + "]"


def _normalize_vector(value: list, expected_dim: int, table: str, column: str) -> list:
    actual_dim = len(value)
    if actual_dim == expected_dim:
        return value
    if actual_dim > expected_dim:
        print(
            f"Note: Truncating {table}.{column} vector from {actual_dim} to {expected_dim} dimensions.",
            file=sys.stderr,
        )
        return value[:expected_dim]
    print(
        f"Note: Padding {table}.{column} vector from {actual_dim} to {expected_dim} dimensions with zeros.",
        file=sys.stderr,
    )
    return value + [0.0] * (expected_dim - actual_dim)


def convert_value(value: Any, column: ColumnInfo, table: str) -> Any:
    if value is None:
        return None

    db_type = column.db_type.lower()
    if db_type == "bytea":
        return parse_bytea(value, table, column.name)
    if db_type == "jsonb":
        return json.dumps(value)
    if db_type.startswith("vector"):
        expected_dim = _parse_vector_dim(db_type)
        if expected_dim is not None and isinstance(value, list):
            value = _normalize_vector(value, expected_dim, table, column.name)
        return format_vector(value, table, column.name)
    return value


def sql_placeholder(column: ColumnInfo) -> str:
    db_type = column.db_type.lower()
    if db_type == "jsonb":
        return "%s::jsonb"
    if db_type in {"timestamp with time zone", "timestamptz"}:
        return "%s::timestamptz"
    if db_type.startswith("vector"):
        return f"%s::{column.db_type}"
    return "%s"


def validate_row(row: Mapping[str, Any], columns: Sequence[ColumnInfo], table: str) -> None:
    column_names = {column.name for column in columns}
    unknown_fields = sorted(set(row.keys()) - column_names)
    if unknown_fields:
        joined = ", ".join(unknown_fields)
        raise RuntimeError(f"JSON row for table '{table}' contains unknown columns: {joined}")

    missing_required = [
        column.name
        for column in columns
        if column.name not in row and not column.nullable and not column.has_default
    ]
    if missing_required:
        joined = ", ".join(missing_required)
        raise RuntimeError(f"JSON row for table '{table}' is missing required columns: {joined}")


def build_insert_sql(schema: str, table: str, row_columns: Sequence[ColumnInfo], use_upsert: bool) -> str:
    column_sql = ", ".join(quote_ident(column.name) for column in row_columns)
    value_sql = ", ".join(sql_placeholder(column) for column in row_columns)
    sql = f"INSERT INTO {qualified_table_name(schema, table)} ({column_sql}) VALUES ({value_sql})"

    if not use_upsert:
        return sql

    update_columns = [column for column in row_columns if column.name != "id"]
    if not update_columns:
        return sql + f" ON CONFLICT ({quote_ident('id')}) DO NOTHING"

    update_sql = ", ".join(
        f"{quote_ident(column.name)} = EXCLUDED.{quote_ident(column.name)}" for column in update_columns
    )
    return sql + f" ON CONFLICT ({quote_ident('id')}) DO UPDATE SET {update_sql}"


def truncate_tables(conn, schema: str, tables: Iterable[str]) -> None:
    table_names = list(tables)
    if not table_names:
        return
    joined = ", ".join(qualified_table_name(schema, table) for table in reversed(table_names))
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE TABLE {joined} RESTART IDENTITY CASCADE")


def import_table(conn, schema: str, table: str, rows: Sequence[Mapping[str, Any]], columns: Sequence[ColumnInfo]) -> int:
    column_by_name = {column.name: column for column in columns}
    imported_count = 0

    with conn.cursor() as cur:
        for row in rows:
            validate_row(row, columns, table)
            row_columns = [column for column in columns if column.name in row]
            sql = build_insert_sql(schema, table, row_columns, use_upsert="id" in row and "id" in column_by_name)
            values = [convert_value(row[column.name], column, table) for column in row_columns]
            cur.execute(sql, values)
            imported_count += 1

    return imported_count


def _coerce_int_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            return int(stripped)
    return None


def _remap_id_value(value: Any, id_map: Mapping[int, int]) -> Any:
    coerced = _coerce_int_id(value)
    if coerced is None:
        return value
    remapped = id_map.get(coerced)
    if remapped is None:
        return value
    if isinstance(value, str):
        return str(remapped)
    return remapped


def fetch_max_id(conn, schema: str, table: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COALESCE(MAX(id), 0) FROM {qualified_table_name(schema, table)}")
        row = cur.fetchone()
    return int(row[0] or 0)


def fetch_existing_ids(conn, schema: str, table: str, candidate_ids: Sequence[int]) -> set[int]:
    if not candidate_ids:
        return set()
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id FROM {qualified_table_name(schema, table)} WHERE id = ANY(%s)",
            (list(candidate_ids),),
        )
        rows = cur.fetchall()
    return {int(row[0]) for row in rows}


def fetch_existing_maps_by_name(conn, schema: str, candidate_names: Sequence[str]) -> Dict[str, int]:
    names = [name for name in candidate_names if name]
    if not names:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT name, id FROM {qualified_table_name(schema, 'maps')} WHERE name = ANY(%s)",
            (names,),
        )
        rows = cur.fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def remap_conflicting_ids(
    conn,
    schema: str,
    dataset: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[int, int]]:
    id_remaps: Dict[str, Dict[int, int]] = {}

    for table_name, rows in dataset.items():
        source_ids: List[int] = []
        for row in rows:
            row_id = _coerce_int_id(row.get("id"))
            if row_id is not None:
                source_ids.append(row_id)
        if not source_ids:
            continue

        conflicting_ids = fetch_existing_ids(conn, schema, table_name, source_ids)
        if not conflicting_ids and table_name != "maps":
            continue

        next_id = fetch_max_id(conn, schema, table_name)
        table_remap: Dict[int, int] = {}
        existing_map_ids_by_name: Dict[str, int] = {}
        if table_name == "maps":
            existing_map_ids_by_name = fetch_existing_maps_by_name(
                conn,
                schema,
                [str(row.get("name") or "").strip() for row in rows],
            )
        for source_id in source_ids:
            if source_id in table_remap:
                continue
            if table_name == "maps":
                row = next((candidate for candidate in rows if _coerce_int_id(candidate.get("id")) == source_id), None)
                map_name = str(row.get("name") or "").strip() if row else ""
                existing_map_id = existing_map_ids_by_name.get(map_name)
                if existing_map_id is not None:
                    if source_id != existing_map_id:
                        table_remap[source_id] = existing_map_id
                    continue
            if source_id not in conflicting_ids:
                continue
            next_id += 1
            table_remap[source_id] = next_id
        if table_remap:
            id_remaps[table_name] = table_remap

    return id_remaps


def rewrite_dataset_ids(
    dataset: Dict[str, List[Dict[str, Any]]],
    id_remaps: Mapping[str, Mapping[int, int]],
) -> None:
    field_remaps: Dict[str, Dict[str, Mapping[int, int]]] = {
        "rooms": {"id": id_remaps.get("rooms", {}), "map_id": id_remaps.get("maps", {})},
        "scenes": {"id": id_remaps.get("scenes", {}), "map_id": id_remaps.get("maps", {})},
        "objects": {"id": id_remaps.get("objects", {})},
        "object_observations": {
            "id": id_remaps.get("object_observations", {}),
            "object_id": id_remaps.get("objects", {}),
            "scene_id": id_remaps.get("scenes", {}),
            "person_id": id_remaps.get("objects", {}),
            "map_id": id_remaps.get("maps", {}),
        },
        "face_observations": {
            "id": id_remaps.get("face_observations", {}),
            "person_id": id_remaps.get("objects", {}),
            "scene_id": id_remaps.get("scenes", {}),
            "object_id": id_remaps.get("objects", {}),
            "map_id": id_remaps.get("maps", {}),
        },
        "interactions": {
            "id": id_remaps.get("interactions", {}),
            "subject_id": id_remaps.get("object_observations", {}),
            "object_id": id_remaps.get("object_observations", {}),
            "map_id": id_remaps.get("maps", {}),
        },
    }

    for table_name, rows in dataset.items():
        table_field_remaps = field_remaps.get(table_name, {})
        if not table_field_remaps:
            continue
        for row in rows:
            for field_name, id_map in table_field_remaps.items():
                if field_name in row and id_map:
                    row[field_name] = _remap_id_value(row[field_name], id_map)


def stamp_dataset_map_ids(dataset: Dict[str, List[Dict[str, Any]]]) -> None:
    map_rows = dataset.get("maps") or []
    if len(map_rows) != 1:
        return

    imported_map_id = _coerce_int_id(map_rows[0].get("id"))
    if imported_map_id is None:
        return

    for room in dataset.get("rooms") or []:
        room["map_id"] = imported_map_id

    for scene in dataset.get("scenes") or []:
        scene["map_id"] = imported_map_id

    for observation in dataset.get("object_observations") or []:
        observation["map_id"] = imported_map_id

    for face in dataset.get("face_observations") or []:
        face["map_id"] = imported_map_id

    for interaction in dataset.get("interactions") or []:
        interaction["map_id"] = imported_map_id


def describe_id_remaps(id_remaps: Mapping[str, Mapping[int, int]]) -> List[str]:
    messages: List[str] = []
    for table_name in KNOWN_TABLE_ORDER:
        table_remap = id_remaps.get(table_name)
        if not table_remap:
            continue
        sample_pairs = list(table_remap.items())[:5]
        sample_text = ", ".join(f"{old}->{new}" for old, new in sample_pairs)
        if len(table_remap) > 5:
            sample_text += ", ..."
        messages.append(
            f"Remapped {len(table_remap)} conflicting {table_name} id(s) to avoid overwriting existing rows: {sample_text}"
        )
    return messages


def reset_sequence(conn, schema: str, table: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_get_serial_sequence(%s, %s)", (f"{schema}.{table}", "id"))
        row = cur.fetchone()
        sequence_name = row[0] if row else None
        if not sequence_name:
            return

        table_sql = qualified_table_name(schema, table)
        cur.execute(
            f"""
            SELECT setval(
                %s,
                COALESCE((SELECT MAX(id) FROM {table_sql}), 1),
                COALESCE((SELECT MAX(id) IS NOT NULL FROM {table_sql}), FALSE)
            )
            """,
            (sequence_name,),
        )


def infer_map_ids(conn, schema: str) -> None:
    """Populate map_id for scenes, observations, and interactions via room spatial containment."""
    min_room_x = "LEAST(r.x1, r.x2, COALESCE(r.x3, r.x1), COALESCE(r.x4, r.x1))"
    max_room_x = "GREATEST(r.x1, r.x2, COALESCE(r.x3, r.x1), COALESCE(r.x4, r.x1))"
    min_room_y = "LEAST(r.y1, r.y2, COALESCE(r.y3, r.y1), COALESCE(r.y4, r.y1))"
    max_room_y = "GREATEST(r.y1, r.y2, COALESCE(r.y3, r.y1), COALESCE(r.y4, r.y1))"
    tables = ["scenes", "object_observations"]
    with conn.cursor() as cur:
        for table in tables:
            cur.execute(
                f"""
                UPDATE {qualified_table_name(schema, table)} t
                SET map_id = r.map_id
                FROM {qualified_table_name(schema, "rooms")} r
                WHERE t.map_id IS NULL
                  AND r.map_id IS NOT NULL
                  AND t.x BETWEEN {min_room_x} AND {max_room_x}
                  AND t.y BETWEEN {min_room_y} AND {max_room_y}
                """
            )
        cur.execute(
            f"""
            UPDATE {qualified_table_name(schema, 'face_observations')} fo
            SET map_id = s.map_id
            FROM {qualified_table_name(schema, 'scenes')} s
            WHERE fo.map_id IS NULL
              AND fo.scene_id = s.id
              AND s.map_id IS NOT NULL
            """
        )
        cur.execute(
            f"""
            UPDATE {qualified_table_name(schema, "interactions")} i
            SET map_id = so.map_id
            FROM {qualified_table_name(schema, "object_observations")} so
            WHERE i.map_id IS NULL
              AND so.id = i.subject_id
              AND so.map_id IS NOT NULL
            """
        )
        cur.execute(
            f"""
            UPDATE {qualified_table_name(schema, "interactions")} i
            SET map_id = oo.map_id
            FROM {qualified_table_name(schema, "object_observations")} oo
            WHERE i.map_id IS NULL
              AND oo.id = i.object_id
              AND oo.map_id IS NOT NULL
            """
        )


def print_summary(imported_counts: Mapping[str, int], dry_run: bool) -> None:
    mode = "Dry run completed; transaction rolled back." if dry_run else "Import completed successfully."
    print(mode)
    for table_name, count in imported_counts.items():
        print(f"  {table_name}: {count} row(s)")


def print_notes(notes: Sequence[str], *, stream=None) -> None:
    if stream is None:
        stream = sys.stdout
    for note in notes:
        print(f"Note: {note}", file=stream)


def fetch_existing_maps(conn, schema: str) -> List[tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT id, name FROM {qualified_table_name(schema, 'maps')} ORDER BY name"
        )
        rows = cur.fetchall()
    return [(int(row[0]), str(row[1])) for row in rows]


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    json_path = Path(args.json_path)
    dataset = load_dataset(json_path)
    notes = collect_import_notes(dataset, truncate_first=args.truncate_first)
    db_driver = get_db_driver()
    conn = db_driver.connect(args.database_url)

    try:
        if getattr(conn, "autocommit", None):
            conn.autocommit = False

        ensure_supported_schema(conn, args.schema)

        if notes:
            print_notes(notes, stream=sys.stderr)

        existing_maps: List[tuple[int, str]] = []
        if args.truncate_first:
            existing_maps = fetch_existing_maps(conn, args.schema)
            if existing_maps:
                existing_labels = ", ".join(f"{name} (id={map_id})" for map_id, name in existing_maps[:10])
                if len(existing_maps) > 10:
                    existing_labels += ", ..."
                print(
                    "Warning: --truncate-first will remove the existing maps/buildings before import: "
                    f"{existing_labels}",
                    file=sys.stderr,
                )
        else:
            id_remaps = remap_conflicting_ids(conn, args.schema, dataset)
            if id_remaps:
                rewrite_dataset_ids(dataset, id_remaps)
                stamp_dataset_map_ids(dataset)
                print_notes(describe_id_remaps(id_remaps), stream=sys.stderr)
            else:
                stamp_dataset_map_ids(dataset)

        table_columns = {
            table_name: fetch_table_columns(conn, args.schema, table_name)
            for table_name in dataset.keys()
        }

        if args.truncate_first:
            truncate_tables(conn, args.schema, dataset.keys())

        imported_counts: Dict[str, int] = {}
        for table_name, rows in dataset.items():
            imported_counts[table_name] = import_table(
                conn,
                args.schema,
                table_name,
                rows,
                table_columns[table_name],
            )

        for table_name in dataset.keys():
            reset_sequence(conn, args.schema, table_name)

        if "rooms" in dataset:
            infer_map_ids(conn, args.schema)

        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()

        print_summary(imported_counts, dry_run=args.dry_run)
        return 0
    except Exception as exc:  # pragma: no cover - CLI error path
        conn.rollback()
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
