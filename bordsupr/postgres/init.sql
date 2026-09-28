CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE maps (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE rooms (
    id BIGSERIAL PRIMARY KEY,
    map_id BIGINT REFERENCES maps(id) ON DELETE CASCADE,
    map_name TEXT,
    name TEXT NOT NULL,
    x1 DOUBLE PRECISION NOT NULL,
    y1 DOUBLE PRECISION NOT NULL,
    x2 DOUBLE PRECISION NOT NULL,
    y2 DOUBLE PRECISION NOT NULL,
    x3 DOUBLE PRECISION,
    y3 DOUBLE PRECISION,
    x4 DOUBLE PRECISION,
    y4 DOUBLE PRECISION
);

CREATE TABLE scenes (
    id BIGSERIAL PRIMARY KEY,
    x DOUBLE PRECISION,
    y DOUBLE PRECISION,
    caption TEXT NOT NULL,
    caption_embedding VECTOR(1024),
    scene_image BYTEA,
    original_scene_image BYTEA,
    source_frame TEXT,
    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE objects (
    id BIGSERIAL PRIMARY KEY,
    class_id BIGINT NOT NULL,
    name TEXT,
    canonical_embedding VECTOR(512),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE object_observations (
    id BIGSERIAL PRIMARY KEY,
    object_id BIGINT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    scene_id BIGINT REFERENCES scenes(id) ON DELETE CASCADE,
    yolo_track_id TEXT,
    class_id BIGINT,
    cropped_image BYTEA NOT NULL,
    mask_image BYTEA,
    original_cropped_image BYTEA,
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
    embedding VECTOR(512) NOT NULL,
    confidence DOUBLE PRECISION,
    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE face_observations (
    id BIGSERIAL PRIMARY KEY,
    person_id BIGINT NOT NULL REFERENCES objects(id) ON DELETE CASCADE,
    scene_id BIGINT REFERENCES scenes(id) ON DELETE SET NULL,
    object_id BIGINT REFERENCES objects(id) ON DELETE SET NULL,
    yolo_track_id TEXT,
    face_image BYTEA,
    person_x_min BIGINT,
    person_y_min BIGINT,
    person_x_max BIGINT,
    person_y_max BIGINT,
    face_x_min BIGINT,
    face_y_min BIGINT,
    face_x_max BIGINT,
    face_y_max BIGINT,
    score DOUBLE PRECISION,
    embedding VECTOR(512) NOT NULL,
    embedding_id TEXT UNIQUE,
    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE interactions (
    id BIGSERIAL PRIMARY KEY,
    action TEXT,
    caption TEXT NOT NULL,
    model_source TEXT,
    confidence DOUBLE PRECISION,
    subject_bbox JSONB,
    object_bbox JSONB,
    subject_id BIGINT REFERENCES object_observations(id) ON DELETE SET NULL,
    object_id BIGINT REFERENCES object_observations(id) ON DELETE SET NULL,
    map_id BIGINT REFERENCES maps(id) ON DELETE SET NULL,
    scene_id BIGINT REFERENCES scenes(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_objects_class_id
ON objects(class_id);

CREATE INDEX idx_objects_canonical_embedding_hnsw
ON objects
USING hnsw (canonical_embedding vector_cosine_ops);

CREATE INDEX idx_rooms_map_id
ON rooms(map_id);

CREATE INDEX idx_object_observations_object_id
ON object_observations(object_id);

CREATE INDEX idx_object_observations_scene_id
ON object_observations(scene_id);

CREATE INDEX idx_object_observations_yolo_track_id
ON object_observations(yolo_track_id);

CREATE INDEX idx_object_observations_class_id
ON object_observations(class_id);

CREATE INDEX idx_object_observations_bbox_x_min
ON object_observations(bbox_x_min);

CREATE INDEX idx_object_observations_bbox_y_min
ON object_observations(bbox_y_min);

CREATE INDEX idx_object_observations_embedding_hnsw
ON object_observations
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_face_observations_person_id
ON face_observations(person_id);

CREATE INDEX idx_face_observations_scene_id
ON face_observations(scene_id);

CREATE INDEX idx_face_observations_yolo_track_id
ON face_observations(yolo_track_id);

CREATE INDEX idx_face_observations_embedding_hnsw
ON face_observations
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX idx_interactions_model_source
ON interactions(model_source);

CREATE INDEX idx_interactions_confidence
ON interactions(confidence);

CREATE INDEX idx_interactions_subject_id
ON interactions(subject_id);

CREATE INDEX idx_interactions_object_id
ON interactions(object_id);

CREATE INDEX idx_scenes_timestamp
ON scenes(timestamp);

CREATE INDEX idx_scenes_source_frame
ON scenes(source_frame);

CREATE INDEX idx_scenes_caption_embedding_hnsw
ON scenes
USING hnsw (caption_embedding vector_cosine_ops);

CREATE INDEX idx_scenes_map_id
ON scenes(map_id);

CREATE INDEX idx_object_observations_map_id
ON object_observations(map_id);

CREATE INDEX idx_face_observations_map_id
ON face_observations(map_id);

CREATE INDEX idx_interactions_map_id
ON interactions(map_id);

CREATE INDEX idx_interactions_scene_id
ON interactions(scene_id);

CREATE INDEX idx_rooms_map_name
ON rooms(map_name);
