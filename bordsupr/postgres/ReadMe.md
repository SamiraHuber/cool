# Start

```
docker compose up -d
```

# Exec the database container

```
docker exec -it pgvector-db bash
```

# Connect to PostreSQL

```
psql -U postgres -d bordsupr
```

# List tables

```
\dt
```

# Check pgvector extension

```
\dx
```

# Inspect a table
```
\d objects
```

# Run a query

```
SELECT * FROM objects LIMIT 5;
```

SELECT COUNT(*) FROM objects;

TRUNCATE TABLE scenes,objects, object_observations RESTART IDENTITY CASCADE;

SELECT object_id, scene_id
FROM object_observations
WHERE object_id = '61c4c57d-595b-41c8-a79a-992bcb7a9e79';

SELECT id, object_id, scene_id, x, y, z, created_at FROM object_observations;