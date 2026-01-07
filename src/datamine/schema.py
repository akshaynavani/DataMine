"""Canonical Arrow schemas shared by the Spark jobs, embedding store and reader.

Raw logs are JSON Lines with one record per camera frame. After extraction every frame
becomes a *sample*: a flat row of frame metadata plus a nested list of 2D box labels.
"""

from __future__ import annotations

import pyarrow as pa

LABEL_STRUCT = pa.struct(
    [
        pa.field("track_id", pa.string()),
        pa.field("category", pa.string()),
        pa.field("x1", pa.float32()),
        pa.field("y1", pa.float32()),
        pa.field("x2", pa.float32()),
        pa.field("y2", pa.float32()),
        pa.field("occluded", pa.bool_()),
    ]
)

SAMPLE_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("log_id", pa.string(), nullable=False),
        pa.field("frame_id", pa.int64(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("camera", pa.string(), nullable=False),
        pa.field("image_uri", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("ego_x", pa.float64()),
        pa.field("ego_y", pa.float64()),
        pa.field("ego_yaw", pa.float64()),
        pa.field("ego_speed", pa.float64()),
        pa.field("weather", pa.string()),
        pa.field("time_of_day", pa.string()),
        pa.field("location", pa.string()),
        pa.field("labels", pa.list_(LABEL_STRUCT)),
        pa.field("num_labels", pa.int32()),
    ]
)

# Columns that must be present for a sample to count as fully labelled.
REQUIRED_COLUMNS = ("sample_id", "log_id", "timestamp_ns", "camera", "image_uri", "labels")

PARTITION_COLUMNS = ("log_id", "camera")


def embedding_schema(dim: int) -> pa.Schema:
    """Schema for the embedding table: one fixed-size float32 vector per sample."""
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("embedding", pa.list_(pa.float32(), dim), nullable=False),
            pa.field("model", pa.string()),
        ]
    )


SELECTION_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("strategy", pa.string(), nullable=False),
        pa.field("score", pa.float32()),
        pa.field("query", pa.string()),
    ]
)


# Spark DDL mirror of the raw JSONL record, used to read logs with an explicit schema
# instead of relying on (slow, error-prone) JSON schema inference.
RAW_FRAME_DDL = """
    log_id STRING,
    frame_id BIGINT,
    timestamp_ns BIGINT,
    camera STRING,
    image_uri STRING,
    width INT,
    height INT,
    ego STRUCT<x: DOUBLE, y: DOUBLE, yaw: DOUBLE, speed: DOUBLE>,
    tags STRUCT<weather: STRING, time_of_day: STRING, location: STRING>,
    labels ARRAY<STRUCT<track_id: STRING, category: STRING, bbox: ARRAY<DOUBLE>, occluded: BOOLEAN>>
"""
