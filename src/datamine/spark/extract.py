"""Extraction job: raw JSONL driving logs -> normalised sample DataFrame.

Extraction is intentionally lossless with respect to label quality: it normalises structure
(flattening ego/tags, converting bbox arrays to typed columns) and drops only records that
cannot be identified at all. Bad boxes and unknown classes survive so that the quality metrics
can measure them; :mod:`datamine.spark.curate` is responsible for filtering.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from datamine.schema import RAW_FRAME_DDL, SAMPLE_SCHEMA
from datamine.storage import to_spark_uri


def read_raw_logs(spark: SparkSession, raw_uri: str) -> DataFrame:
    """Read JSON Lines logs with an explicit schema (no inference pass over the data)."""
    path = to_spark_uri(raw_uri.rstrip("/"))
    if not path.endswith(".jsonl") and not path.endswith(".json"):
        path = f"{path}/*.jsonl"
    return spark.read.schema(RAW_FRAME_DDL).option("mode", "PERMISSIVE").json(path)


def _label_struct(label):
    bbox = label["bbox"]
    return F.struct(
        label["track_id"].alias("track_id"),
        F.lower(F.trim(label["category"])).alias("category"),
        bbox[0].cast("float").alias("x1"),
        bbox[1].cast("float").alias("y1"),
        bbox[2].cast("float").alias("x2"),
        bbox[3].cast("float").alias("y2"),
        F.coalesce(label["occluded"], F.lit(False)).alias("occluded"),
    )


def extract_samples(raw: DataFrame) -> DataFrame:
    """Normalise raw frame records into the canonical sample layout (see ``SAMPLE_SCHEMA``)."""
    identified = raw.where(
        F.col("log_id").isNotNull() & F.col("timestamp_ns").isNotNull() & F.col("camera").isNotNull()
    )

    # Only boxes with exactly four coordinates can be represented; others are malformed input.
    well_formed = F.filter(F.coalesce(F.col("labels"), F.array()), lambda l: F.size(l["bbox"]) == 4)
    labels = F.transform(well_formed, _label_struct)

    df = identified.select(
        F.concat_ws(":", "log_id", "camera", F.col("frame_id").cast("string")).alias("sample_id"),
        "log_id",
        F.coalesce(F.col("frame_id"), F.lit(-1)).cast("bigint").alias("frame_id"),
        "timestamp_ns",
        "camera",
        "image_uri",
        F.col("width").cast("int").alias("width"),
        F.col("height").cast("int").alias("height"),
        F.col("ego.x").alias("ego_x"),
        F.col("ego.y").alias("ego_y"),
        F.col("ego.yaw").alias("ego_yaw"),
        F.col("ego.speed").alias("ego_speed"),
        F.col("tags.weather").alias("weather"),
        F.col("tags.time_of_day").alias("time_of_day"),
        F.col("tags.location").alias("location"),
        labels.alias("labels"),
    ).withColumn("num_labels", F.size("labels").cast("int"))

    # Logs are frequently re-uploaded; keep one record per sample id.
    df = df.dropDuplicates(["sample_id"])
    return df.select(*SAMPLE_SCHEMA.names)


def explode_objects(samples: DataFrame) -> DataFrame:
    """One row per labelled object — handy for per-class statistics and object-level mining."""
    return samples.select(
        "sample_id", "log_id", "camera", "timestamp_ns", "weather", "time_of_day",
        F.explode("labels").alias("obj"),
    ).select(
        "sample_id", "log_id", "camera", "timestamp_ns", "weather", "time_of_day",
        "obj.track_id", "obj.category", "obj.x1", "obj.y1", "obj.x2", "obj.y2", "obj.occluded",
        ((F.col("obj.x2") - F.col("obj.x1")) * (F.col("obj.y2") - F.col("obj.y1"))).alias("area"),
    )


def run_extract(spark: SparkSession, raw_uri: str) -> DataFrame:
    return extract_samples(read_raw_logs(spark, raw_uri))
