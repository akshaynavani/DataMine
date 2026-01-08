"""Curation job: filter extracted samples down to clean, useful training data.

Steps (each is a pure DataFrame -> DataFrame function so they can be composed or tested alone):

1. ``filter_labels``     drop boxes that are degenerate, out of bounds, too small or of an
                         unknown class.
2. ``filter_frames``     drop frames with no image, failing tag filters, or no labels left.
3. ``temporal_subsample`` keep at most one frame per (log, camera, time bucket), which removes
                         the bulk of redundant frames recorded while the ego vehicle is idle.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from datamine.config import CurationConfig
from datamine.schema import PARTITION_COLUMNS, SAMPLE_SCHEMA
from datamine.storage import to_spark_uri


def _valid_box(label, min_area: float, classes: list[str]):
    w = label["x2"] - label["x1"]
    h = label["y2"] - label["y1"]
    in_bounds = (
        (label["x1"] >= 0)
        & (label["y1"] >= 0)
        & (label["x2"] <= F.col("width"))
        & (label["y2"] <= F.col("height"))
    )
    cond = (w > 0) & (h > 0) & in_bounds & ((w * h) >= F.lit(min_area))
    if classes:
        cond = cond & label["category"].isin(*classes)
    return cond


def filter_labels(samples: DataFrame, cfg: CurationConfig) -> DataFrame:
    labels = F.filter("labels", lambda l: _valid_box(l, cfg.min_box_area, cfg.classes))
    return samples.withColumn("labels", labels).withColumn("num_labels", F.size("labels").cast("int"))


def filter_frames(samples: DataFrame, cfg: CurationConfig, drop_empty: bool = True) -> DataFrame:
    df = samples.where(F.col("image_uri").isNotNull() & (F.length("image_uri") > 0))
    for key, value in cfg.require_tags.items():
        df = df.where(F.col(key) == value)
    for key, value in cfg.exclude_tags.items():
        df = df.where(F.col(key).isNull() | (F.col(key) != value))
    if drop_empty:
        df = df.where(F.col("num_labels") > 0)
    return df


def temporal_subsample(samples: DataFrame, stride_ms: int) -> DataFrame:
    if stride_ms <= 0:
        return samples
    stride_ns = stride_ms * 1_000_000
    bucket = F.floor(F.col("timestamp_ns") / F.lit(stride_ns))
    # Within each time bucket keep the frame with the most labels (ties -> earliest).
    w = Window.partitionBy("log_id", "camera", "_bucket").orderBy(F.desc("num_labels"), "timestamp_ns")
    return (
        samples.withColumn("_bucket", bucket)
        .withColumn("_rank", F.row_number().over(w))
        .where(F.col("_rank") == 1)
        .drop("_bucket", "_rank")
    )


def curate(samples: DataFrame, cfg: CurationConfig, drop_empty: bool = True) -> DataFrame:
    df = filter_labels(samples, cfg)
    df = filter_frames(df, cfg, drop_empty=drop_empty)
    df = temporal_subsample(df, cfg.frame_stride_ms)
    return df.select(*SAMPLE_SCHEMA.names)


def write_samples(samples: DataFrame, uri: str, mode: str = "overwrite") -> None:
    """Persist samples as hive-partitioned Parquet (``log_id=.../camera=...``)."""
    (
        samples.repartition(*PARTITION_COLUMNS)
        .sortWithinPartitions("timestamp_ns")
        .write.mode(mode)
        .partitionBy(*PARTITION_COLUMNS)
        .parquet(to_spark_uri(uri))
    )
