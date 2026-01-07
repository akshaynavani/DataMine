"""SparkSession factory with S3A wiring driven by environment variables."""

from __future__ import annotations

import os

from pyspark.sql import SparkSession

from datamine.config import SparkConfig

HADOOP_AWS_PACKAGE = "org.apache.hadoop:hadoop-aws:3.3.4"


def build_spark(cfg: SparkConfig | None = None, enable_s3: bool | None = None) -> SparkSession:
    """Create (or reuse) a SparkSession.

    S3A support is enabled automatically when AWS credentials or a custom endpoint are
    present in the environment. Credentials are resolved by the default AWS provider chain;
    they are never embedded in code or config files.
    """
    cfg = cfg or SparkConfig()
    if enable_s3 is None:
        enable_s3 = bool(
            os.environ.get("AWS_ACCESS_KEY_ID")
            or os.environ.get("AWS_PROFILE")
            or os.environ.get("DATAMINE_S3_ENDPOINT")
        )

    builder = (
        SparkSession.builder.master(cfg.master)
        .appName(cfg.app_name)
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.compression.codec", "zstd")
        # Arrow-accelerated conversion between Spark and pandas/pyarrow.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
    )

    if enable_s3:
        builder = (
            builder.config("spark.jars.packages", HADOOP_AWS_PACKAGE)
            .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
            .config(
                "spark.hadoop.fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.DefaultAWSCredentialsProviderChain",
            )
            .config("spark.hadoop.fs.s3a.fast.upload", "true")
        )
        endpoint = os.environ.get("DATAMINE_S3_ENDPOINT")
        if endpoint:
            builder = builder.config("spark.hadoop.fs.s3a.endpoint", endpoint).config(
                "spark.hadoop.fs.s3a.path.style.access", "true"
            )
        region = os.environ.get("AWS_REGION")
        if region:
            builder = builder.config("spark.hadoop.fs.s3a.endpoint.region", region)

    return builder.getOrCreate()
