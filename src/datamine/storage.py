"""URI-based storage abstraction over local disk and AWS S3 using ``pyarrow.fs``.

Credentials are never passed explicitly: S3 access relies on the AWS default credential
chain (env vars, shared config, instance/IRSA roles). ``DATAMINE_S3_ENDPOINT`` can point at
MinIO/LocalStack for development.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.fs as pafs
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Location:
    filesystem: pafs.FileSystem
    path: str
    is_s3: bool

    def join(self, *parts: str) -> "Location":
        path = "/".join([self.path.rstrip("/"), *[p.strip("/") for p in parts]])
        return Location(self.filesystem, path, self.is_s3)

    @property
    def uri(self) -> str:
        return f"s3://{self.path}" if self.is_s3 else self.path


@lru_cache(maxsize=8)
def _s3_filesystem(region: str | None, endpoint: str | None) -> pafs.S3FileSystem:
    kwargs: dict[str, Any] = {}
    if region:
        kwargs["region"] = region
    if endpoint:
        kwargs["endpoint_override"] = endpoint
        kwargs["scheme"] = "https" if endpoint.startswith("https") else "http"
    return pafs.S3FileSystem(**kwargs)


def resolve(uri: str) -> Location:
    """Turn ``s3://bucket/key`` or a local path into a filesystem + path pair."""
    if uri.startswith("s3://") or uri.startswith("s3a://"):
        path = uri.split("://", 1)[1]
        fs = _s3_filesystem(os.environ.get("AWS_REGION"), os.environ.get("DATAMINE_S3_ENDPOINT") or None)
        return Location(fs, path, True)
    local = os.path.abspath(os.path.expanduser(uri))
    return Location(pafs.LocalFileSystem(), local, False)


def to_spark_uri(uri: str) -> str:
    """Spark's Hadoop S3 connector expects the ``s3a://`` scheme."""
    if uri.startswith("s3://"):
        return "s3a://" + uri[len("s3://"):]
    return uri


def ensure_dir(loc: Location) -> None:
    if not loc.is_s3:
        loc.filesystem.create_dir(loc.path, recursive=True)


def write_table(table: pa.Table, uri: str, partition_cols: Iterable[str] | None = None) -> None:
    """Write an Arrow table as Parquet, optionally hive-partitioned."""
    loc = resolve(uri)
    partition_cols = list(partition_cols or [])
    if partition_cols:
        ensure_dir(loc)
        ds.write_dataset(
            table,
            base_dir=loc.path,
            filesystem=loc.filesystem,
            format="parquet",
            partitioning=partition_cols,
            partitioning_flavor="hive",
            existing_data_behavior="delete_matching",
        )
        return
    parent = loc.path.rsplit("/", 1)[0]
    ensure_dir(Location(loc.filesystem, parent, loc.is_s3))
    pq.write_table(table, loc.path, filesystem=loc.filesystem, compression="zstd")


def read_table(uri: str, columns: list[str] | None = None, filter=None) -> pa.Table:
    loc = resolve(uri)
    dataset = ds.dataset(loc.path, filesystem=loc.filesystem, format="parquet", partitioning="hive")
    return dataset.to_table(columns=columns, filter=filter)


def write_json(obj: Any, uri: str) -> None:
    loc = resolve(uri)
    parent = loc.path.rsplit("/", 1)[0]
    ensure_dir(Location(loc.filesystem, parent, loc.is_s3))
    with loc.filesystem.open_output_stream(loc.path) as out:
        out.write(json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def read_json(uri: str) -> Any:
    loc = resolve(uri)
    with loc.filesystem.open_input_stream(loc.path) as fh:
        return json.loads(fh.read().decode("utf-8"))
