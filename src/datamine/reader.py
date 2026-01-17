"""Low-latency Arrow access to curated samples on S3 or local disk.

``ArrowReader`` keeps one ``pyarrow.dataset.Dataset`` open (file listing + footers are cached),
uses hive partition pruning on ``log_id``/``camera``, pushes row filters down to Parquet row
groups, projects only requested columns, and enables ``pre_buffer`` so S3 reads are coalesced
into a few large range requests instead of many small ones.
"""

from __future__ import annotations

from functools import reduce
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from datamine.schema import PARTITION_COLUMNS, SAMPLE_SCHEMA
from datamine.storage import resolve

_LOOSE_SCHEMA = pa.schema([f.with_nullable(True) for f in SAMPLE_SCHEMA])


def conform(table: pa.Table) -> pa.Table:
    """Reorder/cast a table read from Parquet (possibly Spark-written) to ``SAMPLE_SCHEMA``."""
    cols = [c for c in _LOOSE_SCHEMA.names if c in table.column_names]
    table = table.select(cols)
    return table.cast(pa.schema([_LOOSE_SCHEMA.field(c) for c in cols]))


class ArrowReader:
    def __init__(self, uri: str) -> None:
        loc = resolve(uri)
        partitioning = ds.partitioning(
            pa.schema([(c, pa.string()) for c in PARTITION_COLUMNS]), flavor="hive"
        )
        fmt = ds.ParquetFileFormat(default_fragment_scan_options=ds.ParquetFragmentScanOptions(pre_buffer=True))
        self.uri = uri
        self.dataset = ds.dataset(loc.path, filesystem=loc.filesystem, format=fmt, partitioning=partitioning)

    # -- filters ---------------------------------------------------------------------------
    @staticmethod
    def build_filter(
        log_ids: Sequence[str] | None = None,
        cameras: Sequence[str] | None = None,
        weather: Sequence[str] | None = None,
        time_of_day: Sequence[str] | None = None,
        min_labels: int | None = None,
        time_range_ns: tuple[int, int] | None = None,
    ) -> ds.Expression | None:
        exprs: list[ds.Expression] = []
        if log_ids:
            exprs.append(ds.field("log_id").isin(list(log_ids)))
        if cameras:
            exprs.append(ds.field("camera").isin(list(cameras)))
        if weather:
            exprs.append(ds.field("weather").isin(list(weather)))
        if time_of_day:
            exprs.append(ds.field("time_of_day").isin(list(time_of_day)))
        if min_labels is not None:
            exprs.append(ds.field("num_labels") >= min_labels)
        if time_range_ns is not None:
            lo, hi = time_range_ns
            exprs.append((ds.field("timestamp_ns") >= lo) & (ds.field("timestamp_ns") < hi))
        return reduce(lambda a, b: a & b, exprs) if exprs else None

    # -- reads -----------------------------------------------------------------------------
    def scan(self, columns: list[str] | None = None, limit: int | None = None, **filters) -> pa.Table:
        expr = self.build_filter(**filters)
        if limit is not None:
            table = self.dataset.head(limit, columns=columns, filter=expr)
        else:
            table = self.dataset.to_table(columns=columns, filter=expr)
        return conform(table) if columns is None else table

    def iter_batches(self, columns: list[str] | None = None, batch_size: int = 8192, **filters) -> Iterator[pa.RecordBatch]:
        scanner = self.dataset.scanner(columns=columns, filter=self.build_filter(**filters), batch_size=batch_size)
        yield from scanner.to_batches()

    def get(self, sample_ids: Sequence[str], columns: list[str] | None = None) -> pa.Table:
        """Point lookup by sample id. Ids encode ``log_id:camera:frame`` so we can prune
        partitions before touching any file."""
        sample_ids = list(sample_ids)
        if columns is not None and "sample_id" not in columns:
            columns = ["sample_id", *columns]
        if not sample_ids:
            return conform(self.dataset.schema.empty_table())
        parts = [s.split(":") for s in sample_ids]
        log_ids = sorted({p[0] for p in parts if len(p) == 3})
        cameras = sorted({p[1] for p in parts if len(p) == 3})
        expr = ds.field("sample_id").isin(sample_ids)
        if log_ids and len(log_ids) == len({p[0] for p in parts}):
            expr = expr & ds.field("log_id").isin(log_ids) & ds.field("camera").isin(cameras)
        table = self.dataset.to_table(columns=columns, filter=expr)
        table = conform(table) if columns is None else table
        # Return rows in the order requested.
        order = {sid: i for i, sid in enumerate(sample_ids)}
        ranks = pa.array([order[s] for s in table.column("sample_id").to_pylist()], pa.int64())
        return table.take(pc.sort_indices(ranks))

    def count(self, **filters) -> int:
        return self.dataset.count_rows(filter=self.build_filter(**filters))

    def log_ids(self) -> list[str]:
        tbl = self.dataset.to_table(columns=["log_id"])
        return sorted(pc.unique(tbl.column("log_id")).to_pylist())
