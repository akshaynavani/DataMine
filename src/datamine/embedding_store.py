"""Persist and load embeddings as Parquet, and compute them at scale with Spark.

Vectors are stored as ``FixedSizeList<float32>[dim]`` so they round-trip into a contiguous
NumPy array with zero per-row Python overhead.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np
import pyarrow as pa

from datamine.embeddings.base import Embedder
from datamine.schema import embedding_schema
from datamine.storage import read_table, write_table


def to_arrow(ids: list[str], vectors: np.ndarray, model: str) -> pa.Table:
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    n, dim = vectors.shape
    flat = pa.array(vectors.reshape(-1), type=pa.float32())
    emb = pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.Table.from_arrays(
        [pa.array(ids, pa.string()), emb, pa.array([model] * n, pa.string())],
        schema=embedding_schema(dim),
    )


def from_arrow(table: pa.Table) -> tuple[list[str], np.ndarray]:
    ids = table.column("sample_id").to_pylist()
    col = table.column("embedding").combine_chunks()
    if pa.types.is_fixed_size_list(col.type):
        dim = col.type.list_size
        vectors = col.flatten().to_numpy(zero_copy_only=False).reshape(-1, dim)
    else:  # variable-size list (e.g. written by Spark as array<float>)
        vectors = np.asarray(col.to_pylist(), dtype=np.float32)
    return ids, vectors.astype(np.float32, copy=False)


def embed_table(samples: pa.Table, embedder: Embedder, batch_size: int = 4096) -> pa.Table:
    ids: list[str] = []
    chunks: list[np.ndarray] = []
    for batch in samples.to_batches(max_chunksize=batch_size):
        tbl = pa.Table.from_batches([batch])
        ids.extend(tbl.column("sample_id").to_pylist())
        chunks.append(embedder.embed_samples(tbl))
    vectors = np.vstack(chunks) if chunks else np.zeros((0, embedder.dim), dtype=np.float32)
    return to_arrow(ids, vectors, embedder.name)


def save_embeddings(table: pa.Table, uri: str) -> None:
    write_table(table, uri)


def load_embeddings(uri: str) -> tuple[list[str], np.ndarray]:
    return from_arrow(read_table(uri))


def embed_dataframe(samples_df, embedder_factory: Callable[[], Embedder]):
    """Distributed embedding with ``mapInArrow``: each executor builds its own embedder once
    per partition and streams Arrow record batches through it."""
    out_schema = "sample_id string, embedding array<float>, model string"

    def _embed(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        embedder = embedder_factory()
        for batch in batches:
            tbl = pa.Table.from_batches([batch])
            vecs = embedder.embed_samples(tbl)
            yield pa.RecordBatch.from_arrays(
                [
                    tbl.column("sample_id").combine_chunks(),
                    pa.array(list(vecs), type=pa.list_(pa.float32())),
                    pa.array([embedder.name] * len(tbl), pa.string()),
                ],
                names=["sample_id", "embedding", "model"],
            )

    return samples_df.mapInArrow(_embed, out_schema)
