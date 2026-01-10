"""Cosine-similarity vector index over L2-normalised embeddings.

The default backend is exact brute-force search with NumPy (a matrix multiply plus
``argpartition``), which is fast enough for a few million 512-d vectors on one machine.
If ``faiss`` is installed, ``backend="faiss"`` uses ``IndexFlatIP`` for the same exact
inner-product search with SIMD/multithreading.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from datamine.embeddings.base import l2_normalize


class VectorIndex:
    def __init__(self, dim: int, backend: str = "numpy") -> None:
        self.dim = dim
        self.backend = backend
        self.ids: list[str] = []
        self._vectors = np.zeros((0, dim), dtype=np.float32)
        self._faiss = None
        if backend == "faiss":
            import faiss  # optional dependency

            self._faiss = faiss.IndexFlatIP(dim)
        elif backend != "numpy":
            raise ValueError(f"unknown backend {backend!r}")

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def vectors(self) -> np.ndarray:
        return self._vectors

    def add(self, ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected (n, {self.dim}) vectors, got {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        self.ids.extend(ids)
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._vectors) else vectors
        if self._faiss is not None:
            self._faiss.add(vectors)

    def search(self, queries: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(scores, indices)`` each shaped ``(n_queries, k)``, best match first."""
        queries = l2_normalize(np.atleast_2d(np.asarray(queries, dtype=np.float32)))
        k = min(k, len(self))
        if k == 0:
            empty = np.zeros((len(queries), 0))
            return empty.astype(np.float32), empty.astype(np.int64)
        if self._faiss is not None:
            scores, idx = self._faiss.search(queries, k)
            return scores, idx

        sims = queries @ self._vectors.T
        if k < sims.shape[1]:
            part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        else:
            part = np.tile(np.arange(sims.shape[1]), (len(queries), 1))
        part_scores = np.take_along_axis(sims, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        return np.take_along_axis(sims, idx, axis=1), idx

    def search_ids(self, queries: np.ndarray, k: int = 10) -> list[list[tuple[str, float]]]:
        scores, idx = self.search(queries, k)
        return [[(self.ids[j], float(s)) for j, s in zip(row_i, row_s)] for row_i, row_s in zip(idx, scores)]
