"""Embedder interface shared by the CLIP and hashing backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pyarrow as pa


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, eps)


class Embedder(ABC):
    """Maps samples (Arrow rows) and free-text queries into one shared vector space.

    Implementations must return L2-normalised float32 arrays so cosine similarity reduces to
    a dot product everywhere downstream.
    """

    name: str
    dim: int

    @abstractmethod
    def embed_samples(self, samples: pa.Table) -> np.ndarray:
        """Return an ``(n, dim)`` array, one row per sample in ``samples``."""

    @abstractmethod
    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` array of text embeddings."""
