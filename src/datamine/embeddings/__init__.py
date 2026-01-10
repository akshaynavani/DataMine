"""Embedding backends and factory."""

from __future__ import annotations

from datamine.config import EmbeddingConfig
from datamine.embeddings.base import Embedder, l2_normalize
from datamine.embeddings.hashing import HashingEmbedder


def build_embedder(cfg: EmbeddingConfig) -> Embedder:
    backend = cfg.backend.lower()
    if backend == "hashing":
        return HashingEmbedder(dim=cfg.dim)
    if backend == "clip":
        from datamine.embeddings.clip import ClipEmbedder

        return ClipEmbedder(cfg.clip_model, cfg.clip_pretrained, batch_size=cfg.batch_size)
    raise ValueError(f"unknown embedding backend: {cfg.backend!r}")


__all__ = ["Embedder", "HashingEmbedder", "build_embedder", "l2_normalize"]
