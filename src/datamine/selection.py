"""Embedding-based sample selection strategies.

All functions operate on L2-normalised ``(n, d)`` arrays, so cosine similarity is a dot product
and cosine distance is ``1 - sim``.

* :func:`deduplicate`      greedy near-duplicate removal by similarity threshold.
* :func:`k_center_greedy`  coreset selection maximising coverage of embedding space.
* :func:`query_by_text`    zero-shot retrieval of scenarios described in natural language.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa

from datamine.schema import SELECTION_SCHEMA
from datamine.vector_index import VectorIndex


@dataclass(frozen=True)
class Selected:
    sample_id: str
    strategy: str
    score: float
    query: str | None = None


def deduplicate(embeddings: np.ndarray, threshold: float = 0.97, chunk: int = 1024) -> np.ndarray:
    """Return indices of samples to keep, in input order.

    A sample is dropped when its cosine similarity to any previously kept sample is
    ``>= threshold``. Input order matters (pass samples sorted by time, or by a priority score
    such as label count) because the first member of each duplicate cluster survives.
    """
    n, d = embeddings.shape
    kept = np.empty((n, d), dtype=np.float32)
    n_kept = 0
    keep_idx: list[int] = []
    for start in range(0, n, chunk):
        block = embeddings[start : start + chunk]
        # Similarity of this block against everything kept before it, computed in one matmul.
        prior_max = (block @ kept[:n_kept].T).max(axis=1) if n_kept else np.full(len(block), -1.0)
        block_kept_start = n_kept
        for i, vec in enumerate(block):
            if prior_max[i] >= threshold:
                continue
            if n_kept > block_kept_start and float((kept[block_kept_start:n_kept] @ vec).max()) >= threshold:
                continue
            kept[n_kept] = vec
            n_kept += 1
            keep_idx.append(start + i)
    return np.asarray(keep_idx, dtype=np.int64)


def k_center_greedy(
    embeddings: np.ndarray,
    budget: int,
    already_selected: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy k-center coreset (Sener & Savarese, 2018) under cosine distance.

    Iteratively picks the point farthest from the current selection. Returns the chosen
    indices and, for each, its distance to the selection at the time it was picked (a
    monotonically non-increasing "novelty" score).
    """
    n = len(embeddings)
    budget = min(budget, n)
    if budget <= 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

    min_dist = np.full(n, np.inf, dtype=np.float32)
    chosen: list[int] = []
    gains: list[float] = []

    def _update(idx: int) -> None:
        dist = 1.0 - embeddings @ embeddings[idx]
        np.minimum(min_dist, dist, out=min_dist)

    for idx in already_selected or []:
        _update(int(idx))

    if not already_selected:
        first = int(np.random.default_rng(seed).integers(n))
        chosen.append(first)
        gains.append(1.0)
        _update(first)

    while len(chosen) < budget:
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] <= 0:
            break  # everything remaining is an exact duplicate of the selection
        chosen.append(nxt)
        gains.append(float(min_dist[nxt]))
        _update(nxt)
    return np.asarray(chosen, dtype=np.int64), np.asarray(gains, dtype=np.float32)


def query_by_text(
    index: VectorIndex, text_embeddings: np.ndarray, queries: Sequence[str], top_k: int = 100,
    min_score: float | None = None,
) -> list[Selected]:
    results: list[Selected] = []
    for query, hits in zip(queries, index.search_ids(text_embeddings, top_k)):
        for sample_id, score in hits:
            if min_score is not None and score < min_score:
                continue
            results.append(Selected(sample_id, "text_query", score, query))
    return results


def to_table(selected: Sequence[Selected]) -> pa.Table:
    return pa.Table.from_pylist(
        [{"sample_id": s.sample_id, "strategy": s.strategy, "score": s.score, "query": s.query} for s in selected],
        schema=SELECTION_SCHEMA,
    )


def select(
    ids: Sequence[str],
    embeddings: np.ndarray,
    budget: int,
    dedup_threshold: float = 0.97,
    text_embeddings: np.ndarray | None = None,
    queries: Sequence[str] = (),
    query_top_k: int = 100,
) -> list[Selected]:
    """Full selection recipe: dedup -> targeted text queries -> diversity fill to ``budget``.

    Text-query hits are treated as must-haves (they encode scenarios someone explicitly asked
    for); the remaining budget is spent on k-center diversity seeded with those hits, so the
    coreset covers what the queries did not.
    """
    keep = deduplicate(embeddings, dedup_threshold)
    kept_ids = [ids[i] for i in keep]
    kept_emb = embeddings[keep]

    selected: dict[str, Selected] = {}
    if queries and text_embeddings is not None and len(kept_ids):
        index = VectorIndex(kept_emb.shape[1])
        index.add(kept_ids, kept_emb)
        for hit in query_by_text(index, text_embeddings, queries, query_top_k):
            prev = selected.get(hit.sample_id)
            if prev is None or hit.score > prev.score:
                selected[hit.sample_id] = hit
            if len(selected) >= budget:
                break

    remaining = budget - len(selected)
    if remaining > 0 and len(kept_ids):
        pos = {sid: i for i, sid in enumerate(kept_ids)}
        seeds = [pos[s] for s in selected]
        chosen, gains = k_center_greedy(kept_emb, remaining + len(seeds) if not seeds else remaining, seeds)
        for i, g in zip(chosen, gains):
            sid = kept_ids[int(i)]
            if sid not in selected:
                selected[sid] = Selected(sid, "k_center", float(g))
            if len(selected) >= budget:
                break
    return list(selected.values())
