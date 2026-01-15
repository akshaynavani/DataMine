"""Dataset quality metrics computed directly on Arrow sample tables.

Each metric is a plain function returning numbers/dicts so it can be used standalone, in tests,
or aggregated by :mod:`datamine.report`. Everything is vectorised with ``pyarrow.compute`` and
NumPy; nested label lists are flattened once and mapped back to their parent frames using
``list_parent_indices``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from datamine.schema import REQUIRED_COLUMNS


def _flat_labels(samples: pa.Table) -> tuple[pa.StructArray, np.ndarray]:
    labels = samples.column("labels").combine_chunks()
    parents = pc.list_parent_indices(labels).to_numpy(zero_copy_only=False)
    return labels.flatten(), parents


def label_completeness(samples: pa.Table) -> float:
    """Fraction of samples whose required fields are present and that carry >= 1 label."""
    n = samples.num_rows
    if n == 0:
        return 0.0
    ok = pc.greater(pc.fill_null(samples.column("num_labels"), 0), 0)
    for col in REQUIRED_COLUMNS:
        if col in samples.column_names:
            ok = pc.and_(ok, pc.is_valid(samples.column(col)))
    if "image_uri" in samples.column_names:
        ok = pc.and_(ok, pc.fill_null(pc.greater(pc.utf8_length(samples.column("image_uri")), 0), False))
    return float(pc.sum(pc.cast(ok, pa.int64())).as_py() or 0) / n


def box_validity(samples: pa.Table, classes: Sequence[str] | None = None) -> dict[str, float]:
    """Share of boxes that are geometrically valid, in-bounds and of a known class."""
    flat, parents = _flat_labels(samples)
    total = len(flat)
    if total == 0:
        return {"total_boxes": 0, "validity": 0.0, "degenerate": 0.0, "out_of_bounds": 0.0, "unknown_class": 0.0}

    def col(name: str) -> np.ndarray:
        return flat.field(name).to_numpy(zero_copy_only=False).astype(np.float64)

    x1, y1, x2, y2 = col("x1"), col("y1"), col("x2"), col("y2")
    width = samples.column("width").to_numpy(zero_copy_only=False).astype(np.float64)[parents]
    height = samples.column("height").to_numpy(zero_copy_only=False).astype(np.float64)[parents]

    with np.errstate(invalid="ignore"):
        degenerate = ~((x2 > x1) & (y2 > y1))
        oob = (x1 < 0) | (y1 < 0) | (x2 > width) | (y2 > height)
    degenerate |= np.isnan(x1) | np.isnan(y1) | np.isnan(x2) | np.isnan(y2)
    if classes:
        cats = np.asarray(flat.field("category").to_pylist(), dtype=object)
        unknown = ~np.isin(cats, list(classes))
    else:
        unknown = np.zeros(total, dtype=bool)
    valid = ~(degenerate | oob | unknown)
    return {
        "total_boxes": int(total),
        "validity": float(valid.mean()),
        "degenerate": float(degenerate.mean()),
        "out_of_bounds": float(oob.mean()),
        "unknown_class": float(unknown.mean()),
    }


def normalized_entropy(counts: Sequence[int], k: int | None = None) -> float:
    """Shannon entropy divided by ``log(k)``; 1.0 means perfectly balanced."""
    total = sum(counts)
    k = k or len(counts)
    if total == 0 or k <= 1:
        return 0.0
    h = -sum((c / total) * math.log(c / total) for c in counts if c > 0)
    return h / math.log(k)


def class_distribution(samples: pa.Table, classes: Sequence[str] | None = None) -> dict:
    flat, _ = _flat_labels(samples)
    counts = Counter(flat.field("category").to_pylist()) if len(flat) else Counter()
    keys = list(classes) if classes else sorted(k for k in counts if k)
    per_class = {c: int(counts.get(c, 0)) for c in keys}
    nonzero = [v for v in per_class.values() if v > 0]
    return {
        "counts": per_class,
        "balance": normalized_entropy(list(per_class.values()), len(keys)),
        "imbalance_ratio": (max(nonzero) / min(nonzero)) if nonzero else 0.0,
        "missing_classes": [c for c, v in per_class.items() if v == 0],
    }


def attribute_distribution(samples: pa.Table, column: str) -> dict:
    if column not in samples.column_names:
        return {"counts": {}, "balance": 0.0}
    counts = Counter(v for v in samples.column(column).to_pylist() if v is not None)
    return {"counts": dict(counts), "balance": normalized_entropy(list(counts.values()))}


def temporal_duplicate_rate(samples: pa.Table, stationary_speed: float = 0.5, grid: float = 4.0) -> float:
    """Fraction of frames that repeat the previous frame of the same (log, camera) stream:
    ego is stationary and every box is within ``grid`` pixels of its previous position."""
    if samples.num_rows < 2:
        return 0.0
    cols = ["log_id", "camera", "timestamp_ns", "ego_speed", "labels"]
    rows = samples.select(cols).sort_by([("log_id", "ascending"), ("camera", "ascending"),
                                         ("timestamp_ns", "ascending")]).to_pylist()

    def signature(labels) -> tuple:
        return tuple(sorted(
            (l["track_id"], round((l["x1"] or 0) / grid), round((l["y1"] or 0) / grid),
             round((l["x2"] or 0) / grid), round((l["y2"] or 0) / grid))
            for l in labels or []
        ))

    dupes = 0
    prev_key, prev_sig = None, None
    for r in rows:
        key = (r["log_id"], r["camera"])
        sig = signature(r["labels"])
        if key == prev_key and (r["ego_speed"] or 0.0) < stationary_speed and sig == prev_sig:
            dupes += 1
        prev_key, prev_sig = key, sig
    return dupes / len(rows)


def embedding_duplicate_rate(embeddings: np.ndarray, threshold: float = 0.97) -> float:
    from datamine.selection import deduplicate

    if len(embeddings) == 0:
        return 0.0
    return 1.0 - len(deduplicate(embeddings, threshold)) / len(embeddings)


def embedding_diversity(embeddings: np.ndarray, max_samples: int = 4000, seed: int = 0) -> dict[str, float]:
    """Spread of the dataset in embedding space.

    * ``mean_pairwise_distance``: average cosine distance between random pairs.
    * ``mean_nn_distance``: average distance to the nearest neighbour (low => redundancy).
    """
    n = len(embeddings)
    if n < 2:
        return {"mean_pairwise_distance": 0.0, "mean_nn_distance": 0.0}
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(n, max_samples), replace=False)
    sub = embeddings[idx]
    sims = sub @ sub.T
    m = len(sub)
    off_diag = (sims.sum() - np.trace(sims)) / (m * (m - 1))
    np.fill_diagonal(sims, -np.inf)
    nn = 1.0 - sims.max(axis=1)
    return {"mean_pairwise_distance": float(1.0 - off_diag), "mean_nn_distance": float(nn.mean())}


def coverage(pool: np.ndarray, selected: np.ndarray, radius: float = 0.1, chunk: int = 4096) -> float:
    """Fraction of the pool within cosine distance ``radius`` of at least one selected sample —
    how well a curated subset represents the full dataset."""
    if len(pool) == 0:
        return 0.0
    if len(selected) == 0:
        return 0.0
    covered = 0
    for start in range(0, len(pool), chunk):
        best = (pool[start : start + chunk] @ selected.T).max(axis=1)
        covered += int((1.0 - best <= radius).sum())
    return covered / len(pool)
