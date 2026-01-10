"""Deterministic, dependency-free embedder.

Real CLIP needs images and a GPU-friendly model; this backend lets the full pipeline (and its
tests) run anywhere. It describes each sample as a weighted *bag of semantic tokens* derived
from its metadata and labels — weather, time of day, object classes, crowding, occlusion, and
coarse object positions — and maps each token to a fixed pseudo-random unit vector (seeded by
a stable hash). Text queries are tokenised into the same vocabulary, so a query such as
"pedestrian crossing at night" lands near frames that contain pedestrians at night, mimicking
CLIP's joint text/image space.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np
import pyarrow as pa

from datamine.embeddings.base import Embedder, l2_normalize

SYNONYMS = {
    "pedestrians": "pedestrian", "people": "pedestrian", "person": "pedestrian", "walker": "pedestrian",
    "cars": "car", "vehicle": "car", "vehicles": "car", "trucks": "truck", "lorry": "truck",
    "bike": "cyclist", "bicycle": "cyclist", "cyclists": "cyclist", "biker": "cyclist",
    "rainy": "rain", "raining": "rain", "wet": "rain", "foggy": "fog", "mist": "fog",
    "snowy": "snow", "snowing": "snow", "sunny": "clear", "overcast": "cloudy",
    "nighttime": "night", "dark": "night", "evening": "dusk", "sunset": "dusk", "daytime": "day",
    "busy": "crowded", "dense": "crowded", "traffic": "crowded", "hidden": "occluded",
    "stationary": "stopped", "idle": "stopped", "light": "traffic_light", "lights": "traffic_light",
    "sign": "traffic_sign", "signs": "traffic_sign",
}
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "of", "with", "and", "or", "to", "is", "are", "street",
    "road", "crossing", "scene", "image", "photo", "of", "some", "many", "heavy", "while",
}
GRID = 4  # coarse spatial grid for per-object position tokens


def _stable_seed(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512, position_weight: float = 0.35) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"
        self.position_weight = position_weight
        self._cache: dict[str, np.ndarray] = {}

    # -- token vectors -------------------------------------------------------------------
    def _token_vec(self, token: str) -> np.ndarray:
        vec = self._cache.get(token)
        if vec is None:
            rng = np.random.default_rng(_stable_seed(token))
            vec = rng.standard_normal(self.dim).astype(np.float32)
            vec /= np.linalg.norm(vec)
            self._cache[token] = vec
        return vec

    def _encode(self, bag: Counter) -> np.ndarray:
        out = np.zeros(self.dim, dtype=np.float32)
        for token, weight in bag.items():
            out += weight * self._token_vec(token)
        return out

    # -- samples -------------------------------------------------------------------------
    def sample_tokens(self, row: dict) -> Counter:
        bag: Counter = Counter()
        for key in ("weather", "time_of_day", "location", "camera"):
            if row.get(key):
                bag[str(row[key])] += 1.0
        labels = row.get("labels") or []
        counts = Counter(l["category"] for l in labels if l.get("category"))
        for cat, n in counts.items():
            bag[cat] += 1.0 + math.log(n)
        if len(labels) >= 8:
            bag["crowded"] += 1.0
        if any(l.get("occluded") for l in labels):
            bag["occluded"] += 0.5
        speed = row.get("ego_speed")
        if speed is not None and speed < 0.5:
            bag["stopped"] += 0.5
        w = row.get("width") or 1
        h = row.get("height") or 1
        for l in labels:
            if l.get("x1") is None or l.get("x2") is None:
                continue
            gx = min(GRID - 1, max(0, int(((l["x1"] + l["x2"]) / 2) / w * GRID)))
            gy = min(GRID - 1, max(0, int(((l["y1"] + l["y2"]) / 2) / h * GRID)))
            bag[f"{l['category']}@{gx}{gy}"] += self.position_weight
            # Fine-grained position makes frames recorded while moving distinguishable,
            # whereas frames captured while stopped collapse onto near-identical vectors.
            fx, fy = int(l["x1"] // 32), int(l["y1"] // 32)
            bag[f"{l.get('track_id')}#{fx}_{fy}"] += self.position_weight * 0.5
        return bag

    def embed_samples(self, samples: pa.Table) -> np.ndarray:
        rows = samples.to_pylist()
        if not rows:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._encode(self.sample_tokens(r)) for r in rows]))

    # -- text ----------------------------------------------------------------------------
    @staticmethod
    def text_tokens(text: str) -> Counter:
        words = re.findall(r"[a-z_]+", text.lower())
        bag: Counter = Counter()
        for i, word in enumerate(words):
            if word == "traffic" and i + 1 < len(words) and words[i + 1] in ("light", "lights", "sign", "signs"):
                continue  # handled by the following word's synonym
            word = SYNONYMS.get(word, word)
            if word in STOPWORDS:
                continue
            bag[word] += 1.0
        return bag

    def embed_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._encode(self.text_tokens(t)) for t in texts]))
