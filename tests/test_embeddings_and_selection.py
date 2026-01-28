import numpy as np
import pyarrow as pa
import pytest

from datamine import embedding_store, selection
from datamine.embeddings import HashingEmbedder, l2_normalize
from datamine.vector_index import VectorIndex


def _rand(n, d=16, seed=0):
    return l2_normalize(np.random.default_rng(seed).standard_normal((n, d)))


def test_hashing_embedder_is_deterministic_and_normalized(samples):
    emb = HashingEmbedder(dim=64)
    a = emb.embed_samples(samples.slice(0, 20))
    b = HashingEmbedder(dim=64).embed_samples(samples.slice(0, 20))
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0, rtol=1e-5)


def test_text_query_prefers_matching_scenes(samples):
    emb = HashingEmbedder(dim=256)
    vecs = emb.embed_samples(samples)
    q = emb.embed_text(["pedestrians at night"])[0]
    rows = samples.select(["time_of_day", "labels"]).to_pylist()
    scores = vecs @ q
    top = np.argsort(-scores)[:10]
    matches = [any(l["category"] == "pedestrian" for l in rows[i]["labels"]) for i in top]
    assert sum(matches) >= 7


def test_vector_index_returns_exact_neighbours():
    vecs = _rand(50)
    idx = VectorIndex(16)
    idx.add([f"s{i}" for i in range(50)], vecs)
    scores, ids = idx.search(vecs[[3, 7]], k=5)
    assert ids[0, 0] == 3 and ids[1, 0] == 7
    assert np.all(np.diff(scores, axis=1) <= 1e-6)
    brute = np.argsort(-(vecs @ vecs[3]))[:5]
    assert set(ids[0]) == set(brute)


def test_deduplicate_drops_near_copies():
    base = _rand(10)
    noisy = l2_normalize(base + 1e-3 * np.random.default_rng(1).standard_normal(base.shape))
    emb = np.vstack([base, noisy])
    keep = selection.deduplicate(emb, threshold=0.99)
    assert list(keep) == list(range(10))


def test_k_center_spreads_across_clusters():
    rng = np.random.default_rng(0)
    centers = _rand(4, seed=5)
    pts = l2_normalize(np.repeat(centers, 25, axis=0) + 0.01 * rng.standard_normal((100, 16)))
    chosen, gains = selection.k_center_greedy(pts, 4)
    assert len({int(c) // 25 for c in chosen}) == 4
    assert np.all(np.diff(gains[1:]) <= 1e-6)


def test_select_respects_budget_and_strategies():
    vecs = _rand(200)
    ids = [f"s{i}" for i in range(200)]
    chosen = selection.select(ids, vecs, budget=30, text_embeddings=vecs[:1], queries=["q"], query_top_k=5)
    assert len(chosen) == 30
    assert len({c.sample_id for c in chosen}) == 30
    strategies = {c.strategy for c in chosen}
    assert strategies == {"text_query", "k_center"}
    tbl = selection.to_table(chosen)
    assert tbl.num_rows == 30


def test_embedding_store_roundtrip(tmp_path):
    vecs = _rand(12, d=8)
    ids = [f"s{i}" for i in range(12)]
    table = embedding_store.to_arrow(ids, vecs, "m")
    assert pa.types.is_fixed_size_list(table.schema.field("embedding").type)
    uri = str(tmp_path / "emb" / "e.parquet")
    embedding_store.save_embeddings(table, uri)
    got_ids, got = embedding_store.load_embeddings(uri)
    assert got_ids == ids
    np.testing.assert_allclose(got, vecs, rtol=1e-6)


def test_embed_table_batches(samples):
    tbl = embedding_store.embed_table(samples.slice(0, 33), HashingEmbedder(dim=32), batch_size=10)
    assert tbl.num_rows == 33
    _, vecs = embedding_store.from_arrow(tbl)
    assert vecs.shape == (33, 32)
