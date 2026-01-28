"""Spark integration tests. Skipped automatically when PySpark or a JVM is unavailable."""

import shutil

import pytest

pyspark = pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java runtime not found; skipping Spark tests", allow_module_level=True)

from datamine.config import CurationConfig, PipelineConfig  # noqa: E402
from datamine.pipeline import Pipeline  # noqa: E402
from datamine.reader import ArrowReader  # noqa: E402
from datamine.spark.curate import curate  # noqa: E402
from datamine.spark.extract import run_extract  # noqa: E402
from datamine.synth import SynthConfig, write_logs  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from datamine.config import SparkConfig
    from datamine.spark import build_spark

    session = build_spark(SparkConfig(master="local[2]", shuffle_partitions=4), enable_s3=False)
    yield session
    session.stop()


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("raw")
    write_logs(d, SynthConfig(num_logs=2, frames_per_log=30, corruption_rate=0.1, seed=11))
    return str(d)


def test_extract_normalises_records(spark, raw_dir):
    df = run_extract(spark, raw_dir)
    assert df.count() == 2 * 30 * 4
    row = df.first()
    assert row.sample_id.count(":") == 2
    assert all(isinstance(l.x1, float) for l in row.labels)


def test_curate_removes_invalid_boxes(spark, raw_dir):
    cfg = CurationConfig(frame_stride_ms=0)
    curated = curate(run_extract(spark, raw_dir), cfg).collect()
    assert curated
    for r in curated:
        assert r.image_uri
        assert r.num_labels == len(r.labels) > 0
        for l in r.labels:
            assert l.category in cfg.classes
            assert l.x2 > l.x1 and l.y2 > l.y1
            assert l.x2 <= r.width and l.y2 <= r.height


def test_temporal_stride_reduces_frames(spark, raw_dir):
    dense = curate(run_extract(spark, raw_dir), CurationConfig(frame_stride_ms=0)).count()
    sparse = curate(run_extract(spark, raw_dir), CurationConfig(frame_stride_ms=500)).count()
    assert sparse < dense


def test_full_pipeline(spark, raw_dir, tmp_path):
    cfg = PipelineConfig(raw_uri=raw_dir, output_uri=str(tmp_path / "out"))
    cfg.embedding.dim = 64
    cfg.selection.budget = 20
    cfg.selection.text_queries = ["pedestrian at night"]
    cfg.selection.query_top_k = 5
    result = Pipeline(cfg).run(spark=spark)

    assert result.num_selected == 20
    assert {"extracted", "curated", "selected"} <= result.reports.keys()
    curated = result.reports["curated"].metrics
    extracted = result.reports["extracted"].metrics
    assert curated["box_validity"]["validity"] == pytest.approx(1.0)
    assert extracted["box_validity"]["validity"] <= curated["box_validity"]["validity"]
    assert ArrowReader(str(tmp_path / "out" / "selected")).count() == 20
