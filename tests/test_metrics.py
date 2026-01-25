import numpy as np
import pyarrow as pa
import pytest

from datamine import metrics
from datamine.config import QualityGates
from datamine.report import build_report
from datamine.schema import SAMPLE_SCHEMA


def _row(sid, labels, ts=0, speed=5.0, image="img.jpg"):
    return {
        "sample_id": sid, "log_id": "L", "frame_id": ts, "timestamp_ns": ts, "camera": "front",
        "image_uri": image, "width": 100, "height": 100, "ego_x": 0.0, "ego_y": 0.0, "ego_yaw": 0.0,
        "ego_speed": speed, "weather": "clear", "time_of_day": "day", "location": "x",
        "labels": labels, "num_labels": len(labels),
    }


def _box(cat, x1, y1, x2, y2, tid="t"):
    return {"track_id": tid, "category": cat, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "occluded": False}


def test_box_validity_counts_each_defect():
    t = pa.Table.from_pylist([
        _row("a", [_box("car", 0, 0, 10, 10), _box("car", 5, 5, 5, 5)]),
        _row("b", [_box("car", 0, 0, 200, 10), _box("blob", 0, 0, 10, 10)]),
    ], schema=SAMPLE_SCHEMA)
    v = metrics.box_validity(t, ["car"])
    assert v["total_boxes"] == 4
    assert v["validity"] == pytest.approx(0.25)
    assert v["degenerate"] == pytest.approx(0.25)
    assert v["out_of_bounds"] == pytest.approx(0.25)
    assert v["unknown_class"] == pytest.approx(0.25)


def test_label_completeness():
    t = pa.Table.from_pylist([
        _row("a", [_box("car", 0, 0, 10, 10)]),
        _row("b", []),
        _row("c", [_box("car", 0, 0, 10, 10)], image=None),
        _row("d", [_box("car", 0, 0, 10, 10)]),
    ], schema=SAMPLE_SCHEMA)
    assert metrics.label_completeness(t) == pytest.approx(0.5)


def test_normalized_entropy_bounds():
    assert metrics.normalized_entropy([10, 10, 10]) == pytest.approx(1.0)
    assert metrics.normalized_entropy([10, 0, 0], k=3) == pytest.approx(0.0)


def test_temporal_duplicates_detected_when_stopped():
    box = [_box("car", 10, 10, 20, 20)]
    t = pa.Table.from_pylist([
        _row("a", box, ts=0, speed=0.0), _row("b", box, ts=1, speed=0.0), _row("c", box, ts=2, speed=0.0),
        _row("d", [_box("car", 40, 40, 60, 60)], ts=3, speed=5.0),
    ], schema=SAMPLE_SCHEMA)
    assert metrics.temporal_duplicate_rate(t) == pytest.approx(0.5)


def test_coverage_and_diversity():
    pool = np.eye(4, dtype=np.float32)
    assert metrics.coverage(pool, pool[:2], radius=0.1) == pytest.approx(0.5)
    div = metrics.embedding_diversity(pool)
    assert div["mean_pairwise_distance"] == pytest.approx(1.0)


def test_report_on_synthetic_data_flags_defects(samples):
    rep = build_report(samples, "extracted", classes=["car", "truck", "pedestrian", "cyclist",
                                                       "traffic_light", "traffic_sign"],
                       gates=QualityGates(min_box_validity=0.999))
    assert rep.num_samples == samples.num_rows
    assert rep.metrics["box_validity"]["validity"] < 1.0
    names = {g.name: g for g in rep.gates}
    assert not names["box_validity"].passed
    assert '"stage": "extracted"' in rep.to_json()
