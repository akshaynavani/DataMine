"""Quality report: aggregate metrics, evaluate gates, serialise to JSON (local or S3)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pyarrow as pa

from datamine import metrics
from datamine.config import QualityGates
from datamine.storage import write_json


@dataclass
class GateResult:
    name: str
    value: float
    threshold: float
    comparator: str
    passed: bool


@dataclass
class QualityReport:
    stage: str
    num_samples: int
    metrics: dict[str, Any]
    gates: list[GateResult] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=float)

    def save(self, uri: str) -> None:
        write_json(json.loads(self.to_json()), uri)

    def summary(self) -> str:
        lines = [f"[{self.stage}] samples={self.num_samples} passed={self.passed}"]
        for g in self.gates:
            mark = "PASS" if g.passed else "FAIL"
            lines.append(f"  {mark} {g.name}: {g.value:.4f} {g.comparator} {g.threshold}")
        return "\n".join(lines)


def _gate(name: str, value: float, threshold: float, comparator: str) -> GateResult:
    passed = value >= threshold if comparator == ">=" else value <= threshold
    return GateResult(name, float(value), float(threshold), comparator, bool(passed))


def evaluate_gates(m: dict[str, Any], gates: QualityGates) -> list[GateResult]:
    dup = m.get("embedding_duplicate_rate", m["temporal_duplicate_rate"])
    return [
        _gate("label_completeness", m["label_completeness"], gates.min_label_completeness, ">="),
        _gate("box_validity", m["box_validity"]["validity"], gates.min_box_validity, ">="),
        _gate("class_balance", m["class_distribution"]["balance"], gates.min_class_balance, ">="),
        _gate("duplicate_rate", dup, gates.max_duplicate_rate, "<="),
    ]


def build_report(
    samples: pa.Table,
    stage: str,
    classes: list[str] | None = None,
    gates: QualityGates | None = None,
    embeddings: np.ndarray | None = None,
    dedup_threshold: float = 0.97,
) -> QualityReport:
    m: dict[str, Any] = {
        "label_completeness": metrics.label_completeness(samples),
        "box_validity": metrics.box_validity(samples, classes),
        "class_distribution": metrics.class_distribution(samples, classes),
        "weather": metrics.attribute_distribution(samples, "weather"),
        "time_of_day": metrics.attribute_distribution(samples, "time_of_day"),
        "camera": metrics.attribute_distribution(samples, "camera"),
        "temporal_duplicate_rate": metrics.temporal_duplicate_rate(samples),
    }
    if embeddings is not None and len(embeddings):
        m["embedding_duplicate_rate"] = metrics.embedding_duplicate_rate(embeddings, dedup_threshold)
        m["embedding_diversity"] = metrics.embedding_diversity(embeddings)
    report = QualityReport(stage=stage, num_samples=samples.num_rows, metrics=m)
    if gates is not None:
        report.gates = evaluate_gates(m, gates)
    return report


def selection_report(pool: np.ndarray, selected: np.ndarray, radii=(0.05, 0.1, 0.2)) -> dict[str, Any]:
    """How representative a selected subset is of the pool it was drawn from."""
    return {
        "pool_size": int(len(pool)),
        "selected_size": int(len(selected)),
        "coverage": {f"r={r}": metrics.coverage(pool, selected, r) for r in radii},
        "pool_diversity": metrics.embedding_diversity(pool),
        "selected_diversity": metrics.embedding_diversity(selected),
    }
