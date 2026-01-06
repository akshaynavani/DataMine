"""Pipeline configuration loaded from YAML with environment-variable overrides."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SparkConfig:
    master: str = "local[*]"
    app_name: str = "datamine"
    shuffle_partitions: int = 64


@dataclass
class CurationConfig:
    classes: list[str] = field(
        default_factory=lambda: ["car", "truck", "pedestrian", "cyclist", "traffic_light", "traffic_sign"]
    )
    min_box_area: float = 64.0
    frame_stride_ms: int = 200
    require_tags: dict[str, str] = field(default_factory=dict)
    exclude_tags: dict[str, str] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    backend: str = "hashing"
    dim: int = 512
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    batch_size: int = 64


@dataclass
class SelectionConfig:
    dedup_threshold: float = 0.97
    budget: int = 1000
    text_queries: list[str] = field(default_factory=list)
    query_top_k: int = 100


@dataclass
class QualityGates:
    min_label_completeness: float = 0.95
    min_box_validity: float = 0.98
    min_class_balance: float = 0.5
    max_duplicate_rate: float = 0.2


@dataclass
class PipelineConfig:
    raw_uri: str = "./data/raw"
    output_uri: str = "./data/curated"
    spark: SparkConfig = field(default_factory=SparkConfig)
    curation: CurationConfig = field(default_factory=CurationConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    quality_gates: QualityGates = field(default_factory=QualityGates)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            raw_uri=data.get("raw_uri", cls.raw_uri),
            output_uri=data.get("output_uri", cls.output_uri),
            spark=SparkConfig(**data.get("spark", {})),
            curation=CurationConfig(**data.get("curation", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            selection=SelectionConfig(**data.get("selection", {})),
            quality_gates=QualityGates(**data.get("quality_gates", {})),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> "PipelineConfig":
        data: dict[str, Any] = {}
        if path is not None:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(data)
        cfg.apply_env()
        return cfg

    def apply_env(self) -> None:
        """Override selected fields from DATAMINE_* environment variables."""
        env = os.environ
        self.raw_uri = env.get("DATAMINE_RAW_URI") or self.raw_uri
        self.output_uri = env.get("DATAMINE_OUTPUT_URI") or self.output_uri
        self.spark.master = env.get("DATAMINE_SPARK_MASTER") or self.spark.master
        self.embedding.backend = env.get("DATAMINE_EMBEDDER") or self.embedding.backend
        self.embedding.clip_model = env.get("DATAMINE_CLIP_MODEL") or self.embedding.clip_model
        self.embedding.clip_pretrained = env.get("DATAMINE_CLIP_PRETRAINED") or self.embedding.clip_pretrained

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
