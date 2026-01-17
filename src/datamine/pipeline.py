"""End-to-end orchestration: extract -> curate -> embed -> select -> report.

Output layout under ``output_uri``::

    extracted/   log_id=.../camera=.../*.parquet   all identifiable frames (pre-curation)
    samples/     log_id=.../camera=.../*.parquet   curated samples
    embeddings/embeddings.parquet                  sample_id, embedding[dim], model
    selection/selection.parquet                    sample_id, strategy, score, query
    selected/    log_id=.../camera=.../*.parquet   curated samples chosen for labeling/training
    reports/{extracted,curated,selected}.json, reports/selection.json
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from datamine import embedding_store, selection
from datamine.config import PipelineConfig
from datamine.embeddings import build_embedder
from datamine.reader import ArrowReader
from datamine.report import QualityReport, build_report, selection_report
from datamine.schema import PARTITION_COLUMNS
from datamine.storage import write_json, write_table

log = logging.getLogger(__name__)


@dataclass
class Paths:
    root: str

    def _p(self, *parts: str) -> str:
        return "/".join([self.root.rstrip("/"), *parts])

    @property
    def extracted(self) -> str:
        return self._p("extracted")

    @property
    def samples(self) -> str:
        return self._p("samples")

    @property
    def embeddings(self) -> str:
        return self._p("embeddings", "embeddings.parquet")

    @property
    def selection(self) -> str:
        return self._p("selection", "selection.parquet")

    @property
    def selected(self) -> str:
        return self._p("selected")

    def report(self, name: str) -> str:
        return self._p("reports", f"{name}.json")


@dataclass
class PipelineResult:
    reports: dict[str, QualityReport] = field(default_factory=dict)
    selection_summary: dict | None = None
    num_selected: int = 0


class Pipeline:
    def __init__(self, cfg: PipelineConfig, max_report_rows: int | None = 500_000) -> None:
        self.cfg = cfg
        self.paths = Paths(cfg.output_uri)
        self.max_report_rows = max_report_rows

    # -- stage 1+2: Spark ----------------------------------------------------------------
    def extract_and_curate(self, spark=None) -> None:
        from datamine.spark import build_spark
        from datamine.spark.curate import curate, write_samples
        from datamine.spark.extract import run_extract

        spark = spark or build_spark(self.cfg.spark)
        extracted = run_extract(spark, self.cfg.raw_uri).cache()
        write_samples(extracted, self.paths.extracted)
        curated = curate(extracted, self.cfg.curation)
        write_samples(curated, self.paths.samples)
        extracted.unpersist()
        log.info("wrote extracted samples to %s and curated samples to %s", self.paths.extracted, self.paths.samples)

    # -- stage 3: embeddings -------------------------------------------------------------
    def embed(self, spark=None, distributed: bool = False) -> tuple[list[str], np.ndarray]:
        emb_cfg = self.cfg.embedding
        if distributed:
            from datamine.spark import build_spark
            from datamine.storage import to_spark_uri

            spark = spark or build_spark(self.cfg.spark)
            df = spark.read.parquet(to_spark_uri(self.paths.samples))
            out = embedding_store.embed_dataframe(df, lambda: build_embedder(emb_cfg))
            out.coalesce(1).write.mode("overwrite").parquet(to_spark_uri(self.paths.embeddings))
            return embedding_store.load_embeddings(self.paths.embeddings)

        embedder = build_embedder(emb_cfg)
        samples = ArrowReader(self.paths.samples).scan()
        table = embedding_store.embed_table(samples, embedder, batch_size=emb_cfg.batch_size * 64)
        embedding_store.save_embeddings(table, self.paths.embeddings)
        log.info("embedded %d samples with %s", table.num_rows, embedder.name)
        return embedding_store.from_arrow(table)

    # -- stage 4: selection --------------------------------------------------------------
    def select(self, ids: list[str] | None = None, vectors: np.ndarray | None = None) -> tuple[pa.Table, dict]:
        sel_cfg = self.cfg.selection
        if ids is None or vectors is None:
            ids, vectors = embedding_store.load_embeddings(self.paths.embeddings)

        # Deduplicate in temporal order so the earliest frame of a duplicate run survives.
        reader = ArrowReader(self.paths.samples)
        meta = reader.scan(columns=["sample_id", "timestamp_ns"])
        ts = dict(zip(meta.column("sample_id").to_pylist(), meta.column("timestamp_ns").to_pylist()))
        order = sorted(range(len(ids)), key=lambda i: (ts.get(ids[i], 0), ids[i]))
        ids = [ids[i] for i in order]
        vectors = vectors[order]

        text_emb = None
        if sel_cfg.text_queries:
            text_emb = build_embedder(self.cfg.embedding).embed_text(sel_cfg.text_queries)

        chosen = selection.select(
            ids, vectors, sel_cfg.budget, sel_cfg.dedup_threshold,
            text_embeddings=text_emb, queries=sel_cfg.text_queries, query_top_k=sel_cfg.query_top_k,
        )
        sel_table = selection.to_table(chosen)
        write_table(sel_table, self.paths.selection)

        chosen_ids = sel_table.column("sample_id").to_pylist()
        selected_samples = reader.get(chosen_ids)
        if selected_samples.num_rows:
            write_table(selected_samples, self.paths.selected, partition_cols=PARTITION_COLUMNS)

        pos = {sid: i for i, sid in enumerate(ids)}
        summary = selection_report(vectors, vectors[[pos[s] for s in chosen_ids if s in pos]])
        summary["by_strategy"] = {
            k: int(v) for k, v in zip(*np.unique(sel_table.column("strategy").to_numpy(zero_copy_only=False),
                                                   return_counts=True))
        } if sel_table.num_rows else {}
        write_json(summary, self.paths.report("selection"))
        return sel_table, summary

    # -- stage 5: reports ----------------------------------------------------------------
    def _load_for_report(self, uri: str) -> pa.Table:
        reader = ArrowReader(uri)
        return reader.scan(limit=self.max_report_rows)

    def report(self, stage: str, uri: str, ids: list[str] | None = None, vectors: np.ndarray | None = None) -> QualityReport:
        table = self._load_for_report(uri)
        emb = None
        if ids is not None and vectors is not None and table.num_rows:
            pos = {sid: i for i, sid in enumerate(ids)}
            rows = [pos[s] for s in table.column("sample_id").to_pylist() if s in pos]
            emb = vectors[rows] if rows else None
        rep = build_report(
            table, stage, classes=self.cfg.curation.classes, gates=self.cfg.quality_gates,
            embeddings=emb, dedup_threshold=self.cfg.selection.dedup_threshold,
        )
        rep.save(self.paths.report(stage))
        return rep

    # -- all -----------------------------------------------------------------------------
    def run(self, spark=None, distributed_embed: bool = False) -> PipelineResult:
        result = PipelineResult()
        self.extract_and_curate(spark)
        result.reports["extracted"] = self.report("extracted", self.paths.extracted)
        ids, vectors = self.embed(spark, distributed=distributed_embed)
        result.reports["curated"] = self.report("curated", self.paths.samples, ids, vectors)
        sel_table, summary = self.select(ids, vectors)
        result.selection_summary = summary
        result.num_selected = sel_table.num_rows
        if sel_table.num_rows:
            result.reports["selected"] = self.report("selected", self.paths.selected, ids, vectors)
        return result


def text_search(cfg: PipelineConfig, query: str, k: int = 10) -> list[tuple[str, float]]:
    """Ad-hoc scenario mining: return the ``k`` curated samples closest to a text query."""
    from datamine.vector_index import VectorIndex

    ids, vectors = embedding_store.load_embeddings(Paths(cfg.output_uri).embeddings)
    index = VectorIndex(vectors.shape[1])
    index.add(ids, vectors)
    q = build_embedder(cfg.embedding).embed_text([query])
    return index.search_ids(q, k)[0]


__all__ = ["Pipeline", "PipelineResult", "Paths", "text_search", "pc"]
