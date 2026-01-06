# DataMine — Implementation Plan

**DataMine** is a data-centric pipeline for autonomous-vehicle (AV) perception. It ingests
large-scale driving logs, extracts and curates labeled samples with PySpark, scores dataset
quality, selects the most useful samples using CLIP-style embedding similarity, and stores the
curated result as PyArrow/Parquet columnar data on AWS S3 (or the local filesystem) for fast
random and filtered access.

Tech: Python 3.10+, PySpark, PyArrow, AWS S3 (via `pyarrow.fs` / boto3), CLIP (`open_clip`
optional), NumPy vector similarity (FAISS optional).

## Key design decisions

- **Storage abstraction.** All reads/writes go through `datamine.storage`, which resolves a URI
  (`s3://bucket/prefix` or local path) into a `pyarrow.fs.FileSystem`. Credentials come only from
  the standard AWS environment/credential chain — nothing is hard-coded.
- **Raw log format.** Driving logs are JSON Lines, one record per frame: log id, timestamp, ego
  pose, sensor (camera) id, image URI, weather/time-of-day tags, and a list of 2D box labels. A
  synthetic log generator makes the pipeline runnable without real data.
- **Spark for scale, Arrow for access.** Extraction/curation runs as Spark jobs that write
  partitioned Parquet (`log_id`, `camera`); downstream consumers read with `pyarrow.dataset`
  using predicate pushdown and column projection.
- **Pluggable embedders.** `ClipEmbedder` wraps `open_clip` when installed; `HashingEmbedder` is a
  deterministic, dependency-free fallback so the selection logic is testable on any machine.
- **Selection strategies.** Text-query retrieval (CLIP zero-shot), near-duplicate removal by cosine
  threshold, and greedy k-center coreset selection for diversity, all over L2-normalised vectors.
- **Quality metrics.** Label completeness, box validity, class balance (normalised entropy),
  temporal near-duplicate rate, and embedding-space coverage, emitted as a JSON report.

## Phases

### Phase 1 — Foundation: config, schemas, storage, synthetic logs
Deliverables
- Package skeleton and `pyproject.toml` with optional extras (`clip`, `faiss`, `dev`).
- Pipeline config (`PipelineConfig` dataclass, YAML/env loading).
- Canonical Arrow schemas for raw frames and curated samples.
- Storage layer resolving `s3://` vs. local URIs to `pyarrow.fs`.
- Synthetic driving-log generator (JSONL).

Files: `pyproject.toml`, `.env.example`, `configs/default.yaml`, `src/datamine/__init__.py`,
`src/datamine/config.py`, `src/datamine/schema.py`, `src/datamine/storage.py`,
`src/datamine/synth.py`

### Phase 2 — PySpark extraction & curation
Deliverables
- Spark session factory with S3A configuration from env.
- Extraction job: read JSONL logs, explode per-box labels, normalise fields, filter malformed rows.
- Curation filters: class allow-list, min box area, frame sub-sampling by time stride, tag filters.
- Partitioned Parquet writer (by `log_id`, `camera`).

Files: `src/datamine/spark/session.py`, `src/datamine/spark/extract.py`,
`src/datamine/spark/curate.py`, `src/datamine/spark/__init__.py`

### Phase 3 — Embeddings & vector similarity selection
Deliverables
- `Embedder` protocol, `HashingEmbedder` fallback, `ClipEmbedder` (image + text) via `open_clip`.
- `VectorIndex` (NumPy brute-force cosine, optional FAISS backend) with top-k search.
- Selection: text-query retrieval, near-duplicate removal, greedy k-center coreset.
- Embedding store: writes vectors as a `FixedSizeList<float32>` Parquet column.

Files: `src/datamine/embeddings/{__init__,base,hashing,clip}.py`,
`src/datamine/vector_index.py`, `src/datamine/selection.py`, `src/datamine/embedding_store.py`

### Phase 4 — Dataset quality metrics
Deliverables
- Metrics over Arrow tables: completeness, box validity, class balance entropy, per-class counts,
  temporal duplicate rate, embedding coverage/diversity.
- `QualityReport` dataclass with JSON serialisation and threshold-based pass/fail gates.

Files: `src/datamine/metrics.py`, `src/datamine/report.py`

### Phase 5 — Orchestration, access layer, CLI, tests
Deliverables
- `ArrowReader` for low-latency filtered reads from S3/local Parquet (pushdown + projection).
- End-to-end `Pipeline` orchestration (extract → curate → embed → select → report → write).
- `datamine` CLI (`synth`, `extract`, `embed`, `select`, `report`, `run`).
- Pytest suite (Spark tests auto-skip if Java/PySpark are unavailable).

Files: `src/datamine/reader.py`, `src/datamine/pipeline.py`, `src/datamine/cli.py`,
`tests/*.py`

Post-implementation: `.gitignore`, `README.md` with architecture diagram, run and test docs.
