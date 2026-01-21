"""``datamine`` command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from datamine.config import PipelineConfig


def _load(args) -> PipelineConfig:
    cfg = PipelineConfig.load(args.config)
    if getattr(args, "raw", None):
        cfg.raw_uri = args.raw
    if getattr(args, "out", None):
        cfg.output_uri = args.out
    if getattr(args, "embedder", None):
        cfg.embedding.backend = args.embedder
    if getattr(args, "budget", None):
        cfg.selection.budget = args.budget
    return cfg


def cmd_synth(args) -> int:
    from datamine.synth import SynthConfig, write_logs

    paths = write_logs(args.out, SynthConfig(num_logs=args.logs, frames_per_log=args.frames, seed=args.seed))
    print(f"wrote {len(paths)} logs to {args.out}")
    return 0


def cmd_extract(args) -> int:
    from datamine.pipeline import Pipeline

    Pipeline(_load(args)).extract_and_curate()
    return 0


def cmd_embed(args) -> int:
    from datamine.pipeline import Pipeline

    ids, _ = Pipeline(_load(args)).embed(distributed=args.distributed)
    print(f"embedded {len(ids)} samples")
    return 0


def cmd_select(args) -> int:
    from datamine.pipeline import Pipeline

    table, summary = Pipeline(_load(args)).select()
    print(f"selected {table.num_rows} samples")
    print(json.dumps(summary, indent=2))
    return 0


def cmd_report(args) -> int:
    from datamine import embedding_store
    from datamine.pipeline import Pipeline

    pipe = Pipeline(_load(args), max_report_rows=args.max_rows)
    ids = vectors = None
    if args.with_embeddings:
        ids, vectors = embedding_store.load_embeddings(pipe.paths.embeddings)
    uri = {"extracted": pipe.paths.extracted, "curated": pipe.paths.samples, "selected": pipe.paths.selected}[args.stage]
    rep = pipe.report(args.stage, uri, ids, vectors)
    print(rep.summary())
    return 0 if rep.passed or not args.fail_on_gate else 1


def cmd_run(args) -> int:
    from datamine.pipeline import Pipeline

    result = Pipeline(_load(args)).run(distributed_embed=args.distributed)
    for rep in result.reports.values():
        print(rep.summary())
    print(f"selected {result.num_selected} samples")
    if args.fail_on_gate and "curated" in result.reports and not result.reports["curated"].passed:
        return 1
    return 0


def cmd_query(args) -> int:
    from datamine.pipeline import text_search

    for sample_id, score in text_search(_load(args), args.text, args.k):
        print(f"{score:.4f}  {sample_id}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="datamine", description="AV perception data mining pipeline")
    p.add_argument("-c", "--config", default=None, help="YAML config (default: built-in defaults)")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--raw", help="raw log URI (overrides config)")
        sp.add_argument("--out", help="output URI (overrides config)")
        sp.add_argument("--embedder", choices=["hashing", "clip"])
        return sp

    s = sub.add_parser("synth", help="generate synthetic driving logs")
    s.add_argument("--out", default="./data/raw")
    s.add_argument("--logs", type=int, default=4)
    s.add_argument("--frames", type=int, default=200)
    s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_synth)

    common(sub.add_parser("extract", help="Spark extract + curate")).set_defaults(func=cmd_extract)

    s = common(sub.add_parser("embed", help="compute embeddings for curated samples"))
    s.add_argument("--distributed", action="store_true", help="embed on Spark executors via mapInArrow")
    s.set_defaults(func=cmd_embed)

    s = common(sub.add_parser("select", help="dedup + text query + k-center selection"))
    s.add_argument("--budget", type=int)
    s.set_defaults(func=cmd_select)

    s = common(sub.add_parser("report", help="dataset quality report for a stage"))
    s.add_argument("stage", choices=["extracted", "curated", "selected"])
    s.add_argument("--with-embeddings", action="store_true")
    s.add_argument("--max-rows", type=int, default=500_000)
    s.add_argument("--fail-on-gate", action="store_true")
    s.set_defaults(func=cmd_report)

    s = common(sub.add_parser("run", help="run the full pipeline"))
    s.add_argument("--budget", type=int)
    s.add_argument("--distributed", action="store_true")
    s.add_argument("--fail-on-gate", action="store_true")
    s.set_defaults(func=cmd_run)

    s = common(sub.add_parser("query", help="text search over embedded samples"))
    s.add_argument("text")
    s.add_argument("-k", type=int, default=10)
    s.set_defaults(func=cmd_query)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
