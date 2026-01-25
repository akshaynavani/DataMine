import json

from datamine.config import PipelineConfig
from datamine.schema import SAMPLE_SCHEMA
from datamine.synth import CAMERAS, SynthConfig, write_logs


def test_write_logs_produces_jsonl(tmp_path):
    paths = write_logs(tmp_path, SynthConfig(num_logs=2, frames_per_log=5))
    assert len(paths) == 2
    lines = paths[0].read_text().splitlines()
    assert len(lines) == 5 * len(CAMERAS)
    rec = json.loads(lines[0])
    assert {"log_id", "timestamp_ns", "camera", "labels", "ego", "tags"} <= rec.keys()


def test_sample_table_matches_schema(samples, synth_cfg):
    assert samples.schema.equals(SAMPLE_SCHEMA)
    assert samples.num_rows == synth_cfg.num_logs * synth_cfg.frames_per_log * len(CAMERAS)
    assert len(set(samples.column("sample_id").to_pylist())) == samples.num_rows


def test_config_env_override(tmp_path, monkeypatch):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("output_uri: /tmp/x\nselection:\n  budget: 7\n")
    monkeypatch.setenv("DATAMINE_RAW_URI", "s3://bucket/raw")
    cfg = PipelineConfig.load(cfg_path)
    assert cfg.output_uri == "/tmp/x"
    assert cfg.raw_uri == "s3://bucket/raw"
    assert cfg.selection.budget == 7
