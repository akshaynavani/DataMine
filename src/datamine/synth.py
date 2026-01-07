"""Synthetic driving-log generator.

Produces JSON Lines files that mimic a multi-camera AV log: an ego trajectory, weather and
time-of-day tags, and per-frame 2D box labels for tracked actors. A small fraction of records
are deliberately corrupted (degenerate boxes, missing image URIs, unknown classes) and the ego
occasionally stops, generating near-duplicate frames — so quality metrics and curation filters
have something real to catch.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

CAMERAS = ("front", "front_left", "front_right", "rear")
CLASS_WEIGHTS = {
    "car": 0.45,
    "truck": 0.1,
    "pedestrian": 0.2,
    "cyclist": 0.07,
    "traffic_light": 0.1,
    "traffic_sign": 0.08,
}
WEATHER = ("clear", "cloudy", "rain", "fog", "snow")
TIME_OF_DAY = ("day", "dusk", "night")
LOCATIONS = ("sf_downtown", "phx_suburb", "pit_highway", "lv_strip")


@dataclass
class SynthConfig:
    num_logs: int = 4
    frames_per_log: int = 200
    frame_interval_ms: int = 100
    image_width: int = 1920
    image_height: int = 1080
    max_actors: int = 12
    corruption_rate: float = 0.02
    stop_probability: float = 0.01
    seed: int = 7


class _Actor:
    def __init__(self, rng: random.Random, idx: int, w: int, h: int) -> None:
        self.track_id = f"trk_{idx:04d}"
        self.category = rng.choices(list(CLASS_WEIGHTS), weights=list(CLASS_WEIGHTS.values()))[0]
        scale = {"truck": 2.0, "car": 1.4, "pedestrian": 0.6, "cyclist": 0.8}.get(self.category, 0.4)
        self.bw = rng.uniform(40, 160) * scale
        self.bh = self.bw * rng.uniform(0.6, 2.2 if self.category == "pedestrian" else 1.0)
        self.cx = rng.uniform(0, w)
        self.cy = rng.uniform(h * 0.35, h * 0.85)
        self.vx = rng.gauss(0, 8)
        self.vy = rng.gauss(0, 1.5)

    def step(self) -> None:
        self.cx += self.vx
        self.cy += self.vy

    def bbox(self, w: int, h: int) -> list[float] | None:
        x1, y1 = self.cx - self.bw / 2, self.cy - self.bh / 2
        x2, y2 = self.cx + self.bw / 2, self.cy + self.bh / 2
        x1, y1, x2, y2 = max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def _corrupt(record: dict, rng: random.Random) -> None:
    kind = rng.choice(("degenerate_box", "missing_image", "unknown_class", "out_of_bounds"))
    if kind == "missing_image":
        record["image_uri"] = None
    elif record["labels"]:
        label = rng.choice(record["labels"])
        if kind == "degenerate_box":
            x1, y1, _, _ = label["bbox"]
            label["bbox"] = [x1, y1, x1, y1]
        elif kind == "unknown_class":
            label["category"] = "unlabeled_blob"
        else:
            label["bbox"][2] = record["width"] * 1.5


def generate_log(log_idx: int, cfg: SynthConfig, rng: random.Random) -> list[dict]:
    log_id = f"log_{log_idx:05d}"
    weather = rng.choice(WEATHER)
    tod = rng.choice(TIME_OF_DAY)
    location = rng.choice(LOCATIONS)
    actors = [_Actor(rng, i, cfg.image_width, cfg.image_height) for i in range(rng.randint(3, cfg.max_actors))]
    x = y = yaw = 0.0
    speed = rng.uniform(5, 15)
    stopped_for = 0
    t0 = 1_700_000_000_000_000_000 + log_idx * 3_600_000_000_000
    records: list[dict] = []

    for frame in range(cfg.frames_per_log):
        if stopped_for == 0 and rng.random() < cfg.stop_probability:
            stopped_for = rng.randint(10, 40)
        moving = stopped_for == 0
        stopped_for = max(0, stopped_for - 1)
        cur_speed = speed if moving else 0.0
        dt = cfg.frame_interval_ms / 1000.0
        yaw += rng.gauss(0, 0.01) if moving else 0.0
        x += cur_speed * dt * math.cos(yaw)
        y += cur_speed * dt * math.sin(yaw)
        if moving:
            for a in actors:
                a.step()
        ts = t0 + frame * cfg.frame_interval_ms * 1_000_000

        for cam in CAMERAS:
            labels = []
            for a in actors:
                if cam != "front" and rng.random() < 0.6:
                    continue  # side/rear cameras see a subset of actors
                box = a.bbox(cfg.image_width, cfg.image_height)
                if box is None:
                    continue
                labels.append(
                    {"track_id": a.track_id, "category": a.category, "bbox": box, "occluded": rng.random() < 0.15}
                )
            record = {
                "log_id": log_id,
                "frame_id": frame,
                "timestamp_ns": ts,
                "camera": cam,
                "image_uri": f"s3://av-raw-logs/{log_id}/{cam}/{frame:06d}.jpg",
                "width": cfg.image_width,
                "height": cfg.image_height,
                "ego": {"x": x, "y": y, "yaw": yaw, "speed": cur_speed},
                "tags": {"weather": weather, "time_of_day": tod, "location": location},
                "labels": labels,
            }
            if rng.random() < cfg.corruption_rate:
                _corrupt(record, rng)
            records.append(record)
    return records


def write_logs(out_dir: str | Path, cfg: SynthConfig | None = None) -> list[Path]:
    """Generate ``cfg.num_logs`` JSONL files under ``out_dir`` and return their paths."""
    cfg = cfg or SynthConfig()
    rng = random.Random(cfg.seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(cfg.num_logs):
        path = out / f"log_{i:05d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in generate_log(i, cfg, rng):
                fh.write(json.dumps(rec) + "\n")
        paths.append(path)
    return paths
