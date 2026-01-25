import random

import pytest

from datamine.synth import SynthConfig, generate_log, to_sample_table


@pytest.fixture(scope="session")
def synth_cfg():
    return SynthConfig(num_logs=2, frames_per_log=40, corruption_rate=0.05, stop_probability=0.05, seed=3)


@pytest.fixture(scope="session")
def records(synth_cfg):
    rng = random.Random(synth_cfg.seed)
    out = []
    for i in range(synth_cfg.num_logs):
        out.extend(generate_log(i, synth_cfg, rng))
    return out


@pytest.fixture(scope="session")
def samples(records):
    return to_sample_table(records)
