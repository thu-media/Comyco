import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STREAMLENS_SRC = Path(os.environ.get('STREAMLENS_SRC', ROOT.parent / 'StreamLens/src'))
BBB = STREAMLENS_SRC.parent / 'example/mmsys18/bbb.json'  # same video as synthetic/movie.json
BUFFER_MS = 25000.0  # sabre.py default max buffer
BETA = 10.0  # StreamLens QoE: rebuffer weight
GAMMA = 1.0  # StreamLens QoE: switch weight


def load_movie(path=BBB):
    m = json.load(open(path))
    return dict(
        sizes=np.asarray(m['segment_sizes_bits'], dtype=np.float64),
        seg_ms=float(m['segment_duration_ms']),
        bitrates=[float(b) for b in m['bitrates_kbps']],
    )


def load_network(path, multiplier=1.0):
    n = json.load(open(path))
    return (
        [float(p['duration_ms']) for p in n],
        [float(p['bandwidth_kbps']) * multiplier for p in n],
        [float(p['latency_ms']) for p in n],
    )


def make_sim(movie, traces, buffer_ms=BUFFER_MS):
    import slcore
    return slcore.Sim(movie['sizes'].tolist(), movie['seg_ms'], movie['bitrates'], buffer_ms, BETA, GAMMA, traces)


class QoE:
    # session QoE as printed by sabre.py: mean_utility - 10 * rebuf_ratio - 1 * switch_rate
    def __init__(self, seg_ms):
        self.seg_ms = seg_ms
        self.util = self.rebuf_ms = self.download_ms = self.wait_ms = 0.0
        self.switches = 0
        self.last_buffer_ms = 0.0

    def add(self, out, buffer_after_ms):
        dl, _ttfb, rebuf, wait, _size, _r, util, switched, _done = out
        self.util += util
        self.rebuf_ms += rebuf
        self.download_ms += dl
        self.wait_ms += wait
        self.switches += int(switched)
        self.last_buffer_ms = buffer_after_ms

    def summary(self):
        # total play time = every download + every wait + playing out the final buffer
        T = self.download_ms + self.wait_ms + self.last_buffer_ms
        mean_util = self.util / (T / self.seg_ms)
        rebuf_ratio = self.rebuf_ms / T
        switch_rate = self.switches / (T / 1000.0)
        return dict(
            qoe=mean_util - BETA * rebuf_ratio - GAMMA * switch_rate,
            mean_utility=mean_util,
            rebuf_ratio=rebuf_ratio,
            rebuf_s=self.rebuf_ms / 1000.0,
            switch_rate=switch_rate,
            switches=self.switches,
            total_play_s=T / 1000.0,
        )
