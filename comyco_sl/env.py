import numpy as np

from .common import QoE, make_sim
from .state import StateBuilder

FIRST_QUALITY = 0  # what StreamLens' Abr.get_first_quality() returns by default
MAX_STEPS = 320


class Env:
    # one episode = one pass over the video from a random point of a random trace
    # seek_prob > 0 adds random seeks (miss: empty buffer, hit: 1-5 chunks); off by default
    def __init__(self, movie, traces, family_index=None, seed=0, seek_prob=0.0):
        self.movie = movie
        self.sim = make_sim(movie, traces)
        self.sb = StateBuilder(movie['bitrates'])
        self.rng = np.random.default_rng(seed)
        self.family_index = family_index
        self.seek_prob = seek_prob
        self.n = self.sim.n_segments
        self.seg_ms = movie['seg_ms']

    def pick_trace(self):
        if self.family_index is None:
            return int(self.rng.integers(self.sim.n_traces))
        fam = self.family_index[int(self.rng.integers(len(self.family_index)))]
        return int(fam[int(self.rng.integers(len(fam)))])

    def reset(self, trace_idx=None, start_ms=None):
        ti = self.pick_trace() if trace_idx is None else trace_idx
        if start_ms is None:
            start_ms = 0.0 if self.family_index is None else float(self.rng.uniform(0, 600000))
        self.sim.reset(ti, start_ms)
        self.sb.reset()
        self.qoe = QoE(self.seg_ms)
        self.steps = 0
        out = self.sim.step(FIRST_QUALITY)
        self.sb.push(FIRST_QUALITY, out[4], out[0])
        self.qoe.add(out, self.sim.buffer_ms)
        self.seek_plan = []
        if self.seek_prob > 0 and self.rng.random() < self.seek_prob:
            for _ in range(int(self.rng.integers(1, 4))):
                at = int(self.rng.integers(4, 150))
                tgt = int(self.rng.integers(0, self.n))
                hit = self.rng.random() < 0.5
                buf = float(self.rng.integers(1, 6)) * self.seg_ms if hit else 0.0
                self.seek_plan.append((at, tgt, buf))
            self.seek_plan.sort()
        return self.obs()

    def obs(self):
        return self.sb.observe(self.sim.buffer_ms, self.sim.next_sizes(), self.n - self.sim.chunk)

    def step(self, action):
        out = self.sim.step(int(action))
        self.sb.push(int(action), out[4], out[0])
        self.qoe.add(out, self.sim.buffer_ms)
        self.steps += 1
        done = out[8] or self.steps >= MAX_STEPS
        if not done and self.seek_plan and self.steps >= self.seek_plan[0][0]:
            _, tgt, buf = self.seek_plan.pop(0)
            self.sim.seek(tgt, buf)
        return (None if done else self.obs()), out[5], done, out
