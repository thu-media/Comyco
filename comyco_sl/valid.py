import numpy as np

from . import traces
from .common import load_movie
from .env import Env
from .policy import forward_np

N_TRACES = 24
SCALES = [1.0, 2.0, 3.0, 4.0]  # test/ traces are slow (median 1.35 Mbps), scaled to the range of the StreamLens network
SEEK_PROB = 0.5
SEED = 0


class Valid:
    # held-out HSDPA traces from cooked_test_traces (never used for training), with the same seek augmentation
    def __init__(self):
        files = sorted((traces.TRACES / 'cooked_test_traces').iterdir())
        picks = np.random.default_rng(SEED).choice(len(files), N_TRACES, replace=False)
        self.traces = [traces._from_time_mbps(files[i], SCALES[k % len(SCALES)]) for k, i in enumerate(picks)]
        self.movie = load_movie()

    def score(self, weights):
        # mean session QoE of the greedy policy; a fresh Env with a fixed seed gives every call the same seek plans
        env = Env(self.movie, self.traces, None, seed=SEED, seek_prob=SEEK_PROB)
        qoe = []
        for i in range(N_TRACES):
            obs = env.reset(i, 0.0)
            done = False
            while not done:
                obs, _, done, _ = env.step(int(np.argmax(forward_np(weights, obs))))
            qoe.append(env.qoe.summary()['qoe'])
        return float(np.mean(qoe))
