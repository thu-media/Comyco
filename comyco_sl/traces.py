import pickle
from pathlib import Path

import numpy as np

TRACES = Path(__file__).resolve().parents[1]  # cooked_traces and cooked_test_traces next to train.py
MAHIMAHI_LATENCY_MS = 80.0  # Comyco's simulator RTT
TRAIN_SCALES = [0.5, 1.0, 2.0, 3.0, 4.0]  # bandwidth multipliers used to widen the training set
CACHE = Path(__file__).resolve().parent.parent / 'data/traces.pkl'


def _from_time_mbps(path, scale=1.0, latency_ms=MAHIMAHI_LATENCY_MS):
    # lines of '<time_s> <Mbps>'; sample k is held over (t[k-1], t[k]]
    a = np.loadtxt(path)
    if a.ndim != 2 or len(a) < 3:
        return None
    t, bw = a[:, 0], a[:, 1]
    dur = np.diff(np.concatenate([[0.0], t])) * 1000.0
    keep = dur > 0
    dur, bw = dur[keep], bw[keep]
    if len(dur) < 2:
        return None
    return dur.tolist(), (bw * 1000.0 * scale).tolist(), [latency_ms] * len(dur)


def train_families():
    # the repository's cooked_traces, each trace at every scale in TRAIN_SCALES
    files = sorted((TRACES / 'cooked_traces').iterdir())
    traces = []
    for scale in TRAIN_SCALES:
        for f in files:
            tr = _from_time_mbps(f, scale)
            if tr is not None:
                traces.append(tr)
    return {'hsdpa_scaled': traces}


def cached():
    if CACHE.exists():
        return pickle.load(open(CACHE, 'rb'))
    obj = train_families()
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump(obj, open(CACHE, 'wb'))
    return obj
