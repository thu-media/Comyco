import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from comyco_sl.common import BBB, STREAMLENS_SRC, QoE, load_movie, load_network, make_sim  # noqa: E402

# our C++ simulator vs StreamLens' sabre.py on identical action sequences
LAUNCHER = Path(__file__).resolve().parent.parent / 'scripts/sabre_launcher.py'


def run_sabre(net_path, actions, extra=(), multiplier=1.0):
    with tempfile.TemporaryDirectory() as d:
        a, o = os.path.join(d, 'a.json'), os.path.join(d, 'o.json')
        json.dump([int(x) for x in actions], open(a, 'w'))
        env = dict(os.environ, SCRIPTED_ACTIONS=a, SABRE_SUMMARY=o)
        cmd = [sys.executable, str(LAUNCHER), '-n', str(net_path), '-m', str(BBB), '-a', 'scripted',
               '-nm', str(multiplier), *extra]
        p = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-2000:])
        return json.load(open(o))


def run_ours(net_path, actions, multiplier=1.0):
    movie = load_movie()
    sim = make_sim(movie, [load_network(net_path, multiplier)])
    sim.reset(0, 0.0)
    q = QoE(movie['seg_ms'])
    rows = []
    for i, a in enumerate(actions):
        dec_buf = sim.buffer_ms
        out = sim.step(int(a))
        q.add(out, sim.buffer_ms)
        rows.append((i, dec_buf, out[0], out[1]))
    return q.summary(), rows


def compare(name, net_path, actions, extra=(), multiplier=1.0):
    ref = run_sabre(net_path, actions, extra, multiplier)
    ours, rows = run_ours(net_path, actions, multiplier)
    worst = 0.0
    for rec in ref['records']:  # chunks >= 1
        _, dec_buf, dl, ttfb = rows[rec['seg']]
        worst = max(worst, abs(rec['decision_buffer_ms'] - dec_buf), abs(rec['time'] - dl), abs(rec['ttfb'] - ttfb))
    keys = ['qoe', 'mean_utility', 'rebuf_ratio', 'rebuf_s', 'switches', 'switch_rate', 'total_play_s']
    diffs = {k: abs(ref[k] - ours[k]) for k in keys}
    ok = worst < 1e-3 and max(diffs.values()) < 1e-6 * max(1.0, abs(ref['total_play_s']))
    print('[%s] %s: per-chunk max|diff|=%.2e ms | QoE sabre=%.6f ours=%.6f | rebuf %.3fs vs %.3fs | switches %d vs %d' % (
        'PASS' if ok else 'FAIL', name, worst, ref['qoe'], ours['qoe'], ref['rebuf_s'], ours['rebuf_s'],
        ref['switches'], ours['switches']))
    return ok


def main():
    net = STREAMLENS_SRC / 'synthetic/network.json'
    n = len(load_movie()['sizes'])
    rng = np.random.default_rng(0)
    cases = {
        'const q0': [0] * n,
        'const q5': [5] * n,
        'const q9': [9] * n,
        'random seed0': rng.integers(0, 10, n),
        'random seed1': rng.integers(0, 10, n),
        'sawtooth': [(i // 3) % 10 for i in range(n)],
    }
    ok = True
    for name, acts in cases.items():
        ok &= compare(name, net, acts)
    g3 = sorted((STREAMLENS_SRC.parent / 'example/mmsys18/3Glogs').glob('*.json'))[3]
    ok &= compare('3G trace random', g3, rng.integers(0, 10, n))
    sd = sorted((STREAMLENS_SRC.parent / 'example/mmsys18/sd_fs').glob('*.json'))[7]
    ok &= compare('sd_fs trace random', sd, rng.integers(0, 10, n))
    ok &= compare('synthetic, -noa', net, rng.integers(0, 10, n), extra=('-noa',))
    ok &= compare('network x0.4', net, rng.integers(0, 10, n), multiplier=0.4)
    print('ALL PASS' if ok else 'SOME FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
