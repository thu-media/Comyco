import json
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

# runs StreamLens' paper workflow (5 seek scenarios, linear vs nonlinear+prefetch) for every checkpoint
# and ranks them by the mean QoE under nonlinear buffering + prefetch
# usage: python scripts/sweep_paper.py models/base results/sweep
SL = Path(os.environ.get('STREAMLENS_SRC', Path(__file__).resolve().parents[2] / 'StreamLens/src'))
SCENARIOS = ['seeks', 'seeks_prefetch_hit', 'seeks_mixed', 'seeks_linear_hit_nonlinear_miss', 'seeks_linear_miss_nonlinear_hit']
WORKERS = 6


def run(job):
    weights, out = job
    Path(out).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, COMYCO_WEIGHTS=str(weights))
    cmd = [sys.executable, 'run_comparison.py', '-n', 'synthetic/network.json', '-m', 'synthetic/movie.json',
           '-sc', ','.join('synthetic/%s.json' % s for s in SCENARIOS), '-pc', 'synthetic/test_prefetch_config.json',
           '-a', 'comyco', '-o', str(out)]
    subprocess.run(cmd, cwd=SL, env=env, capture_output=True, check=True)
    rows = []
    for sc in SCENARIOS:
        d = json.load(open(Path(out) / sc / 'comparison_comyco.json'))
        rows.append((d['without_buffer_py']['summary']['qoe_score'], d['with_buffer_py']['summary']['qoe_score']))
    n = len(rows)
    return sum(r[0] for r in rows) / n, sum(r[1] for r in rows) / n


def main():
    model_dir, out_root = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    eps = sorted(int(p.stem.split('_')[-1]) for p in model_dir.glob('nn_model_ep_*.npz'))
    jobs = [(model_dir / ('nn_model_ep_%d.npz' % e), out_root / ('ep%d' % e)) for e in eps]
    with Pool(WORKERS) as pool:
        res = pool.map(run, jobs)
    scores = {e: r for e, r in zip(eps, res)}
    json.dump(scores, open(out_root / 'scores.json', 'w'))
    ranked = sorted(scores.items(), key=lambda kv: -kv[1][1])
    print('rank  epoch   linear QoE   nonlinear+prefetch QoE')
    for i, (e, (a, b)) in enumerate(ranked[:10]):
        print('%4d  %5d   %10.3f   %10.3f' % (i + 1, e, a, b))
    last = max(eps)
    print('last  %5d   %10.3f   %10.3f' % (last, *scores[last]))


if __name__ == '__main__':
    main()
