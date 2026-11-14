import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from comyco_sl.common import BBB, STREAMLENS_SRC, load_movie, load_network  # noqa: E402
from comyco_sl.env import Env  # noqa: E402
from comyco_sl.policy import forward_np, load_weights  # noqa: E402

# the plugin inside StreamLens (no seeks) must pick the same bitrates as the greedy policy in our simulator
# usage: python tests/test_plugin_vs_sim.py models/ent1/best.npz
NET = STREAMLENS_SRC / 'synthetic/network.json'


def main():
    weights = str(Path(sys.argv[1]).resolve())
    w = load_weights(weights)
    env = Env(load_movie(), [load_network(NET)], None, seed=0)
    obs, done, acts = env.reset(0, 0.0), False, [0]
    while not done:
        a = int(np.argmax(forward_np(w, obs)))
        acts.append(a)
        obs, _, done, _ = env.step(a)
    ours = env.qoe.summary()
    with tempfile.TemporaryDirectory() as d:
        o = os.path.join(d, 'o.json')
        env_vars = dict(os.environ, COMYCO_WEIGHTS=weights, SABRE_SUMMARY=o)
        cmd = [sys.executable, str(ROOT / 'scripts/sabre_launcher.py'), '-n', str(NET), '-m', str(BBB), '-a', 'comyco']
        p = subprocess.run(cmd, env=env_vars, capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(p.stderr[-1500:])
        sl = json.load(open(o))
    same = sl['comyco_actions'] == acts
    print('actions identical=%s  QoE streamlens=%.6f sim=%.6f' % (same, sl['qoe'], ours['qoe']))
    sys.exit(0 if same and abs(sl['qoe'] - ours['qoe']) < 1e-6 else 1)


if __name__ == '__main__':
    main()
