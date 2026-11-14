import shutil
import sys
from pathlib import Path

# picks the checkpoint with the highest validation QoE (held-out HSDPA traces, see comyco_sl/valid.py)
# from a training log and copies it to <dir>/best.npz
# usage: python scripts/pick_best.py models/auto1k
d = Path(sys.argv[1])
rows = []
for line in open(d / 'log.tsv').read().splitlines()[1:]:
    p = line.split('\t')
    rows.append((float(p[7]), float(p[5]), float(p[4]), int(p[0])))
rows.sort(reverse=True)
val, nonlinear, linear, epoch = rows[0]
print('best  epoch %d  validation %.3f  (B nonlinear+prefetch %.3f, linear %.3f)' % (epoch, val, nonlinear, linear))
print('top 5:', ', '.join('ep%d=%.3f' % (r[3], r[0]) for r in rows[:5]))
shutil.copy(d / ('nn_model_ep_%d.npz' % epoch), d / 'best.npz')
shutil.copy(d / ('nn_model_ep_%d.pth' % epoch), d / 'best.pth')
print('copied to', d / 'best.npz')
