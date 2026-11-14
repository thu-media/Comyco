import json
import sys

from sweep_paper import run

# B evaluation of one checkpoint through StreamLens' paper workflow (5 seek scenarios): mean QoE over the scenarios
# with the linear buffer and with the multi-region buffer + prefetch
# usage: python scripts/paper_eval.py weights.npz out_dir score.json
weights, out, score = sys.argv[1:4]
linear, nonlinear = run((weights, out))
json.dump({'linear': linear, 'nonlinear': nonlinear}, open(score, 'w'))
