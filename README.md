# Comyco for StreamLens

Training code of the Comyco ABR that ships with [StreamLens](https://github.com/Slothbetty/StreamLens) (`-a comyco`).
It is the imitation-learning recipe of [Comyco](https://github.com/thu-media/Comyco), ported to the StreamLens QoE and its seek-aware session
model, and trained on the HSDPA traces in `cooked_traces` (copied from `src/cooked_traces` of Comyco).

- **Simulator.** `core/sim.cc` reproduces the linear-buffer path of StreamLens' `sabre.py` (`tests/test_vs_sabre.py` checks this).
  It also hosts the beam-search expert.
- **Expert.** Beam search (15 chunks ahead) on a discounted bandwidth. The target of the network is the regret of every
  first action, learned with an MSE loss, and the policy acts with `argmax`.
- **Discount.** The bandwidth discount of the expert is adapted by hill climbing on a validation score (`comyco_sl/discount.py`).
- **Seeks.** Half of the training sessions contain random seeks, so the policy sees empty buffers mid-session.
- **Evaluation.** Every checkpoint is evaluated in the background through the StreamLens paper workflow
  (5 seek scenarios, linear buffer vs. multi-region buffer + prefetch).

## Differences from Comyco

The first column is the original Comyco in `src/` of the `master` branch of [thu-media/Comyco](https://github.com/thu-media/Comyco), the second the PyTorch version
[Comyco-lin](https://github.com/godka/comyco-lin), the third this code.

| | Comyco (`master`) | Comyco-lin | this code |
|---|---|---|---|
| Framework | TensorFlow / TFLearn | PyTorch | PyTorch |
| Environment | Pensieve-style chunk simulator, 6 bitrates, 48 chunks | same | StreamLens' simulator, 10 bitrates, 199 chunks, with seeks (empty buffer, or 1 to 5 chunks kept) in half of the sessions |
| Reward | VMAF-based QoE (video quality, rebuffer, asymmetric smoothness) | linear QoE (bitrate, rebuffer, smoothness) | StreamLens' QoE (utility, rebuffer ratio, switch rate) |
| State | 7 x 8, with the VMAF of the last and of the next chunks | 6 x 8 | 6 x 10 |
| Network | one fully connected or 1-D convolution branch per state row, merged by a GRU, softmax output | Pensieve's actor, softmax output | Pensieve's actor, one score per action |
| Training target | one-hot label of the best first action | same | regret of every first action relative to the best one (clipped at 10, divided by 5) |
| Loss | cross entropy plus an entropy term (weight 1e-3) | cross entropy plus an entropy term (weight 0.15) | MSE |
| Acting while collecting data | sample from the softmax | same | `argmax(score / 0.25 + Gumbel noise)`; evaluation uses plain `argmax` |
| Expert | MPC over all 6^8 bitrate sequences of the next 8 chunks | beam search, one beam of 5000 nodes, 15 chunks ahead | beam search, one beam of 300 nodes per first action (3000 in total), 15 chunks ahead, so no action is crowded out |
| Bandwidth seen by the expert | the true future bandwidth | the true future bandwidth times a fixed 0.9 | the true future bandwidth times a discount adapted by hill climbing on a validation score |
| Training traces | `cooked_test_traces` (HSDPA), the same traces it is tested on | `cooked_traces` (HSDPA), as they are | `cooked_traces` at 0.5, 1, 2, 3 and 4 times the bandwidth |
| Evaluation during training | `cooked_test_traces` every 10 videos | `cooked_test_traces` | a validation set of held-out HSDPA traces (scaled 1 to 4 times), plus an evaluation of every checkpoint in StreamLens |

Compared with Comyco-lin, the network itself (Pensieve's actor) is unchanged apart from the missing softmax and the number of actions.

## Setup

```
git clone https://github.com/Slothbetty/StreamLens ../StreamLens   # or set STREAMLENS_SRC to its src/ directory
pip install torch numpy pybind11 setuptools
cd core && python setup.py build_ext --inplace && cd ..
```

StreamLens needs to contain the `comyco` ABR (`src/comyco.py`) for the evaluation.

## Train

```
python train.py
```

Checkpoints and a `log.tsv` (loss, validation score, discount, StreamLens QoE) are written to `SUMMARY_DIR` (see `train.py`).
Training overwrites that directory, so change it before a new run.

## Deploy

```
scripts/deploy_to_streamlens.sh models/step05/nn_model_ep_950.npz
```

copies the weights to `src/comyco_weights.npz` of the StreamLens checkout.

## Tests

```
python tests/test_vs_sabre.py
python tests/test_plugin_vs_sim.py
```
