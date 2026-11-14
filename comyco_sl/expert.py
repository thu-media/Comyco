import numpy as np

REGRET_CLIP = 10.0  # score gaps beyond this are cut off, so one huge stall does not dominate the regression
REGRET_SCALE = 5.0


def expert_scores(sim, top_k, horizon, bw_scale):
    # score of every first action relative to the best one (best = 0, others negative), clipped and scaled
    # the best value is the same for every action, so subtracting it does not change which action is best
    values, _ = sim.get_values_strat(top_k, horizon, bw_scale)
    v = np.asarray(values, dtype=np.float32)
    return np.maximum(v - v.max(), -REGRET_CLIP) / REGRET_SCALE
