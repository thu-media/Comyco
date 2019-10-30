import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import load_trace
import dt as a3c
import fixed_env as env
from statsmodels.stats.weightstats import DescrStatsW


S_INFO = 4  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 1  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log_sim_dt'
TEST_TRACES = './cooked_test_traces/'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = sys.argv[1]

def main():
    actor = a3c.ActorNetwork(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                max_depth=6)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        actor.load(NN_MODEL)
        print("Testing model restored.")
        actor.tree_to_code(['last_vmaf', 'buf', 'thr_avg', 'thr_std'], 'comyco-pitree-fixed.py')

if __name__ == '__main__':
    main()
