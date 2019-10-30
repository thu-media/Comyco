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
CHUNK_TIL_VIDEO_END_CAP = 48.0
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
FUTURE_P = 5

def mean_var(throu, delay):
    thr_, delay_ = np.array(throu), np.array(delay)
    weighted_stats = DescrStatsW(thr_, weights=delay_)
    return weighted_stats.mean, weighted_stats.std

def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'w')


    actor = a3c.ActorNetwork(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                max_depth=6)

    # restore neural net parameters
    if NN_MODEL is not None:  # NN_MODEL is the path to file
        actor.load(NN_MODEL)
        print("Testing model restored.")
        actor.export('comyco-pitree')
        # actor.export()
    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    last_chunk_vmaf = None

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []

    throu_array, delay_array = [], []
    video_count = 0

    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, next_video_chunk_vmaf, \
        end_of_video, video_chunk_remain, video_chunk_vmaf = \
            net_env.get_video_chunk(bit_rate)
            
        if last_chunk_vmaf is None:
            last_chunk_vmaf = video_chunk_vmaf

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        reward = 0.8469011 * video_chunk_vmaf - 28.79591348 * rebuf + 0.29797156 * \
            np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - 1.06099887 * \
            np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - \
            2.661618558192494
        r_batch.append(reward)

        last_bit_rate = bit_rate
        last_chunk_vmaf = video_chunk_vmaf

        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write(str(time_stamp / M_IN_K) + '\t' +
                        str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                        str(buffer_size) + '\t' +
                        str(rebuf) + '\t' +
                        str(video_chunk_size) + '\t' +
                        str(delay) + '\t' +
                        str(reward) + '\n')
        log_file.flush()

        state = np.zeros([S_INFO, S_LEN])
        throughput = video_chunk_size / delay / M_IN_K
        throu_array.append(throughput)
        delay_array.append(delay)
        if len(throu_array) >= FUTURE_P:
            throu_array.pop(0)
            delay_array.pop(0)
        mean, var = mean_var(throu_array, delay_array)
        # this should be S_INFO number of terms
        # state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[0, -1] = video_chunk_vmaf
        state[1, -1] = buffer_size  # 10 sec
        state[2, -1] = mean
        state[3, -1] = var  # 10 sec

        action_prob = actor.predict(
            np.reshape(state, (-1, S_INFO, S_LEN)))

        bit_rate = np.argmax(action_prob)

        s_batch.append(state)


        if end_of_video:
            log_file.write('\n')
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here
            last_chunk_vmaf = None

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)

            throu_array, delay_array = [], []
            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    main()
