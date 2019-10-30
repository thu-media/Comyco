#PiTree: Our implementation

import os
import numpy as np
#import opt_env as env
import envcpp as env
import time
import warnings
import load_trace
import dt as a3c
import dt_pool as pool
from statsmodels.stats.weightstats import DescrStatsW

S_INFO = 4
S_LEN = 1
A_DIM = 6
BITRATE_LEVELS = A_DIM
KERNEL = 32
LR_RATE = 1e-4
DEFAULT_QUALITY = 1
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
MODEL_PRINT_INTERVAL = 2000
MODEL_SAVE_INTERVAL = MODEL_PRINT_INTERVAL * 2
MODEL_TEST_INTERVAL = MODEL_SAVE_INTERVAL * 5
BUFFER_NORM_FACTOR = 5.0
RANDOM_SEED = 42
TRAINING_TIMES = 30
FUTURE_P = 5
ENTROPY_THRES = 0.3

BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
RAND_RANGE = 1000
TRAIN_SEQ_LEN = 100  # take as a train batch

TRAIN_TRACES = './cooked_traces/'
LOG_FILE = './results2/'
# only validate in envivo
VIDEO_SIZE_FILE = './envivo/size/video_size_'
VMAF = './envivo/vmaf/video'
CHUNK_TIL_VIDEO_END_CAP = 48.0

def mean_var(throu, delay):
    thr_, delay_ = np.array(throu), np.array(delay)
    weighted_stats = DescrStatsW(thr_, weights=delay_)
    return weighted_stats.mean, weighted_stats.std

def loopmain():
    pool_ = pool.pool()

    video_size = {}  # in bytes
    vmaf_size = {}
    for bitrate in range(BITRATE_LEVELS):
        video_size[bitrate] = []
        vmaf_size[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
        with open(VMAF + str(BITRATE_LEVELS - bitrate)) as f:
            for line in f:
                vmaf_size[bitrate].append(float(line))

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(
        TRAIN_TRACES)
    net_env = env.Environment(TRAIN_TRACES)
    with open(LOG_FILE + 'agent', 'w') as log_file:
        actor = a3c.ActorNetwork(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 max_depth=6)
        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        a_real_batch = [action_vec]
        r_batch = []
        
        time_stamp = 0

        throu_array, delay_array = [], []
        epoch = 0

        while True:
            net_env.get_video_chunk(int(bit_rate))

            #next_video_chunk_sizes, next_video_chunk_vmaf, \
            delay, sleep_time, buffer_size, rebuf, video_chunk_size, \
                end_of_video, video_chunk_remain, video_chunk_vmaf = \
                net_env.delay0, net_env.sleep_time0, net_env.return_buffer_size0, net_env.rebuf0, \
                net_env.video_chunk_size0, net_env.end_of_video0, net_env.video_chunk_remain0, net_env.video_chunk_vmaf0

            next_video_chunk_sizes = []
            for i in range(A_DIM):
                next_video_chunk_sizes.append(
                    video_size[i][net_env.video_chunk_counter])

            next_video_chunk_vmaf = []
            for i in range(A_DIM):
                next_video_chunk_vmaf.append(
                    vmaf_size[i][net_env.video_chunk_counter])

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            if last_chunk_vmaf is None:
                last_chunk_vmaf = video_chunk_vmaf

            reward = 0.8469011 * video_chunk_vmaf - 28.79591348 * rebuf + 0.29797156 * \
                np.abs(np.maximum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - 1.06099887 * \
                np.abs(np.minimum(video_chunk_vmaf - last_chunk_vmaf, 0.)) - \
                2.661618558192494

            r_batch.append(reward)

            last_bit_rate = bit_rate
            last_chunk_vmaf = video_chunk_vmaf

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

            net_env.get_optimal(float(last_chunk_vmaf))
            action_real = int(net_env.optimal)
            # force robust
            if actor.compute_entropy(action_prob) > ENTROPY_THRES:
                action_cumsum = np.cumsum(action_prob)
                bit_rate = (action_cumsum > np.random.randint(
                    1, RAND_RANGE) / float(RAND_RANGE)).argmax()
            else:
                bit_rate = np.random.randint(A_DIM)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            

            action_real_vec = np.zeros(A_DIM)
            action_real_vec[action_real] = 1
            
            pool_.submit(state, action_real_vec)
                
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(VIDEO_BIT_RATE[action_real]) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del a_real_batch[:]
                throu_array, delay_array = [], []
                # so that in the log we know where video ends
                log_file.write('\n')

            # store the state and action into batches
            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                last_chunk_vmaf = None
                #chunk_index = 0

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                action_real_vec = np.zeros(A_DIM)
                action_real_vec[action_real] = 1

                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                a_real_batch.append(action_real_vec)

                epoch += 1
                if epoch % 10 == 0:
                    print(time.time())
                    training_s_batch, training_a_batch = pool_.get()
                    if training_s_batch.shape[0] > 0:
                        actor.train(np.array(training_s_batch),
                            np.array(training_a_batch))
                    actor.save('pitree/pitree' + str(epoch) + '.model')
                    os.system('python dt_test.py ' + 'pitree/pitree' + str(epoch) + '.model')
                    os.system('python plot_results.py >> tab.log')
                #d_batch.append(np.zeros((3, 5)))

            else:
                s_batch.append(state)

                action_vec = np.zeros(A_DIM)
                action_vec[bit_rate] = 1

                a_batch.append(action_vec)
                a_real_batch.append(action_vec)


def main():
    # create result directory
    if not os.path.exists(LOG_FILE):
        os.makedirs(LOG_FILE)

    loopmain()


if __name__ == '__main__':
    main()
