import os
import numpy as np
import tensorflow as tf
import envcpp as env
import time
import load_trace
import pool
import libcomyco

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

S_INFO = 7
S_LEN = 8
A_DIM = 6
LR_RATE = 1e-4
DEFAULT_QUALITY = 1
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
MODEL_TEST_INTERVAL = 10

RANDOM_SEED = 42
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
RAND_RANGE = 1000
TRAIN_TRACES = './cooked_test_traces/'
LOG_FILE = './results/'

# fixed to envivo
VIDEO_SIZE_FILE = './envivo/size/video_size_'
VMAF = './envivo/vmaf/video'
CHUNK_TIL_VIDEO_END_CAP = 48.0


def loopmain(sess, actor):
    video_size = {}  # in bytes
    vmaf_size = {}
    for bitrate in range(A_DIM):
        video_size[bitrate] = []
        vmaf_size[bitrate] = []
        with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
            for line in f:
                video_size[bitrate].append(int(line.split()[0]))
        with open(VMAF + str(A_DIM - bitrate)) as f:
            for line in f:
                vmaf_size[bitrate].append(float(line))
    net_env = env.Environment(TRAIN_TRACES)
    with open(LOG_FILE + 'agent', 'w') as log_file:
        #last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY
        last_chunk_vmaf = None

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        a_real_batch = [action_vec]
        r_batch = []

        entropy_record = []
        time_stamp = 0
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver(max_to_keep=100000)
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

            state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = video_chunk_vmaf / 100.
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / \
                float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
            state[4, :A_DIM] = np.array(
                next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, :A_DIM] = np.array(
                next_video_chunk_vmaf) / 100.  # mega byte
            state[6, -1] = np.minimum(video_chunk_remain,
                                      CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            action_prob, bit_rate = actor.predict(
                np.reshape(state, (-1, S_INFO, S_LEN)))

            net_env.get_optimal(float(last_chunk_vmaf))
            action_real = int(net_env.optimal)

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            action_real_vec = np.zeros(A_DIM)
            action_real_vec[action_real] = 1
            
            actor.submit(state, action_real_vec)
            actor.train()

            entropy_record.append(actor.compute_entropy(action_prob[0]))
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(VIDEO_BIT_RATE[action_real]) + '\t' +
                           str(entropy_record[-1]) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            # report experience to the coordinator
            if end_of_video:
                del s_batch[:]
                del a_batch[:]
                del r_batch[:]
                del a_real_batch[:]
                #del d_batch[:]
                del entropy_record[:]

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
                if epoch % MODEL_TEST_INTERVAL == 0:
                    actor.save('models/nn_model_ep_' + \
                        str(epoch) + '.ckpt')
                    os.system('python rl_test.py ' + 'models/nn_model_ep_' + \
                        str(epoch) + '.ckpt')
                    os.system('python plot_results.py >> results.log')
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
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    actor = libcomyco.libcomyco(sess,
            S_INFO=S_INFO, S_LEN=S_LEN, A_DIM=A_DIM,
            LR_RATE=LR_RATE)
    # modify for single agent
    loopmain(sess, actor)


if __name__ == '__main__':
    main()
