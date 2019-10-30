import numpy as np
import itertools
A_DIM = 6
MILLISECONDS_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
VIDEO_CHUNCK_LEN = 4000.0  # millisec, every time add this amount to buffer
BITRATE_LEVELS = 6
MPC_FUTURE_CHUNK_COUNT = 5
BUFFER_THRESH = 60.0 * MILLISECONDS_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
PACKET_SIZE = 1500  # bytes

VIDEO_SIZE_FILE = './envivo/size/video_size_'
VMAF = './envivo/vmaf/video'
CHUNK_TIL_VIDEO_END_CAP = 48.0
TOTAL_VIDEO_CHUNKS = 48


class Environment:
    def __init__(self, all_cooked_time, all_cooked_bw, random_seed=RANDOM_SEED):
        assert len(all_cooked_time) == len(all_cooked_bw)

        self.CHUNK_COMBO_OPTIONS = []
        for combo in itertools.product(range(A_DIM), repeat=MPC_FUTURE_CHUNK_COUNT):
            self.CHUNK_COMBO_OPTIONS.append(combo)
            
        np.random.seed(random_seed)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # pick a random trace file
        self.trace_idx = 0
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        self.mahimahi_start_ptr = 1
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # in bytes
        self.vmaf_size = {}
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            self.vmaf_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))
            with open(VMAF + str(BITRATE_LEVELS - bitrate)) as f:
                for line in f:
                    self.vmaf_size[bitrate].append(float(line))

        self.virtual_mahimahi_ptr = self.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.last_mahimahi_time

    def reset_download_time(self):
        self.virtual_mahimahi_ptr = self.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.last_mahimahi_time

    def get_download_time(self, video_chunk_size):

        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.virtual_mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.virtual_mahimahi_ptr] \
                - self.virtual_last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.virtual_last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.virtual_last_mahimahi_time = self.cooked_time[self.virtual_mahimahi_ptr]
            self.virtual_mahimahi_ptr += 1

            if self.virtual_mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.virtual_mahimahi_ptr = 1
                self.virtual_last_mahimahi_time = 0
        delay += LINK_RTT / 1000.
        return delay

    def get_optimal(self, last_video_vmaf):
        video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.video_chunk_counter
        last_index = int(CHUNK_TIL_VIDEO_END_CAP - video_chunk_remain - 1)
        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if (TOTAL_VIDEO_CHUNKS - last_index - 1 < MPC_FUTURE_CHUNK_COUNT):
            future_chunk_length = TOTAL_VIDEO_CHUNKS - last_index - 1

        max_reward = -100000000
        reward_ = 0.
        send_data = 0
        start_buffer = self.buffer_size / MILLISECONDS_IN_SECOND

        for combo in self.CHUNK_COMBO_OPTIONS:
            #combo = full_combo[0:future_chunk_length]
            #curr_rebuffer_time = 0
            #curr_buffer = start_buffer
            self.reset_download_time()

            curr_rebuffer_time = 0.
            curr_buffer = start_buffer
            vmaf_sum = 0.
            vmaf_smoothness0 = 0.
            vmaf_smoothness1 = 0.
            vmaf_last = last_video_vmaf

            for position in range(future_chunk_length):
                chunk_quality = combo[position]
                index = last_index + position + 1
                download_time = self.get_download_time(
                    self.video_size[chunk_quality][index])

                if (curr_buffer < download_time):
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0.
                else:
                    curr_buffer -= download_time
                curr_buffer += 4.
                vmaf_current = self.vmaf_size[chunk_quality][index]
                vmaf_sum += vmaf_current
                vmaf_smoothness0 += np.abs(
                    np.maximum(vmaf_current - vmaf_last, 0.))
                vmaf_smoothness1 += np.abs(
                    np.minimum(vmaf_current - vmaf_last, 0.))
                vmaf_last = vmaf_current

                remaining = future_chunk_length - position - 1
                reward_ = 0.8469011 * vmaf_sum - 28.79591348 * curr_rebuffer_time + 0.29797156 * \
                    vmaf_smoothness0 - 1.06099887 * vmaf_smoothness1 - 2.661618558192494
                reward_est = reward_ + remaining * 100.0
                if reward_est < max_reward:
                    break

            if (reward_ >= max_reward):
                max_reward = reward_
                send_data = combo[0]
        #print(send_data, max_reward)
        return send_data

    def get_video_chunk(self, quality):

        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]
        video_chunk_vmaf = self.vmaf_size[quality][self.video_chunk_counter]

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        while True:  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] \
                * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] \
                - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                    throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] \
                    - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNKS - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNKS:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            self.trace_idx += 1
            if self.trace_idx >= len(self.all_cooked_time):
                self.trace_idx = 0

            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # randomize the start point of the video
            # note: trace file starts with time 0
            self.mahimahi_ptr = self.mahimahi_start_ptr
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(
                self.video_size[i][self.video_chunk_counter])
                
        next_video_chunk_vmaf = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_vmaf.append(
                self.vmaf_size[i][self.video_chunk_counter])

        return delay, \
            sleep_time, \
            return_buffer_size / MILLISECONDS_IN_SECOND, \
            rebuf / MILLISECONDS_IN_SECOND, \
            video_chunk_size, \
            next_video_chunk_sizes, \
            next_video_chunk_vmaf, \
            end_of_video, \
            video_chunk_remain, \
            video_chunk_vmaf
