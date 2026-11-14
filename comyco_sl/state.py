import numpy as np

# rows: last bitrate, buffer, throughput, download time, next chunk sizes, chunks left
S_INFO = 6
S_LEN = 10  # must be >= number of bitrate levels
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0


class StateBuilder:
    def __init__(self, bitrates_kbps, s_len=S_LEN):
        self.bitrates = [float(b) for b in bitrates_kbps]
        self.max_br = max(self.bitrates)
        self.a_dim = len(self.bitrates)
        self.s_len = s_len
        assert s_len >= self.a_dim
        self.reset()

    def reset(self):
        self.state = np.zeros((S_INFO, self.s_len), dtype=np.float32)

    def push(self, bit_rate, size_bits, delay_ms):
        self.state = np.roll(self.state, -1, axis=1)
        self.state[0, -1] = self.bitrates[bit_rate] / self.max_br
        self.state[2, -1] = (size_bits / 8.0) / max(delay_ms, 1e-6) / M_IN_K  # MB/s
        self.state[3, -1] = delay_ms / M_IN_K / BUFFER_NORM_FACTOR

    def observe(self, buffer_ms, next_sizes_bits, chunks_left):
        state = self.state.copy()
        state[1, -1] = buffer_ms / M_IN_K / BUFFER_NORM_FACTOR
        state[4, :] = 0.0
        state[4, :self.a_dim] = np.asarray(next_sizes_bits, dtype=np.float32) / 8.0 / 1e6  # MB
        state[5, -1] = min(float(chunks_left), CHUNK_TIL_VIDEO_END_CAP) / CHUNK_TIL_VIDEO_END_CAP
        return state
