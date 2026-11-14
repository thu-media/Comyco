import numpy as np


class pool:
    def __init__(self, obs_dim=[6, 10], act_dim=10, size=100000):
        self.obs_buf = np.zeros([size, obs_dim[0], obs_dim[1]], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def submit(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def clear(self):
        self.ptr, self.size = 0, 0

    def get(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return self.obs_buf[idxs], self.acts_buf[idxs]
