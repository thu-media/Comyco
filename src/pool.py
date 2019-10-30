import numpy as np


class pool:
    def __init__(self, pool_size = 50000, a_dim = 6):
        self.pool_size = pool_size
        self.a_dim = a_dim
        self.buffer = []
        # for _ in range(self.a_dim):
        #     self.buffer.append([])

    def submit_core(self, s, a):
        _a = np.argmax(a)
        _len = len(self.buffer)
        if _len < self.pool_size:
            self.buffer.append((s, a))
        else:
            i_ = np.random.randint(_len)
            self.buffer[i_] = (s, a)

    def submit(self, s, a):
        #for _s, _a in zip(s, a):
            #__a = np.argmax(_a)
        self.submit_core(s, a)
    
    def clean(self):
        self.buffer = []
        
    def get(self, batch_size = 128):
        _len = len(self.buffer)
        if _len < batch_size:
            return np.array([]), np.array([])
        np.random.shuffle(self.buffer)
        batch_ = self.buffer[:batch_size]
        x_batch, y_batch = [],[]
        for (x, y) in batch_:
            x_batch.append(x)
            y_batch.append(y)
        return np.array(x_batch), np.array(y_batch)
#test done
if __name__ == "__main__":
    pool_ = pool()
    for i in range(1000):
        pool_.submit([i], [i % 6])
    _s, _a = pool_.get()
    for s,a in zip(_s, _a):
        print(s, a)