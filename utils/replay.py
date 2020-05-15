import os
import random
import pickle
from collections import deque

import numpy as np

from utils.config import data_path


class ReplayBuffer:
    def __init__(self, buf_size, path=None):
        self.buf_size = buf_size
        self.buffer = deque(maxlen=buf_size)

        if path is not None:
            self.load(path)

    def get_batch(self, batch_size):
        size = batch_size if len(self.buffer) > batch_size else len(self.buffer)
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def append(self, transition):
        self.buffer.append(transition)

    def save(self, path):
        p = os.path.join(data_path, path)
        if not os.path.exists(p):
            os.mkdir(p)
        np.save(os.path.join(p, "data"), self.buffer)

    def load(self, path):
        p = os.path.join(data_path, path, "data.npy")
        data = np.load(p)
        self.buffer.extend(data)
