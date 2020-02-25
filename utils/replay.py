from collections import deque
import random


class ReplayBuffer:
    def __init__(self, buf_size):
        self.buf_size = buf_size
        self.buffer = deque(maxlen=buf_size)

    def get_batch(self, batch_size):
        size = batch_size if len(self.buf_size) > batch_size else len(self.buf_size)
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

    def append(self, transition):
        self.buffer.append(transition)
