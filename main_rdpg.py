import os

from algo.rdpg import Rdpg
from utils.config import data_path, RDPG

if __name__ == '__main__':
    agent = Rdpg((224, 224, 3), 2, mode=RDPG)
    agent.train_v2()
    # agent.test(15)
