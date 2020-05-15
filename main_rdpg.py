import os

from algo.rdpg import Rdpg
from utils.config import data_path, RDPG

if __name__ == '__main__':

    # p = os.path.join(data_path, RDPG_discrete, "data.npy")

    agent = Rdpg((80, 160, 3), 2, mode=RDPG)
    # agent.exploration_learn(5000)
    # agent.load()
    agent.train()
    # agent.test(15)
