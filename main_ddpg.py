import os

from algo.ddpg import Ddpg
from utils.config import data_path, DDPG

if __name__ == '__main__':

    # p = os.path.join(data_path, RDPG_discrete, "data.npy")

    agent = Ddpg((84, 84, 3), 2, mode=DDPG)
    # agent.exploration_learn(5000)
    # agent.load()
    agent.train()
    # agent.test(15)
