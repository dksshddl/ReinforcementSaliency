import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from custom_env.envs import CustomEnv
from networks.resnet import Resnet

from dataset import Sal360

max_ep_length = 5_000


def data_gen():
    env = CustomEnv()
    while True:
        ob, ac, target_video = env.reset(trajectory=True, inference=False)
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done:
                break
            yield np.array([ob]), np.array([ac])
            ob = next_ob
            ac = next_ac


if __name__ == '__main__':
    data = Sal360()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    writer_path = os.path.join("log", "resnet")

    env = CustomEnv()
    state_dim = list(env.observation_space.shape)
    action_dim = list(env.action_space.shape)

    model = Resnet(state_dim, session)
    model.model.fit_generator(data_gen(), 99, 5000)
    #
    # ep_length = 0
    # global_step = 0
    # losses = []
    # while ep_length < max_ep_length:
    #     ob, ac, target_video = env.reset(trajectory=True, inference=False)
    #     loss = []
    #     # batch_ob, batch_ac = [], []
    #     print("{}, {}".format(ep_length, target_video))
    #     while True:
    #         next_ob, reward, done, next_ac = env.step(ac)
    #
    #         padded_ob = tf.keras.preprocessing.sequence.pad_sequences([ob], maxlen=8, dtype=tf.float32, padding="post")
    #
    #         if done:
    #             break
    #
    #         l = model.optimize(padded_ob, ac, global_step)
    #
    #         loss.append(l)
    #         ob = next_ob
    #         ac = next_ac
    #
    #     model.reset_state()
        # losses.append(np.mean(loss))
        # plt.plot(losses)
        # plt.show()
        # if ep_length is not 0 and ep_length % 25 == 0:
        #     model.save()
        # ep_length += 1
