import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from custom_env.envs import CustomEnv
from algo.resnet import Resnet

max_ep_length = 5_000

if __name__ == '__main__':

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    writer_path = os.path.join("log", "resnet")

    env = CustomEnv()
    state_dim = list(env.observation_space.shape)
    action_dim = list(env.action_space.shape)

    model = Resnet(state_dim, action_dim)
    ep_length = 0
    losses = []
    while ep_length < max_ep_length:
        ob, ac, target_video = env.reset()
        loss = []
        # batch_ob, batch_ac = [], []
        print("{}, {}".format(ep_length, target_video))
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            ac = [_ + 1 for _ in ac]
            if done:
                break
            l = model.train_on_batch(np.array([ob]), np.array([ac]))
            loss.append(l)
            # batch_ob.append(ob)
            # batch_ac.append(ac)
            ob = next_ob
            ac = next_ac
        # print("batch ob: ", np.shape(batch_ob))
        # print("batch ac: ", np.shape(batch_ac))
        # model.train_on_batch(np.array(batch_ob), np.array(batch_ac))
        model.reset_state()
        losses.append(np.mean(loss))
        plt.plot(losses)
        plt.show()
        if ep_length is not 0 and ep_length % 50 == 0:
            model.save(ep_length)
        ep_length += 1
