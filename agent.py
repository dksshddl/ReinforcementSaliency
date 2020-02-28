import os

import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

from custom_env.envs import CustomEnv
from algo.ddpg import DDPG
from utils import config
from utils.replay import ReplayBuffer


class Agent:
    def __init__(self, env, sess=None):
        if env is None:
            self.env = CustomEnv()
        else:
            self.env = env

        state_dim = list(self.env.observation_space.shape)  # [224,224,3]
        action_dim = list(self.env.action_space.shape)  # [2]
        action_max = env.action_space.high
        action_min = env.action_space.low
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.model = DDPG(self.sess, state_dim, action_dim, action_max[0], action_min[0])
        self.replay_buffer = ReplayBuffer(3000)
        self.sess.run(tf.global_variables_initializer())

    def train(self, max_epochs):
        epoch = 0
        global_step = 0
        train_step = 0
        train_indicator = False
        w = open(os.path.join("log", config.DDPG_MOBILE, "reward.txt"), "a")
        while epoch < max_epochs:
            ob, _, target_video = self.env.reset()
            ep_reward = 0
            while True:
                # TODO OU noise
                ac = self.model.actor.predict(np.array([ob]))
                next_ob, reward, done, _ = self.env.step(ac)
                global_step += 1
                # TODO replay buffer
                if global_step >= 1000:
                    if global_step == 1000:
                        train_indicator = True
                        print("train start")
                    transition = (ob, ac, reward, next_ob, done)
                    self.replay_buffer.append(transition)

                    batch = self.replay_buffer.get_batch(30)  # --> [30, :, :, :, :]

                    obs = np.asarray([e[0] for e in batch])
                    acs = np.asarray([e[1] for e in batch]).reshape([-1, 2])
                    rewards = np.asarray([e[2] for e in batch]).reshape([-1, 1])
                    next_obs = np.asarray([e[3] for e in batch])
                    dones = np.asarray([e[4] for e in batch]).reshape([-1, 1])
                    y_t = np.asarray([e[2] for e in batch])
                    target_q_value = self.model.target_critic.predict([next_obs, acs])
                    for k in range(len(batch)):
                        if dones[k]:
                            y_t[k] = rewards[k]
                        else:
                            y_t[k] = rewards[k] + 0.99 * target_q_value[k]
                    y_t = y_t.reshape([-1, 1])

                    for data in zip(obs, acs, next_obs, y_t):
                        train_step += 1
                        o, a, no, y = map(lambda _: np.array([_]), data)
                        self.model.train_critic(o, a, y, train_step)
                        a_for_grad = self.model.actor.predict(o)
                        grads = self.model.gradients(o, a_for_grad)
                        self.model.train_actor(o, grads)
                        self.model.target_actor_train()
                        self.model.target_critic_train()

                if np.shape(next_ob) == (6, 224, 224, 3):
                    transition = (ob, ac, reward, next_ob, done)
                    self.replay_buffer.append(transition)
                    ob = next_ob

                if done:
                    break

                ep_reward += reward
            print("{}'s reward is {} # {}".format(target_video, ep_reward, global_step))
            if train_indicator:
                w.write("{} {}\n".format(target_video, ep_reward))
            # self.model.reset_state()
            if epoch is not 0 and epoch % 50 is 0:
                self.model.save(epoch)

            epoch += 1
        w.close()

    def test(self, max_epochs):
        epoch = 0
        while epoch < max_epochs:
            acs, pred_acs = [], []
            ob, ac, target_video = self.env.reset("test")
            pred_ac = self.model.predict(ob)
            acs.append(ac)
            pred_acs.append(pred_ac)
            while True:
                next_ob, done, reward, next_ac = self.env.step(ac)
                if done:
                    break
                else:
                    pred_ac = self.model.predict(ob)
                    acs.append(ac)
                    pred_acs.append(pred_ac)

                    ob = next_ob
                    ac = next_ac


if __name__ == '__main__':
    tf.keras.backend.clear_session()  # For easy reset of notebook state.

    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    session = tf.Session(config=config_proto)

    my_env = CustomEnv()
    a = Agent(my_env, session)
    a.train(5000)
