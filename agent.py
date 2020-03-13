import os

import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

from custom_env.envs import CustomEnv
from algo.ddpg import DDPG
from utils import config
from utils.replay import ReplayBuffer

train_step = 0


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
        train_indicator = False
        w = open(os.path.join("log", config.DDPG_MOBILE, "reward.txt"), "a")
        while epoch < max_epochs:

            if epoch >= 100 or len(self.replay_buffer) >= 2000:
                pass

            ob, _, target_video = self.env.reset()
            ep_reward = 0
            buffer = [[], [], [], [], []]
            while True:
                # TODO OU noise
                ac = self.model.actor.predict(np.array([ob]))
                next_ob, reward, done, _ = self.env.step(ac)

                if done:
                    break

                global_step += 1
                # TODO replay buffer
                if global_step >= 1000:
                    if global_step == 1000:
                        train_indicator = True
                        print("train start")

                    batches = self.replay_buffer.get_batch(15)  # --> [30, :, :, :, :]

                    for batch in batches:  # batch --> [5, 5, 5, 5, 5]
                        obs = np.asarray(batch[0])
                        acs = np.asarray(batch[1])
                        rewards = np.asarray(batch[2])
                        next_obs = np.asarray(batch[3])
                        dones = np.asarray(batch[4])
                        y_t = np.copy(rewards)
                        # print(np.shape(o), np.shape(a), np.shape(r), np.shape(no), np.shape(dones))
                        target_q_value = self.model.target_critic.predict([next_obs, acs])
                        for k in range(len(batch)):
                            if dones[k]:
                                y_t[k] = rewards[k]
                            else:
                                y_t[k] = rewards[k] + 0.99 * target_q_value[k]
                        y_t = y_t.reshape([-1, 1])
                        train_step += 1
                        self.model.train_critic(obs, acs, y_t, train_step)
                        a_for_grad = self.model.actor.predict(obs)
                        grads = self.model.gradients(obs, a_for_grad)
                        self.model.train_actor(obs, grads)
                        self.model.target_actor_train()
                        self.model.target_critic_train()

                if np.shape(next_ob) == (6, 224, 224, 3):
                    buffer[0].append(ob)
                    buffer[1].append(ac[0])
                    buffer[2].append([reward])
                    buffer[3].append(next_ob)
                    buffer[4].append([done])
                    # print(np.shape(buffer[0]), np.shape(buffer[1]), np.shape(buffer[3]))
                    # transition = (ob, ac, reward, next_ob, done)
                    if len(buffer[0]) == 5:
                        self.replay_buffer.append(buffer)
                        buffer = [[], [], [], [], []]
                    ob = next_ob
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

    def update(self):
        global train_step
        batches = self.replay_buffer.get_batch(15)  # --> [30, :, :, :, :]

        for batch in batches:  # batch --> [5, 5, 5, 5, 5]
            obs = np.asarray(batch[0])
            acs = np.asarray(batch[1])
            rewards = np.asarray(batch[2])
            next_obs = np.asarray(batch[3])
            dones = np.asarray(batch[4])
            y_t = np.copy(rewards)
            # print(np.shape(o), np.shape(a), np.shape(r), np.shape(no), np.shape(dones))
            target_q_value = self.model.target_critic.predict([next_obs, acs])
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + 0.99 * target_q_value[k]
            y_t = y_t.reshape([-1, 1])
            self.model.train_critic(obs, acs, y_t, train_step)
            a_for_grad = self.model.actor.predict(obs)
            grads = self.model.gradients(obs, a_for_grad)
            self.model.train_actor(obs, grads)
            self.model.target_actor_train()
            self.model.target_critic_train()
            train_step += 1


if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)])
        except RuntimeError as e:
            # 프로그램 시작시에 가장 장치가 설정되어야만 합니다
            print(e)

    # tf.keras.backend.clear_session()  # For easy reset of notebook state.
    #
    # config_proto = tf.ConfigProto()
    # off = rewriter_config_pb2.RewriterConfig.OFF
    # config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    # session = tf.Session(config=config_proto)

    my_env = CustomEnv()
    a = Agent(my_env, tf.Session())
    a.train(5000)
