import os
import datetime

import numpy as np
import tensorflow as tf
import gym
import cv2
from matplotlib import pyplot as plt

from tensorflow.core.protobuf import rewriter_config_pb2

from custom_env.envs import CustomEnv
from algo.ddpg import DDPG
from utils import config
from utils.replay import ReplayBuffer
from utils.ou_noise import OUNoise


class Agent:
    def __init__(self, env: gym.Env, sess=None):
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
        self.replay_buffer = ReplayBuffer(5000)
        self.sess.run(tf.global_variables_initializer())
        self.train_step = 0

    def load(self, path):
        print("laod weights from ", path)
        self.model.load_all(path)

    def train(self, max_epochs):
        epoch = 0
        global_step = 0
        steps = 0
        w = open(os.path.join("log", config.DDPG_RESNET, "reward.txt"), "a")
        self.noise = OUNoise(2)

        while epoch < max_epochs:

            ob, _, target_video = self.env.reset(trajectory=False)
            ob = tf.keras.preprocessing.sequence.pad_sequences([ob], padding='post', value=256, maxlen=30)
            history = []
            ep_reward = 0
            while True:
                buffer = [[], [], [], [], []]
                # TODO OU noise

                pred_ac = self.model.actor.predict(ob)
                noise_ac = pred_ac.squeeze() + self.noise.noise()
                next_ob, reward, done, next_ac = self.env.step(noise_ac)

                next_ob = tf.keras.preprocessing.sequence.pad_sequences([next_ob], padding='post', value=256, maxlen=30)

                reward = reward * 10
                buffer[0].append(ob[0])
                buffer[1].append(noise_ac)
                buffer[2].append([reward])
                buffer[3].append(next_ob[0])
                buffer[4].append([done])

                global_step += 1
                steps += 1
                ep_reward += reward

                self.update(buffer)

                # self.replay_buffer.append(buffer)

                if done:
                    break
                ob = next_ob

            print("{}'s reward is {} # {}".format(target_video, ep_reward, global_step))

            self.model.writer.add_summary(
                tf.Summary(value=[tf.Summary.Value(tag=target_video, simple_value=ep_reward)]), global_step)

            w.write("{} {}\n".format(target_video, ep_reward))
            self.model.save()

            if epoch is not 0 and epoch % 25 is 0:
                self.model.save(epoch)

            epoch += 1
        w.close()

    def test(self, max_epochs):
        epoch = 0
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out_path = config.output_path
        while epoch < max_epochs:
            acs, pred_acs, rewards = [], [], []
            ob, ac, target_video = self.env.reset(video_type="validation", trajectory=True)
            print(np.shape(ob))
            now = datetime.datetime.now().strftime("%d_%H-%M-%S")
            writer = cv2.VideoWriter(os.path.join(out_path, target_video + "_" + str(now) + ".mp4"),
                                     fourcc, 24, (3840, 1920))
            plt.title(target_video)
            plt.ylabel("reward")
            plt.xlabel("step")
            pred_ac = self.model.predict(np.array([ob]))
            pred_acs.append(pred_ac)
            reward_sum = 0
            while True:
                next_ob, reward, done, next_ac = self.env.step(pred_ac)
                distance = pred_ac / next_ac
                print(next_ac, pred_ac, distance)
                self.env.render(mode="viewport", writer=writer)
                rewards.append(reward)
                if done:
                    break
                else:
                    pred_ac = self.model.predict(np.array([ob]))
                    pred_acs.append(pred_ac)
                    reward_sum += reward
                    ob = next_ob
                    ac = next_ac
            writer.release()
            self.model.reset_state()
            plt.plot(rewards, 'r:')
            plt.show()
            print("end ", target_video, " reward is ", reward_sum)
            epoch += 1

    def update(self, transition):
        self.replay_buffer.append(transition)

        if len(self.replay_buffer) >= 1_000:

            batches = self.replay_buffer.get_batch(32)  # --> [40, :, :, :, :]
            print("train start ", np.shape(batches))

            for batch in batches:  # batch --> [5, 5, 5, 5, 5]
                obs = np.asarray(batch[0])
                acs = np.asarray(batch[1])
                rewards = np.asarray(batch[2])
                next_obs = np.asarray(batch[3])
                dones = np.asarray(batch[4])

                # next_action = self.model.target_actor.predict(np.array(next_obs))
                # next_action = next_action.reshape([-1, 2])
                # target_q = self.model.target_critic.predict([np.array(next_obs), np.array(next_action)])
                # target_q = target_q.reshape([-1, 1])
                #
                # for i in range(len(y_t)):
                #     if dones[i]:
                #         y_t[i] = rewards[i][0]
                #     else:
                #         y_t[i] = rewards[i][0] + 0.99 * target_q[i][0]
                # y_t = np.reshape(y_t, [-1, 1])
                # acs.reshape([-1, 2])
                # self.train_step += 1
                # self.model.train_critic(np.array(obs), np.array(acs), np.array(y_t), self.train_step)
                # a_for_grad = self.model.actor.predict(np.array(obs))
                # grads = self.model.gradients(np.array(obs), a_for_grad)
                # self.model.train_actor(np.array(obs), grads)
                # self.model.target_actor_train()
                # self.model.target_critic_train()
                # self.model.reset_state()

                for o, a, r, no, d in zip(obs, acs, rewards, next_obs, dones):
                    self.train_step += 1
                    next_action = self.model.target_actor.predict(np.array([no]))
                    target_q = self.model.target_critic.predict([np.array([no]), np.array(next_action)])
                    r, target_q = np.squeeze(r), np.squeeze(target_q)
                    if np.squeeze(d):
                        yt = r
                    else:
                        yt = r + 0.99 * target_q
                    yt = np.reshape(yt, [-1, 1])
                    self.model.train_critic(np.array([o]), np.array([a]), np.array(yt), self.train_step)
                    a_for_grad = self.model.actor.predict(np.array([o]))
                    grads = self.model.gradients(np.array([o]), a_for_grad)
                    self.model.train_actor(np.array([o]), grads)
                    self.model.target_actor_train()
                    self.model.target_critic_train()

                self.model.reset_state()

        if np.squeeze(transition[4]):
            self.noise.reset()


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            # 프로그램 시작시에 가장 장치가 설정되어야만 합니다
            print(e)

    p = os.path.join(config.weight_path, config.DDPG_RESNET, "model_ddpg.ckpt")
    my_env = CustomEnv()

    a = Agent(my_env, tf.Session())
    a.train(5000)
    # a.load(p)
    # a.test(5)
