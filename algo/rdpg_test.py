import datetime
from sys import getsizeof

import tensorflow as tf
import numpy as np
import cv2

from networks.Actor import Actor
from networks.Critic import Critic
from utils.config import *

# Recurrent Deterministic Policy Gradient
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

gamma = 0.95


class Rdpg:
    def __init__(self, env):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
                print(e)

        self.sess = tf.Session()

        self.obs_dim = list(env.observation_space.shape)
        self.acs_dim = list(env.action_space.shape)

        self.actor = Actor(self.sess, self.obs_dim, self.acs_dim, "actor")
        self.actor_target = Actor(self.sess, self.obs_dim, self.acs_dim, "actor_target")

        self.critic = Critic(self.sess, self.obs_dim, self.acs_dim, "critic")
        self.critic_target = Critic(self.sess, self.obs_dim, self.acs_dim, "critic_target")

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        writer_path = os.path.join(log_path, CARRACING)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.compat.v1.summary.FileWriter(writer_path, self.sess.graph)
        self.exploration = True

        # self.soft_target_update(tau=1)
        self.env = env
        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(5000)

    def exploration_learn(self, max_epochs):
        epoch = 0
        global_step = 0
        self.exploration = True
        while epoch < max_epochs:
            if self.exploration:
                ob, ac, target_video = self.env.reset(trajectory=True, fx=0.3, fy=0.3, saliency=True, inference=False)
            else:
                ob, ac, target_video = self.env.reset(trajectory=False, fx=0.3, fy=0.3, saliency=True, inference=False)
            history = {"observations": [ob], "actions": [], "rewards": []}
            ep_reward = 0
            while True:
                h_t = np.concatenate(history["observations"])
                if self.exploration:
                    # pred_ac = self.actor.get_action(np.array([h_t]))
                    # noise_ac = np.squeeze(pred_ac) + self.noise.noise()
                    next_ob, reward, done, next_ac = self.env.step(ac)
                else:
                    ac = self.actor.get_action(np.array([h_t]))
                    next_ob, reward, done, _ = self.env.step(ac)
                reward = reward * 10

                if done:
                    history["observations"].append(next_ob)  # 마지막 처리
                    break
                history["observations"].append(next_ob)
                history["actions"].append(ac)
                history["rewards"].append(reward)
                ep_reward += reward

                if self.exploration:
                    ac = next_ac

            print(f"episode {target_video}, reward {ep_reward}")

            reward_summary = tf.Summary(value=[tf.Summary.Value(tag=target_video, simple_value=ep_reward)])
            self.writer.add_summary(reward_summary, global_step)
            self.learn(history)
            self.save()

            global_step += 1

    def train(self, max_epochs):
        epoch = 0
        global_step = 0
        while True:
            ob = self.env.reset()
            # ob, ac, target_video = self.env.reset(trajectory=False, fx=0.3, fy=0.3, saliency=True, inference=False)
            history = {"observations": ob, "actions": [], "rewards": []}
            ep_reward = 0
            while True:
                h_t = history["observations"]
                pred_ac = self.actor.get_action(np.array([h_t]))
                pred_ac = np.reshape(pred_ac, [-1] + self.acs_dim)
                noise_ac = pred_ac[-1] + self.noise.noise()
                next_ob, reward, done, info = self.env.step(noise_ac)
                # reward = reward * 10  # scaling reward

                if done:
                    break
                history["observations"] += next_ob
                history["actions"].append(noise_ac)
                history["rewards"].append(reward)
                ep_reward += reward

            print(f"episode {CARRACING}, reward {ep_reward} # of {epoch}")
            print(f"info: {info}")
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag=CARRACING, simple_value=ep_reward)])
            self.writer.add_summary(reward_summary, global_step)
            if ep_reward > 500:
                self.save()
                break
            self.learn(history)
            self.save()

            global_step += 1

    def test(self, max_epochs=10, save_path=None):
        if save_path is None:
            save_path = os.path.join(weight_path, RDPG, "model_rdpg.ckpt")
        self.saver.restore(self.sess, save_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        epoch = 0

        while epoch < max_epochs:
            ob, ac, target_video = self.env.reset(trajectory=True, fx=1, fy=1, saliency=True, inference=True)
            history = {"observations": [ob]}
            print(f"start test {target_video}")
            out = os.path.join(output_path, RDPG)
            if not os.path.exists(out):
                os.mkdir(out)
            now = datetime.datetime.now().strftime("%d_%H-%M-%S")

            writer = cv2.VideoWriter(os.path.join(out, target_video + "_" + str(now) + ".mp4"),
                                     fourcc, fps[target_video], (3840, 1920))
            ep_reward = 0
            while True:
                h_t = np.concatenate(history["observations"])
                pred_ac = self.actor.get_action(np.array([h_t]))
                pred_ac = pred_ac[-1]
                print(pred_ac)
                # noise_ac = np.squeeze(pred_ac) + self.noise.noise()
                next_ob, reward, done, next_ac = self.env.step(pred_ac[-1])
                reward = reward * 10

                self.env.render(writer=writer)
                if done:
                    break
                ep_reward += reward
                history["observations"].append(next_ob)
                history["actions"].append(ac)
                history["rewards"].append(reward)
            print(f"{target_video}'s total reward is {ep_reward}")

    def save(self, epochs=None):
        model_save_path = os.path.join(weight_path, RDPG)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if epochs is None:
            self.saver.save(self.sess, os.path.join(model_save_path, "model_rdpg.ckpt"))
        else:
            self.saver.save(self.sess, os.path.join(model_save_path, f"model_rdpg_{epochs}.ckpt"))

    def soft_target_update(self, tau=0.001):
        critic_weight = self.critic.get_weights()
        critic_target_weight = self.critic_target.get_weights()

        actor_weight = self.actor.get_weights()
        actor_target_weight = self.actor_target.get_weights()

        for i in range(len(critic_weight)):
            critic_target_weight[i] = tau * critic_weight[i] + (1 - tau) * critic_target_weight[i]
        for i in range(len(actor_weight)):
            actor_target_weight[i] = tau * actor_weight[i] + (1 - tau) * actor_target_weight[i]

        self.critic_target.set_weights(critic_target_weight)
        self.actor_target.set_weights(actor_target_weight)

    def learn(self, buf_data):
        self.replay_buffer.append(buf_data)

        batch_size = 10
        if len(self.replay_buffer) >= 100:
            self.exploration = False
            batches = self.replay_buffer.get_batch(batch_size)  # --> [40, :, :, :, :]
            print("train start ", np.shape(batches))

            for history in batches:  # batch --> [5, 5, 5, 5, 5]
                observations = history["observations"][:-1]  # [batch_size, o_t, dim]
                next_observation = np.roll(history["observations"], -1)[:-1]  # [batch_size, o_t1, dim]
                trajectory_length = len(observations)
                actions = history["actions"]  # [batch_size, a_t, 2]
                rewards = history["rewards"]  # [batch_size, r, 1]
                rewards = np.reshape(rewards, [-1, 1])
                a_t1 = self.actor_target.get_action(np.array([next_observation]))
                q_t1 = self.critic_target.get_q_value(np.array([next_observation]), a_t1)
                y_t = rewards + gamma * q_t1

                a_for_grad = self.actor.get_action(np.array([observations]))
                gradients = self.critic.get_q_gradient(np.array([observations]), a_for_grad)
                gradients = np.reshape(gradients, [1, -1, 2]) / trajectory_length
                self.critic.train(np.array([observations]), np.array([actions]), y_t)
                self.actor.train(np.array([observations]), gradients)

        self.soft_target_update()


def make_generator(items):
    length = len(items)


if __name__ == '__main__':
    agent = Rdpg((84, 84, 3), 2)
    agent.train(5000)
