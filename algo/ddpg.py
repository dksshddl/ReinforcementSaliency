import datetime
from tempfile import TemporaryFile

import tensorflow as tf
import numpy as np
import cv2

from networks.Actor_ddpg import Actor
from networks.Critic_ddpg import Critic
from utils.config import *
from custom_env.envs import CustomEnv

# Recurrent Deterministic Policy Gradient
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

gamma = 0.99

action_arr = [
    [0, 0],
    [0.0275, 0],
    [-0.0275, 0],
    [0, 0.035],
    [0, -0.035]
]


class Ddpg:
    def __init__(self, obs_dim, acs_dim, mode=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_memory_growth(gpus[0], True)
            except RuntimeError as e:
                # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
                print(e)

        if mode is not None:
            self.mode = mode
        else:
            self.mode = RDPG_discrete
        self.sess = tf.Session()

        self.obs_dim = list(obs_dim)
        self.acs_dim = acs_dim

        self.actor = Actor(self.sess, self.obs_dim, self.acs_dim, "actor")
        self.actor_target = Actor(self.sess, self.obs_dim, self.acs_dim, "actor_target")

        self.critic = Critic(self.sess, self.obs_dim, self.acs_dim, "critic")
        self.critic_target = Critic(self.sess, self.obs_dim, self.acs_dim, "critic_target")

        self.saver = tf.train.Saver()

        self.sess.run(tf.global_variables_initializer())

        writer_path = os.path.join(log_path, self.mode)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.compat.v1.summary.FileWriter(writer_path, self.sess.graph)
        self.exploration = True

        self.env = CustomEnv()
        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(10_000)

    def train(self):
        global_step = 0
        epoch = 0

        self.exploration = True

        while True:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=0.3, fy=0.3, saliency=True, inference=False)
            transition = [[], [], [], []]
            ep_reward = 0
            while True:
                ac = self.actor.get_action(np.array([ob]))
                ac = np.reshape(ac, [-1, 2])
                ac = ac[-1] + self.noise.noise()

                next_ob, reward, done, next_ac = self.env.step(ac)
                reward = reward * 10  # scaling reward

                if done:
                    self.noise.reset()
                    epoch += 1
                    if epoch >= 100:
                        self.exploration = False
                    break

                transition[0] = [ob]
                transition[1] = ac
                transition[2] = reward
                transition[3] = [next_ob]

                ob = next_ob
                self.learn_batch(transition)
                transition = [[], [], [], []]

                ep_reward += reward

            print(f"episode {target_video}, reward {ep_reward}, step {global_step}, epochs {epoch}")
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag=target_video, simple_value=ep_reward)])
            self.writer.add_summary(reward_summary, global_step)
            self.save()
            if not self.exploration and ep_reward > 1500:
                break
            global_step += 1

    def test(self, max_epochs=10, save_path=None):
        if save_path is None:
            save_path = os.path.join(weight_path, self.mode, "model_rdpg.ckpt")
        self.saver.restore(self.sess, save_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        epoch = 0

        while epoch < max_epochs:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=1, fy=1, saliency=True, inference=True, target_video="11_Abbottsford.mp4")
            history = ob
            print(f"start test {target_video}")
            out = os.path.join(output_path, self.mode)
            if not os.path.exists(out):
                os.mkdir(out)
            now = datetime.datetime.now().strftime("%d_%H-%M-%S")

            writer = cv2.VideoWriter(os.path.join(out, target_video + "_" + str(now) + ".mp4"),
                                     fourcc, fps[target_video], (3840, 1920))
            ep_reward = 0
            while True:
                pred_ac = self.actor.get_action(np.array([history]))
                pred_ac = np.reshape(pred_ac, [-1, 2])
                # noise_ac = np.squeeze(pred_ac) + self.noise.noise()
                next_ob, reward, done, next_ac = self.env.step(pred_ac[-1])
                reward = reward * 10

                self.env.render(writer=writer)
                if done:
                    break
                ep_reward += reward
                history += next_ob
                # history["actions"].append(ac)
                # history["rewards"].append(reward)
            print(f"{target_video}'s total reward is {ep_reward}")
            epoch += 1

    def save(self, epochs=None):
        model_save_path = os.path.join(weight_path, self.mode)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if epochs is None:
            self.saver.save(self.sess, os.path.join(model_save_path, "model_rdpg.ckpt"))
        else:
            self.saver.save(self.sess, os.path.join(model_save_path, f"model_rdpg_{epochs}.ckpt"))

    def load(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(weight_path, self.mode, "model_rdpg.ckpt")
        self.saver.restore(self.sess, save_path)

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

        batch_size = 128
        if len(self.replay_buffer) >= 100:
            self.exploration = False
            batches = self.replay_buffer.get_batch(batch_size)  # --> [40, :, :, :, :]
            print("train start ", np.shape(batches))

            for history in batches:  # batch --> [5, 5, 5, 5, 5]
                observations = history[0][:-1]  # [batch_size, o_t, dim]
                next_observation = np.roll(history[0], -1)[:-1]  # [batch_size, o_t1, dim]

                actions = history[1]  # [batch_size, a_t, 2]
                rewards = history[2]  # [batch_size, r, 1]
                rewards = np.reshape(rewards, [1, -1, 1])

                # critic update
                a_t1 = self.actor_target.get_action(np.array([next_observation]))
                q_t1 = self.critic_target.get_q_value(np.array([next_observation]), a_t1)
                # y_t = rewards + gamma * q_t1[0]
                y_t = rewards + q_t1  # episodic
                self.critic.train(np.array([observations]), np.array([actions]), y_t)

                # actor update
                a_for_grad = self.actor.get_action(np.array([observations]))
                gradients = self.critic.get_q_gradient(np.array([observations]), a_for_grad)
                gradients = np.reshape(gradients, [1, -1, 2])
                self.actor.train(np.array([observations]), gradients)

        self.soft_target_update()

    def learn_batch(self, buf_data):
        self.replay_buffer.append(buf_data)

        batch_size = 64
        if not self.exploration:
            batches = self.replay_buffer.get_batch(batch_size)  # --> [batch_size, :, :, :, :]
            # print("train start ", np.shape(batches))

            batches = np.array(batches)

            ob = batches[:, 0]  # 32,
            ob = np.vstack(ob)

            action = batches[:, 1]  # 32, 2
            action = np.vstack(action)

            reward = batches[:, 2].reshape([batch_size, 1])  # 32, 1
            reward = np.vstack(reward)

            next_ob = batches[:, 3]
            next_ob = np.vstack(next_ob)
            # print(np.shape(history[0]), np.shape(history[1]))
            # next_hist = [np.roll(h, -1)[:-1] for h in history]
            # history = [h[:-1] for h in history]

            at = self.actor_target.get_action(np.array(next_ob))
            qt = self.critic_target.get_q_value(np.array(next_ob), at)
            yt = np.array(reward) + np.array(qt)
            self.critic.train(ob, action, yt)

            a_for_grad = self.actor.get_action(ob)
            # gradients = self.critic.get_q_gradient(history, a_for_grad)
            grads = self.critic.get_q_gradient(ob, a_for_grad)
            grads = np.reshape(grads, [-1, 2])
            self.actor.train(ob, grads)


class CriticGenerator(tf.keras.utils.Sequence):
    def __init__(self, x1, x2, y, batch_size):
        self.x1, self.x2, self.y = x1, x2, y
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, item):
        return [np.array([self.x1[item]]), np.array([self.x2[item]])], self.y[item]


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, y, batch_size):
        self.x, self.y = x, y
        self.batch_size = batch_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, item):
        return np.array([self.x[item]]), self.y[item]


if __name__ == '__main__':
    agent = Ddpg((84, 84, 3), 2)
    agent.train(5000)
