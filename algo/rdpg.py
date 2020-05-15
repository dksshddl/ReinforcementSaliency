import datetime
import time
from tempfile import TemporaryFile

import tensorflow as tf
import numpy as np
import cv2

from networks.Actor import Actor
from networks.Critic import Critic
from utils.config import *
from custom_env.envs import CustomEnv

# Recurrent Deterministic Policy Gradient
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

gamma = 0.95  # discount factor


class Rdpg:
    def __init__(self, obs_dim, acs_dim, mode=None):
        tf.compat.v1.enable_eager_execution()
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
        # self.sess = tf.Session()

        self.obs_dim = list(obs_dim)
        self.acs_dim = acs_dim

        self.actor = Actor(self.obs_dim, self.acs_dim, "actor")
        self.actor_target = Actor(self.obs_dim, self.acs_dim, "actor_target")

        self.critic = Critic(self.obs_dim, self.acs_dim, "critic")
        self.critic_target = Critic(self.obs_dim, self.acs_dim, "critic_target")

        # self.saver = tf.train.Saver()
        # tf.keras.
        # self.sess.run(tf.global_variables_initializer())

        writer_path = os.path.join(log_path, self.mode)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.summary.create_file_writer(writer_path)
        self.exploration = True

        # self.soft_target_update(tau=1)
        self.env = CustomEnv()
        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(20_000)

    def train(self):
        global_step = 0
        epoch = 0
        self.exploration = True
        self.writer.set_as_default()
        while True:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=0.3, fy=0.3, saliency=True, inference=False)
            history = [ob, [], [], []]
            ep_reward = 0
            while True:
                h_t = history[0]
                ac = self.actor.get_action(np.array([h_t]))
                # ac = np.reshape(ac, [-1, 2])
                ac = ac[-1] + self.noise.noise()
                next_ob, reward, done, next_ac = self.env.step(ac)
                reward = reward * 10  # scaling reward

                if done:
                    self.noise.reset()
                    if global_step >= 500:
                        self.exploration = False
                    break
                # hist = history[0][-1] + next_ob
                history[0] += next_ob
                history[1].append(ac)
                history[2].append([reward])

                ep_reward += reward

            print(f"episode {target_video}, reward {ep_reward}, step {global_step}, epochs {epoch}")
            # print(f"obs {np.shape(history[0])} acs {np.shape(history[1])} rewards {np.shape(history[2])}")
            tf.summary.scalar(target_video, ep_reward, global_step)
            if not self.exploration and ep_reward > 1500:
                self.save()
                break
            self.save()
            self.learn_batch(history, global_step)

            global_step += 1

    def test(self, max_epochs=10, save_path=None):
        if save_path is None:
            save_path = os.path.join(weight_path, self.mode, "model_rdpg.ckpt")
        self.saver.restore(self.sess, save_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        epoch = 0

        while epoch < max_epochs:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=1, fy=1, saliency=True, inference=True,
                                                  target_video="11_Abbottsford.mp4")
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
        self.actor.save(self.mode)
        self.actor_target.save(self.mode)
        self.critic.save(self.mode)
        self.critic_target.save(self.mode)

    def load(self, save_path=None):
        self.actor.restore(self.mode)
        self.actor_target.restore(self.mode)
        self.critic.restore(self.mode)
        self.critic_target.restore(self.mode)

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

        batch_size = 32
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
                y_t = rewards + gamma * q_t1[0]
                # y_t = rewards + q_t1  # episodic
                self.critic.train(np.array([observations]), np.array([actions]), y_t)

                # actor update
                a_for_grad = self.actor.get_action(np.array([observations]))
                gradients = self.critic.get_q_gradient(np.array([observations]), a_for_grad)
                gradients = np.reshape(gradients, [1, -1, 2])
                self.actor.train(np.array([observations]), gradients)

        self.soft_target_update()

    def learn_batch(self, buf_data, global_step):
        self.replay_buffer.append(buf_data)
        mse_loss = tf.keras.losses.MeanSquaredError()
        var_list_fn = lambda m: m.model.trainable_weights
        batch_size = 4
        if not self.exploration:
            batches = self.replay_buffer.get_batch(batch_size)  # --> [batch_size, action, reward]

            window = 1
            window_size = 5
            print("train start ", np.shape(batches))

            batch_critic_grads = []
            batch_actor_grads = []
            batch_critic_loss = []
            t0 = time.time()
            for batch in batches:
                critic_loss = []

                ob = batch[0]  # 32,
                next_ob = ob[window_size:]
                ob = ob[:-window_size]
                action = batch[1]

                reward = batch[2]
                t10 = time.time()
                max_len = len(ob)
                q_grads = []
                a_grads = []
                while True:
                    n_o = next_ob[:window * window_size]
                    o = ob[:window * window_size]
                    a = action[window - 1]
                    n_o = tf.convert_to_tensor([n_o], dtype=tf.float32)
                    o = tf.convert_to_tensor([o], dtype=tf.float32)
                    a = tf.convert_to_tensor([a], dtype=tf.float32)
                    with tf.GradientTape() as critic_tape:
                        critic_tape.watch(n_o)
                        critic_tape.watch(o)
                        critic_tape.watch(a)
                        at1 = self.actor_target.model(n_o)
                        qt1 = self.critic_target.model([n_o, at1])
                        yt = tf.convert_to_tensor(reward[window-1]) + qt1  # episodic reward
                        qt = self.critic.model([o, a], training=True)
                        q_loss = mse_loss(yt, qt)
                        critic_loss.append(float(q_loss))
                    grads = critic_tape.gradient(q_loss, var_list_fn(self.critic))
                    q_grads.append(grads)  # (step, 15)
                    with tf.GradientTape(persistent=True) as actor_tape:
                        actor_tape.watch(o)
                        a_for_grad = self.actor.model(o, training=True)
                        q_for_grad = self.critic.model([o, a_for_grad], training=True)
                    grad_q = actor_tape.gradient(q_for_grad, a_for_grad)
                    grad_a = actor_tape.gradient(a_for_grad, var_list_fn(self.actor), -grad_q)

                    a_grads.append(grad_a)  # (step, 11)
                    del actor_tape
                    if window * 5 >= max_len:
                        t1 = time.time()
                        print("end compute grads : ", t1 - t10)
                        break
                    window += 1
                q_grads = np.transpose(q_grads)
                a_grads = np.transpose(a_grads)
                q_grads = q_grads.tolist()
                a_grads = a_grads.tolist()
                q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
                a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]

                batch_critic_loss.append(np.mean(critic_loss))
                batch_critic_grads.append(q_grads)  # (batch_size, 15)
                batch_actor_grads.append(a_grads)  # (batch_szie, 11)

            tt0 = time.time()
            print("end batch compute : ", tt0 - t0)
            q_grads = np.transpose(batch_critic_grads)
            a_grads = np.transpose(batch_actor_grads)
            q_grads = q_grads.tolist()
            a_grads = a_grads.tolist()
            q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
            a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
            tf.summary.scalar("critic loss", np.mean(batch_critic_loss), step=global_step)
            self.critic.optimizer.apply_gradients(zip(q_grads, var_list_fn(self.critic)))
            self.actor.optimizer.apply_gradients(zip(a_grads, var_list_fn(self.actor)))
            tt1 = time.time()
            print("end optimize : ", tt1 - tt0)


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
    agent = Rdpg((84, 84, 3), 2)
    agent.train(5000)
