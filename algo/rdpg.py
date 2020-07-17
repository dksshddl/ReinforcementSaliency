import datetime
import tracemalloc

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import cv2

from networks.Actor import Actor
from networks.Critic import Critic
from utils.config import *
from custom_env.envs import CustomEnv

# Recurrent Deterministic Policy Gradient
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

gamma = 0.99  # discount factor
var_list_fn = lambda m: m.model.trainable_weights
mse_loss = tf.keras.losses.MeanSquaredError()

class Rdpg:
    def __init__(self, obs_dim, acs_dim, mode=None):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # tf.config.experimental.set_virtual_device_configuration(
                #     gpus[0],
                #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 8)])
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

        # feature = tf.keras.applications.MobileNetV2(include_top=False)  # 1280
        # feature_target = tf.keras.applications.MobileNetV2(include_top=False)  # 1280

        # feature = tf.keras.applications.DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling="max")  # 2048
        # feature_target = tf.keras.applications.DenseNet121(include_top=False, input_shape=(224, 224, 3), pooling="max") # 2048

        # feature = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3), pooling="max")  # 2048
        # feature_target = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3),
        #                                                    pooling="max")  # 2048
        # decode = tf.keras.applications.mobilenet_v2.decode_predictions
        feature = None
        feature_target = None

        self.actor = Actor(self.obs_dim, self.acs_dim, "actor", feature=feature)
        self.critic = Critic(self.obs_dim, self.acs_dim, "critic", feature=feature)

        self.actor_target = Actor(self.obs_dim, self.acs_dim, "actor_target", feature=feature_target)
        self.critic_target = Critic(self.obs_dim, self.acs_dim, "critic_target", feature=feature_target)

        self.soft_target_update(1)

        writer_path = os.path.join(log_path, self.mode)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.summary.create_file_writer(writer_path)
        self.exploration = True

        self.env = CustomEnv()
        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(150)

    # def train(self):
    #     global_step = 0
    #     self.exploration = True
    #     self.writer.set_as_default()
    #     while True:
    #         ob, ac, target_video = self.env.reset(trajectory=False, fx=1, fy=1, saliency=True, inference=False)
    #         history = [ob, [], [], []]
    #         ep_reward = 0
    #         while True:
    #             h_t = history[0]
    #             ac = self.actor.get_action(np.array([h_t]))
    #             # ac = np.reshape(ac, [-1, 2])
    #             ac = ac[-1] + self.noise.noise()
    #             next_ob, reward, done, next_ac = self.env.step(ac)
    #             # reward = reward * 10  # scaling reward
    #
    #             if done:
    #                 self.noise.reset()
    #
    #                 self.actor.reset_state()
    #                 self.critic.reset_state()
    #                 self.actor_target.reset_state()
    #                 self.critic_target.reset_state()
    #
    #                 if global_step > 1:
    #                     self.exploration = False
    #                 break
    #             history[0] += next_ob
    #             history[1].append(ac)
    #             history[2].append([reward])
    #
    #             ep_reward += reward
    #
    #         print(f"episode {target_video}, reward {ep_reward}, step {global_step}")
    #         # print(f"obs {np.shape(history[0])} acs {np.shape(history[1])} rewards {np.shape(history[2])}")
    #         tf.summary.scalar(target_video, ep_reward, global_step)
    #         if not self.exploration and ep_reward > 1500:
    #             self.save()
    #             break
    #         self.save()
    #         self.learn_batch(history, global_step)
    #
    #         global_step += 1

    def test(self, max_epochs=10, save_path=None):
        if save_path is None:
            save_path = os.path.join(weight_path, self.mode, "model_rdpg.ckpt")
        self.saver.restore(self.sess, save_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        epoch = 0

        while epoch < max_epochs:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=1, fy=1, saliency=True, inference=True)
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
        self.critic.save(self.mode)

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

    # def learn_batch(self, buf_data, global_step):
    #     self.replay_buffer.append(buf_data)
    #     batch_size = 4
    #     if not self.exploration:
    #         batches = self.replay_buffer.get_batch(batch_size)  # --> [batch_size, action, reward]
    #
    #         print("train start ", np.shape(batches))
    #
    #         batch_critic_grads = []
    #         batch_actor_grads = []
    #         batch_critic_loss = []
    #         t0 = time.time()
    #         for batch in batches:
    #             window = 1
    #             window_size = 5
    #
    #             critic_loss = []
    #
    #             ob = batch[0]  # 32,
    #             next_ob = ob[window_size:]
    #             ob = ob[:-window_size]
    #             action = batch[1]
    #             reward = batch[2]
    #
    #             t10 = time.time()
    #             max_len = len(ob)
    #             q_grads = []
    #             a_grads = []
    #             while True:
    #                 n_o = next_ob[:window * window_size]
    #                 o = ob[:window * window_size]
    #                 n_o = tf.convert_to_tensor([n_o], dtype=tf.float32)
    #                 o = tf.convert_to_tensor([o], dtype=tf.float32)
    #                 a = tf.convert_to_tensor([action[window - 1]], dtype=tf.float32)
    #                 r = tf.convert_to_tensor(reward[window - 1], dtype=tf.float32)
    #                 with tf.GradientTape() as critic_tape:
    #                     at1 = self.actor_target.model(n_o, training=False)
    #                     qt1 = self.critic_target.model([n_o, at1], training=False)
    #                     # yt = r + qt1  # episodic reward
    #                     yt = r + gamma * qt1
    #
    #                     qt = self.critic.model([o, a], training=True)
    #                     q_loss = mse_loss(yt, qt)
    #                     critic_loss.append(float(q_loss))
    #                 grads = critic_tape.gradient(q_loss, var_list_fn(self.critic))
    #                 q_grads.append(grads)  # (step, 15)
    #                 with tf.GradientTape(persistent=True) as actor_tape:
    #                     a_for_grad = self.actor.model(o, training=True)
    #                     q_for_grad = self.critic.model([o, a_for_grad], training=True)
    #                 grad_q = actor_tape.gradient(q_for_grad, a_for_grad)
    #                 grad_a = actor_tape.gradient(a_for_grad, var_list_fn(self.actor), -grad_q)
    #
    #                 a_grads.append(grad_a)  # (step, 11)
    #                 del actor_tape
    #                 if window * 5 >= max_len:
    #                     t1 = time.time()
    #                     print("end compute grads : ", t1 - t10)
    #                     break
    #                 window += 1
    #             q_grads = np.transpose(q_grads)
    #             a_grads = np.transpose(a_grads)
    #             q_grads = q_grads.tolist()
    #             a_grads = a_grads.tolist()
    #             q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
    #             a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
    #
    #             batch_critic_loss.append(np.mean(critic_loss))
    #             batch_critic_grads.append(q_grads)  # (batch_size, 15)
    #             batch_actor_grads.append(a_grads)  # (batch_szie, 11)
    #
    #         tt0 = time.time()
    #         print("end batch compute : ", tt0 - t0)
    #         q_grads = np.transpose(batch_critic_grads)
    #         a_grads = np.transpose(batch_actor_grads)
    #         q_grads = q_grads.tolist()
    #         a_grads = a_grads.tolist()
    #         q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
    #         a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
    #         tf.summary.scalar("critic loss", np.mean(batch_critic_loss), step=global_step)
    #         self.critic.optimizer.apply_gradients(zip(q_grads, var_list_fn(self.critic)))
    #         self.actor.optimizer.apply_gradients(zip(a_grads, var_list_fn(self.actor)))
    #         tt1 = time.time()
    #         print("end optimize : ", tt1 - tt0)
    #         self.soft_target_update()

    def train_v2(self):
        # tracemalloc.start()
        global_step = 0
        self.exploration = True
        self.writer.set_as_default()
        while True:
            ob, ac, target_video = self.env.reset(trajectory=False, fx=1, fy=1,
                                                  saliency=True, inference=False)
            # ob = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(ob))
            # ob = np.array(ob, dtype=np.float32) / 255.  # 10, 160, 320, 3
            history = [[], [], []]
            ep_reward = 0
            # print(np.shape(ob))
            while True:
                ac = self.actor.get_action(np.array([ob]))
                ac = ac[-1] + self.noise.noise()

                next_ob, reward, done, next_ac = self.env.step(ac)
                # next_ob = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(next_ob))
                # next_ob = np.array(next_ob, dtype=np.float32) / 255.
                # reward = reward * 10  # scaling reward
                if done:
                    history[0].append(next_ob)

                    self.noise.reset()
                    self.actor.reset_state()
                    self.critic.reset_state()
                    self.actor_target.reset_state()
                    self.critic_target.reset_state()
                    if global_step >= 30:
                        self.exploration = False
                    break
                history[0].append(ob)
                history[1].append([ac])
                history[2].append([reward])
                ep_reward += reward
                ob = next_ob

            print(f"episode {target_video}, reward {ep_reward}, step {global_step}")
            # print(f"obs {np.shape(history[0])} acs {np.shape(history[1])} rewards {np.shape(history[2])}")
            tf.summary.scalar(target_video, ep_reward, global_step)
            if not self.exploration and ep_reward > 1500:
                self.save()
                break
            self.save()
            self.learn_batch_v2(history, global_step)
            global_step += 1

    def learn_batch_v2(self, buf_data, global_step):
        self.replay_buffer.append(buf_data)
        batch_size = 4

        if not self.exploration:
            batches = self.replay_buffer.get_batch(batch_size)  # --> [batch_size, action, reward]

            print("train start ", np.shape(batches))

            batch_critic_grads = []
            batch_actor_grads = []
            batch_critic_loss = []

            # t0 = time.time()
            for batch in batches:

                critic_loss = []

                ob = np.array(batch[0])  # 32,
                next_ob = ob[1:]
                ob = ob[:-1]
                action = batch[1]
                reward = batch[2]

                # t10 = time.time()

                q_grads = []
                a_grads = []
                print(np.shape(ob), np.shape(reward), np.shape(next_ob), np.shape(reward), np.shape(action))
                for o, n_o, a, r in zip(ob, next_ob, action, reward):
                    n_o = tf.convert_to_tensor([n_o], dtype=tf.float32)
                    o = tf.convert_to_tensor([o], dtype=tf.float32)
                    a = tf.convert_to_tensor([a], dtype=tf.float32)
                    r = tf.convert_to_tensor(r, dtype=tf.float32)
                    at1 = self.actor_target.model(n_o, training=False)
                    at1 = tf.convert_to_tensor([at1], dtype=tf.float32)
                    qt1 = self.critic_target.model([n_o, at1], training=False)
                    yt = r + gamma * qt1

                    # with tf.GradientTape() as tape:
                    #     # yt = r + qt1  # episodic reward
                    #     qt = self.critic.model([o, a], training=True)
                    #     q_loss = mse_loss(yt, qt)
                    # grads = tape.gradient(q_loss, var_list_fn(self.critic))
                    # with tf.GradientTape(persistent=True) as tape:
                    #     a_for_grad = self.actor.model(o, training=True)
                    #     q_for_grad = self.critic.model([o, a_for_grad], training=True)
                    # grad_q = tape.gradient(q_for_grad, a_for_grad)
                    # grad_a = tape.gradient(a_for_grad, var_list_fn(self.actor), -grad_q)
                    # del tape
                    grads, grad_a, q_loss = self.compute_grad(o, a, yt)
                    critic_loss.append(q_loss)
                    q_grads.append(grads)  # (step, 15)
                    a_grads.append(grad_a)  # (step, 11)

                q_grads = np.transpose(q_grads)
                a_grads = np.transpose(a_grads)
                q_grads = q_grads.tolist()
                a_grads = a_grads.tolist()
                q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
                a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]

                batch_critic_loss.append(np.mean(critic_loss))
                batch_critic_grads.append(q_grads)  # (batch_size, 15)
                batch_actor_grads.append(a_grads)  # (batch_szie, 11)

            # tt0 = time.time()
            # print("end batch compute : ", tt0 - t0)
            q_grads = np.transpose(batch_critic_grads)
            a_grads = np.transpose(batch_actor_grads)
            q_grads = q_grads.tolist()
            a_grads = a_grads.tolist()
            q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
            a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
            tf.summary.scalar("critic loss", np.mean(batch_critic_loss), step=global_step)
            self.critic.optimizer.apply_gradients(zip(q_grads, var_list_fn(self.critic)))
            self.actor.optimizer.apply_gradients(zip(a_grads, var_list_fn(self.actor)))
            # tt1 = time.time()
            # print("end optimize : ", tt1 - tt0)
            self.soft_target_update()
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')
        # print("Top 10")
        # for stat in top_stats[:10]:
        #     print(stat)

    @tf.function
    def compute_grad(self, o, a, yt):
        with tf.GradientTape() as tape:
            # yt = r + qt1  # episodic reward
            qt = self.critic.model([o, a], training=True)
            q_loss = mse_loss(yt, qt)
        grads = tape.gradient(q_loss, var_list_fn(self.critic))
        with tf.GradientTape(persistent=True) as tape:
            a_for_grad = self.actor.model(o, training=True)
            a_for_grad = tf.convert_to_tensor([a_for_grad])
            q_for_grad = self.critic.model([o, a_for_grad], training=True)
        grad_q = tape.gradient(q_for_grad, a_for_grad)
        grad_a = tape.gradient(a_for_grad, var_list_fn(self.actor), -grad_q)
        del tape
        return grads, grad_a, float(q_loss)
