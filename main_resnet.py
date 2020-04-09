import os
import datetime

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from custom_env.envs import CustomEnv
from networks.resnet import Resnet
from dataset import Sal360
from utils.config import *

max_ep_length = 5_000


# 이거 쓰면 thread not safe 에러 --> stackoverflow에서 내부로유래
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, video_type="train"):
#         self.env = CustomEnv(video_type=video_type)
#
#         ob, ac, target_video = self.env.reset(video_type=video_type, trajectory=True, inference=False, fx=1, fy=1,
#                                               saliency=True)
#         self.video_type = video_type
#         self.ac = ac
#         self.history = []
#
#     def __len__(self):
#         return 98
#
#     def __getitem__(self, item):
#         next_ob, reward, done, next_ac = self.env.step(self.ac)
#         self.ac = next_ac
#         return np.array([next_ob]), np.array([next_ac])
#
#     def on_epoch_end(self):
#         ob, ac, target_video = self.env.reset(video_type=self.video_type, trajectory=True, inference=False, fx=1, fy=1,
#                                               saliency=True)
#         self.ac = ac


def test_gen(test_range):
    env = CustomEnv(video_type="test")
    for _ in range(test_range):
        ob, ac, target_video = env.reset(video_type="test", trajectory=True, inference=False, fx=0.3, fy=0.3,
                                         saliency=False)
        val = 1
        # history = ob
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done or val == 99:
                break
            yield np.array([ob]), np.array([ac])
            val += 1
            # history = np.concatenate([history, next_ob])
            ac = next_ac
            ob = next_ob


def val_gen():
    env = CustomEnv(video_type="validation")
    while True:
        ob, ac, target_video = env.reset(video_type="validation", trajectory=True, inference=False, fx=0.3, fy=0.3,
                                         saliency=False)
        # history = ob
        val = 1
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done or val == 99:
                break
            # history = np.concatenate([history, next_ob])
            yield np.array([ob]), np.array([ac])
            val += 1
            ac = next_ac
            ob = next_ob


def train_gen():
    env = CustomEnv(video_type="train")
    while True:
        ob, ac, target_video = env.reset(video_type="train", trajectory=True, inference=False, fx=0.3, fy=0.3,
                                         saliency=False)
        val = 1
        # history = ob
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done or val == 99:
                break
            yield np.array([ob]), np.array([ac])
            val += 1
            # history = np.concatenate([history, next_ob])
            ac = next_ac
            ob = next_ob


def learn():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    writer_path = os.path.join("log", "resnet")

    custom_env = CustomEnv("test")
    state_dim = list(custom_env.observation_space.shape)
    action_dim = list(custom_env.action_space.shape)

    model = Resnet(state_dim, session)

    tf_writer = tf.summary.FileWriter(writer_path, session.graph)

    tensor_board_path = os.path.join(log_path, "supervised")
    tensor_board_test_path = os.path.join(tensor_board_path, "test")
    model_weight_path = os.path.join(weight_path, "supervised")

    if not os.path.exists(tensor_board_path):
        os.mkdir(tensor_board_path)
    if not os.path.exists(tensor_board_test_path):
        os.mkdir(tensor_board_test_path)
    if not os.path.exists(model_weight_path):
        os.mkdir(model_weight_path)

    tensor_boarder = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "supervised"), write_graph=True,
                                                    batch_size=99,
                                                    update_freq="batch")
    tensor_boarder_test = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "supervised", "test"),
                                                         batch_size=99,
                                                         update_freq="batch")
    model_saver = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_weight_path, "model.ckpt"), monitor="val_loss", save_best_only=True)

    # early_stopper = tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=50)

    model.model.fit_generator(train_gen(), 99, 5000, validation_data=val_gen(), validation_steps=99, shuffle=False,
                              callbacks=[tensor_boarder, model_saver])

    size = 10
    model.model.evaluate_generator(test_gen(size), [tensor_boarder_test])


def test():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    custom_env = CustomEnv("test")
    state_dim = list(custom_env.observation_space.shape)
    action_dim = list(custom_env.action_space.shape)

    model = Resnet(state_dim, session)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    out = os.path.join(output_path, "supervised")
    if not os.path.exists(out):
        os.mkdir(out)

    tensor_boarder_test = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "supervised", "test"),
                                                         batch_size=99,
                                                         update_freq="batch")
    size = 10
    model.model.evaluate_generator(test_gen(size), steps=99, callbacks=[tensor_boarder_test])

    for i in range(10):
        obs, acs, target_videos = custom_env.reset(video_type="test", trajectory=True, inference=True, fx=1, fy=1)

        writer = cv2.VideoWriter(os.path.join(out, target_videos + "_" + str(now) + ".mp4"),
                                 fourcc, fps[target_videos], (3840, 1920))
        # history = obs
        true_acs = []
        pred_acs = []

        while True:
            pred_ac = model.model.predict(np.array([obs]))
            # pred_acs.append(pred_ac)
            # true_acs.append(acs)
            next_obs, rewards, dones, next_acs = custom_env.step(pred_ac)
            # print(pred_ac, acs)
            if dones:
                break
            custom_env.render(writer=writer)
            obs = next_obs
            acs = next_acs
            # history = np.concatenate([history, next_obs])
        writer.release()
        # true_acs = np.reshape(true_acs, [-1, 2]).transpose()
        # pred_acs = np.reshape(pred_acs, [-1, 2]).transpose()
        # fig = plt.figure()
        # fig2 = plt.figure()
        # ax = Axes3D(fig)
        # ax.set_xlabel('X axis')
        # ax.set_ylabel('Y axis')
        # ax.set_zlabel('Z axis')
        # ax.plot(true_acs[0], true_acs[1])
        # ax.set_title("true")
        # ax2 = Axes3D(fig2)
        # ax2.set_xlabel('X axis')
        # ax2.set_ylabel('Y axis')
        # ax2.set_zlabel('Z axis')
        # ax2.plot(pred_acs[0], pred_acs[1])
        # ax2.set_title("predict")
        # plt.show()



if __name__ == '__main__':
    # for x, y in train_gen():
    #     print(np.shape(x), np.shape(y))

    learn()
    # test()
    # for i, (x, y) in enumerate(train_gen()):
    #     losss = model.model.train_on_batch(x, y)
    #     print(f"loss: {losss})
    #     reward_summary = tf.Summary(value=[tf.Summary.Value(tag="train_loss", simple_value=losss)])
    #     tf_writer.add_summary(reward_summary, i)
    #     if i % 30 == 0:
    #         model.save(step=i)

    # train = DataGenerator("train")
    # val = DataGenerator("validation")
    # test = DataGenerator("test")

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


class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class AccuracyHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EvalutateCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
