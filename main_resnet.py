import os
import datetime

import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from custom_env.envs import CustomEnv
from networks.resnet import Resnet
from dataset import Sal360
from utils.config import *

max_ep_length = 5_000


def test_gen(test_range):
    env = CustomEnv(video_type="test")
    for _ in range(test_range):
        ob, ac, target_video = env.reset(video_type="test", trajectory=True, inference=False, fx=1, fy=1)
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done:
                break
            yield np.array([ob]), np.array([ac])
            ob = next_ob
            ac = next_ac


def val_gen():
    env = CustomEnv(video_type="validation")
    while True:
        ob, ac, target_video = env.reset(video_type="validation", trajectory=True, inference=False, fx=1, fy=1)
        while True:
            next_ob, reward, done, next_ac = env.step(ac)
            if done:
                break
            yield np.array([ob]), np.array([ac])
            ob = next_ob
            ac = next_ac


def train_gen():
    env = CustomEnv(video_type="train")
    while True:
        ob, ac, target_video = env.reset(video_type="train", trajectory=True, inference=False, fx=1, fy=1)
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
    early_stopper = tf.keras.callbacks.EarlyStopping(min_delta=0.0001, patience=50)

    model.model.fit_generator(train_gen(), 99, 5000, 2, validation_data=val_gen(), validation_steps=99, shuffle=False,
                              callbacks=[tensor_boarder, model_saver, early_stopper])

    size = 10
    model.model.evaluate_generator(test_gen(size), 99, [tensor_boarder_test])

    env = CustomEnv(video_type="test")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    out = os.path.join(output_path, "supervised")
    if not os.path.exists(out):
        os.mkdir(out)

    for i in range(10):
        ob, ac, target_video = env.reset(video_type="test", trajectory=True, inference=False, fx=1, fy=1)

        writer = cv2.VideoWriter(os.path.join(output_path, target_video + "_" + str(now) + ".mp4"),
                                 fourcc, fps[target_video], (3840, 1920))
        while True:
            pred_ac = model.model.predict(np.array([ob]))
            next_ob, reward, done, next_ac = env.step(pred_ac)
            if done:
                break
            env.render(writer=writer)
            a, b = np.array([ob]), np.array([ac])
            ob = next_ob
            ac = next_ac

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


class EvalutateCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
