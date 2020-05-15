import os
import datetime
import copy
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
stand_dict = {
    "01_PortoRiverside.mp4": [[-0.00014235, 0.00015636], [0.01457013, 0.12722483]],
    "02_Diner.mp4": [[0.00035419, 0.00058077], [0.01710453, 0.13510216]],
    "03_PlanEnergyBioLab.mp4": [[1.59446661e-04, 2.62226536e-05], [0.01516597, 0.13287633]],
    "04_Ocean.mp4": [[-0.00010823, 0.00070909], [0.02380825, 0.14271857]],
    "05_Waterpark.mp4": [[4.19333558e-04, -6.62591863e-05], [0.01721657, 0.11431039]],
    "06_DroneFlight.mp4": [[0.00096727, -0.00010646], [0.02123786, 0.11721233]],
    "07_GazaFishermen.mp4": [[8.66520711e-05, -1.80686089e-03], [0.0256012, 0.13320691]],
    "08_Sofa.mp4": [[0.00014075, 0.00061949], [0.02242027, 0.11336181]],
    "09_MattSwift.mp4": [[-0.00080271, -0.0011968], [0.0249275, 0.13625535]],
    "10_Cows.mp4": [[-0.00026264, 0.00067812], [0.01957265, 0.12899429]],
    "11_Abbottsford.mp4": [[0.00067969, 0.00065381], [0.02147157, 0.14254084]],
    "12_TeatroRegioTorino.mp4": [[7.01761207e-05, -2.21156444e-04], [0.01814762, 0.1319638]],
    "13_Fountain.mp4": [[0.00071678, 0.00064842], [0.02222087, 0.13347862]],
    "14_Warship.mp4": [[-0.00022952, -0.00096409], [0.02872411, 0.09542387]],
}


def preprocessing(action):
    action[0] *= 5
    action[1] = action[1] * 4 if -0.25 < action[1] < 0.25 else action[1] * 2
    return np.array([action])


def depreprocessing(aa):
    aa[1] /= 5
    aa[1] = aa / 2 if aa[1] >= 1 else aa / 4
    return np.array([aa])


# def preprocessing(mean, std):
#     def function(action):
#         action = np.array(action)
#         action = (action - mean) / std
#         return action
#     return function
#
# def depreprocessing(mean, std):
#     def function(action):
#         action = np.array(action)
#         action = (action * std) + mean
#         return action
#     return function

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
            yield np.array([ob]), preprocessing(ac)
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
            yield np.array([ob]), preprocessing(ac)
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
            yield np.array([ob]), preprocessing(ac)
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
    # model.restore()
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
    model.set_weight()
    model_saver = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_weight_path, "model.ckpt"), save_weights_only=True)

    early_stopper = tf.keras.callbacks.EarlyStopping(min_delta=0.001, patience=5)

    model.model.fit_generator(train_gen(), 99, 5000, validation_data=val_gen(), validation_steps=99, shuffle=False,
                              callbacks=[tensor_boarder, model_saver, early_stopper])

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
    model.set_weight()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    out = os.path.join(output_path, "supervised")
    if not os.path.exists(out):
        os.mkdir(out)

    tensor_boarder_test = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(log_path, "supervised", "test"),
                                                         batch_size=99,
                                                         update_freq="batch")
    size = 10
    # model.model.evaluate_generator(test_gen(size), steps=99, callbacks=[tensor_boarder_test])

    for i in range(10):
        obs, acs, target_videos = custom_env.reset(video_type="test", trajectory=True, inference=True, fx=1, fy=1)

        writer = cv2.VideoWriter(os.path.join(out, target_videos + "_" + str(now) + ".mp4"),
                                 fourcc, fps[target_videos], (3840, 1920))
        # history = obs
        true_acs = []
        pred_acs = []

        while True:
            pred_ac = model.model.predict(np.array([obs]))
            print(pred_ac, acs)
            # pred_ac = depreprocessing(pred_ac)
            # print(pred_ac, acs)
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

    # learn()
    test()
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


class ModelSaver(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class EvalutateCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
