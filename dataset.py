import csv
import os
import random
import pprint

import tensorflow as tf
import cv2
import math
import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from utils.viewport import Viewport
from utils.config import scanpaths_H_path, video_path
from utils.dataset_utils import *
from utils.binary import *

width = 3840
height = 1920

# all sequence -> 20s
fps_list = {'01_PortoRiverside.mp4': 25, '02_Diner.mp4': 30, '03_PlanEnergyBioLab.mp4': 25,
            '04_Ocean.mp4': 30, '05_Waterpark.mp4': 30, '06_DroneFlight.mp4': 25, '07_GazaFishermen.mp4': 25,
            '08_Sofa.mp4': 24, '09_MattSwift.mp4': 30, '10_Cows.mp4': 24, '11_Abbottsford.mp4': 30,
            '12_TeatroRegioTorino.mp4': 30, '13_Fountain.mp4': 30, '14_Warship.mp4': 25, '15_Cockpit.mp4': 25,
            '16_Turtle.mp4': 30, '17_UnderwaterPark.mp4': 30, '18_Bar.mp4': 25, '19_Touvet.mp4': 30}


def regular_action(action_data):
    x_val, y_val = action_data
    x_val = x_val if abs(x_val) > 0.01 else 0
    y_val = y_val if abs(y_val) > 0.01 else 0
    return x_val, y_val


TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


class Sal360:
    def __init__(self):

        self.train, self.validation, self.test = self.load_sal360v2()

        self.target_videos = os.listdir(video_path)
        self.train_video_path = os.path.join(video_path, "train", "3840x1920")
        self.test_video_path = os.path.join(video_path, "test", "3840x1920")

        self.saliency_info = get_SalMap_info()

        self.time_step = 0

        self.video = None
        self.saliency_map = None

    def load_sal360v2(self):

        data = []
        actions = []
        data_dict = {}
        for file in sorted(os.listdir(scanpaths_H_path)):
            data_dict[file] = []
            with open(os.path.join(scanpaths_H_path, file)) as f:
                row = csv.reader(f)
                data.append([list(map(lambda x: [float(i) for i in x], list(row)[1:]))])

        np_data = np.array(data).reshape((-1, 7))
        # np_data = [i[6] - i[5] for i in np_data]
        # np4 = [i == 4.0 for i in np_data] 5 --> 34336
        # np5 = [i == 5.0 for i in np_data] 6 --> 42542
        # np6 = [i == 6.0 for i in np_data] 7 --> 6
        # np7 = [i == 7.0 for i in np_data] 8 --> 7
        for idx in range(len(np_data)):
            if np_data[idx][0] == 99.:
                action = (0, 0)
            else:
                action = (np_data[idx + 1][2] - np_data[idx][2], np_data[idx + 1][1] - np_data[idx][1])
            actions.append(action)

        actions = np.array(actions)

        np_data = np_data.reshape((19, 57, 100, 7))

        actions = actions.reshape((19, 57, 100, 2))

        _x_train, x_test = np_data[:14, :, :, :], np_data[14:, :, :, :]
        _y_train, y_test = actions[:14, :, :, :], actions[14:, :, :, :]
        x_train, x_validation = _x_train[:, :45, :, :], _x_train[:, 45:, :, :]
        y_train, y_validation = _y_train[:, :45, :, :], _y_train[:, 45:, :, :]

        print(np.shape(x_train), np.shape(y_train))
        print(np.shape(x_validation), np.shape(y_validation))
        print(np.shape(x_test), np.shape(y_test))

        x_train_dict, y_train_dict = {}, {}
        x_val_dict, y_val_dict = {}, {}
        x_test_dict, y_test_dict = {}, {}

        train_length = len(x_train)
        for index, file in enumerate(sorted(os.listdir(scanpaths_H_path))):
            file = (file.split(".")[0] + ".mp4").replace("_fixations", "")
            # file = (file[0] + ".mp4").replace("_fixations", "")
            # file = file.replace("_fixations", "")
            if index < train_length:
                """train data"""
                x_train_dict[file] = x_train[index]
                y_train_dict[file] = y_train[index]
                """validation data"""
                x_val_dict[file] = x_validation[index]
                y_val_dict[file] = y_validation[index]
            else:
                """"test data"""
                x_test_dict[file] = x_test[index - train_length]
                y_test_dict[file] = y_test[index - train_length]
        # print("shape : (# of video, # of person, # of data per video, # of data)")
        # print("shape of train set x, y : ", x_train.shape, y_train.shape)
        # print("shape of validation set x, y : ", x_validation.shape, y_validation.shape)
        # print("shape of test set x, y : ", x_test.shape, y_test.shape)
        return (x_train_dict, y_train_dict), (x_val_dict, y_val_dict), (x_test_dict, y_test_dict)

    def plot_state_data(self, data):
        assert np.shape(data) == 25, 2
        data = np.transpose(data)
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        ax.plot(data[0], data[1], range(100))
        plt.title("test")
        plt.show()

    def kl_divergence(self):

        state = np.array([(i[1] * 3840, i[2] * 1920) for i in self.data]).reshape([19, 57, 100, 2])
        state_prob = np.array([(i[1], i[2]) for i in self.data]).reshape([19, 57, 100, 2])

        base = state[0][0]  # 100, 2
        base_T = state[0][0].transpose()  # 2, 100
        base_prob = state_prob[0][0]
        base_prob_T = state_prob[0][0].transpose()

        def l2norm(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        for i, ii in zip(state_prob[0], state[0]):  # 57 2 100  # KLdivergence, L2norm
            i_T = i.transpose()
            distance = [l2norm(p1, p2) for p1, p2 in zip(base, ii)]

            # kld_x = stats.entropy(pk=base_prob_T[0], qk=i_T[0])
            # kld_y = stats.entropy(pk=base_prob_T[1], qk=i_T[1])

            # print("KL divergence (x, y) : ({}, {})".format(kld_x, kld_y))
            # print("avg L2 norm : {}".format(min(distance)))

    def next(self, type="train"):
        pass

    def choice_dir(self, s):
        if s == TRAIN:
            path = self.train_video_path
            x_dict, y_dict = self.train
        elif s == VALIDATION:
            path = self.train_video_path
            x_dict, y_dict = self.validation
        elif s == TEST:
            path = self.test_video_path
            x_dict, y_dict = self.test
        else:
            raise ValueError(f"train type must be train, validation or test but got {s}")

        return x_dict, y_dict, path

    def select_trajectory(self, mode="train", target_video=None, randomness=True):
        self.time_step = 0
        if target_video is None:
            if mode == TRAIN:
                self.target_video = random.choice(os.listdir(self.train_video_path))
            elif mode == TEST:
                self.target_video = random.choice(os.listdir(self.test_video_path))
            elif mode == VALIDATION:
                self.target_video = random.choice(os.listdir(self.train_video_path))
        else:
            self.target_video = target_video

        x_dict, y_dict, path = self.choice_dir(mode)

        if randomness:
            ran_idx = random.randint(0, len(x_dict[self.target_video]) - 1)
            ran_x, ran_y = x_dict[self.target_video][ran_idx], y_dict[self.target_video][ran_idx]
            self.x_iter, self.y_iter = iter(ran_x), iter(ran_y)
            self.video = self.get_video(path)
            self.saliency_map = self.get_saliency_map()
            return self.target_video
        else:
            if self.video is None:
                self.video = self.get_video(path, fx=0.3, fy=0.3)
                print(np.shape(self.video))
                # self.saliency_map = self.get_saliency_map()
                self.x_data, self.y_data = iter(x_dict[self.target_video]), iter(y_dict[self.target_video])

            try:
                self.x_iter, self.y_iter = iter(next(self.x_data)), iter(next(self.y_data))
            except StopIteration:
                self.x_data, self.y_data = iter(x_dict[self.target_video]), iter(y_dict[self.target_video])  # ring
                self.x_iter, self.y_iter = iter(next(self.x_data)), iter(next(self.y_data))

            return self.target_video

    def get_video(self, path, fx=0.3, fy=0.3):
        cap = cv2.VideoCapture(os.path.join(path, self.target_video))
        return read_whole_video(cap, fx=fx, fy=fy)

    def get_saliency_map(self):
        return read_SalMap(self.saliency_info[self.target_video])

    def get_expert_trajectory(self, target_video, mode="train"):
        self.target_video = target_video

        # self.saliency_map = self.get_saliency_map()
        total_ob, total_ac, total_done = [], [], []
        while True:
            try:
                obs, acs, rewards, dones = [], [], [], []

                # reset env
                self.select_trajectory(target_video=target_video, randomness=False)
                state, _, lat, lng, ac, done = self.next_data(trajectory=True)

                view = Viewport(3840 * 0.3, 1920 * 0.3)
                view.set_center((lat, lng), normalize=True)
                ob = [cv2.resize(view.get_view(f), (84, 84)) for f in state]

                obs.append(ob)
                acs.append(ac)
                dones.append(done)

                while True:
                    next_state, _, lat, lng, next_ac, done = self.next_data(trajectory=True)
                    next_ob = [cv2.resize(view.get_view(f), (84, 84)) for f in next_state]

                    view.set_center((lat, lng), normalize=True)

                    obs.append(next_ob)
                    acs.append(next_ac)
                    dones.append(done)
                    if done:
                        break

                obs = tf.keras.preprocessing.sequence.pad_sequences(obs, padding='post', value=256, maxlen=8)
                total_ob.append(obs)
                total_ac.append(acs)
                total_done.append(dones)
            except StopIteration:
                print(np.shape(total_ob), np.shape(total_ac), np.shape(total_done))
                return total_ob, total_ac, total_done

    def next_data(self, trajectory=True):
        x_data, y_data = next(self.x_iter), next(self.y_iter)
        lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])
        if trajectory:
            done = True if x_data[0] == 99 else False
            self.state = self.video[start_frame - 1:end_frame]
            if self.saliency_map is not None:
                self.saliency_state = self.saliency_map[start_frame - 1:end_frame]
            else:
                self.saliency_state = None
            return self.state, self.saliency_state, lat, lng, y_data, done
        else:
            frame_step = fps_list[self.target_video]
            self.state = self.video[self.time_step:self.time_step + frame_step]
            self.saliency_state = self.saliency_map[self.time_step:self.time_step + frame_step]
            self.time_step += frame_step
            done = True if frame_step * 20 == self.time_step else False
            return self.state, self.saliency_state, lat, lng, y_data, done


# class DataGenerator:
#     # train, validation, test = Sal360.load_sal360_dataset()
#     train, validation, test = Sal360.load_sal360v2()
#
#     @staticmethod
#     def generator(img_w, img_h, type="train", resolution='3840x1920', tiling=False, return_batch=True, normalize=False):
#         if type == "train":
#             gen = DataGenerator.train
#             videos = os.listdir(os.path.join('sample_videos', type, resolution))
#             videos = videos[:15]
#         elif type == "test":
#             gen = DataGenerator.test
#             videos = os.listdir(os.path.join('sample_videos', type, resolution))
#             videos = videos[15:]
#         elif type == "validation":
#             gen = DataGenerator.validation
#             type = 'train'  # train과 validation의 폴더는 같아
#             videos = os.listdir(os.path.join('sample_videos', type, resolution))
#             videos = videos[:15]
#         else:
#             raise ValueError("invalid value(train, test, validation)")
#
#         width, height = list(map(lambda x: int(x), resolution.split('x')))
#         view = Viewport(width, height)
#         x_dict, y_dict = gen[0], gen[1]
#
#         idx = 0
#         while True:
#             video = random.choice(videos)
#             random_x, random_y = random.choice(x_dict[video]), random.choice(y_dict[video])  # 100, 7
#             video_path = os.path.join('sample_videos', type, resolution, video)
#             cap = cv2.VideoCapture(video_path)
#             # trace_length = fps_list[video]
#             batch_x, batch_y = None, []
#             actions = np.array([0., 0.])
#             for idx, random_data in enumerate(zip(random_x, random_y)):  # _x : 7,  _y : 2,
#                 _x, _y = random_data
#                 frame_idx = _x[6] - _x[5] + 1
#                 frames = []
#                 actions += _y
#                 if (idx is not 0 and idx % 5 is 0) or idx is 99:
#                     batch_y.append(actions)
#                     actions = np.array([0., 0.])
#
#                 while len(frames) < frame_idx and cap.isOpened():
#                     ret, frame = cap.read()
#                     if ret:
#                         w, h = _x[1] * width, _x[2] * height
#                         view.set_center(np.array([w, h]))
#                         frame = view.get_view(frame)
#                         frame = cv2.resize(frame, (img_w, img_h))
#                         frames.append(frame)
#                         if len(frames) == frame_idx:
#                             if batch_x is None:
#                                 batch_x = frames
#                             else:
#                                 batch_x = np.concatenate((batch_x, frames))
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
#                         # if index == frame_idx:
#                         #     w, h = _x[1] * width, _x[2] * height
#                         #     view.set_center(np.array([w, h]))
#                         #     frame = view.get_view(frame)
#                         #     frame = cv2.resize(frame, (img_w, img_h))
#                         #
#                         #     sequenceX.append(frame)
#                         #
#                         #     if tiling:
#                         #         tiles = view.find_tile()
#                         #     else:
#                         #         if not normalize:
#                         #             _y = np.array([_y[0] * width, _y[1] * height], dtype=np.float64)
#                         #         __y = [0] * 4
#                         #
#                         #         __y[0] = _y[0] if _y[0] >= 0 else 0
#                         #         __y[1] = _y[0] * -1 if _y[0] < 0 else 0
#                         #
#                         #         __y[2] = _y[1] if _y[1] >= 0 else 0
#                         #         __y[3] = _y[1] * -1 if _y[1] < 0 else 0
#                         #         sequenceY.append(__y)
#                         #
#                         #     index = 0
#                         #     # if len(sequenceX) == trace_length:
#                         #     if tiling:
#                         #         yield np.array([sequenceX]), np.array([tiles])
#                         #     else:
#                         #         batch_x.append(sequenceX)
#                         #         batch_y.append(sequenceY)
#                         #         yield np.array([sequenceX]), np.array([sequenceY])
#                         #         sequenceY = []
#                         #     sequenceX = []
#                         #     continue
#                     else:
#                         if len(frames) != 0:
#                             batch_x = np.concatenate((batch_x, frames))
#                         break
#
#             cap.release()
#             cv2.destroyAllWindows()
#
#             error = len(batch_x) - fps_list[video] * 20
#             if error:
#                 batch_x = batch_x[:len(batch_x) - error]
#
#             batch_x = np.reshape(batch_x, (20, -1, img_w, img_h, 3))
#
#             for x_data, y_data in zip(batch_x, batch_y):
#                 y = [0] * 4
#
#                 if y_data[0] >= 0:
#                     y[0] = 1
#                 else:
#                     y[1] = 1
#                 if y_data[1] >= 0:
#                     y[2] = 1
#                 else:
#                     y[3] = 1
#                 yield np.array([x_data]), np.array([y])
#
#     @staticmethod
#     def generator_for_batch(img_w, img_h, type="train", resolution='3840x1920', tiling=False, return_batch=True,
#                             normalize=False):
#         if type == "train":
#             gen = DataGenerator.train
#             dir_path = "train"
#             videos = os.listdir(os.path.join('sample_videos', dir_path, resolution))
#             videos = videos[:15]
#         elif type == "test":
#             gen = DataGenerator.test
#             dir_path = "test"
#             videos = os.listdir(os.path.join('sample_videos', dir_path, resolution))
#             videos = videos[15:]
#         elif type == "validation":
#             gen = DataGenerator.validation
#             dir_path = "train"
#             videos = os.listdir(os.path.join('sample_videos', dir_path, resolution))
#             videos = videos[:15]
#         else:
#             raise ValueError("invalid value(train, test, validation)")
#
#         width, height = list(map(lambda x: int(x), resolution.split('x')))
#         view = Viewport(width, height)
#         x_dict, y_dict = gen[0], gen[1]
#
#         while True:
#             """ random video select """
#             video = random.choice(videos)
#             random_idx = random.randint(0, len(x_dict[video]))
#             random_x, random_y = x_dict[video][random_idx], x_dict[video][random_idx]  # (99, 7), (99, 2)
#
#             video_path = os.path.join('sample_videos', dir_path, resolution, video)
#             cap = cv2.VideoCapture(video_path)
#             print(type, " video ", video)
#             for idx, random_data in enumerate(zip(random_x, random_y)):
#                 _x, _y = random_data  # _x : (7,)  _y : (2,)
#                 frame_idx = _x[6] - _x[5] + 1  # 5 ~ 8의 window size (time)
#                 frames = []  # store frame
#
#                 while len(frames) < frame_idx and cap.isOpened():
#                     ret, frame = cap.read()
#                     if ret:
#                         w, h = _x[1] * width, _x[2] * height
#                         view.set_center(np.array([w, h]))
#                         frame = view.get_view(frame)
#                         frame = cv2.resize(frame, (img_w, img_h))
#                         frames.append(frame)
#                         if len(frames) == frame_idx:
#                             yield np.array([frames]), np.array([_y])
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             break
#                     else:
#                         break
#
#             # 위치 중요
#             cap.release()
#             cv2.destroyAllWindows()


if __name__ == '__main__':
    a = Sal360()
    a.get_expert_trajectory(target_video="10_Cows.mp4")
    # gen = DataGenerator.generator_for_batch(224, 224, type='validation')
    # gen = DataGenerator.generator(224, 224)
    # for x, y in gen:
    pass
