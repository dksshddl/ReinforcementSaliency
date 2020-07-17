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
VALIDATION = "val"
TEST = "test"


class Sal360:
    def __init__(self):

        self.train, self.validation, self.test = self.load_sal360v2()

        self.target_videos = os.listdir(video_path)
        self.prev_video = None
        self.train_video_path = os.path.join(video_path, "train", "3840x1920")
        self.test_video_path = os.path.join(video_path, "test", "3840x1920")

        # self.train_video_path = os.path.join(video_path, "train", "320x160")
        # self.test_video_path = os.path.join(video_path, "test", "320x160")

        self.saliency_info = get_SalMap_info()

        self.time_step = 0

        self.video = None
        self.saliency_map = None
        self.target_video = None
        self.dict_index = 0
        self.index = 0

    # def get_expert_train(self, target_video, index):
    #     print("get expert")
    #     if self.video is None:
    #         self.video = self.get_video(self.train_video_path, 1, 1, target_video=target_video)
    #
    #     x_train, y_train = self.train[0][target_video], self.train[1][target_video]
    #     x, y = x_train[index], y_train[index]
    #     x_iter, y_iter = iter(x), iter(y)
    #     expert_x, expert_y = [], []
    #
    #     view = Viewport(3840, 1920)
    #     while True:
    #         try:
    #             x_data, y_data = next(x_iter), next(y_iter)
    #             lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])
    #             view.set_center((lat, lng), normalize=True)
    #             state = self.video[start_frame - 1:end_frame]
    #             ob = [cv2.resize(view.get_view(f), (160, 160)) for f in state]
    #             expert_x.append(ob)
    #             expert_y.append(y_data)
    #         except StopIteration:
    #             break
    #     print(np.shape(expert_x), np.shape(expert_y))
    #     return expert_x, expert_y

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

        # print(np.shape(x_train), np.shape(y_train))
        # print(np.shape(x_validation), np.shape(y_validation))
        # print(np.shape(x_test), np.shape(y_test))

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

    def scatter_data(self, data):
        data = np.transpose(data)

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

    def get_whole_data(self):
        x_train_data, y_train_data = [], []
        x_test_data, y_test_data = [], []

        self.train_video_path = os.path.join(video_path, "train", "320x160")
        self.test_video_path = os.path.join(video_path, "test", "320x160")

        train = sorted(os.listdir(self.train_video_path))
        test = sorted(os.listdir(self.test_video_path))
        print(train)
        print(test)
        index = 0
        for t in train:
            if index == 0:
                break
            sal = read_SalMap(self.saliency_info[t])
            v = self.get_video(self.train_video_path, 1, 1, t)
            if len(v) == 501:
                v = v[:-1]
                sal = sal[:-1]
            if len(v) == 601:
                v = v[:-1]
                sal = sal[:-1]
            shape = np.shape(v)
            width, height = shape[2], shape[1]
            y_data = [cv2.resize(f, (width, height)) for f in sal]
            del sal
            y_data = np.expand_dims(y_data, 3).astype(np.float)

            x_train_data.append(v)
            y_train_data.append(y_data)
            print(np.shape(y_data), np.shape(v), t)
            # index += 1
        t_idx = 0
        for t in test:
            if t_idx == 1:
                break
            t_idx += 1

            sal = read_SalMap(self.saliency_info[t])
            v = self.get_video(self.test_video_path, 1, 1, t)

            if len(v) == 501:
                v = v[:-1]
                sal = sal[:-1]
            if len(v) == 601:
                v = v[:-1]
                sal = sal[:-1]

            shape = np.shape(v)
            width, height = shape[2], shape[1]
            y_data = [cv2.resize(f, (width, height)) for f in sal]
            del sal
            y_data = np.expand_dims(y_data, 3).astype(np.float)

            x_test_data.append(v)
            y_test_data.append(y_data)
            print(np.shape(y_data), np.shape(v), t)

        return (np.array(x_train_data), np.array(y_train_data)), (np.array(x_test_data), np.array(y_test_data))

    def select_trajectory(self, fx, fy, mode="train", target_video=None, randomness=False, saliency=False,
                          trajectory=False):
        # pop_value = [1, 5, 9, 4, 7]
        pop_value = []
        self.index = 0
        self.time_step = 0
        x_dict, y_dict, path = self.choice_dir(mode)
        if target_video is not None:
            self.target_video = target_video
        else:
            p = os.listdir(list(x_dict.keys()))
            self.target_video = random.choice(p)
        if trajectory:
            if randomness:
                ran_idx = random.randint(0, len(x_dict[self.target_video]) - 1)
                ran_x, ran_y = x_dict[self.target_video][ran_idx], y_dict[self.target_video][ran_idx]
                self.x_iter, self.y_iter = iter(ran_x), iter(ran_y)
                if self.prev_video == self.target_video:
                    pass
                else:
                    self.video = self.get_video(path, fx, fy)
                    if saliency:
                        self.saliency_map = self.get_saliency_map()
                return self.target_video
            else:
                self.index = 0
                self.dict_index += 1

                if self.dict_index >= 57:
                    self.dict_index = 0

                if self.target_video == self.prev_video:
                    self.x_data, self.y_data = x_dict[self.target_video][self.dict_index], y_dict[self.target_video][
                        self.dict_index]
                else:
                    print("new video!")
                    self.prev_video = self.target_video
                    self.video = self.get_video(path, fx, fy)
                    self.x_data, self.y_data = x_dict[self.target_video][self.dict_index], y_dict[self.target_video][
                        self.dict_index]
                # print(self.prev_video, self.target_video, np.shape(self.x_data), self.dict_index)

                if saliency:
                    self.saliency_map = self.get_saliency_map()
                else:
                    self.saliency_map = None
                return self.target_video
        else:
            if self.prev_video == self.target_video:
                pass
            else:
                self.video = self.get_video(path, fx, fy)
                self.saliency_map = self.get_saliency_map()
            return self.target_video

    def get_video(self, path, fx=0.3, fy=0.3, target_video=None):
        if target_video is None:
            target_video = self.target_video
        cap = cv2.VideoCapture(os.path.join(path, target_video))
        return read_whole_video(cap, fx=fx, fy=fy)

    def get_saliency_map(self):
        return read_SalMap(self.saliency_info[self.target_video])

    def get_data(self, trajectory, saliency, inference, mode="train", target_video=None):
        self.select_trajectory(fx=1, fy=1, trajectory=trajectory, saliency=saliency, target_video=target_video,
                               mode=mode)
        if mode == "train":
            data = self.train[0][self.target_video]
        elif mode == "val":
            data = self.validation[0][self.target_video]
        elif mode == "test":
            data = self.test[0][self.target_video]
        else:
            raise ValueError(f"mode must be train, validation and test but got {mode}")
        return self.video, data

    def get_expert_trajectory(self, target_video, mode="train"):
        # self.target_video = target_video

        # self.saliency_map = self.get_saliency_map()
        total_ob, total_ac, total_done = [], [], []
        print(np.shape(total_ob), np.shape(total_ac), np.shape(total_done))
        while True:
            try:
                obs, acs, rewards, dones = [], [], [], []

                # reset env
                self.select_trajectory(1, 1, target_video=target_video, randomness=False)
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
                # obs = tf.keras.preprocessing.sequence.pad_sequences(obs, padding='post', value=256, maxlen=8)
                total_ob.append(obs)
                total_ac.append(acs)
                total_done.append(dones)
            except StopIteration:
                print(np.shape(total_ob), np.shape(total_ac), np.shape(total_done))
                return total_ob, total_ac, total_done

    def next_data(self, trajectory=True):
        if trajectory:
            x_data, y_data = self.x_data[self.index], self.y_data[self.index]
            self.index += 1
            lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])
            done = True if int(x_data[0]) == 99 else False
            state = self.video[start_frame - 1:end_frame]
            if self.saliency_map is not None:
                saliency_state = self.saliency_map[start_frame - 1:end_frame]
            else:
                saliency_state = None
            return state, saliency_state, lat, lng, y_data, done
        else:
            fps = fps_list[self.target_video]
            frame_step = 5
            state = self.video[self.time_step:self.time_step + frame_step]
            # self.video = self.video[frame_step:]
            # if self.saliency_map is not None:
            saliency_state = self.saliency_map[self.time_step:self.time_step + frame_step]
            # self.saliency_map = self.saliency_map[frame_step:]
            # else:
            #     saliency_state = None
            self.time_step += frame_step
            # done = len(self.video) < 5 and len(state) == 5
            done = True if fps * 20 == self.time_step else False
            # return self.state, self.saliency_state, lat, lng, y_data, done
            return state, saliency_state, None, None, None, done


if __name__ == '__main__':
    a = Sal360()

    trainset = a.train[0]
    testtset = a.test[0]

    for v in os.listdir(a.train_video_path): # 45 100
        tt = trainset[v]
        print(np.shape(tt))
        n = v.split(".")[0]
        plt.title(f"{n}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("longitude (x)")
        plt.ylabel("latitude (y)")
        for tra in tt: # 100 7
            tra = np.array(tra)
            print(np.shape(tra[:, 1]))
            plt.scatter(tra[:, 1], tra[:, 2])
        plt.show()



    # target_videos = sorted(os.listdir(os.path.join(video_path, "train", "3840x1920")))
    # x_dict, y_dict = a.train
    # standard = {}
    # # item = [v for k, v in y_dict.items()]
    #
    # # item = np.reshape(item, [-1, 2])
    # f = open("val.txt", "a")
    # for video in target_videos:
    #     plt.figure()
    #     plt.title(video)
    #     # plt.xlim(xmin=-1, xmax=1)
    #     # plt.ylim(ymin=-1, ymax=1)
    #
    #     x = np.array(x_dict[video])  # 45 7
    #     fig_lat = plt.figure()
    #     fig_lng = plt.figure()
    #
    #     for data in x:
    #         lat, lng = x[2], x[1]
    #         plt.figu
    #     # mean = np.mean(item, axis=0)
    #     # std = np.std(item, axis=0)
    #     # f.write(f"{video} {mean} {std}\n")
    #
    #     for action in y_dict[video]:
    #         # fnc = preprocessing(mean, std)
    #         acs = list(map(fnc, action))
    #         acs = np.array(acs)
    #         plt.scatter(acs[:, 0], acs[:, 1])  # x, y
#         # plt.savefig(os.path.join("plot", "trajectory", video + "_std_global.png"), format="png")
#     f.close()
#     # a.get_expert_trajectory(target_video="10_Cows.mp4")
#     # gen = DataGenerator.generator_for_batch(224, 224, type='validation')
#     # gen = DataGenerator.generator(224, 224)
#     # for x, y in gen:
#     pass
