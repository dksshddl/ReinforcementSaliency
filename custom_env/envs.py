import os
import heapq

import gym
from gym import spaces

from utils.dataset_utils import read_whole_video
from dataset import *
from utils.binary import *
from utils.config import *
from utils.equirectangle import Equirectangular, NFOV
from utils.replay import ReplayBuffer
from utils.viewport import TileViewport
import matplotlib.pyplot as plt

n_samples = 8
width = 224
height = 224
n_channels = 3

action_arr = [
    [0, 0],
    [0.0275, 0],
    [-0.0275, 0],
    [0, 0.035],
    [0, -0.035]
]


def mean(arr):
    return np.mean(arr)


def depreprocessing(action):
    action[1] /= 5
    action[1] = action / 2 if action[1] >= 1 else action / 4
    return np.array([action])


class CustomEnv(gym.Env):

    def __init__(self, video_type="train"):
        self.dataset = Sal360()
        self.width, self.height = 3840, 1920
        # discrete
        # self.action_space = spaces.Discrete(5)  # -1,1 사이의 1x2 벡터
        # continuous
        self.action_space = spaces.Box(-1, 1, (2,))
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, n_channels))

        # self.observation_space = spaces.Discrete(5)
        self.view = None  # viewport's area / current state
        self.saliency_view = None
        self.observation = None
        self.trajectory = False
        self.start_frame, self.end_frame = None, None

        self.get_view = NFOV(224, 224)

    def step(self, acs=None):

        if self.trajectory:
            obs, saliency, lat, lng, action, done = self.dataset.next_data()

            # discrete
            # acs = np.argmax(acs)
            # acs = action_arr[acs]

            if acs is None:
                self.inference_view.active = False
                self.view.set_center((lng, lat), normalize=True)
                self.saliency_view.set_center((lng, lat), normalize=True)
            else:
                self.inference_view.active = True
                self.inference_view.set_center(acs, normalize=True)
                self.saliency_infer_view.set_center(acs, normalize=True)
                self.view.set_center((lng, lat), normalize=True)
                self.saliency_view.set_center((lng, lat), normalize=True)

            if self.inference:
                # self.observation = [cv2.resize(self.inference_view.get_view(f), (width, height)) for f in obs]
                self.observation = obs
                # self.observation = [self.inference_view.get_view(f) for f in obs]
                if saliency is not None:
                    saliency_observation = [self.saliency_infer_view.get_view(f) for f in saliency]
            else:
                # self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
                # self.observation = [ab.toNFOV(f, np.array([1., 1.])) for f in obs]

                self.observation = obs
                if saliency is not None:
                    self.saliency = [cv2.resize(f, (32, 32)) for f in saliency]
                else:
                    self.saliency = None
                # self.observation = [self.view.get_view(f) for f in obs]
                if saliency is not None:
                    saliency_observation = [self.saliency_view.get_view(f) for f in saliency]

            # if saliency is not None:
            #     total_sum = np.sum(saliency)
            #     observation_sum = np.sum(saliency_observation)
            #
            #     reward = observation_sum / total_sum
            # else:
            #     reward = None
            return self.observation, self.saliency, done, (lng, lat)
        else:
            obs, sal, _, _, _, done = self.dataset.next_data(self.trajectory)
            sal = np.expand_dims(sal, 2)
            # discrete
            # acs = np.argmax(acs)
            # acs = action_arr[acs]
            self.view.move(acs)
            self.saliency_view.move(acs)
            # self.observation = [cv2.resize(f, (width, height)) for f in obs]
            # self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
            self.observation = [self.get_view.toNFOV(f, np.array(self.view.center)) for f in obs]
            # self.observation = obs
            # self.observation = [cv2.resize(self.view.get_view(obs), (width, height))]
            # self.observation = [self.view.get_view(f) for f in obs]
            saliency_observation = [self.saliency_view.get_view(f) for f in sal]
            total_sum = np.sum(sal)
            observation_sum = np.sum(saliency_observation)
            # reward = observation_sum
            reward = observation_sum / total_sum
            del sal, saliency_observation

            return self.observation, reward, done, None

    def reset(self, video_type="train", trajectory=False, target_video=None, randomness=True, inference=True, fx=1,
              fy=1, saliency=True):
        self.inference = inference
        self.trajectory = trajectory

        self.view = TileViewport(self.width * 1, self.height * 1, 4, 4)
        self.saliency_view = TileViewport(2048, 1024, 6, 6)  # saliency w, h
        self.inference_view = TileViewport(self.width * 1, self.height * 1, 6, 6)
        self.saliency_infer_view = TileViewport(2048, 1024, 6, 6)

        if self.trajectory:
            video = self.dataset.select_trajectory(fx, fy, trajectory=True, mode=video_type, randomness=randomness,
                                                   saliency=saliency,
                                                   target_video=target_video)

            obs, saliency, lat, lng, action, done = self.dataset.next_data()
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)
            self.inference_view.set_center((lat, lng), normalize=True)
            self.saliency_infer_view.set_center((lat, lng), normalize=True)

            if saliency is None:
                self.saliency = None
            else:
                self.saliency = [cv2.resize(f, (32, 32)) for f in saliency]

            # self.observation = [cv2.resize(f, (width, height)) for f in obs]
            self.observation = obs

            # self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
            # self.observation = [self.view.get_view(f) for f in obs]
            return self.observation, (lng, lat), video
        else:
            video = self.dataset.select_trajectory(fx, fy, video_type, saliency=saliency, target_video=target_video)
            obs, sal, _, _, _, done = self.dataset.next_data(self.trajectory)
            self.view.set_center((0.5, 0.5))
            self.saliency_view.set_center((0.5, 0.5))
            del sal
            # self.observation = [cv2.resize(f, (width, height)) for f in obs]
            # self.observation = obs
            self.observation = [self.get_view.toNFOV(f, np.array(self.view.center)) for f in obs]
            # self.observation = obs
            # self.observation = [cv2.resize(self.view.get_view(obs), (width, height))]
            # self.observation = [self.view.get_view(f) for f in obs]
            # self.observation = [cv2.resize(self.view.get_view(obs), (width, height))]
            # self.observation = [self.view.get_view(f) for f in obs]
            return self.observation, None, video

    def render(self, mode='viewport', writer=None):
        if mode == 'viewport':
            if self.trajectory:

                # rec = self.view.get_rectangle_point()
                rec = self.view.get_rectangle_point()
                infer_rec = self.inference_view.get_rectangle_point()

                tile_p = self.view.tile_info()
                tile_infer_p = self.inference_view.tile_info()
                for f in self.observation:
                    if self.inference_view.active:
                        f = draw_tile(f, tile_p, color=(0, 255, 0))
                        # f = draw_tile(f, tile_infer_p, color=(0, 0, 255))
                        f = draw_viewport(f, rec, color=(0, 255, 0))  # Blue
                        # f = draw_viewport(f, infer_rec, color=(0, 0, 255))  # Red
                    else:
                        f = draw_tile(f, tile_p, color=(0, 255, 0))
                        f = draw_viewport(f, rec, color=(0, 255, 0))  # Blue
                    if writer is not None:
                        writer.write(f)
                    else:
                        cv2.imshow("render", f)
            else:
                rec = self.view.get_rectangle_point()
                for f in self.dataset.state:
                    f = draw_viewport(f, rec)
                    if writer is not None:
                        writer.write(f)
                    else:
                        cv2.imshow("render", f)
        elif mode == 'tile':
            point = self.saliency_view.tile_info()
            vp = self.saliency_view.get_rectangle_point()
            # print(np.shape(self.dataset.state))
            for f in self.dataset.saliency_state:
                f = draw_tile(f, point)
                f = draw_viewport(f, vp, color=(255, 0, 0))
                if writer is None:
                    cv2.imshow("render", f)
                else:
                    writer.write(f)
        else:
            for f in self.observation:
                if writer is not None:
                    writer.write(f)
                else:
                    cv2.imshow("render", f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.waitKey(32)


def draw_tile(target_frame, rectangle_points, color=(0, 255, 0), grid=True):
    overlay = target_frame.copy()
    # print(f"draw tile {rectangle_points}")
    for point in rectangle_points:
        overlay = cv2.rectangle(overlay, point[0], point[1], color, thickness=cv2.FILLED)  # draw tile
        if grid:
            overlay = cv2.rectangle(overlay, point[0], point[1], color=(0, 0, 0), thickness=3)  # draw grid
    result = cv2.addWeighted(overlay, 0.5, target_frame, 0.5, 0)
    return result


def draw_viewport(target_frame, rectangle_point, color=(0, 0, 255), thickness=5):
    if len(rectangle_point) == 2:
        frame = cv2.rectangle(target_frame, rectangle_point[0], rectangle_point[1], color, thickness)
    elif len(rectangle_point) == 4:
        frame = cv2.rectangle(target_frame, rectangle_point[0], rectangle_point[1], color, thickness)  # left
        frame = cv2.rectangle(frame, rectangle_point[2], rectangle_point[3], color, thickness)  # right
    else:
        raise ValueError(
            f"Invalid rectangle point. rectangle point length must be 2 or 4 but got {len(rectangle_point)}")
    return frame


def embed_frame(observation):
    if len(observation) > n_samples:
        observation = observation[:n_samples]
    elif len(observation) < n_samples:
        embed = np.zeros(shape=(width, height, n_channels))
        for _ in range(n_samples - len(observation)):
            observation = np.concatenate([observation, [embed]])
    return observation


def ttest(max_epochs=1):
    env = CustomEnv()
    epoch = 0

    buffer = ReplayBuffer(3000)
    viewport_info = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    while epoch < max_epochs:
        obs, acs, next_obs, rewards, dones = [], [], [], [], []
        ob, ac, target_video = env.reset(trajectory=True, fx=1, fy=1, inference=False, saliency=False, randomness=False,
                                         target_video="01_PortoRiverside.mp4")
        writer = cv2.VideoWriter(os.path.join("saliency", "test", target_video + "_" + ".mp4"),
                                 fourcc, fps[target_video], (3840, 1920))
        t = np.transpose(env.saliency_view.tile)
        print(t)
        viewport_info.append(t)
        while True:
            next_ob, reward, done, next_ac = env.step([0.1, 0.1])
            t = np.transpose(env.saliency_view.tile)
            print(t)
            viewport_info.append(t)
            env.render(writer=writer)
            # transition = (ob, ac, reward, next_ob, done)
            print(np.shape(ob), np.shape(next_ob), ac, next_ac, reward, done)

            # if len(buffer) >= 30:
            #     t = buffer.get_batch(30)
            #     t = [e[3] for e in t]
            #     if not np.shape(t) == (30, 6, 224, 224, 3):
            #         raise ValueError("batch Error...", np.shape(t))

            obs.append(ob)
            acs.append(ac)
            next_obs.append(next_ob)
            rewards.append(reward)
            dones.append(done)
            if done:
                break

            ob = next_ob
            ac = next_ac
            #     env.render()

        print("epoch # of ", epoch)
        print("obs shape: ", np.shape(obs))
        print("acs shape: ", np.shape(acs), acs[0])
        print("next_obs shape:", np.shape(next_obs))
        print("rewards shape: ", np.shape(rewards))
        print("dones shape: ", np.shape(dones))
        epoch += 1


def sal(saliency_info, target_video):
    saliency_map = read_SalMap(saliency_info[target_video])  # [total_frame, 2048, 1024]
    if len(saliency_map) == 601:
        saliency_map = saliency_map[:-1]
    if len(saliency_map) == 501:
        saliency_map = saliency_map[:-1]
    frame_length = len(saliency_map)  # total frame
    seqnce_length = 20  # 20초

    # tmp_viewport = TileViewport(2048, 1024, 8, 8)
    # rectangle_points = []
    # for i in range(8):
    #     for j in range(8):
    #         rectangle_points.append(tmp_viewport.tile_point(i, j))
    # for ff in saliency_map:
    #     cv2.imshow("test", ff)
    #
    #     overlay = ff.copy()
    #     # print(f"draw tile {rectangle_points}")
    #     for point in rectangle_points:
    #         overlay = cv2.rectangle(overlay, point[0], point[1], color=(255, 255, 255), thickness=3)  # draw grid
    #     result = cv2.addWeighted(overlay, 0.5, ff, 0.5, 0)
    #     cv2.imshow("test", result)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #     cv2.waitKey(32)
    # cv2.destroyAllWindows()

    segments = [2, 4, 5, 10]  # seconds
    tiles = ["4x4", "8x4", "8x8"]
    print(f"start target video {target_video}")
    save_path = os.path.join("tile_HE_info", "z-score", target_video.split(".")[0])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for segment in segments:
        t = seqnce_length // segment
        pack = frame_length // t
        segmented_saliency_map = np.reshape(saliency_map, [t, pack, 2048, 1024])
        print(f"segment saliency shape = {np.shape(segmented_saliency_map)}")
        segmented_saliency_map = np.sum(segmented_saliency_map, axis=1)
        max_val = np.max(segmented_saliency_map)
        min_val = np.min(segmented_saliency_map)
        print(max_val, min_val)
        # segmented_saliency_map = list(map(lambda x: (np.array(x) - np.mean(x)) / (np.std(x) + 1e-7), segmented_saliency_map))
        # segmented_saliency_map = list(map(lambda x: (np.array(x) - min_val) / (max_val - min_val + 1e-7), segmented_saliency_map))

        segmented_saliency_map = np.array(segmented_saliency_map)
        print(f"sum of segment saliency shape = {np.shape(segmented_saliency_map)}")
        result = []
        seg = os.path.join(save_path, str(segment))
        if not os.path.exists(seg):
            os.mkdir(seg)

        for tile in tiles:
            n, m = map(int, tile.split("x"))
            viewport = TileViewport(2048, 1024, n, m)
            tiling = np.zeros([t, n, m])
            tt = os.path.join(seg, tile)
            if not os.path.exists(tt):
                os.mkdir(tt)
            for w in range(n):
                for h in range(m):
                    point = viewport.tile_point(w, h)
                    x1, x2 = point[0][0], point[1][0]
                    y1, y2 = point[0][1], point[1][1]

                    score = segmented_saliency_map[:, x1:x2, y1:y2]  # seg_size, tile_w, tile_h
                    score_total = [np.sum(i).astype(np.int) for i in score]
                    tiling[:, w, h] = score_total
                    result.append(score_total)
                    # print(f"shape {np.shape(score
            ppp = os.path.join(tt, target_video.split(".")[0] + "_tile_info.npy")
            _d = tiling.reshape([-1, m, n])
            _d = [(x - np.mean(x)) / np.std(x) for x in _d]
            _d = np.array(_d)
            print(_d)
            np.save(ppp, _d)


if __name__ == '__main__':
    ttest()

# if __name__ == '__main__':
#     saliency_info = get_SalMap_info()
#     videos = list(fps.keys())
#     for v in videos:
#         sal(saliency_info, v)
# if __name__ == '__main__':
#     # model = saliecny_model()
#     binary_loss = tf.keras.losses.binary_crossentropy
#     # opt = tf.keras.optimizers.Adam(2e-6)
#     dataset = Sal360()
#     sal_info = get_SalMap_info()
#     train_path = os.path.join(video_path, "train", "3840x1920")
#     test_path = os.path.join(video_path, "test", "320x160")
#     train_videos = sorted(os.listdir(train_path))
#     test_videos = sorted(os.listdir(test_path))
#     # writer_path = os.path.join("log", "test")
#     # if not os.path.exists(writer_path):
#     #     os.mkdir(writer_path)
#     # writer = tf.summary.create_file_writer(writer_path)
#     # writer.set_as_default()
#     # env = CustomEnv()
#     # checkpoint_directory = os.path.join("weights", "test")
#     # checkpoint = tf.train.Checkpoint(optimizer=opt, model=model, )
#     # if not os.path.exists(checkpoint_directory):
#     #     os.mkdir(checkpoint_directory)
#     # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
#
#     # tensorboard_callback = tf.keras.callbacks.TensorBoard(os.path.join("log", "test"), update_freq='epoch')
#     data = os.path.join("data")
#     ab = NFOV(224, 224)
#     viewport_path = os.path.join("tile_info", "min_max")
#
#     # (14, 45, 100, 7)(14, 45, 100, 2) train
#     # (14, 12, 100, 7)(14, 12, 100, 2) validation
#     # (5, 57, 100, 7)(5, 57, 100, 2) test
#     viewport1 = TileViewport(3840, 1920, 4, 4)
#     viewport2 = TileViewport(3840, 1920, 5, 5)
#     viewport3 = TileViewport(3840, 1920, 6, 6)
#     for video in train_videos:
#         x_train = dataset.train[0][video]
#         x_val = dataset.validation[0][video]
#         # x_test = dataset.test[0][video]
#         # cap = cv2.VideoCapture(os.path.join(train_path, video))
#         # videos = read_whole_video(cap)
#         p = os.path.join(viewport_path, video.split(".")[0])
#         if not os.path.exists(p):
#             os.mkdir(p)
#         # saliency_map = read_SalMap(sal_info[video])
#         # trajectories = dataset.validation[0][video]
#         nonetile = [None for _ in range(100)]
#         tiles = [[None for _ in range(100)] for _ in range(3)]
#         print(f"start video {video}")
#         hit_rate = [[[] for _ in range(4)] for _ in range(3)]
#         t44 = []
#         t55 = []
#         t66 = []
#         for trajectory in x_train:  # 45
#             t44_tmp = []
#             t55_tmp = []
#             t66_tmp = []
#             for index, pp in enumerate(trajectory):  # 100
#                 lat, lng, start_frame, end_frame = pp[2], pp[1], int(pp[5]), int(pp[6])
#                 viewport1.set_center((lng, lat), normalize=True)
#                 viewport2.set_center((lng, lat), normalize=True)
#                 viewport3.set_center((lng, lat), normalize=True)
#                 add = [viewport1.tile, viewport2.tile, viewport3.tile]
#                 t44_tmp.append(viewport1.tile)
#                 t55_tmp.append(viewport2.tile)
#                 t66_tmp.append(viewport3.tile)
#                 for ii, (i, j) in enumerate(zip(tiles, add)):
#                     if i[index] is None:
#                         if ii != 1:
#                             i[index] = np.array(j).transpose()
#                         else:
#                             i[index] = np.array(j)
#                     else:
#                         if ii != 1:
#                             i[index] += np.array(j).transpose()
#                         else:
#                             i[index] += np.array(j)
#
#             xx = os.path.join(data, video.split(".")[0] + "_x_train_data.npy")
#             yy = os.path.join(data, video.split(".")[0] + "_y_train_data.npy")
#
#         for trajectory in x_val:
#             tiles = [[None for _ in range(100)] for _ in range(3)]
#             for index, pp in enumerate(trajectory):  # 100
#                 lat, lng, start_frame, end_frame = pp[2], pp[1], int(pp[5]), int(pp[6])
#                 viewport1.set_center((lng, lat), normalize=True)
#                 viewport2.set_center((lng, lat), normalize=True)
#                 viewport3.set_center((lng, lat), normalize=True)
#                 # print(lat, lng)
#                 add = [viewport1.tile, viewport2.tile, viewport3.tile]
#                 for ii, (i, j) in enumerate(zip(tiles, add)):
#                     if i[index] is None:
#                         if ii != 1:
#                             i[index] = np.array(j).transpose()
#                         else:
#                             i[index] = np.array(j)
#                     else:
#                         if ii != 1:
#                             i[index] += np.array(j).transpose()
#                         else:
#                             i[index] += np.array(j)
#
#             for index, segment in enumerate([2, 4, 5, 10]):
#                 video_length = 20
#                 segment_length = 20 // segment
#                 tt1 = np.reshape(tiles[0], [100 // segment_length, 20 // segment, 4, 4])
#                 tt2 = np.reshape(tiles[1], [100 // segment_length, 20 // segment, 5, 5])
#                 tt3 = np.reshape(tiles[2], [100 // segment_length, 20 // segment, 6, 6])
#                 tt1 = np.sum(tt1, axis=0)
#                 tt2 = np.sum(tt2, axis=0)
#                 tt3 = np.sum(tt3, axis=0)
#
#                 tt1 = [((x - np.mean(x)) / (np.std(x) + 1e-7)) for x in tt1]
#                 tt2 = [((x - np.mean(x)) / (np.std(x) + 1e-7)) for x in tt2]
#                 tt3 = [((x - np.mean(x)) / (np.std(x) + 1e-7)) for x in tt3]
#                 tt2 = np.transpose(tt2, [0, 2, 1])
#                 tt1 = list(map(lambda x: x >= 0, tt1))
#                 tt2 = list(map(lambda x: x >= 0, tt2))
#                 tt3 = list(map(lambda x: x >= 0, tt3))
#
#                 p1 = os.path.join(viewport_path, video.split(".")[0], str(segment), "4x4")
#                 p2 = os.path.join(viewport_path, video.split(".")[0], str(segment), "5x5")
#                 p3 = os.path.join(viewport_path, video.split(".")[0], str(segment), "6x6")
#                 a1 = os.path.join(p1, video.split(".")[0] + "_tile_info.npy")
#                 a2 = os.path.join(p2, video.split(".")[0] + "_tile_info.npy")
#                 a3 = os.path.join(p3, video.split(".")[0] + "_tile_info.npy")
#                 c1 = np.load(a1).reshape([len(tt1), -1]).tolist()
#                 c2 = np.load(a2).reshape([len(tt2), -1]).tolist()
#                 c3 = np.load(a3).reshape([len(tt3), -1]).tolist()
#
#                 # c1 = list(map(lambda x: x >= 0, c1))
#                 # c2 = list(map(lambda x: x >= 0, c2))
#                 # c3 = list(map(lambda x: x >= 0, c3))
#                 c11 = np.copy(c2).tolist()
#                 c12 = np.copy(c2).tolist()
#                 c13 = np.copy(c2).tolist()
#                 # i11 = []
#                 # i12 = []
#                 # i13 = []
#                 for k in range(len(c11)):
#                     i11 = []
#                     for i in range(10):
#                         idx = c11[k].index(max(c11[k]))
#                         i11.append(idx)
#                         c11[k][idx] = -99999
#
#                     for j in range(len(c11[k])):
#                         if j in i11:
#                             c11[k][j] = True
#                         else:
#                             c11[k][j] = False
#                 for k in range(len(c12)):
#                     i11 = []
#                     for i in range(15):
#                         idx = c12[k].index(max(c12[k]))
#                         i11.append(idx)
#                         c12[k][idx] = -99999
#                     for j in range(len(c12[k])):
#                         if j in i11:
#                             c12[k][j] = True
#                         else:
#                             c12[k][j] = False
#                 for k in range(len(c13)):
#                     i11 = []
#                     for i in range(20):
#                         idx = c13[k].index(max(c13[k]))
#                         i11.append(idx)
#                         c13[k][idx] = -99999
#                     for j in range(len(c13[k])):
#                         if j in i11:
#                             c13[k][j] = True
#                         else:
#                             c13[k][j] = False
#
#                 c11 = np.reshape(c11, np.shape(tt2))
#                 c12 = np.reshape(c12, np.shape(tt2))
#                 c13 = np.reshape(c13, np.shape(tt2))
#                 # print(np.shape(c12))
#                 bb1 = np.count_nonzero(tt1)
#                 bb2 = np.count_nonzero(tt2)
#                 bb3 = np.count_nonzero(tt3)
#
#                 d1 = np.logical_and(c11, tt2)
#                 d2 = np.logical_and(c12, tt2)
#                 d3 = np.logical_and(c13, tt2)
#                 non1 = np.count_nonzero(d1)
#                 non2 = np.count_nonzero(d2)
#                 non3 = np.count_nonzero(d3)
#                 e1 = segment_length * 5 * 5
#
#                 # e2 = segment_length * 4 * 4
#                 # e3 = segment_length * 4 * 4
#                 rate1 = non1 / bb2
#                 rate2 = non2 / bb2
#                 rate3 = non3 / bb2
#                 hit_rate[0][index].append(rate1)
#                 hit_rate[1][index].append(rate2)
#                 hit_rate[2][index].append(rate3)
#
#                 # print(np.shape(tt1), np.shape(c1), non1 / e1)
#                 # print(np.shape(tt2), np.shape(c2), non2 / e2)
#                 # print(np.shape(tt3), np.shape(c3), non3 / e3)
#
#         mean_ratio = list(map(lambda x: list(map(np.mean, x)), hit_rate))
#         # print(mean_ratio)
#         plt.xlabel(f"segment size (s)")
#         plt.ylabel(f"hit ratio")
#         # plt.ylim((0.5, 1))
#         title = video.split(".")[0].split("_")[1]
#         plt.title(f"{title} (5x5)")
#         plt.ylim(0, 1)
#         format = ["s-r", "^:b", "o--g"]
#         labels = ["queue=10", "queue=15", "queue=20"]
#         for ratio, ff, label in zip(mean_ratio, format, labels):
#             plt.plot([2, 4, 5, 10], ratio, ff, label=label)
#         plt.legend(loc="lower right")
#         plt.savefig(os.path.join("tile_info", "5x5", f"{title}_5x5.png"))
#         plt.show()
#         print(np.transpose(np.shape(tiles[0])))
#         for segment in [2, 4, 5, 10]:
#             segment_length = 20 // segment
#             tt1 = np.reshape(tiles[0], [100 // segment_length, segment_length, 4, 4])
#             tt2 = np.reshape(tiles[1], [100 // segment_length, segment_length, 5, 5])
#             tt3 = np.reshape(tiles[2], [100 // segment_length, segment_length, 6, 6])
#             # tt1 = np.reshape(tiles[0], [segment, 100 // segment, 4, 4])
#             # tt2 = np.reshape(tiles[1], [segment, 100 // segment, 8, 4])
#             # tt3 = np.reshape(tiles[2], [segment, 100 // segment, 8, 8])
#             tt1 = np.sum(tt1, axis=0)
#             tt2 = np.sum(tt2, axis=0)
#             tt3 = np.sum(tt3, axis=0)
#             tt1 = [((x - np.mean(x)) / (np.std(x) + 1e-7)).tolist() for x in tt1]
#             tt2 = [((x - np.mean(x)) / (np.std(x) + 1e-7)).tolist() for x in tt2]
#             tt3 = [((x - np.mean(x)) / (np.std(x) + 1e-7)).tolist() for x in tt3]
#
#             p1 = os.path.join(viewport_path, video.split(".")[0], str(segment), "4x4")
#             p2 = os.path.join(viewport_path, video.split(".")[0], str(segment), "5x5")
#             p3 = os.path.join(viewport_path, video.split(".")[0], str(segment), "6x6")
#
#             if not os.path.exists(p1):
#                 os.makedirs(p1)
#             if not os.path.exists(p2):
#                 os.makedirs(p2)
#             if not os.path.exists(p3):
#                 os.makedirs(p3)
#             a1 = os.path.join(p1, video.split(".")[0] + "_tile_info.npy")
#             a2 = os.path.join(p2, video.split(".")[0] + "_tile_info.npy")
#             a3 = os.path.join(p3, video.split(".")[0] + "_tile_info.npy")
#             np.save(a1, np.array(tt1))
#             np.save(a2, np.transpose(tt2, [0, 2, 1]))
#             np.save(a3, np.array(tt3))
#             c1 = np.load(a1)
#             c2 = np.load(a2)
#             c3 = np.load(a3)
#             # print(np.shape(tt1))
#             # print(np.shape(tt2))
#             # print(np.shape(tt3))
#             print(np.shape(c1), np.shape(c2), np.shape(c3))
