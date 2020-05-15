import os

import gym
from gym import spaces

from dataset import *
from utils.binary import *
from utils.config import *
from utils.replay import ReplayBuffer
from utils.viewport import TileViewport

n_samples = 8

width = 160
height = 80
n_channels = 3

action_arr = [
    [0, 0],
    [0.0275, 0],
    [-0.0275, 0],
    [0, 0.035],
    [0, -0.035]
]


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

    def step(self, acs=None):

        if self.trajectory:
            obs, saliency, lat, lng, action, done = self.dataset.next_data()

            # discrete
            # acs = np.argmax(acs)
            # acs = action_arr[acs]

            self.inference_view.move(acs)
            self.saliency_infer_view.move(acs)
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)

            if self.inference:
                self.observation = [cv2.resize(self.inference_view.get_view(f), (width, height)) for f in obs]
                # self.observation = [self.inference_view.get_view(f) for f in obs]
                if saliency is not None:
                    saliency_observation = [self.saliency_infer_view.get_view(f) for f in saliency]
            else:
                self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
                # self.observation = [self.view.get_view(f) for f in obs]
                if saliency is not None:
                    saliency_observation = [self.saliency_view.get_view(f) for f in saliency]

            if saliency is not None:
                total_sum = np.sum(saliency)
                observation_sum = np.sum(saliency_observation)

                reward = observation_sum / total_sum
            else:
                reward = None
            return self.observation, reward, done, action
        else:
            obs, saliency, _, _, _, done = self.dataset.next_data(self.trajectory)
            # discrete
            # acs = np.argmax(acs)
            # acs = action_arr[acs]
            self.view.move(acs)
            self.saliency_view.move(acs)
            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
            # self.observation = [cv2.resize(self.view.get_view(obs), (width, height))]

            # self.observation = [self.view.get_view(f) for f in obs]
            saliency_observation = [self.saliency_view.get_view(f) for f in saliency]
            # saliency_observation = [self.saliency_view.get_view(saliency)]

            total_sum = np.sum(saliency)
            observation_sum = np.sum(saliency_observation)

            reward = observation_sum / total_sum

            return self.observation, reward, done, None

    def reset(self, video_type="train", trajectory=False, target_video=None, randomness=True, inference=True, fx=0.3,
              fy=0.3, saliency=True):
        self.inference = inference
        self.trajectory = trajectory

        self.view = TileViewport(self.width * fx, self.height * fy, 8, 4)
        self.saliency_view = TileViewport(2048, 1024, 8, 4)  # saliency w, h
        self.inference_view = TileViewport(self.width * fx, self.height * fy, 8, 4)
        self.saliency_infer_view = TileViewport(2048, 1024, 8, 4)

        if self.trajectory:
            video = self.dataset.select_trajectory(fx, fy, mode=video_type, randomness=randomness, saliency=False)

            obs, saliency, lat, lng, action, done = self.dataset.next_data()
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)
            self.inference_view.set_center((lat, lng), normalize=True)
            self.saliency_infer_view.set_center((lat, lng), normalize=True)

            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
            # self.observation = [self.view.get_view(f) for f in obs]
            return self.observation, action, video
        else:
            video = self.dataset.select_trajectory(fx, fy, video_type, saliency=saliency, target_video=target_video)
            obs, saliency, _, _, _, done = self.dataset.next_data(self.trajectory)

            self.view.set_center((0.5, 0.5))
            self.saliency_view.set_center((0.5, 0.5))

            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in obs]
            # self.observation = [cv2.resize(self.view.get_view(obs), (width, height))]
            # self.observation = [self.view.get_view(f) for f in obs]
            return self.observation, None, video

    def render(self, mode='viewport', writer=None):
        if mode == 'viewport':
            if self.trajectory:

                rec = self.view.get_rectangle_point()
                infer_rec = self.inference_view.get_rectangle_point()

                for f in self.dataset.state:
                    f = draw_viewport(f, rec, color=(255, 0, 0))  # Blue
                    f = draw_viewport(f, infer_rec, color=(0, 0, 255))  # Red

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
            point = self.view.tile_info()
            vp = self.view.get_rectangle_point()
            # print(np.shape(self.dataset.state))
            for f in self.dataset.state:
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    while epoch < max_epochs:
        obs, acs, next_obs, rewards, dones = [], [], [], [], []
        ob, ac, target_video = env.reset(trajectory=True, fx=1, fy=1, inference=False, saliency=False)
        writer = cv2.VideoWriter(os.path.join("saliency", "test", target_video + "_" + ".mp4"),
                                 fourcc, fps[target_video], (3840, 1920))
        while True:
            next_ob, reward, done, next_ac = env.step([0.1, 0.1])
            env.render(mode="tile", writer=writer)
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


if __name__ == '__main__':
    ttest(5)
