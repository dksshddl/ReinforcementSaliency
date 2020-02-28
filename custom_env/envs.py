import os

import gym
from gym import spaces

from dataset import *
from utils.binary import *
from utils.config import *

n_samples = 8
width = 224
height = 224
n_channels = 3
frame_step = 6


class CustomEnv(gym.Env):

    def __init__(self, video_type="train"):
        self.train, self.validation, self.test = Sal360.load_sal360v2()
        self.width, self.height = 3840, 1920
        self.x_dict, self.y_dict = None, None
        self.x_iter, self.y_iter = None, None
        self.target_videos = None
        self.target_video = None
        self.files = os.listdir(video_path)
        self.action_space = spaces.Box(-1, 1, [2])  # -1,1 사이의 1x2 벡터
        self.observation_space = spaces.Box(low=0, high=255, shape=(width, height, n_channels))

        self.view = None  # viewport's area / current state
        self.saliency_view = None
        self.observation = None
        self.video_path = None
        self.files = None
        self.trajectory = False
        self.time_step = 0
        self.saliency_info = get_SalMap_info()
        self.set_dataset(video_type)

    def set_dataset(self, video_type="train"):
        if video_type == "train":
            self.x_dict, self.y_dict = self.train
            self.video_path = os.path.join(video_path, "train", "3840x1920")
        elif video_type == "validation":
            self.x_dict, self.y_dict = self.validation
            self.video_path = os.path.join(video_path, "train", "3840x1920")
        elif video_type == "test":
            self.x_dict, self.y_dict = self.test
            self.video_path = os.path.join(video_path, "test", "3840x1920")
        else:
            raise NotImplementedError
        self.target_videos = os.listdir(self.video_path)

    def step(self, action=None):
        x_data, y_data = next(self.x_iter), next(self.y_iter)
        step_idx, lat, lng, start_frame, end_frame = int(x_data[0]), x_data[2], x_data[1], int(x_data[5]), int(
            x_data[6])
        done = True if step_idx == 99 else False

        if self.trajectory:
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)
            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in
                                self.video[start_frame - 1:end_frame]]

            saliency_observation = [self.saliency_view.get_view(f) for f in self.saliency[start_frame - 1:end_frame]]
            assert len(self.observation) == len(saliency_observation)

            total_sum = np.sum(self.saliency[start_frame - 1:end_frame])
            observation = self.observation.copy()
            embed = np.zeros(shape=(width, height, n_channels)) + 256.
            for _ in range(n_samples - len(observation)):
                observation = np.concatenate([observation, [embed]])
            assert len(observation) == n_samples

            observation_sum = np.sum(saliency_observation)
            reward = observation_sum / total_sum

            return observation, reward, done, y_data
        else:
            self.view.move(action)
            self.saliency_view.move(action)
            self.view.set_center((lat, lng), normalize=True)
            self.saliency_view.set_center((lat, lng), normalize=True)
            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in
                                self.video[self.time_step:self.time_step+frame_step]]
            if len(self.observation) < frame_step:
                return None, None, True, None
            else:
                saliency_observation = [self.saliency_view.get_view(f) for f in self.saliency[self.time_step:self.time_step+frame_step]]
                total_sum = np.sum(self.saliency[self.time_step:self.time_step+frame_step])
                observation_sum = np.sum(saliency_observation)
                reward = observation_sum / total_sum
                self.time_step += frame_step
                return self.observation, reward, done, None

    def reset(self, video_type="train", trajectory=False):
        self.trajectory = trajectory
        self.time_step = 0
        self.view = Viewport(self.width, self.height)
        self.saliency_view = Viewport(2048, 1024)
        self.set_dataset(video_type)
        self.target_video = random.choice(self.target_videos)
        print(self.target_video + " start")
        ran_idx = random.randint(0, len(self.x_dict[self.target_video]) - 1)
        ran_x, ran_y = self.x_dict[self.target_video][ran_idx], self.y_dict[self.target_video][ran_idx]
        self.saliency = read_SalMap(self.saliency_info[self.target_video])
        self.video = []
        cap = cv2.VideoCapture(os.path.join(self.video_path, self.target_video))
        while True:
            ret, frame = cap.read()
            if ret:
                self.video.append(frame)
            else:
                cap.release()
                break
        self.x_iter, self.y_iter = iter(ran_x), iter(ran_y)
        x_data, y_data = next(self.x_iter), next(self.y_iter)
        lat, lng, start_frame, end_frame = x_data[2], x_data[1], int(x_data[5]), int(x_data[6])
        self.view.set_center((lat, lng), normalize=True)
        if self.trajectory:
            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in
                                self.video[start_frame - 1:end_frame]]
            embed = np.zeros(shape=(width, height, n_channels)) + 256.
            observation = self.observation.copy()
            for _ in range(n_samples - len(observation)):
                observation = np.concatenate([observation, [embed]])
            assert len(observation) == n_samples
            return observation, y_data, self.target_video
        else:
            self.observation = [cv2.resize(self.view.get_view(f), (width, height)) for f in
                                self.video[self.time_step:self.time_step+frame_step]]
            return self.observation, y_data, self.target_video

    def render(self, mode='human'):
        for f in self.observation:
            cv2.imshow("render", f)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def ttest(max_epochs=0):
    env = CustomEnv()
    epoch = 0
    while epoch < max_epochs:
        obs, acs, next_obs, rewards, dones = [], [], [], [], []
        ob, ac, target_video = env.reset(trajectory=False)
        while True:
            next_ob, reward, done, next_ac = env.step([0.1, 0.1])
            env.render()
            print(np.shape(ob), np.shape(next_ob), ac, next_ac, reward, done)
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
    ttest(1)
