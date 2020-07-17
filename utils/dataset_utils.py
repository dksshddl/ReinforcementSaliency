import numpy as np
import cv2
from utils.config import *


def embed_frame(observation, n_samples=8, width=224, height=224, n_channels=3):
    if len(observation) > n_samples:
        observation = observation[:n_samples]
    elif len(observation) < n_samples:
        embed = np.zeros(shape=(width, height, n_channels))
        for _ in range(n_samples - len(observation)):
            observation = np.concatenate([observation, [embed]])
    return observation


def read_whole_video(cap, fx=0.3, fy=0.3):
    video = []
    while True:
        ret, frame = cap.read()
        if ret:
            # frame = cv2.resize(frame, dsize=(224, 224))  # 원래 이미지의 fx, fy배

            video.append(frame)
            # cv2.imshow("test",frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            cap.release()
            break
    # cv2.destroyAllWindows()
    return video


if __name__ == '__main__':
    cap = cv2.VideoCapture(os.path.join(video_path, "train", "3840x1920", "01_PortoRiverSide.mp4"))
    read_whole_video(cap)
