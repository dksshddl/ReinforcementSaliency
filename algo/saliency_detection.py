import datetime
import os

import cv2
import numpy as np

from custom_env.envs import CustomEnv
from utils.config import *
from utils.dataset_utils import read_whole_video

saliency = cv2.saliency.StaticSaliencyFineGrained_create()

epoch = 0
env = CustomEnv()

train_video_path = os.path.join(video_path, "train", "3840x1920")
test_video_path = os.path.join(video_path, "test", "3840x1920")

train_video = os.listdir(train_video_path)
test_video = os.listdir(test_video_path)

for video in train_video:
    print(f"start {video}")
    path = os.path.join(train_video_path, video)
    cap = cv2.VideoCapture(path)
    videos = read_whole_video(cap, 1, 1)
    out = os.path.join("output_video", "saliency")
    if not os.path.exists(out):
        os.mkdir(out)
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(os.path.join(out, video + "_" + str(now) + ".mp4"),
                             fourcc, fps[video], (3840, 1920), 0)
    writer_thressh = cv2.VideoWriter(os.path.join(out, video + "_thresh_" + str(now) + ".mp4"),
                                     fourcc, fps[video], (3840, 1920), 0)
    for frame in videos:
        success, saliencyMap = saliency.computeSaliency(frame)
        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        writer.write(saliencyMap)
        writer_thressh.write(threshMap)
    writer.release()
for video in test_video:
    print(f"start {video}")
    path = os.path.join(test_video_path, video)
    cap = cv2.VideoCapture(path)
    videos = read_whole_video(cap, 1, 1)
    out = os.path.join("output_video", "saliency")
    if not os.path.exists(out):
        os.mkdir(out)
    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    writer = cv2.VideoWriter(os.path.join(out, video + "_" + str(now) + ".mp4"),
                             fourcc, fps[video], (3840, 1920), 0)
    writer_thressh = cv2.VideoWriter(os.path.join(out, video + "_thresh_" + str(now) + ".mp4"),
                                     fourcc, fps[video], (3840, 1920), 0)
    for frame in videos:
        success, saliencyMap = saliency.computeSaliency(frame)
        threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
        writer.write(saliencyMap)
        writer_thressh.write(threshMap)
    writer.release()

