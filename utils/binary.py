import os
import re
import time

import numpy as np
import cv2

from utils.config import saliency_map_H_path, fps

dtypes = {16: np.float16,
          32: np.float32,
          64: np.float64}


def get_SalMap_info():
    data = os.listdir(saliency_map_H_path)
    data = [file for file in data if file.endswith(".bin")]
    get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
    sal_map_info = {}
    for filename in data:
        info = get_file_info.findall(filename)[0]  # 각 영상의 name, width, height, dtype
        sal_map_info[info[0] + '.mp4'] = [filename] + list(map(lambda x: int(x), info[1:]))
    return sal_map_info


def readAll():
    data = os.listdir(saliency_map_H_path)
    data = [file for file in data if file.endswith(".bin")]
    get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
    sal_map_info = {}

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    # fourcc = cv2.CV_FOURCC(*'DIVX')

    for filename in data:
        info = get_file_info.findall(filename)[0]  # 각 영상의 name, width, height, dtype

        writer = cv2.VideoWriter(os.path.join("saliency", info[0] + ".mp4"),
                                 fourcc, fps[info[0] + '.mp4'], (2048, 1024), 0)

        with open(os.path.join(saliency_map_H_path, filename), "rb") as f:
            print(info, filename)
            width, height, n_frame, dtype = int(info[1]), int(info[2]), int(info[3]), int(info[4])
            sal_map_info[info[0] + '.mp4'] = []
            for i in range(n_frame):
                f.seek(width * height * i * (dtype // 8))
                data = np.fromfile(f, count=width * height, dtype=dtypes[dtype])
                data = np.reshape(data, [height, width])
                data.astype(np.float)
                cv2.imshow("test", data)
                data = data / 25
                frame = np.zeros([height, width, 1])
                frame[:, :, 0] = data
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(np.shape(data), np.shape(frame))
                sal_map_info[info[0] + '.mp4'].append(data)
                writer.write(data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cv2.waitKey(32)
    return sal_map_info


def read_SalMap(info):
    full = []
    with open(os.path.join(saliency_map_H_path, info[0]), "rb") as f:
        # Position read pointer right before target frame
        width, height, n_frame, dtype = info[1], info[2], info[3], info[4]
        for i in range(n_frame):
            f.seek(width * height * i * (dtype // 8))
            #  Read from file the content of one frame
            data = np.fromfile(f, count=width * height, dtype=dtypes[dtype])
            # Reshape flattened data to 2D image
            full.append(data.reshape([height, width]))
    return full


if __name__ == '__main__':
    readAll()