import os
import re

import numpy as np

from utils.config import saliency_map_H_path


dtypes = {16: np.float16,
          32: np.float32,
          64: np.float64}


def get_SalMap_info():
    data = os.listdir(saliency_map_H_path)
    data = [file for file in data if file.endswith(".bin")]
    print(data)
    get_file_info = re.compile("(\d+_\w+)_(\d+)x(\d+)x(\d+)_(\d+)b")
    # 각 영상의 name, width, height, dtype
    sal_map_info = {}
    for i in data:
        info = get_file_info.findall(i)[0]
        sal_map_info[info[0] + '.mp4'] = [i] + list(map(lambda x: int(x), info[1:]))
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
