import os

# base dataset path
base_path = os.path.join("C:", os.sep, "Users", "dkssh", "SaliencyData")

# Head only path
saliency_map_H_path = os.path.join(base_path, "dataset", "H", "SalMaps")
scanpaths_H_path = os.path.join(base_path, "dataset", "H", "Scanpaths")

# Head + eye path
saliency_map_HE_path = os.path.join(base_path, "dataset", "HE", "SalMaps")
scanpaths_HE_L_path = os.path.join(base_path, "dataset", "HE", "Scanpaths", "L")
scanpaths_HE_R_path = os.path.join(base_path, "dataset", "HE", "Scanpaths", "R")

# stimuli path
video_path = os.path.join(base_path, "sample_videos")
