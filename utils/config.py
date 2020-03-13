import os

DDPG_MOBILE = "ddpg_mobile"
DDPG_RESNET = "ddpg_resnet"
DDPG_CONVLSTM = "ddpg_convLSTM"
DDPG = "ddpg"


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

# log, weight path
log_path = os.path.join("./log")
weight_path = os.path.join("./weights")

fps = {
    "01_PortoRiverside.mp4": 25,
    "02_Diner.mp4": 30,
    "03_PlanEnergyBioLab.mp4": 25,
    "04_Ocean.mp4": 30,
    "05_Waterpark.mp4": 30,
    "06_DroneFlight.mp4": 25,
    "07_GazaFishermen.mp4": 25,
    "08_Sofa.mp4": 24,
    "09_MattSwift.mp4": 30,
    "10_Cows.mp4": 24,
    "11_Abbottsford.mp4": 30,
    "12_TeatroRegioTorino.mp4": 30,
    "13_Fountain.mp4": 30,
    "14_Warship.mp4": 25,
    "15_Cockpit.mp4": 25,
    "16_Turtle.mp4": 30,
    "17_UnderwaterPark.mp4": 30,
    "18_Bar.mp4": 25,
    "19_Touvet.mp4": 30
}
