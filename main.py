import os


############################## Directory Setting ##############################


DIR_IMAGE = 'image'
DIR_VIDEO = 'video'
DIR_MESH = 'mesh'

for directory in [DIR_IMAGE, DIR_VIDEO, DIR_MESH]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


############################## Parameter Setting ##############################


# TUM RGB-D
# DATA_NAME = 'rgbd_dataset_freiburg1_xyz'
# MAP_SIZE = (640, 480)
# CAM_INTRINSIC = (591.1, 590.1, 331.0, 234.0)
# PYRAMID_LEVEL = 3

# RealSense D455 (color to depth align)
# DATA_NAME = 'rs2'
# MAP_SIZE = (1280, 720)
# CAM_INTRINSIC = (642.2863159179688, 642.2863159179688, 635.0969848632812, 360.14208984375)
# PYRAMID_LEVEL = 3

# RealSense D455 (depth to color align)
DATA_NAME = 'rs2'
MAP_SIZE = (1280, 800)
CAM_INTRINSIC = (637.4338989257812, 636.5624389648438, 637.0335693359375, 410.9555358886719)
PYRAMID_LEVEL = 3

BG_VOLUME_SIZE = (128, 128, 128)
BG_CENTER_POS = (0.0, 0.0, 2.0)
BG_VOXEL_SIZE = 0.02

OBJ_VOLUME_SIZE = 32
OBJ_VOLUME_PERCENT = 10
OBJ_VOLUME_SCALE = 1.5

CHECK_SCORE = 0.5
CHECK_IOU_MAX = 0.5
CHECK_CENTER = 0.05
CHECK_SIZE = 0.1
