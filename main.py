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
