from typing import Tuple
import numpy as np
import time

import pyrealsense2 as rs2


############################## Parameter Setting ##############################


COLOR_WIDTH = 1280
COLOR_HEIGHT = 800
COLOR_FRAME_RATE = 30

DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
DEPTH_FRAME_RATE = 30

ALIGN_TO = 'COLOR'  # 'COLOR' or 'DEPTH'
