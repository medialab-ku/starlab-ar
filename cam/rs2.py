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


############################## Function Definition ##############################


def get_frames() -> Tuple[np.ndarray, np.ndarray]:

    # get frames
    frames = pipeline.wait_for_frames()

    # align frames
    aligned_frames = align.process(frames)

    # get aligned frames
    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()

    # convert frames to NumPy objects
    color_frame = np.asanyarray(color_frame.get_data())
    depth_frame = np.asanyarray(depth_frame.get_data())

    return color_frame, depth_frame

def get_timestamp() -> str:

    return '{:.6f}'.format(time.time())

def get_frame_size() -> Tuple[int, int]:

    return intrinsics.width, intrinsics.height

def get_intrinsics() -> Tuple[float, float, float, float]:

    return intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
