import taichi as ti
import numpy as np


@ti.kernel
def set_background(img_buff: ti.types.ndarray(), new_buff: ti.types.ndarray()):
    # find green pixels
    w = img_buff.shape[0]
    h = img_buff.shape[1]
    for i, j in ti.ndrange(w, h):
        if img_buff[i, j, 0] == 0.0 and img_buff[i, j, 1] == 1.0 and img_buff[i, j, 2] == 0.0:
            continue
        else:
            new_buff[i, j, 0] = img_buff[i, j, 0]
            new_buff[i, j, 1] = img_buff[i, j, 1]
            new_buff[i, j, 2] = img_buff[i, j, 2]