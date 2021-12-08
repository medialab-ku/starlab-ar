import os
import ctypes
import numpy as np


############################## DLL Import ##############################


kf = ctypes.CDLL(os.path.dirname(__file__) + '/build/libkf.so')


############################## Class Definition ##############################


class Frame:

    def __init__(self, size: tuple, level: int, intrinsic: tuple):
        self._id = create_frame(size, level, intrinsic)
        self._size = size
        self._level = level
        self._intrinsic = intrinsic

    def __del__(self):
        delete_frame(self._id)

    @property
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def level(self) -> int:
        return self._level

    @property
    def intrinsic(self) -> tuple:
        return self._intrinsic

    def preprocess(self, depth_map: np.ndarray) -> None:
        frame_preprocess(self._id, depth_map.astype(np.float32))

    def mask(self, mask: np.ndarray) -> None:
        frame_mask(self._id, mask.astype(np.bool8))

    def clear(self) -> None:
        frame_clear(self._id)

    def get_depth_map(self) -> np.ndarray:
        return get_frame_depth_map(self._size, self._id)

    def get_vertex_map(self) -> np.ndarray:
        return get_frame_vertex_map(self._size, self._id)

    def get_normal_map(self) -> np.ndarray:
        return get_frame_normal_map(self._size, self._id)

class Object:

    def __init__(self, size: tuple, center: tuple, voxel_size: float):
        self._id = create_object(size, center, voxel_size)
        self._size = size
        self._center = center
        self._voxel_size = voxel_size

    def __del__(self):
        delete_object(self._id)

    @property
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def center(self) -> tuple:
        return self._center

    @property
    def voxel_size(self) -> float:
        return self._voxel_size

    def integrate(self, frame: Frame) -> None:
        object_integrate_frame(self._id, frame.id)

    def get_tsdf_volume(self) -> np.ndarray:
        return get_object_tsdf_volume(self._size, self._id)

    def get_weight_volume(self) -> np.ndarray:
        return get_object_weight_volume(self._size, self._id)

    def get_binomial_volume(self) -> np.ndarray:
        return get_object_binomial_volume(self._size, self._id)

    def get_polygon_volume(self) -> np.ndarray:
        return get_object_polygon_volume(self._size, self._id)
