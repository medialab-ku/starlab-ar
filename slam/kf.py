import os
import ctypes
import numpy as np


############################## DLL Import ##############################


kf = ctypes.CDLL(os.path.dirname(__file__) + '/build/libkf.so')


############################## Function Definition ##############################


def hello_world() -> None:
    kf_hello_world = kf.hello_world
    kf_hello_world()


def create_frame(size: tuple, level: int, intrinsic: tuple) -> int:
    kf_create_frame = kf.create_frame
    kf_create_frame.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    kf_create_frame.restype = ctypes.c_int
    return kf_create_frame(size[0], size[1], level,
                           intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])


def create_object(size: tuple, center: tuple, voxel_size: float) -> int:
    kf_create_object = kf.create_object
    kf_create_object.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    kf_create_object.restype = ctypes.c_int
    return kf_create_object(size[0], size[1], size[2],
                            center[0], center[1], center[2], voxel_size)


def create_camera(size: tuple, intrinsic: tuple) -> int:
    kf_create_camera = kf.create_camera
    kf_create_camera.argtypes = [
        ctypes.c_int, ctypes.c_int,
        ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
    kf_create_camera.restype = ctypes.c_int
    return kf_create_camera(size[0], size[1],
                            intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])


def delete_frame(id: int) -> None:
    kf_delete_frame = kf.delete_frame
    kf_delete_frame.argtypes = [ctypes.c_int]
    kf_delete_frame(id)


def delete_object(id: int) -> None:
    kf_delete_object = kf.delete_object
    kf_delete_object.argtypes = [ctypes.c_int]
    kf_delete_object(id)


def delete_camera(id: int) -> None:
    kf_delete_camera = kf.delete_camera
    kf_delete_camera.argtypes = [ctypes.c_int]
    kf_delete_camera(id)


def get_camera_depth_map(size: tuple, camera_id: int) -> np.ndarray:
    kf_get_camera_depth_map = kf.get_camera_depth_map
    kf_get_camera_depth_map.argtypes = [ctypes.c_int]
    kf_get_camera_depth_map.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, shape=(size[1], size[0]))
    return kf_get_camera_depth_map(camera_id)


def get_camera_vertex_map(size: tuple, camera_id: int) -> np.ndarray:
    kf_get_camera_vertex_map = kf.get_camera_vertex_map
    kf_get_camera_vertex_map.argtypes = [ctypes.c_int]
    kf_get_camera_vertex_map.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, shape=(size[1], size[0], 3))
    return kf_get_camera_vertex_map(camera_id)


def get_camera_normal_map(size: tuple, camera_id: int) -> np.ndarray:
    kf_get_camera_normal_map = kf.get_camera_normal_map
    kf_get_camera_normal_map.argtypes = [ctypes.c_int]
    kf_get_camera_normal_map.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, shape=(size[1], size[0], 3))
    return kf_get_camera_normal_map(camera_id)


def get_camera_instance_map(size: tuple, camera_id: int) -> np.ndarray:
    kf_get_camera_instance_map = kf.get_camera_instance_map
    kf_get_camera_instance_map.argtypes = [ctypes.c_int]
    kf_get_camera_instance_map.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_int, shape=(size[1], size[0]))
    return kf_get_camera_instance_map(camera_id)


def get_camera_pose(camera_id: int) -> np.ndarray:
    kf_get_camera_pose = kf.get_camera_pose
    kf_get_camera_pose.argtypes = [ctypes.c_int]
    kf_get_camera_pose.restype = np.ctypeslib.ndpointer(
        dtype=ctypes.c_float, shape=(4, 4))
    return np.transpose(kf_get_camera_pose(camera_id))


def get_camera_track_error(camera_id: int) -> float:
    kf_get_camera_track_error = kf.get_camera_track_error
    kf_get_camera_track_error.argtypes = [ctypes.c_int]
    kf_get_camera_track_error.restype = ctypes.c_float
    return kf_get_camera_track_error(camera_id)


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

class Camera:
    def __init__(self, size: tuple, intrinsic: tuple):
        self._id = create_camera(size, intrinsic)
        self._size = size
        self._intrinsic = intrinsic

    def __del__(self):
        delete_camera(self._id)

    @property
    def id(self) -> int:
        return self._id

    @property
    def size(self) -> tuple:
        return self._size

    @property
    def intrinsic(self) -> tuple:
        return self._intrinsic

    def raycast(self, object: Object) -> None:
        camera_raycast_object(self._id, object.id)

    def track(self, frame: Frame) -> None:
        camera_track_frame(self._id, frame.id)

    def clear(self) -> None:
        camera_clear(self._id)

    def get_depth_map(self) -> np.ndarray:
        return get_camera_depth_map(self._size, self._id)

    def get_vertex_map(self) -> np.ndarray:
        return get_camera_vertex_map(self._size, self._id)

    def get_normal_map(self) -> np.ndarray:
        return get_camera_normal_map(self._size, self._id)

    def get_instance_map(self) -> np.ndarray:
        return get_camera_instance_map(self._size, self._id)

    def get_pose(self) -> np.ndarray:
        return get_camera_pose(self._id)

    def get_track_error(self) -> float:
        return get_camera_track_error(self._id)
