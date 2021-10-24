import numpy as np
import cv2


def show_depth_map(name: str, map: np.ndarray) -> None:
    image = 255 * (map / 4)
    image = image.astype(np.uint8)
    cv2.imshow(name, image)
    cv2.waitKey(1)

def write_depth_map(path: str, map: np.ndarray) -> None:
    image = 255 * (map / 4)
    image = image.astype(np.uint8)
    cv2.imwrite(path, image)
    cv2.waitKey(1)