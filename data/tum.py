import os
import cv2


############################## Parameter Setting ##############################


TUM_PATH = os.path.dirname(__file__) + '/tum/'
ASSOCIATE_SCRIPT = 'associate.py'
ASSOCIATIONS_FILE = 'associations.txt'
DEPTH_SCALE = 1.0 / 5000.0


############################## Class Definition ##############################


class DataLoader:

    def __init__(self, name: str):

        # set sequence path
        self._path = TUM_PATH + name + '/'

        # initialize frame count
        self._count = -1

        # initialize rgb and depth info list
        self._rgb_times = []
        self._rgb_paths = []
        self._depth_times = []
        self._depth_paths = []

        # parse rgb and depth info
        self._parse_info()

    @property
    def path(self):
        return self._path

    @property
    def count(self):
        return self._count

    def _parse_info(self):

        # create associations text file
        if not os.path.exists(self._path + ASSOCIATIONS_FILE):
            os.system('python3' + ' ' +
                      TUM_PATH + ASSOCIATE_SCRIPT + ' ' +
                      self._path + 'rgb.txt' + ' ' +
                      self._path + 'depth.txt' + ' > ' +
                      self._path + ASSOCIATIONS_FILE)

        # open associations text file
        with open(self._path + ASSOCIATIONS_FILE) as f:
            lines = f.readlines()

        # read rgb and depth info
        for line in lines:

            # skip comment
            if line.startswith('#'):
                continue

            # parse line
            rgb_time, rgb_path, depth_time, depth_path = line.split()

            # add rgb and depth info to list
            self._rgb_times.append(rgb_time)
            self._rgb_paths.append(rgb_path)
            self._depth_times.append(depth_time)
            self._depth_paths.append(depth_path)

    def empty(self):

        # check frame count
        return self._count + 1 >= len(self._rgb_times)

    def next(self) -> tuple:

        # check next frame exist
        if self.empty():
            return None, None, None, None

        # increase count
        self._count += 1

        # get rgb and depth info
        rgb_time = self._rgb_times[self._count]
        rgb_image = cv2.imread(self._path + self._rgb_paths[self._count],
                               cv2.IMREAD_ANYCOLOR)
        depth_time = self._depth_times[self._count]
        depth_image = cv2.imread(self._path + self._depth_paths[self._count],
                                 cv2.IMREAD_ANYDEPTH) * DEPTH_SCALE

        # return next frame info (tuple)
        return rgb_time, rgb_image, depth_time, depth_image
