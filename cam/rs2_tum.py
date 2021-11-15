import cam.rs2

import cv2
import os
import shutil


# folder name
FOLDER_ROOT = 'data/tum/rs2'
FOLDER_COLOR = 'rgb'
FOLDER_DEPTH = 'depth'

# file name
FILE_COLOR = 'rgb.txt'
FILE_DEPTH = 'depth.txt'
FILE_ASSOC = 'associations.txt'

# depth multiplier
DEPTH_MULTIPLIER = 5

# remove(clean) folders
if os.path.exists(FOLDER_ROOT):
    shutil.rmtree(FOLDER_ROOT)

# make folders
os.mkdir(FOLDER_ROOT)
os.mkdir(os.path.join(FOLDER_ROOT, FOLDER_COLOR))
os.mkdir(os.path.join(FOLDER_ROOT, FOLDER_DEPTH))

# open files
file_color = open(os.path.join(FOLDER_ROOT, FILE_COLOR), 'wt')
file_depth = open(os.path.join(FOLDER_ROOT, FILE_DEPTH), 'wt')
file_assoc = open(os.path.join(FOLDER_ROOT, FILE_ASSOC), 'wt')

# print info
print('Frame Size:', cam.rs2.get_frame_size())
print('Intrinsic Parameters:', cam.rs2.get_intrinsics())

# main loop
while cv2.waitKey(1) != ord('q'):

    # get timestamp
    timestamp = cam.rs2.get_timestamp()

    # get color and depth frame
    color, depth = cam.rs2.get_frames()

    # convert color and depth frames to display
    color = color[:, :, ::-1]  # RGB to BGR
    depth = depth * DEPTH_MULTIPLIER  # apply multiplier

    # show color and depth images
    cv2.imshow('Color', color)
    cv2.imshow('Depth', depth)

    # write color and depth images
    cv2.imwrite(os.path.join(FOLDER_ROOT, FOLDER_COLOR, timestamp + '.png'), color)
    cv2.imwrite(os.path.join(FOLDER_ROOT, FOLDER_DEPTH, timestamp + '.png'), depth)

    # write list files
    file_color.write(timestamp + ' ' + os.path.join(FOLDER_COLOR, timestamp + '.png') + '\n')
    file_depth.write(timestamp + ' ' + os.path.join(FOLDER_DEPTH, timestamp + '.png') + '\n')
    file_assoc.write(timestamp + ' ' + os.path.join(FOLDER_COLOR, timestamp + '.png') + ' ' +
                     timestamp + ' ' + os.path.join(FOLDER_DEPTH, timestamp + '.png') + '\n')

# close files
file_color.close()
file_depth.close()
file_assoc.close()
