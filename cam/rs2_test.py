import cam.rs2
import cv2


# depth multiplier to display
DEPTH_MULTIPLIER = 5


# print info
print('Frame Size:', cam.rs2.get_frame_size())
print('Intrinsic Parameters:', cam.rs2.get_intrinsics())

# main loop
while cv2.waitKey(1) != ord('q'):

    # print timestamp
    timestamp = cam.rs2.get_timestamp()
    print('Timestamp:', timestamp)

    # get color and depth frame
    color, depth = cam.rs2.get_frames()

    # convert color and depth frames to display
    color = color[:, :, ::-1]  # RGB to BGR
    depth = depth * DEPTH_MULTIPLIER  # apply multiplier

    # show color and depth images
    cv2.imshow('Color', color)
    cv2.imshow('Depth', depth)
