import cam.rs2 as rs2
import seg.mrcnn as mrcnn
import cv2


while cv2.waitKey(1) != ord('q'):

    rgb, depth = rs2.get_frames()
    rgb = rgb[:, :, ::-1]

    seg = mrcnn.run(rgb)
    viz = mrcnn.visualize(rgb, seg)

    cv2.imshow("Color", rgb)
    cv2.imshow("Depth", depth * 5)
    cv2.imshow("Segmentation", viz)
