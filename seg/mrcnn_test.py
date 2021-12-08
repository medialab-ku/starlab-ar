import seg.mrcnn
import cv2


# read input image
image_input = cv2.imread('data/tum/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png')

# run Mask R-CNN
result = seg.mrcnn.run(image_input)

# visualize result
image_output = seg.mrcnn.visualize(image_input, result)

# show input and output images
cv2.imshow('Mask R-CNN Input', image_input)
cv2.imshow('Mask R-CNN Output', image_output)

# wait key input
cv2.waitKey()

# write input and output images
cv2.imwrite('image/mrcnn_input.png', image_input)
cv2.imwrite('image/mrcnn_output.png', image_output)
