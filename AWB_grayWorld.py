import cv2 as cv
import numpy as np

src = cv.imread("test.jpg")
#cv.imshow(src)

b_avg = np.mean(src[:, :, 0])
g_avg = np.mean(src[:, :, 1])
r_avg = np.mean(src[:, :, 2])
# averaged gray
k = (b_avg + g_avg + r_avg)/3

b_gain = k / b_avg
g_gain = k / g_avg
r_gain = k / r_avg


src1 = np.zeros(src.shape)
src1[:, :, 0] = src[:, :, 0] * b_gain
src1[:, :, 1] = src[:, :, 1] * g_gain
src1[:, :, 2] = src[:, :, 2] * r_gain


src1 = src1.astype(np.uint8)

img = np.hstack([src, src1])
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
cv.imshow('input_image', img)
cv.waitKey(0)
cv.destroyAllWindows()