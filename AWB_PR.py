import cv2 as cv
import numpy as np

#src = cv.imread("test.jpg")
src = cv.imread("AS.png")
cv.imshow('orig_image', src)


b_gain = 255 / np.max(src[:, :, 0])
g_gain = 255 / np.max(src[:, :, 1])
r_gain = 255 / np.max(src[:, :, 2])

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