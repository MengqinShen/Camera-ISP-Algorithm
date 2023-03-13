import cv2 as cv
import numpy as np

src0 = cv.imread("AS.png")
#src0 = cv.imread("test.jpg")
src = src0.astype(np.uint16)

# get average value of each color
b_ave = np.mean(src[:, :, 0])
g_ave = np.mean(src[:, :, 1])
r_ave = np.mean(src[:, :, 2])

# max value of each color
b_max = np.max(src[:, :, 0])
g_max = np.max(src[:, :, 1])
r_max = np.max(src[:, :, 2])

# calculate factors using QCGP equation
k_ave = (b_ave + g_ave + r_ave)/3
k_max = (b_max + g_max + r_max)/3
k_matrix = np.mat([[k_ave], [k_max]])

# converted matrix
b_coefficient_matrix = np.mat([[b_ave * b_ave, b_ave],
                               [b_max * b_max, b_max]])
b_conversion_matrix = b_coefficient_matrix.I * k_matrix

b = (src[:, :, 0]).transpose()
bb = (src[:, :, 0] * src[:, :, 0]).transpose()
b = np.stack((bb, b), axis=0).transpose()
b_des = np.dot(b, np.array(b_conversion_matrix))
b_des = b_des.astype(np.uint8).reshape([280, 471])

#
g_coefficient_matrix = np.mat([[g_ave * g_ave, g_ave],
                               [g_max * g_max, g_max]])
g_conversion_matrix = g_coefficient_matrix.I * k_matrix

g = (src[:, :, 1]).transpose()
gg = (src[:, :, 1] * src[:, :, 1]).transpose()
g = np.stack((gg, g), axis=0).transpose()
g_des = np.dot(g, np.array(g_conversion_matrix))
g_des = g_des.astype(np.uint8).reshape([280, 471])

#
r_coefficient_matrix = np.mat([[r_ave * r_ave, r_ave],
                               [r_max * r_max, r_max]])
r_conversion_matrix = r_coefficient_matrix.I * k_matrix

r = (src[:, :, 2]).transpose()
rr = (src[:, :, 2] * src[:, :, 2]).transpose()
r = np.stack((rr, r), axis=0).transpose()
r_des = np.dot(r, np.array(r_conversion_matrix))
r_des = r_des.astype(np.uint8).reshape([280, 471])

#
src1 = np.zeros(src.shape).astype(np.uint8)
src1[:, :, 0] = b_des
src1[:, :, 1] = g_des
src1[:, :, 2] = r_des


img = np.hstack([src0, src1])
cv.namedWindow("AWB", cv.WINDOW_AUTOSIZE)
cv.imshow("AWB", img)
cv.waitKey(0)
cv.destroyAllWindows()