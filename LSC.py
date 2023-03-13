import cv2 as cv
from mat4py import loadmat
import scipy.io
import numpy as np

lscRefImg = cv.imread('lscRefImg.jpg')
corTab = loadmat('corTab.mat')
#print(np.shape(lscRefImg),np.shape(corTab))
corImg = np.dot(lscRefImg, corTab)

img = np.hstack([lscRefImg, corImg])
cv.namedWindow('org vs corrected', cv.WINDOW_NORMAL)
cv.imshow('input_image', img)
cv.waitKey(0)
cv.destroyAllWindows()

