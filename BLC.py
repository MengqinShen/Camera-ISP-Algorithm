import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

path = "HisiRAW_4208x3120_8bits_RGGB.raw"
row = 4208
col = 3120
bits = 8

npimg = np.fromfile(path, dtype=np.uint8)
imageSize = (row, col)
npimg = npimg.reshape(imageSize)
#plt.plot(npimg[:,2:col])
#plt.show()

R  = npimg[0:2:row-1, 0:2:col-1]
Gr = npimg[0:2:row-1, 1:2:col-1]
Gb = npimg[1:2:row-1, 0:2:col-1]
B  = npimg[1:2:row-1, 1:2:col-1]

#print(np.mean(npimg))
#print(npimg[1:2:row, 1])
#print(np.mean(R),np.mean(Gr),np.mean(Gb),np.mean(B))

R_mean = round(np.mean(R))
Gr_mean = round(np.mean(Gr))
Gb_mean = round(np.mean(Gb))
B_mean = round(np.mean(B))

cR  = R-R_mean
cGr = Gr-Gr_mean
cGb = Gb-Gb_mean
cB  = B-B_mean

cData = np.zeros(np.shape(npimg))
cData[0:2:row-1, 0:2:col-1] = cR
cData[0:2:row-1, 1:2:col-1] = cGr
cData[1:2:row-1, 0:2:col-1] = cGb
cData[1:2:row-1, 1:2:col-1] = cB

img = np.hstack([npimg, cData])
cv.namedWindow('input_image', cv.WINDOW_NORMAL)
cv.imshow('input_image', img)
cv.waitKey(0)
cv.destroyAllWindows()

#cv.imshow('input_image', npimg)

cv.imshow('output_image', npimg)
cv.waitKey(0)
cv.imshow('output_image', cData)
cv.waitKey(0)
cv.destroyAllWindows()
#plt.imshow(npimg)
#plt.show()
#plt.imshow(cData)
#plt.show()
