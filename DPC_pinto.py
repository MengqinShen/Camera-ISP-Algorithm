import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv

def judgeDefectPixel(aroundP, currentP, Th):
    ## get the  median  value of the around list
    medianV = np.median(aroundP)
    ## get the difference between the around pixel and the current pixel
    diff = aroundP - np.ones(1, len(aroundP)) * currentP
    ## if all difference bigger than 0 or all smaller than 0 and all abs of the diff are bigger than Th,
    # that pixel is a defect pixel and replace it with the median
    pos_diff = [y for y in diff if y > 0]
    neg_diff = [y for y in diff if y < 0]
    abs_diff = [y for y in diff if abs(y) > Th]
    if (len(pos_diff) == len(aroundP)) or (len(neg_diff) == len(aroundP)):
        if len(abs_diff) == len(aroundP):
            correctP = medianV
#        else:
#            correctP = currentP

    return currentP


row = 4208
col = 3120

path = 'HisiRAW_4208x3120_8bits_RGGB.raw'
rawData = np.fromfile(path, dtype=np.uint8)
#print(np.shape(rawData))
imageSize = (row, col)
rawData = rawData.reshape(imageSize)

[height, width] =np.shape(rawData)
expandNum = 2
img_expand = np.zeros([height+expandNum*2, width+expandNum*2])
img_expand[expandNum:height+expandNum, expandNum:width+expandNum] = rawData[:,:]
img_expand[0:expandNum, expandNum:width+expandNum] = rawData[0:expandNum,:]
img_expand[height+expandNum:height+expandNum*2, expandNum:width+expandNum] = rawData[height-expandNum+1:height,:]
img_expand[:,0:expandNum] = img_expand[:, expandNum:2*expandNum]
img_expand[:,width+expandNum+1:width+2*expandNum] = img_expand[:, width+1:width+expandNum]


Th = 30

disImg = np.zeros([height, width])
for i in range(expandNum, 2, height+expandNum):
    for j in range(expandNum,2, width+expandNum):
        ## R get the pixel around the current R pixel
        around_R_pixel = [img_expand[i-2, j-2], img_expand[i-2, j], img_expand[i-2, j+2],img_expand[i, j-2], img_expand[i, j+2], img_expand[i+2, j-2], img_expand[i+2, j], img_expand[i+2, j+2]]
        print(around_R_pixel)
        disImg[i-expandNum, j-expandNum] = judgeDefectPixel(around_R_pixel, img_expand[i, j], Th)
        # Gr get the pixel around the current Gr pixel
        around_Gr_pixel = [img_expand[i-1, j], img_expand[i-2, j+1], img_expand[i-1, j+2], img_expand[i, j-1],img_expand[i, j+3], img_expand[i+1, j], img_expand[i+2, j+1], img_expand[i+1, j+2]]
        disImg[i-expandNum, j-expandNum+1] = judgeDefectPixel(around_Gr_pixel, img_expand[i, j+1], Th)
        # B get the pixel around the current B pixel
        around_B_pixel = [img_expand[i-1, j-1], img_expand[i-1, j+1], img_expand[i-1, j+3], img_expand[i+1, j-1], img_expand[i+1, j+3], img_expand[i+3, j-1], img_expand[i+3, j+1], img_expand[i+3, j+3]]
        disImg[i-expandNum+1, j-expandNum+1] = judgeDefectPixel(around_B_pixel, img_expand[i+1, j+1], Th)
        # Gb get the pixel around the current Gb pixel
        around_Gb_pixel = [img_expand[i, j-1], img_expand[i-1, j], img_expand[i, j+1], img_expand[i+1, j-2], img_expand[i+1, j+2], img_expand[i+2, j-1], img_expand[i+3, j], img_expand[i+2, j+1]]
        disImg[i-expandNum+1, j-expandNum] = judgeDefectPixel(around_Gb_pixel, img_expand[i+1, j], Th)


#img = np.hstack([rawData, disImg])
cv.namedWindow('raw_image', cv.WINDOW_KEEPRATIO)
cv.imshow('raw_image', rawData)
#cv.resizeWindow('raw_image window', 1000, 800)
cv.waitKey(0)
cv.namedWindow('corrected_image', cv.WINDOW_KEEPRATIO)
cv.imshow('corrected_image', disImg)
#cv.resizeWindow('corrected_image window', 1000, 800)
cv.waitKey(0)
cv.destroyAllWindows()

