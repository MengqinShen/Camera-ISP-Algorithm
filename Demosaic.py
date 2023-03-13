import numpy as np
import cv2 as cv
import rawpy

# filePath = 'kodim19_8bits_RGGB.raw'
# width  = 512 #512
# height = 768 #768
#
# bayerData = np.fromfile(filePath, dtype=np.uint8)
# imageSize = (height,width)
# bayerData = bayerData.reshape(imageSize)

def demosaic(bayerData):
## expand image in order to make it easy to calcu late edge pixels
    height,width = np.shape(bayerData)
    bayerPadding = np.zeros((height + 2,width+2))
    bayerPadding[1:height+1,1:width+1] = bayerData
    bayerPadding[0,:] = bayerPadding[2,:]
    bayerPadding[height+1,:] = bayerPadding[height,:]
    bayerPadding[:,0] = bayerPadding[:,2]
    bayerPadding[:,width+1] = bayerPadding[:,width]

## main code of imterpolation
    imDst = np.zeros([height+2, width+2, 3])
#print(np.shape(imDst),imDst[2][2][2])
    for ver in range(1, height + 1):
        for hor in range(1, width + 1):
    # G B -> R
            if ver%2 == 1 and hor%2 == 1 :
                imDst[ver,hor,0] = bayerPadding[ver, hor]
    # G -> R
                imDst[ver, hor, 1] = (bayerPadding[ver-1, hor] + bayerPadding[ver+1, hor] + bayerPadding[ver, hor-1] + bayerPadding[ver, hor+1])/4
    #B -> R
                imDst[ver, hor, 2] = (bayerPadding[ver-1, hor-1] + bayerPadding[ver-1, hor+1] + bayerPadding[ver+1, hor-1] + bayerPadding[ver+1, hor+1])/4
    # G R -> B
            elif ver%2 == 0 and hor%2 == 0:
                imDst[ver, hor, 2] = bayerPadding[ver, hor]
    # G -> B
                imDst[ver, hor, 1] = (bayerPadding[ver-1, hor] + bayerPadding[ver+1, hor] + bayerPadding[ver, hor-1] + bayerPadding[ver, hor+1])/4
    # R -> B
                imDst[ver, hor, 0] = (bayerPadding[ver-1, hor-1] + bayerPadding[ver-1, hor+1] + bayerPadding[ver+1, hor-1] + bayerPadding[ver+1, hor+1])/4
            elif ver%2 == 1 and hor%2 == 0:
                imDst[ver, hor, 1] = bayerPadding[ver, hor]
    # R -> Gr
                imDst[ver, hor, 0] = (bayerPadding[ver, hor-1] + bayerPadding[ver, hor+1])/2
    # B -> Gr
                imDst[ver, hor, 2] = (bayerPadding[ver-1, hor] + bayerPadding[ver+1, hor])/2
            elif ver%2 == 0 and hor%2 == 1:
                imDst[ver, hor, 1] = bayerPadding[ver, hor]
    # B -> Gb
                imDst[ver, hor, 2] = (bayerPadding[ver, hor-1] + bayerPadding[ver, hor+1])/2
    # R -> Gb
                imDst[ver, hor, 0] = (bayerPadding[ver-1, hor] + bayerPadding[ver+1, hor])/2


    imDst = imDst[1:height+1,1:width+1,:]
    imDst_new = np.zeros((height,width,3))
    imDst_new[:,:,0] = imDst[:,:,2]
    imDst_new[:,:,1] = imDst[:,:,1]
    imDst_new[:,:,2] = imDst[:,:,0]
    imDst_new = np.uint8(imDst_new)
    return imDst_new

with rawpy.imread("sample.DNG") as raw:
    bayerData = raw.raw_image.copy()
    img = raw.postprocess()

# cv.namedWindow('custom window', cv.WINDOW_KEEPRATIO)
# cv.imshow('custom window', bayerData)
# cv.resizeWindow('custom window', 1000, 800)
# cv.waitKey(0)

imDst_new = demosaic(bayerData)

cv.namedWindow('demosaicing window', cv.WINDOW_KEEPRATIO)
cv.imshow('demosaicing window', imDst_new)
cv.resizeWindow('demosaicing window', 1000, 800)
cv.waitKey(0)
# print(imDst[:,30:40,:])
# cv.namedWindow('demosaicing windowG', cv.WINDOW_KEEPRATIO)
# cv.imshow('demosaicing windowG', imDst[:,:,1])
# cv.resizeWindow('demosaicing windowG', 1000, 800)
# cv.waitKey(0)
#
# cv.namedWindow('demosaicing windowB', cv.WINDOW_KEEPRATIO)
# cv.imshow('demosaicing windowB', imDst[:,:,2])
# cv.resizeWindow('demosaicing windowB', 1000, 800)
# cv.waitKey(0)
#
# cv.namedWindow('demosaicing windowR', cv.WINDOW_KEEPRATIO)
# cv.imshow('demosaicing windowR', imDst[:,:,0])
# cv.resizeWindow('demosaicing windowR', 1000, 800)
# cv.waitKey(0)