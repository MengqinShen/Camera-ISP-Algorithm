import rawpy
import numpy as np
import cv2 as cv
from  scipy.linalg import fractional_matrix_power

#def extraInfoFromRaw(file):
#    with rawpy.imread(file) as raw:
#    bayer = raw.raw_image.copy()
#    rgb = raw.postprocess()
#    blackLevel = raw.black_level_per_channel.copy()
#    whiteLevel = raw.white_level
#    ccm = [0.5,0.5,0.5]
#    return bayer,rgb,blackLevel,whiteLevel,ccm

def normalizeImg(rawData, black,white):
    rawMin = np.min(rawData)
    rawMax = np.max(rawData)
    nData = rawData-rawMin
    nData = nData /(rawMax-rawMin)
    nData = nData * (white-black)
    nData = nData +black


    #nData = (rawData-rawMin)/(rawMax-rawMin)*(white-black)+black
    return nData

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
    imDst_new = np.uint16(imDst_new)
    return imDst_new


def AWB_grayWorld(data):
    b_avg = np.mean(data[:, :, 0])
    g_avg = np.mean(data[:, :, 1])
    r_avg = np.mean(data[:, :, 2])
    # averaged gray
    k = (b_avg + g_avg + r_avg) / 3

    b_gain = k / b_avg
    g_gain = k / g_avg
    r_gain = k / r_avg

    awbData = np.zeros(data.shape)
    awbData[:, :, 2] = data[:, :, 2] * r_gain
    awbData[:, :, 1] = data[:, :, 1] * g_gain
    awbData[:, :, 0] = data[:, :, 0] * b_gain

    return awbData
def csc(rgbData,ccm):
    # height,width,channel = np.shape(rgbData)
    cscData = rgbData
    # ccm = np.array([1,0,0,
    #                 0,1,0,
    #                 0,0,1])

    # ccm = ccm.reshape(3,3)
    # cscData = np.matmul(rgbData, ccm)
    return cscData
def correctGamma(cscData,gamma):
    # height,width,channel = np.shape(cscData)
    finalData = np.zeros(np.shape(cscData))
    # for i in range(0,height):
    #     for j in range(0,width):
    #         for k in range(0,channel):
    #             finalData[i,j,k] = (cscData[i,j,k]/(2**14))**gamma*(2**8)
    # finalData[:,:,0] = fractional_matrix_power(np.array(cscData[:,:,0]),gamma)
    # finalData[:,:,1] = fractional_matrix_power(cscData[:,:,1],gamma)
    # finalData[:,:,2] = fractional_matrix_power(cscData[:,:,2],gamma)
    finalData = np.power(cscData/(2**14), gamma)*(2**8)
    imDst_new = np.uint8(finalData)
    return imDst_new

#def plotRawAndCorrectedImg(raw,cRaw):
#    img = np.hstack([raw,cRaw])
#    cv.imshow('raw vs corrected', img)
#    cv.waitKey(0)
#    cv.destroyAllWindows()

#file = "sample.DNG "
gamma = 1/2.2
ccm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
with rawpy.imread("sample.DNG") as raw:
    bayerData = raw.raw_image.copy()
    rgb = raw.postprocess()
    blackLevel = raw.black_level_per_channel.copy()
    whiteLevel = raw.white_level


#bayer,rgb,blackLevel,whiteLevel,ccm = extraInfoFromRaw(file)
nData = normalizeImg(bayerData, blackLevel[0],whiteLevel)
print('normalization is done!')
#print(np.min(bayer),np.max(bayer),np.min(nData),np.max(nData))

rgbData = demosaic(bayerData)
print('demosaicing is done!')
# cv.namedWindow('demosaicing window', cv.WINDOW_KEEPRATIO)
# cv.imshow('demosaicing window', rgbData)
# cv.resizeWindow('demosaicing window', 1000, 800)
# cv.waitKey(0)


awbData = AWB_grayWorld(rgbData)
print('AWB is done!')
awbData_new = np.uint16(awbData)
cv.namedWindow('AWB Img', cv.WINDOW_KEEPRATIO)
cv.imshow('AWB Img', awbData_new)
cv.resizeWindow('AWB Img', 1000, 800)
cv.waitKey(0)

cscData = csc(awbData,ccm)
print('CSC is done!')
cscData_new = np.uint16(cscData)
cv.namedWindow('CSC Img', cv.WINDOW_KEEPRATIO)
cv.imshow('CSC Img', cscData_new)
cv.resizeWindow('CSC Img', 1000, 800)
cv.waitKey(0)

finalData = correctGamma(cscData,gamma)
print(np.shape(finalData))
print(np.min(finalData),np.max(finalData),np.max(cscData))
cv.namedWindow('gamma Img', cv.WINDOW_KEEPRATIO)
cv.imshow('gamma Img', finalData)
cv.resizeWindow('gamma Img', 1000, 800)
cv.waitKey(0)
print('gamma is tuned!')




# cv.namedWindow('rgb', cv.WINDOW_KEEPRATIO)
# cv.imshow('rgb', rgb)
# cv.resizeWindow('rgb', 1000, 800)
# cv.waitKey(0)
# cv.destroyAllWindows()
# cv.namedWindow('bayer', cv.WINDOW_KEEPRATIO)
# cv.imshow('bayer', bayer)
# cv.resizeWindow('bayer', 1000, 800)
# cv.waitKey(0)



