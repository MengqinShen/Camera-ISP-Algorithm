import numpy as np
import cv2 as cv

imagefile = 'AS.png'
#  Read the image file and normalize to maximum for image type(uint8 or uint16)
imageRGB = cv.imread(imagefile)

# just an assumption to test code
gamma = 2.2
ccmnum = [[0.5,0,0],[0,0.5,0],[0,0,0.5]]
#print(np.max(imageRGB))
imageRGB = np.double(imageRGB)/np.max(imageRGB)
#cv.imshow('Original image: gamma = '+ str(gamma),imageRGB)
#cv.waitKey(0)


#% Linearize the image (apply inverse of encoding gamma)
linearRGB = imageRGB*(1/gamma)

# Change to 2D to apply matrix; then change back.
[my, mx, mc] = np.shape(linearRGB)
linearRGB = linearRGB.reshape(my*mx,mc)
correctedRGB = np.dot(linearRGB,ccmnum)

# Place limits on output
correctedRGB[correctedRGB >1]=1
correctedRGB[correctedRGB <0]=0
# Apply gamma for sRGB, Adobe RGB color space.
correctedRGB = correctedRGB * gamma
# Deal with saturated pixels. Not perfect, but this is what cameras do. Related to "purple fringing".
#correctedRGB(linearRGB==1) = 1
correctedRGB = correctedRGB.reshape(my, mx, mc)

#cv.imshow('Corrected image: gamma = '+ str(gamma),correctedRGB)
#cv.waitKey(0)
img = np.hstack([imageRGB, correctedRGB])
cv.imshow('CCM test ',img)
cv.waitKey(0)