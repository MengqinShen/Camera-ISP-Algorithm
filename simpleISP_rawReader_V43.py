from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from statistics import mean
def plot_image4(image_1, image_2,image_3,image_4,title_1="Orignal",title_2="New Image1",title_3="New Image2",title_4="New Image3"):
    plt.figure(figsize=(10,8))
    plt.subplot(2, 2, 1)
    plt.imshow(image_1)
    plt.title(title_1)
    plt.subplot(2, 2, 2)
    plt.imshow(image_2)
    plt.title(title_2)
    plt.subplot(2, 2, 3)
    plt.imshow(image_3)
    plt.title(title_3)
    plt.subplot(2, 2, 4)
    plt.imshow(image_4)
    plt.title(title_4)
    plt.show()
def plot_rawImage2(image_1, image_2,title_1="Orignal",title_2="New Image"):
    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_1, cmap='gray')
    plt.title(title_1)
    plt.subplot(1, 2, 2)
    plt.imshow(image_2, cmap='gray')
    plt.title(title_2)
    plt.show()

def wbscalematrix(m, n, wb_scales, align):
    # Makes a white-balance scaling matrix for an image of size m-by-n
    # from the individual RGB white balance scalar multipliers [wb_scales] = [R_scale G_scale B_scale].
    # [align] is string indicating the 2x2 Bayer arrangement:

    scalematrix = wb_scales[1] * np.ones((m,n)) # Initialize to all green values

    # Fill in the scales for the red and blue pixels across the matrix
    if (align == 'rggb'):
        scalematrix[0::2, 0::2] = wb_scales[0]  # r
        scalematrix[1::2, 1::2] = wb_scales[2]  # b

    elif (align == 'bggr'):
        scalematrix[1::2, 1::2] = wb_scales[0] # r
        scalematrix[0::2, 0::2] = wb_scales[2] # b
    elif (align == 'grbg'):
        scalematrix[0::2, 1::2] = wb_scales[0] # r
        scalematrix[0::2, 1::2] = wb_scales[2] # b
    elif (align == 'gbrg'):
        scalematrix[1::2, 0::2] = wb_scales[0] # r
        scalematrix[0::2, 1::2] = wb_scales[2] # b
    return scalematrix


def apply_cmatrix(img, cmatrix):
    # Applies color transformation CMATRIX to RGB input IM.
    # Finds the appropriate weighting of the old color planes to form the new color planes,
    # equivalent to but much more efficient than applying a matrix transformation to each pixel.
    if (img.shape[2] != 3):
        raise ValueError('Apply cmatrix to RGB image only.')

    r = cmatrix[0,0] * img[:,:,0] + cmatrix[0,1] * img[:,:,1] + cmatrix[0,2] * img[:,:,2]
    g = cmatrix[1,0] * img[:,:,0] + cmatrix[1,1] * img[:,:,1] + cmatrix[1,2] * img[:,:,2]
    b = cmatrix[2,0] * img[:,:,0] + cmatrix[2,1] * img[:,:,1] + cmatrix[2,2] * img[:,:,2]
    corrected = np.stack((r,g,b), axis=2)
    return corrected


def debayering(input):
    # Bilinear Interpolation of the missing pixels
    # Assumes a Bayer CFA in the 'rggb' layout
    #   R G R G
    #   G B G B
    #   R G R G
    #   G B G B
    # Input: Single-channel rggb Bayered image  # Returns: A debayered 3-channels RGB image
    img = input.astype(np.double)
    m = img.shape[0]
    n = img.shape[1]

    # Create indicator masks that tells where each of the color pixels are in the bayered input image
    # 1 indicates presence of that color, 0 otherwise
    red_mask = np.tile([[1,0],[0,0]], (int(m/2), int(n/2)))

    # indicator masks for the green and blue channels
    green_mask = np.tile([[0,1],[1,0]], (int(m/2), int(n/2)))
    blue_mask = np.tile([[0,0],[0,1]], (int(m/2), int(n/2)))

    r = np.multiply(img, red_mask)
    g = np.multiply(img, green_mask)
    b = np.multiply(img, blue_mask)


    # Fill in the missing values in r,g,b with filtering - convolution - to implement bilinear interpolation.
    # Interpolating green:
    filter_g = 0.25 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_g = convolve2d(g, filter_g, 'same')
    g = g + missing_g

    # Interpolating blue:
    # Step 1:
    filter1 = 0.25 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    missing_b1 = convolve2d(b, filter1, 'same')
    # Step 2:
    filter2 = 0.25 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_b2 = convolve2d(b + missing_b1, filter2, 'same')
    b = b + missing_b1 + missing_b2


    # Interpolation for the red at the missing points
    # Step 1:
    filter3 = 0.25 * np.array([[1,0,1],[0,0,0],[1,0,1]])
    missing_r1 = convolve2d(r, filter3, 'same')
    # Step 2:
    filter4 = 0.25 * np.array([[0,1,0],[1,0,1],[0,1,0]])
    missing_r2 = convolve2d(r + missing_r1, filter4, 'same')
    r = r + missing_r1 + missing_r2

    output = np.stack((r,g,b), axis=2)
    return output




# Step 0: Convert RAW file to TIFF
# 'black': the black point in the RAW image
black = 0
saturation = 16383
# 'wb_multipliers': white balance multipliers for each of the R, G, B channels
wb_multipliers = [1.117041, 1.000000, 1.192484]

# Use the output file from command "dcraw -4 -D -T <raw_file_name>"
raw_data = Image.open('sample.tiff')
# The line below should display a blank image.
# raw_data.show()

raw = np.array(raw_data).astype(np.double)

# # Step 1: Normalization, Map raw to range [0,1].
linear_bayer = (raw-black)/(saturation-black)
print(np.max(raw),np.max(linear_bayer),np.min(raw),np.min(linear_bayer))
linear_bayer[linear_bayer > 1.0] = 1.0 # Always keep image clipped b/w 0-1
linear_bayer[linear_bayer < 0.0] = 0.0
# plt.imshow(linear_bayer, cmap='gray')
# plt.show()

# # Step 2: White balancing
mask = wbscalematrix(linear_bayer.shape[0], linear_bayer.shape[1], wb_multipliers, 'rggb')
balanced_bayer = np.multiply(linear_bayer, mask)

# plt.imshow(balanced_bayer, cmap='gray')
# plt.show()


# # Step 3: Debayering (also called demosaicing)
lin_rgb = debayering(balanced_bayer)
# plt.imshow(cv2.cvtColor(lin_rgb, cv2.COLOR_BGR2RGB))
# plt.show()

# # Step 4: Color space conversion --- we do this one for you.
# # Convert to sRGB. xyz2cam is found in dcraw's source file adobe_coeff.
rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
           [0.2126729, 0.7151522, 0.0721750],
           [0.0193339, 0.1191920, 0.9503041]])
xyz2cam = np.array([[0.6653, -0.1486, -0.0611],
           [-0.4221, 1.3303, 0.0929],
           [-0.0881, 0.2416, 0.7226]])
rgb2cam = xyz2cam * rgb2xyz # Assuming previously defined matrices
denom = np.tile(np.reshape(np.sum(rgb2cam,axis=1),(3,-1)), (1,3))
rgb2cam = np.divide(rgb2cam, denom) # Normalize rows to 1
cam2rgb = np.linalg.inv(rgb2cam)
lin_srgb = apply_cmatrix(lin_rgb, cam2rgb)
lin_srgb[lin_srgb > 1.0] = 1.0 # Always keep image clipped b/w 0-1
lin_srgb[lin_srgb < 0.0] = 0.0



# # Step 5: Brightness and gamma correction
r, g, b = lin_srgb[:, :, 0], lin_srgb[:, :, 1], lin_srgb[:, :, 2]
gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
delta_luminance = np.max(gray) / 4 - np.mean(gray)

# Scale the image by multiplying each pixel by the ratio of target to mean
bright_srgb = lin_srgb + delta_luminance
#
# plt.imshow(bright_srgb)
# plt.show()
gamma = 1/2.0
nl_srgb = np.power(bright_srgb, gamma)
# plt.imshow(nl_srgb)
# plt.show()
plot_rawImage2(raw,balanced_bayer,title_1="Raw",title_2="Raw AWB")
plot_image4(lin_rgb,lin_srgb,bright_srgb,nl_srgb,title_1="demosaicing",title_2="CSC correction",title_3="brightness correction",title_4="gamma correction")