import numpy
import cv2
from matplotlib import pyplot

bgrKids = cv2.imread('kids.jpg', -1) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bgrAnimal = cv2.imread('animal.jpg', -1)
bgrCity = cv2.imread('city.jpg', -1)
bgrLandscape = cv2.imread('landscape.jpg', -1)
bgrNature = cv2.imread('nature.jpg', -1)
bgrPortrait = cv2.imread('portrait.jpg', -1)
bgrReef = cv2.imread('reef.jpg', -1)
bgrScenery = cv2.imread('scenery.jpg', -1)

bgrImage = bgrPortrait

b,g,r = cv2.split(bgrImage) # Split bgr image
rgbImage = cv2.merge([r,g,b]) # Merge using rgb order


cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('image', bgrImage) # Show [1] image in a [0] window
cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result.jpg', bgrImage) # Save [1] image in the working directory with specified name [0]

def thresholdBinary (img):
    # Pasar una imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholdBin ahora es la imagen a la que aplico el operador threshold binario
    ret, thresholdBin = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # regresar la imagen
    return thresholdBin

def thresholdBinaryInv(img):
    # Pasar una imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholdBinInv ahora es la imagen a la que aplico el operador threshold binario invertido
    ret, thresholdBinInv = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)

    # regresar la imagen
    return thresholdBinInv

def contrastStretch(img):
    # Pasar una imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #dimensiones de la imagen
    row = gray_img.shape[0]
    col = gray_img.shape[1]

    #min and max values of gray image
    smallest = numpy.amin(gray_img)
    biggest = numpy.amax(gray_img)

    #recorrido de la imagen
    for y in range(0, row):
        for x in range(0, col):
            pixelValue = gray_img[y, x]
            gray_img[y, x] = ((pixelValue - smallest)/(biggest - smallest))*255

    #regresar la imagen
    return gray_img

# POSSIBLY the Inverted Gray Scale Threshold Operator
def grayInverse(img):
    # Pasar una imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert gray scale image
    invertedImg = cv2.bitwise_not(gray_img)

    #regresar la imagen
    return invertedImg

# POSSIBLY the Inverse Operator
def colorInverse(img):
    b, g, r = cv2.split(img)  # Split bgr image
    b = 255 - b # invert b component
    g = 255 - g # invert g component
    r = 255 - r # invert r component
    bgrImgInverted = cv2.merge([b, g, r])  # Merge using bgr order
    return bgrImgInverted

# original image in gray scale
originalGray = cv2.cvtColor(bgrImage, cv2.COLOR_BGR2GRAY)
# originalGray with contrast stretch
res1 = contrastStretch(bgrImage)
# original with threshold binary
res2 = thresholdBinary(bgrImage)
# original with threshold binary inverted
res3 = thresholdBinaryInv(bgrImage)
# original with gray scale inverted
res4 = grayInverse(bgrImage)
# original with inverted colors
res5 = colorInverse(bgrImage)

cv2.namedWindow('thresholdBinary', cv2.WINDOW_NORMAL)
cv2.imshow('thresholdBinary', res2)
cv2.namedWindow('thresholdBinaryInverted', cv2.WINDOW_NORMAL)
cv2.imshow('thresholdBinaryInverted', res3)
cv2.namedWindow('originalGray', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('originalGray', originalGray) # Show [1] image in a [0] window
cv2.namedWindow('contrastStretch', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('contrastStretch', res1) # Show [1] image in a [0] window
cv2.namedWindow('grayScaleInverted', cv2.WINDOW_NORMAL)
cv2.imshow('grayScaleInverted', res4)
cv2.namedWindow('colorInverted', cv2.WINDOW_NORMAL)
cv2.imshow('colorInverted', res5)
cv2.namedWindow('ORIGINAL', cv2.WINDOW_NORMAL)
cv2.imshow('ORIGINAL', bgrImage)
cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

# cv2addition = cv2.add(bgrReef, bgrCity) # Saturated operation
