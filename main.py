import numpy as np
import cv2
from matplotlib import pyplot as plt

def imageAdition(img1, img2):
    img3 = cv2.add(img1,img2)
    return img3

def imageConvolution(img):
    kernel = np.ones((7, 7), np.float32) / 50
    dst = cv2.filter2D(img, -1, kernel)
    return dst

def imageSubstraction(img1, img2):
    img3 = cv2.subtract(img1,img2)
    return img3

def imageRotaton(img, ):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst
#------------------------------------Load Images------------------------------------
#------------------------------------Load Images------------------------------------
#------------------------------------Load Images------------------------------------

bgrKids = cv2.imread('kids.jpg', -1) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bgrAnimal = cv2.imread('animal.jpg', -1)
bgrCity = cv2.imread('city.jpg', -1)
bgrLandscape = cv2.imread('landscape.jpg', -1)
bgrNature = cv2.imread('nature.jpg', -1)
bgrPortrait = cv2.imread('portrait.jpg', -1)
bgrReef = cv2.imread('reef.jpg',0)
bgrJaja = cv2.imread('jaja.jpg',0)

res = imageAdition(bgrAnimal, bgrCity)
res2 = imageSubstraction(bgrAnimal, bgrCity)

res3 = imageConvolution(bgrAnimal)

res4 = imageRotaton(bgrAnimal)

cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('image', res) # Show [1] image in a [0] window
cv2.namedWindow('substraction', cv2.WINDOW_NORMAL)
cv2.imshow('substraction', res2)
cv2.namedWindow('convolution', cv2.WINDOW_NORMAL)
cv2.imshow('convolution', res3)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', bgrAnimal)
cv2.namedWindow('rotation', cv2.WINDOW_NORMAL)
cv2.imshow('rotation', res4)
cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result.jpg', res) # Save [1] image in the working directory with specified name [0]

#pyplot.imshow(rgbImage, cmap='gray', interpolation='bicubic') # Load [0] image ... Loaded in RGB.
#pyplot.show() # Display figure

# cv2addition = cv2.add(bgrReef, bgrCity) # Saturated operation
