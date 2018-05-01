import numpy
import cv2
from matplotlib import pyplot
import numpy as np;

im_in = cv2.imread("animal.jpg", cv2.IMREAD_GRAYSCALE);
th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
cv2.imshow("Thresholded Image", im_th)
cv2.waitKey(0)

image = cv2.imread('animal.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray_image.png", gray_image)
cv2.imshow('color_image',image)
cv2.imshow('gray_image',gray_image)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()


bgrKids = cv2.imread('kids.jpg', -1) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bgrAnimal = cv2.imread('animal.jpg', -1)
bgrCity = cv2.imread('city.jpg', -1)
bgrLandscape = cv2.imread('landscape.jpg', -1)
bgrNature = cv2.imread('nature.jpg', -1)
bgrPortrait = cv2.imread('portrait.jpg', -1)
bgrReef = cv2.imread('reef.jpg', -1)

bgrImage = bgrPortrait

b,g,r = cv2.split(bgrImage) # Split bgr image
rgbImage = cv2.merge([r,g,b]) # Merge using rgb order


cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('image', bgrImage) # Show [1] image in a [0] window
cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result.jpg', bgrImage) # Save [1] image in the working directory with specified name [0]

pyplot.imshow(rgbImage, cmap='gray', interpolation='bicubic') # Load [0] image ... Loaded in RGB.
pyplot.show() # Display figure

# cv2addition = cv2.add(bgrReef, bgrCity) # Saturated operation
