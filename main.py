import numpy as np
import cv2
from matplotlib import pyplot

def imageAdeition(img1, img2):
    img3 = cv2.add(img1,img2)
    return img3

bgrKids = cv2.imread('kids.jpg', -1) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bgrAnimal = cv2.imread('animal.jpg', -1)
bgrCity = cv2.imread('city.jpg', -1)
bgrLandscape = cv2.imread('landscape.jpg', -1)
bgrNature = cv2.imread('nature.jpg', -1)
bgrPortrait = cv2.imread('portrait.jpg', -1)
bgrReef = cv2.imread('reef.jpg', -1)

bgrImage = bgrPortrait

#b,g,r = cv2.split(bgrImage) # Split bgr image
#rgbImage = cv2.merge([r,g,b]) # Merge using rgb order

res = imageAdeition(bgrAnimal,bgrCity)

cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('image', res) # Show [1] image in a [0] window
cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result.jpg', res) # Save [1] image in the working directory with specified name [0]

#pyplot.imshow(rgbImage, cmap='gray', interpolation='bicubic') # Load [0] image ... Loaded in RGB.
#pyplot.show() # Display figure

# cv2addition = cv2.add(bgrReef, bgrCity) # Saturated operation
