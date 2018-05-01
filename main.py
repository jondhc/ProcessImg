import numpy
import cv2
from matplotlib import pyplot

# To apply:
# Low-pass filter
# Band-pass filter
# High-pass filter
# Noise reduction filter (average, median)
# Gradient operators (Sobel, Prewitt, Isotropic, Compass, line detection)
# Laplacian operators
# Histogram analysis and tresholding

########################################################i
def averaging(image):
    print("Averaging applied")
    kernel = numpy.ones((5, 5), numpy.float32) / 25
    dst = cv2.filter2D(image, -1, kernel)

    pyplot.subplot(121), pyplot.imshow(image), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(dst), pyplot.title('Averaging')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def blurring(image):
    print("Blurring applied")
    blur = cv2.blur(image, (20, 20))
    pyplot.subplot(121), pyplot.imshow(image), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(blur), pyplot.title('Blurred')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def gaussianBlurring(image):
    print("Gaussian Blurring applied")
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    pyplot.subplot(121), pyplot.imshow(image), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(blur), pyplot.title('Gaussian Blurred')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def medianBlur(image):
    print("Median Blurring applied")
    blur = cv2.medianBlur(image, 5)
    pyplot.subplot(121), pyplot.imshow(image), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(blur), pyplot.title('Median Blurred')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def bilateralFiltering(image):
    print("Bilateral Filtering applied")
    blur = cv2.bilateralFilter(image, 9, 75, 75)
    pyplot.subplot(121), pyplot.imshow(image), pyplot.title('Original')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(blur), pyplot.title('Bilateral Filtering')
    pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def laplacian(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    pyplot.subplot(2, 2, 1), pyplot.imshow(image, cmap='gray')
    pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(2, 2, 2), pyplot.imshow(laplacian, cmap='gray')
    pyplot.title('Laplacian'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(2, 2, 3), pyplot.imshow(sobelx, cmap='gray')
    pyplot.title('Sobel X'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(2, 2, 4), pyplot.imshow(sobely, cmap='gray')
    pyplot.title('Sobel Y'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

########################################################f

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
#cv2.imshow('image', bgrImage) # Show [1] image in a [0] window
# cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result.jpg', bgrImage) # Save [1] image in the working directory with specified name [0]

pyplot.imshow(rgbImage, cmap='gray', interpolation='bicubic') # Load [0] image ... Loaded in RGB.
pyplot.show() # Display figure


########################################################i
averaging(rgbImage)
blurring(rgbImage)
gaussianBlurring(rgbImage)
medianBlur(rgbImage)
bilateralFiltering(rgbImage)
laplacian(rgbImage)
########################################################f

