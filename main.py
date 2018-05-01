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
    print("Laplacian applied")
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

def horizontalSobel(image):
    print("Horizontal sobel applied")
    sobelx8u = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)
    sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel64f = numpy.absolute(sobelx64f)
    sobel_8u = numpy.uint8(abs_sobel64f)
    pyplot.subplot(1, 3, 1), pyplot.imshow(image, cmap='gray')
    pyplot.title('Original'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(1, 3, 2), pyplot.imshow(sobelx8u, cmap='gray')
    pyplot.title('Sobel CV_8U'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(1, 3, 3), pyplot.imshow(sobel_8u, cmap='gray')
    pyplot.title('Sobel abs(CV_64F)'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()

def canny(image):
    print("Canny algorithm applied")
    edges = cv2.Canny(image, 100, 200)
    pyplot.subplot(121), pyplot.imshow(image, cmap='gray')
    pyplot.title('Original Image'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.subplot(122), pyplot.imshow(edges, cmap='gray')
    pyplot.title('Edge Image'), pyplot.xticks([]), pyplot.yticks([])
    pyplot.show()
########################################################f

bgrKids = cv2.imread('kids.jpg', -1) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bgrAnimal = cv2.imread('animal.jpg', -1)
bgrCity = cv2.imread('city.jpg', -1)
bgrLandscape = cv2.imread('landscape.jpg', -1)
bgrNature = cv2.imread('nature.jpg', -1)
bgrPortrait = cv2.imread('portrait.jpg', -1)
bgrReef = cv2.imread('reef.jpg', -1)

bgrImage = bgrKids

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
bnwKids = cv2.imread('kids.jpg', 0) # Read image [0], [1] flag -1 is used to load it unchanged. Loaded in BGR.
bnwAnimal = cv2.imread('animal.jpg', 0)
bnwCity = cv2.imread('city.jpg', 0)
bnwLandscape = cv2.imread('landscape.jpg', 0)
bnwNature = cv2.imread('nature.jpg', 0)
bnwPortrait = cv2.imread('portrait.jpg', 0)
bnwReef = cv2.imread('reef.jpg', 0)

bnwImage = bnwKids


averaging(rgbImage)
blurring(rgbImage)
gaussianBlurring(rgbImage)
medianBlur(rgbImage)
bilateralFiltering(rgbImage)
laplacian(bnwImage) # B&W photo needs to be passed.
horizontalSobel(bnwImage) # B&W photo needs to be passed.
canny(bnwImage) # B&W photo needs to be passed.
########################################################f

