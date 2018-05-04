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


# To apply:
# Low-pass filter
# Band-pass filter
# High-pass filter
# Noise reduction filter (average, median)
# Gradient operators (Sobel, Prewitt, Isotropic, Compass, line detection)
# Laplacian operators
# Histogram analysis and tresholding

########################################################i

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

def imageRotaton(img, degrees):
    rows, cols = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

def circleDetecion(img, img2):
    src = cv2.medianBlur(img,5)
    #src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, 100 , param1=50,param2=30,
                               minRadius=0,maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print("nuevo:\n\n")
        for i in circles[0,:]:
            print(i)
            cv2.circle(img2,(i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(img,(i[0], i[1]), i[2], (0,255,0), 2)
            cv2.circle(img2, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv2.circle(img,(i[0], i[1]), i[2], (0,255,0), 2)
        return img2, circles

    #cv2.imshow('detected circles', img)
    print("sin circulos")

def thresholdBinary (img):
    # Pasar una imagen a escala de grises
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # thresholdBin ahora es la imagen a la que aplico el operador threshold binario
    ret, thresholdBin = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # regresar la imagen
    return thresholdBin
def distance(x1, y1, x2, y2):

    xs = (x2 - x1) * (x2 - x1)
    ys = (y2 - y1) * (y2 -y1)
    rs = xs + ys
    res = pow(rs, 1/2)
    return res

def colorDetection(img, circle):
    lista = []
    total = [0,0,0]
    count = 0
    radius = circle[0][0][2]
    x1 = circle[0][0][0]
    y1 = circle[0][0][1]
    counti = 0
    countj = 0
    for i in range (0,499):
        for j in range (0,499):
            distanceS = distance(x1, y1, i, j)
            print(j)
            print(distanceS)
            if distanceS < radius:
                if np.all(img[i][j] > 100):
                    b,g,r = img[i][j]
                    lista.append([r,g,b])
                    total[0] = total[0] + b
                    total[1] = total[1] + g
                    total[2] = total[2] + r

                    count = count + 1
    total[0] = total[0] / count
    total[1] = total[1] / count
    total[2] = total[2] / count
    print(total[0])
    print(total[1])
    print(total[2])


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

def recognition(image):
    face_cascade = cv2.CascadeClassifier('/Users/jondhc/Documents/Python/Image_processing/venv/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/jondhc/Documents/Python/Image_processing/venv/lib/python3.6/site-packages/cv2/data/haarcascade_eye.xml')
    img = image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 10)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # Set new window [0], with flags [1]
    cv2.imshow('img', img)
    cv2.waitKey(0)


########################################################f

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
#cv2.imshow('image', bgrImage) # Show [1] image in a [0] window
# cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
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
recognition(bgrImage)
########################################################f

