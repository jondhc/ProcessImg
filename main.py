import numpy as np
import cv2
from matplotlib import pyplot as plt
from datetime import datetime

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

    #cv2.imshow('detected circles', img)
    return img2, circles

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
                if np.all(img[i][j] != 0) or np.all(img[i][j]) != 255:
                    r,g,b = img[i][j]
                    lista.append([r,g,b])
                    total[0] = total[0] + r
                    total[1] = total[1] + g
                    total[2] = total[2] + b

                    count = count + 1
    total[0] = total[0] / count
    total[1] = total[1] / count
    total[2] = total[2] / count
    print(total[0])
    print(total[1])
    print(total[2])






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

bgrOjo = cv2.imread('ojo.jpg',-1)
bgrOjo2 = cv2.imread('ojo2.jpg',-1)
bgrOjo3 = cv2.imread('ojo3.jpg',-1)
bgrOjo4 = cv2.imread('ojo4.jpg', -1)


"""res = imageAdition(bgrAnimal, bgrCity)
res2 = imageSubstraction(bgrAnimal, bgrCity)"""



"""cv2.namedWindow('image', cv2.WINDOW_NORMAL) # Set new window [0], with flags [1]
cv2.imshow('image', res) # Show [1] image in a [0] window
cv2.namedWindow('substraction', cv2.WINDOW_NORMAL)
cv2.imshow('substraction', res2)
cv2.namedWindow('convolution', cv2.WINDOW_NORMAL)
cv2.imshow('convolution', res3)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', bgrAnimal)"""

instanteInicial = datetime.now()
print("Iniciando")


res1_1 = thresholdBinary(bgrOjo)
res1_2, data1= circleDetecion(res1_1, bgrOjo)
cv2.namedWindow('ojo1', cv2.WINDOW_NORMAL)
cv2.imshow('ojo1', res1_2)
colorDetection(bgrOjo,data1)
print (data1)
"""res2_1 = thresholdBinary(bgrOjo2)
res2_2, data2 = circleDetecion(res2_1, bgrOjo2)
cv2.namedWindow('ojo2', cv2.WINDOW_NORMAL)
cv2.imshow('ojo2', res2_2)

res3_1 = thresholdBinary(bgrOjo3)
res3_2, data3 = circleDetecion(res3_1, bgrOjo3)
cv2.namedWindow('ojo3', cv2.WINDOW_NORMAL)
cv2.imshow('ojo3', res3_2)



res4_1 = thresholdBinary(bgrOjo4)
res4_2, data4= circleDetecion(res4_1, bgrOjo4)
cv2.namedWindow('ojo4', cv2.WINDOW_NORMAL)
cv2.imshow('ojo4', res4_2)
"""
instanteFinal = datetime.now()
time = instanteFinal - instanteInicial
print(time.seconds)

cv2.waitKey(0) # Wait [0] miliseconds for a keyboard event for the program to continue
cv2.destroyAllWindows() # Destroy created windows

cv2.imwrite('result2.jpg', res1_2) # Save [1] image in the working directory with specified name [0]

#pyplot.imshow(rgbImage, cmap='gray', interpolation='bicubic') # Load [0] image ... Loaded in RGB.
#pyplot.show() # Display figure

# cv2addition = cv2.add(bgrReef, bgrCity) # Saturated operation
