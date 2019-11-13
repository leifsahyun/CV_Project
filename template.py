#Initial template code for 2019 computer vision project
#Python 3.6, OpenCV 2
#Leif Sahyun 11/13/2019
import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt

#Read image
img_uri = 'example.jpg'
img = cv2.imread(img_uri, cv2.IMREAD_COLOR)

#Split image into colors for later processing
red = copy.copy(img[..., 0])
green = copy.copy(img[..., 1])
blue = copy.copy(img[..., 2])

#Combine image colors in bgr order expected by OpenCV
bgrImg = np.zeros(img.shape, dtype=np.uint8)
bgrImg[..., 0] = blue
bgrImg[..., 1] = green
bgrImg[..., 2] = red

##### Code in this section could probably be replaced by Canny edge detection #####

#Gaussian blur
blurSize = 5
blurImg = cv2.GaussianBlur(bgrImg, (blurSize, blurSize), 0)

#Convert to grayscale as first step to detect edges
grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)
cv2.imshow("first", blurImg)

#Use Sobel method to find edges
sobelImg = cv2.Sobel(grayImg, cv2.CV_16S, 1, 1)
sobel8UImg = sobelImg.astype(np.uint8)

#Threshold edges, keeping only those above threshold value
#Edges are black so that white regions can be more easily labeled
edgeThreshValue = 3
ret, sobelBinaryImg = cv2.threshold(sobel8UImg, edgeThreshValue, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("binary", sobelBinaryImg)
#Edges variable contains the image of the edges for future use
edges = sobelBinaryImg

##### End edge detection section #####

#Noise removal
kernel = np.ones((3,3),np.uint8)
denoised = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel, iterations = 1)
denoised = cv2.morphologyEx(denoised,cv2.MORPH_ERODE,kernel, iterations = 1)
cv2.imshow("pretty", denoised)

#Label regions in the denoised image
ret, labeled = cv2.connectedComponents(denoised)
labeled = labeled+1
plt.imshow(labeled,cmap='Pastel1')
plt.show()

#Segment image using watershed algorithm
segmented = cv2.watershed(blurImg,labeled)
plt.imshow(segmented,cmap='Pastel1')
plt.show()
