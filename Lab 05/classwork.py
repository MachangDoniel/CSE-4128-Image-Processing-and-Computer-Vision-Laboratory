import cv2
import numpy as np

def calculate_descriptors(image,i):
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    border = img - eroded
    cv2.imshow('Border'+str(i), border)
    cv2.imshow('Input image'+str(i), img)


image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png']

for i in range(len(image_name)):
    img = cv2.imread(image_name[i], 0)
    calculate_descriptors(img,i)

cv2.waitKey(0)
cv2.destroyAllWindows()    