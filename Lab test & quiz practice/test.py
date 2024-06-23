# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:55:01 2024

@author: Doniel
"""

import numpy as np
import cv2

img = cv2.imread("Lena.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image",img)

img = cv2.copyMakeBorder(src= img, top= 2, bottom= 2, left= 2, right= 2, borderType = cv2.BORDER_CONSTANT)
cv2.imshow("GrayScaled Image",img)

kernel =(1/273) * np.array([[1, 4, 7, 4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1, 4, 7, 4,1]])

print("Image size: ",img.shape[1],"*",img.shape[0])
out=img.copy() # or use out=img
print("Out as matrix")
print(out)

n=int(kernel.shape[0]/2)

for x in range(n, img.shape[0]-n):
    for y in range(n, img.shape[1]-n):
        res=0
        for j in range(-n, n+1):
            for i in range(-n, n+1):
                f= kernel.item(i,j)
                ii= img.item(x-i,y-j)
                res+= (f*ii)
        
        out[x,y]=res

print(out)
cv2.imshow("Non-normalized Output image",out)

cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)
out= np.round(out).astype(np.uint8)

print(out)
cv2.imshow("Normalized Output image",out)

cv2.waitKey(0)
cv2.destroyAllWindows()