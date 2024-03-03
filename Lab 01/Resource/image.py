import numpy as np
import cv2

img = cv2.imread('lena.jpg',cv2.IMREAD_GRAYSCALE)
img_bordered = cv2.copyMakeBorder(src=img, top=25, bottom=25, left=25, right=25,borderType= cv2.BORDER_CONSTANT)
cv2.imshow('grayscaled image',img)
cv2.imshow('bordered image',img_bordered)
#out=img.copy()
out = np.zeros((512,512)) #, dtype=np.uint8)
print(img.max())
print(img.min())

cv2.imshow('output image',out)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a = img.item(i,j)
        out[i,j] = a-100
        #out.itemset((i,j),a+122) #255-a)
cv2.waitKey(0)      
cv2.imshow('output image',out)
print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)
print(out)
cv2.imshow('normalised output image',out)


cv2.waitKey(0)
cv2.destroyAllWindows()