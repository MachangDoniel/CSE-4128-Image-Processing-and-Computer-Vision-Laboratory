import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('lena.jpg')
print(img.shape)

#%%
#gray = np.ones((128,128,3), dtype=np.uint8)
cv2.imshow("Color",img)
b1,g1,r1 = cv2.split(img)
#print(b.shape)
cv2.imshow("Green",g1)
cv2.imshow("Red",r1)
cv2.imshow("Blue",b1)
cv2.waitKey(0)

b = img.copy()
# set green and red channels to 0
b[:, :, 1] = 0
b[:, :, 2] = 0

g = img.copy()
# set blue and red channels to 0
g[:, :, 0] = 0
g[:, :, 2] = 0

r = img.copy()
# set blue and green channels to 0
r[:, :, 0] = 0
r[:, :, 1] = 0

cv2.imshow('B-RGB', b)
cv2.imshow('G-RGB', g)
cv2.imshow('R-RGB', r)
merged = cv2.merge((b1,g1,r1))
cv2.imshow("Merged",merged )
cv2.waitKey(0)
img2 = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
cv2.imshow("hsv",img2 )
img3 = cv2.cvtColor(img2,cv2.COLOR_HSV2RGB)
#cv2.imshow("rgb",img3 )

#%%
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("GRAY", gray)

#%%
cv2.waitKey(0)
plt.figure(1)
im = plt.imread("Lena.jpg")
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
plt.title("GRAY IMG")
plt.imshow(gray, cmap = 'gray', vmin=0, vmax=255)
plt.show()

plt.figure(2)
im = plt.imread("Lena.jpg")
plt.title("GRAY IMG2")
plt.imshow(im)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()