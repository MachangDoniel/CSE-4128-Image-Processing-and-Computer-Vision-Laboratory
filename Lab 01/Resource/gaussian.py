import numpy as np
import cv2

img = cv2.imread('box.jpg',0)
cv2.imshow('grayscaled image',img)
#void cv::GaussianBlur(InputArray src,OutputArray dst,Size ksize,double sigmaX,double sigmaY = 0,int borderType = BORDER_DEFAULT)
#blur = cv2.GaussianBlur(img,(5,5),1)
#cv2.imshow('output image1',blur)




kernel =(1/273) * np.array([[1, 4, 7, 4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1, 4, 7, 4,1]])

#void cv::filter2D(InputArray src, OutputArray dst,int ddepth,InputArray kernel,Point anchor = Point(-1,-1),double delta = 0,intborderType = BORDER_DEFAULT )	
out = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
cv2.imshow("output 2",out)


cv2.waitKey(0)
cv2.destroyAllWindows() 