import cv2
import numpy as np
from create_kernel import *
from extractor_merger import *
from convolution import *

def convolution_rgb(image, kernel, kernel_center=(-1,-1)):
    red, green, blue = extract_rgb(image)

    red_conv = convolution(image=red, kernel=kernel, kernel_center=kernel_center)
    red_nor = normalize(red_conv)
    
    green_conv = convolution(image=green, kernel=kernel, kernel_center=kernel_center)
    green_nor = normalize(green_conv)
    
    blue_conv = convolution(image=blue, kernel=kernel, kernel_center=kernel_center)
    blue_nor = normalize(blue_conv)
    
    cv2.imshow("Red", red_nor)
    cv2.imshow("green", green_nor)
    cv2.imshow("blue", blue_nor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    merged = merge_rgb(red=red_nor, green=green_nor, blue=blue_nor)
    cv2.imshow("Original image", image)
    cv2.imshow("Merged image", merged)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    return merged


#image_path = '.\images\\shape.jpg'
#image = cv2.imread( image_path )

#kernel = createGaussianKernel(sigmaX=1,sigmaY=1)

#convolution_rgb(image=image, kernel=kernel, kernel_center=(-1,-1))
