import cv2
import numpy as np
from create_kernel import *
from convolution import *
import math
from enum import Enum



class InputImageType(Enum):
    GRAY = 1
    RGB_HSV = 2

def perform_sobel(imagePath, conv_type = InputImageType.GRAY, kernel_center = (-1,-1)):
    
    if conv_type == InputImageType.GRAY:
        image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
        kernel_horiz = createSobelKernel(horiz=True)
        image_horiz = convolution(image=image, kernel=kernel_horiz, kernel_center=kernel_center)
        
        #image_horiz = normalize(image_horiz)
        cv2.imshow('Horiz image', image_horiz)
        cv2.waitKey(0)

        kernel_vert = createSobelKernel(horiz=False)
        image_vert  = convolution(image=image, kernel=kernel_vert, kernel_center=kernel_center)
        #image_vert = normalize(image_vert)

        cv2.imshow('Vertical image', image_vert)
        cv2.waitKey(0)
        
        height, width = image.shape
        out = np.zeros_like(image, dtype='float32')
        
        for x in range(0,height):
            for y in range(0, width):
                dx = image_horiz[x,y]
                dy = image_vert[x,y]
                
                res = math.sqrt( dx**2 + dy**2 )
                out[x,y] = res

        out = normalize(out)

        cv2.imshow('Output image', out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def showSobelKernel():
    kernel_horiz = createSobelKernel(horiz=True)
    print("Horizontal sobel kernel")
    print(kernel_horiz)

    kernel_vert = createSobelKernel(horiz=False)
    print("Vertical sobel kernel")
    print(kernel_vert)

#image_path = '.\images\\lena.jpg'
#perform_sobel(imagePath=image_path, conv_type=InputImageType.GRAY, kernel_center=(0,0))
