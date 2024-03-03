import numpy as np
import cv2
from create_kernel import *
from extractor_merger import *
from convolution import *
from sobel import *
from rgb_convolution import *
from hsv_convolution import *

from enum import Enum
 
class InputImageType(Enum):
    GRAY = 1
    RGB_HSV = 2
    
def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)
    
    return difference

def performConvolution(imagePath, kernel, imageType = InputImageType.GRAY, kernel_center = (-1,-1)): # kernel_center = (y,x) # ( h, w )
    
    if imageType == InputImageType.GRAY:
        image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
        
        out_conv = convolution(image=image, kernel=kernel, kernel_center=kernel_center)
        out_nor = normalize(out_conv)

        cv2.imshow('Input image', image)
        cv2.imshow('Output image', out_nor)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(imagePath)        
        
        rgb1 = convolution_rgb(image=image,kernel=kernel,kernel_center=kernel_center)
        
        rgb2 = convolution_hsv(image=image, kernel=kernel, kernel_center=kernel_center)
        
        diff = find_difference(rgb1, rgb2)
        cv2.imshow("RGB from RGB",rgb1)
        cv2.imshow("RGB from HSV",rgb2)
        cv2.imshow("Difference", diff)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def showAllFilter():
    # kernel = createGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5)
    # print("Gaussian filter")
    # print(kernel)

    # kernel = createMeanKernel()
    # print("Mean filter")
    # print(kernel)

    # kernel = createLaplacianKernel()
    # print("Laplacian filter")
    # print(kernel)

    # kernel = createLogKernel(sigma = 1.4)
    # print("LoG filter")
    # print(kernel)

    kernel = createSobelKernel()
    print("Sobel filter")
    print(kernel)

    performConvolution('.\images\\cat.jpg',kernel=kernel, imageType=InputImageType.GRAY, kernel_center = (1,1))


    image1 = cv2.imread('.\images\\table_1.jpg')
    image2 = cv2.imread('.\images\\table_2.jpg')
    dif = find_difference(image1=image1, image2=image2)

    cv2.imshow("Image_1", image1)
    cv2.imshow("Image_2", image2)
    cv2.imshow("Difference", dif)
    cv2.waitKey(0)

def choose_option(list, message = "Select an option", error_message="Invalid index. Restarting..\n"):
    for i in range( len(list) ):
        print(f"{i}. {list[i]}", end=" | ")
    print()
    
    print(message, end='')
    index = int(input())
    
    if( index >= len(list) ):
        if error_message != None:
            print(error_message)
        return -1
    
    val = list[index]
    print(f"`{val}` is selected <-----------------------\n")
    return index

def take_and_create_kernel(kernel_name):
    kernel = None
    formatted_kernel = None
    
    if kernel_name.lower() == "gaussian":
        print("Enter sigma-x: ", end=' ')
        sigma_x = float(input())
        print("Enter sigma-y: ", end=' ')
        sigma_y = float(input())
        
        kernel, formatted_kernel = createGaussianKernel(sigmaX=sigma_x, sigmaY=sigma_y)
    
    if kernel_name.lower() == 'mean':
        print("Enter number of rows: ", end=' ')
        rows = int(input())
        
        print("Enter number of cols: ", end=' ')
        cols = int(input())
        
        kernel, formatted_kernel = createMeanKernel(rows=rows, cols=cols)

    if kernel_name.lower() == 'laplacian':
        options = ['Negative', 'Positive']
        index = choose_option(options, message="Select center sign(Default -): ", error_message=None)
        
        kernel, formatted_kernel = createLaplacianKernel(negCenter=(index == 0))

    if kernel_name.lower() == "log":
        print("Enter sigma: ", end=' ')
        sigma = float(input())
        
        kernel, formatted_kernel = createLogKernel(sigma=sigma)

    print("Formatted kernel")
    print(formatted_kernel)
    
    print("Actual kernel")
    print(kernel)
    
    return kernel


def get_kernel_center():
    print("Enter center-y (or -1 for default):", end=' ')
    center_y = int(input())
    
    print("Enter center-x:", end=' ')
    center_x = int(input())
    
    return (center_y, center_x)

    

def start():
    showAllFilter()

    image_names = ['box.jpg', 'cat.jpg', 'lena.jpg', 'shape.jpg', 'table_1.jpg', 'table_2.jpg']
    kernel_names = ["Gaussian", "Mean", "Laplacian", "LoG", "Sobel"]
    conv_type = ["GrayScale", "HSV & RGB Difference"]
    
    while( True ):

        index = 1
        image_name = image_names[index]
        image_path = '.\images\\'+image_name

        index = 0
        kernel_name = kernel_names[index]
        
        if kernel_name.lower() == "sobel" :
            showSobelKernel()
            kernel_center = get_kernel_center()
            perform_sobel(imagePath=image_path, conv_type=0, kernel_center=kernel_center)
        else:
            kernel = take_and_create_kernel(kernel_name=kernel_name)
            kernel_center = get_kernel_center()
                
            performConvolution( imagePath=image_path, imageType=2, kernel=kernel, kernel_center=kernel_center )
            
        print("Completed")

        
        
        
start()