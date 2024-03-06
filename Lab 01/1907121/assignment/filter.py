# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:26:40 2024

@author: Doniel
"""

import numpy as np
import cv2

from kernel_generator import *
from extractor_merger import *
from convolution import *
from sobel import *
from rgb_convolve import convolve_rgb
from hsv_convolve import convolve_hsv

from enum import Enum

class InputImageType(Enum):
    GRAY = 1
    RGB_HSV = 2
    
def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)
    return difference


def choose_option(list, message="select an option: ", error_message="Invalid index, Restarting...\n"):
    for i in range(len(list)):
        print(f"{i}. {list[i]}")
    print()
    
    print(message, end=" ")
    index=int(input())
    if(index>=len(list)):
        if (error_message!=None):
            print(error_message)
        return -1
    val= list[index]
    print(f"'{val}'<--- is selected")
    return index


def performConvol(imagePath, kernel, imageType = InputImageType.GRAY, kernel_center = (-1,-1)): # kernel_center = (y,x) # ( h, w )
    
    if imageType == InputImageType.GRAY:
        image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
        
        out_conv = convolve(image=image, kernel=kernel, kernel_center=kernel_center)
        out_nor = normalize(out_conv)
        
        cv2.imshow('Input image', image)
        cv2.imshow('Output image', out_nor)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        image = cv2.imread(imagePath)        
        
        rgb1 = convolve_rgb(image=image,kernel=kernel,kernel_center=kernel_center)
        rgb2 = convolve_hsv(image=image, kernel=kernel, kernel_center=kernel_center)
        
        diff = find_difference(rgb1, rgb2)
        
        cv2.imshow("RGB from RGB",rgb1)
        cv2.imshow("RGB from HSV",rgb2)
        cv2.imshow("Difference", diff)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def take_and_generate_kernel(kernel_name):
    kernel = None
    formatted_kernel = None
    
    if kernel_name.lower() == "gaussian":
        print("Enter sigma-x: ", end=' ')
        sigma_x = float( input() )
        print("Enter sigma-y: ", end=' ')
        sigma_y = float( input() )
        kernel = generateGaussianKernel(sigmaX=sigma_x,sigmaY=sigma_y)
    
    elif kernel_name.lower() == 'mean':
        print("Enter number of rows: ", end=' ')
        rows = int( input() )
        print("Enter number of cols: ", end=' ')
        cols = int( input() )
        kernel = generateMeanKernel(rows=rows, cols=cols)

    elif kernel_name.lower() == 'laplacian':
        options = ['Negative', 'Positive']
        index = choose_option(options,message="Select center sign(Default -): ", error_message=None)
        kernel= generateLaplacianKernel(negCenter= (index == 0))

    elif kernel_name.lower() == "log":
        print("Enter sigma: ", end=' ')
        sigma = float( input() )
        kernel = generateLogKernel(sigma=sigma)

    
    print("Actual kernel")
    print(kernel)
    return kernel


def get_kernel_center():
    print("Enter -1 to use the actual center")
    print("Enter center-x: ", end=' ')
    center_x = int(input())
    if(center_x == -1):
        return (-1,-1)
    
    print("Enter center-y:", end=' ')
    center_y = int(input())
    
    return (center_y, center_x)

def pre():
    kernel = generateGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5)
    print("Gaussian filter")
    print(kernel)

    kernel = generateMeanKernel()
    print("Mean filter")
    print(kernel)

    kernel = generateLaplacianKernel()
    print("Laplacian filter")
    print(kernel)

    kernel = generateLogKernel(sigma = 1.4)
    print("LoG filter")
    print(kernel)

    kernel = generateSobelKernel()
    print("Sobel filter")
    print(kernel)

    performConvol('.\images\\shape.jpg',kernel=kernel, imageType=InputImageType.GRAY, kernel_center = (1,1))


    image1 = cv2.imread('.\images\\table_1.jpg')
    image2 = cv2.imread('.\images\\table_2.jpg')
    dif = find_difference(image1=image1, image2=image2)

    cv2.imshow("Image_1", image1)
    cv2.imshow("Image_2", image2)
    cv2.imshow("Difference", dif)
    cv2.waitKey(0)



def start():
    main_options= ["continue","exit"]
    image_names = ['box.jpg', 'cat.jpg', 'lena.jpg', 'shape.jpg', 'table_1.jpg', 'table_2.jpg', 'rimi.jpg']
    conv_type = ["GrayScale", "HSV & RGB Difference"]
    kernel_names = ["Gaussian", "Mean", "Laplacian", "LoG", "Sobel"]
    
    
    while(True):
        index= choose_option(main_options,"select 0 to continue: ")
        if(index!=0):
            continue
        index= choose_option(image_names,"select an image: ")
        if(index==-1):
            continue
        
        image_name= image_names[index]
        image_path= '.\images\\'+image_name
        
        index= choose_option(conv_type,"select operation type: ")
        if(index==-1):
            continue
        operation_type= InputImageType.GRAY if(index == 0) else InputImageType.RGB_HSV
        
        index= choose_option(kernel_names, "select a kernel: ")
        if(index==-1):
            continue
        kernel_name=kernel_names[index]
        
        if kernel_name.lower() == "sobel" :
            # showSobelKernel()
            kernel_center = get_kernel_center()
            perform_sobel(imagePath=image_path, conv_type=operation_type, kernel_center=kernel_center)
        else:
            kernel = take_and_generate_kernel(kernel_name=kernel_name)
            kernel_center = get_kernel_center()
            
            performConvol( imagePath=image_path, imageType=operation_type, kernel=kernel, kernel_center=kernel_center )
        
        print("Completed")
        
        
pre()
start()
    
    
    