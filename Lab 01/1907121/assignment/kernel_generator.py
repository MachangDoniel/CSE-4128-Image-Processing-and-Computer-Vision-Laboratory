# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 19:58:22 2024

@author: Doniel
"""

import numpy as np
import math
# import scipy.ndimages as ndimage

def generateGaussianKernel(sigmaX, sigmaY, MUL = 5):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1
    
    print("n of the kernel: ",h,"*",w)

    cx = w // 2  # 2
    cy = h // 2  # 2

    kernel = np.zeros((w, h))
    c = 1 / ( 2 * np.pi * sigmaX * sigmaY )
    
    for x in range(w):
        for y in range(h):
            dx = x - cx  # -2,-1,0,1,2
            dy = y - cy  # -2,-1,0,1,2
            
            x_part = (dx*dx) / (sigmaX * sigmaX)
            y_part = (dy*dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp( - 0.5 * (x_part + y_part) )

    formatted_kernel = kernel / np.min(kernel)
    formatted_kernel = formatted_kernel.astype(int)

    print("Formatted gaussian filter")
    print(formatted_kernel)
    return (kernel)

def generateMeanKernel(rows = 3, cols = 3):#odd
    formatted_kernel = np.zeros( (rows, cols) )

    for x in range(0, rows):
        for y in range(0, cols):
            formatted_kernel[x,y] = 1

    
    kernel = formatted_kernel / (rows * cols)
    
    print("Formatted mean filter")
    print(formatted_kernel)
    return (kernel)

# def generateLaplacianKernel(negCenter = True, n=3):   
#     other_values= 1 if negCenter else -1
    
#     kernel= other_values* np.ones((n,n))
#     center= n//2
#     kernel[center, center]= -other_values*(n*n-1)
    
#     print("Lapcican filter")
#     print(kernel)
#     return (kernel)

def generateLaplacianKernel(negCenter = True, n = 3):
    n = n | 1             # n must be odd
    kernel = np.zeros((n,n))
    center = n * n -1
    if negCenter:
        for i in range(n):
            for j in range(n):
                kernel[i][j] = 1
        kernel[n // 2][n // 2] = - center
    else:
        for i in range(n):
            for j in range(n):
                kernel[i][j] = -1
        kernel[n // 2][n // 2] = center

    print("Lapcian Kernel")
    print(kernel)
    return kernel

def generateLogKernel(sigma, MUL = 7): 
    n = int(sigma * MUL)
    n = n | 1
    
    kernel = np.zeros((n,n))

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    
    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            
            part2 = (dx**2 + dy**2) / (2 * sigma**2)
            
            kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)
    
    
    mn = np.min(np.abs(kernel))
    formatted_kernel = (kernel / mn).astype(int)
    print("Formatted LoG kernel")
    print(formatted_kernel)
    
    return (kernel)

def generateSobelKernel( horiz = True ):
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    sobel_y = np.array([
        [1, 2, 1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    
    print("Horizontal" if horiz else "Vetical", end=" ")
    print("Sobel Filter")
    print(sobel_x if horiz else sobel_y)
    return sobel_x if horiz else sobel_y


def testKernel():
    kernel = generateGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5)
    
    kernel = generateMeanKernel( rows = 5, cols = 5 )
    
    kernel = generateLaplacianKernel(negCenter = True, n=3)
    
    kernel = generateLogKernel(1.4)
    print(kernel)
    
    kernel = generateSobelKernel( horiz = True )
    
    print("-")



testKernel()