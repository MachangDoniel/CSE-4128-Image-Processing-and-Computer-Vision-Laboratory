import numpy as np
import math

def generateGuassianKernel(sigmaX = 1,sigmay = 1,mul=5):
    # constant part
    constant = 1 / (2 * np.pi * sigmaX * sigmay)
    h = int(mul * sigmaX) | 1
    w = int(mul * sigmay) | 1
    kernel = np.zeros((h, w))

    # variable part
    centerx = h // 2   # floor division
    centery = w // 2   # floor division
    for i in range(h):
        for j in range(w):
            x = i - centerx
            y = j - centery
            x_part = (x * x) / (sigmaX * sigmaX)
            y_part = (y * y) / (sigmay * sigmay)
            variable = -0.5 *(x_part + y_part)
            kernel[i][j] = constant * math.exp(variable)
    
    kernel = kernel / np.min(kernel)
    kernel = kernel.astype(np.uint8)
    print("Gaussion Kernel")
    print(kernel)
    kernel = kernel / np.sum(kernel)
    return kernel

def generateMeanKernel(rows = 3,columns = 3):
    kernel = np.ones((rows,columns))
    print("Mean Kernel")
    print(kernel)
    kernel = kernel / np.sum(kernel)
    return kernel

def generateLaplacianKernel(centerNeg = True, size = 3):
    size = size | 1             # size must be odd
    kernel = np.zeros((size,size))
    center = size * size -1
    if centerNeg:
        for i in range(size):
            for j in range(size):
                kernel[i][j] = 1
        kernel[size // 2][size // 2] = - center
    else:
        for i in range(size):
            for j in range(size):
                kernel[i][j] = -1
        kernel[size // 2][size // 2] = center

    print("Lapcian Kernel")
    print(kernel)
    return kernel

def generateLogKernel(sigma, mul = 7):
    n = int(mul * sigma) | 1
    kernel = np.zeros((n, n))
    center = n // 2
    constant = - 1 / (np.pi * sigma**4)
    for i in range(n):
        for j in range(n):
            x = (i-center)
            y = (j-center)
            variable = (x*x + y*y) / (2 * sigma**2)
            kernel[i][j] = constant * (1 - variable) * np.exp(-variable)
    
    formatted_kernel = kernel / np.min(np.abs(kernel))
    formatted_kernel = formatted_kernel.astype(int)
    print("Log Kernel")
    print(formatted_kernel)
    print(np.sum(kernel))
    return kernel

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







def showKernel():
    kernel = generateGuassianKernel(1,1)
    print(kernel)
    kernel = generateMeanKernel(3,3)
    print(kernel)
    kernel = generateLaplacianKernel(True,3)
    print(kernel)
    kernel = generateLaplacianKernel(False,3)
    print(kernel)
    kernel = generateLogKernel(1.4)
    print(kernel)
    kernel = generateSobelKernel(True)
    print(kernel)
    kernel = generateSobelKernel(False)
    print(kernel)

showKernel()