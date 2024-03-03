import math
import numpy as np
import scipy.ndimage as ndimage

def createGaussianKernel(sigmaX, sigmaY, MUL = 5):
    # make the height & width odd by performing bitwise or with 1
    # h = int(sigmaY * MUL)
    # w = int(sigmaX * MUL)
    h = int(sigmaY * MUL) | 1
    w = int(sigmaX * MUL) | 1
    
    # print("Kernel Size: ",w,h)

    # center_x,center_y -> centre of kernel is calculated by floor division of w and h
    center_x = w // 2
    center_y = h // 2 
    # print("Center: ",center_x,center_y)

    kernel = np.zeros((w, h))
    # print("Kernel Initialize with zeros")
    # print(kernel)
    divider = 2 * 3.1416 * sigmaX * sigmaY
    # print("Divider: ",divider)
    
    for x in range(w):
        for y in range(h):
            dx = x - center_x
            dy = y - center_y
            
            x_part = (dx*dx) / (sigmaX * sigmaX)
            y_part = (dy*dy) / (sigmaY * sigmaY)

            kernel[x][y] = math.exp(- 0.5 * (x_part + y_part))/ divider

    formatted_kernel = kernel / np.min(kernel)
    # print("Actual Formatted gaussian filter")
    # print(formatted_kernel)
    formatted_kernel = formatted_kernel.astype(int)

    # print("Actual gaussian filter")
    # print(kernel)
    # print("Formatted gaussian filter")
    # print(formatted_kernel)
    
    return (kernel, formatted_kernel)

def createMeanKernel(rows = 3, cols = 3):
    formatted_kernel = np.zeros((rows, cols))

    for x in range(0, rows):
        for y in range(0, cols):
            formatted_kernel[x,y] = 1.0

    kernel = formatted_kernel / (rows * cols)

    # print("Actual Mean filter")
    # print(kernel)
    # print("Formatted Mean filter")
    # print(formatted_kernel)

    return (kernel, formatted_kernel)

def createLaplacianKernel( negativeCenter = True ):
    # assuming size of the kernel 3 kernel
    n = 3   
    other_values = 1 if negativeCenter else -1
    kernel = other_values * np.ones( (n, n) )
    center = n // 2
    
    kernel[center, center] = - other_values * ( n*n - 1 )
    
    formatted_kernel = kernel

    # print("Actual Laplacian filter")
    # print(kernel)
    # print("Formatted Laplacian filter")
    # print(formatted_kernel)

    return (kernel,formatted_kernel)

def createLogKernel(sigma, MUL = 5):
    # n = int(sigma * MUL)
    n = int(sigma * MUL) | 1 
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

    # print("Actual Log filter")
    # print(kernel)
    # print("Formatted LoG kernel")
    # print(formatted_kernel)
    
    return (kernel, formatted_kernel)


def createSobelKernel( horiz = True ):
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

    return sobel_x if horiz else sobel_y




def checkKernel():
    # kernel, formatted_kernel = createGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 6)
    # print(kernel)
    # print(formatted_kernel)

    # kernel, formatted_kernel = createMeanKernel( rows = 5, cols = 5 )
    # print(kernel)
    # print(formatted_kernel)

    # kernel,formatted_kernel = createLaplacianKernel( negativeCenter = False )
    # print(kernel)
    # print(formatted_kernel)

    # kernel,formatted_kernel = createLogKernel(1.4)
    # print(kernel)
    # print(formatted_kernel)

    # kernel = createSobelKernel( horiz = True )
    # print(kernel)
    # kernel = createSobelKernel( horiz = False )
    # print(kernel)
    
    print("Kernel Create Succesfull!")



checkKernel()
