import numpy as np
import math
import numpy as np
import scipy.ndimage as ndimage

def generateGaussianKernel(sigmaX, sigmaY, MUL = 7):
    w = int(sigmaX * MUL) | 1
    h = int(sigmaY * MUL) | 1
    
    #print(w,h)

    cx = w // 2
    cy = h // 2 

    kernel = np.zeros((w, h))
    c = 1 / ( 2 * 3.1416 * sigmaX * sigmaY )
    
    for x in range(w):
        for y in range(h):
            dx = x - cx
            dy = y - cy
            
            x_part = (dx*dx) / (sigmaX * sigmaX)
            y_part = (dy*dy) / (sigmaY * sigmaY)

            kernel[x][y] = c * math.exp( - 0.5 * (x_part + y_part) )

    formatted_kernel = kernel / np.min(kernel)
    formatted_kernel = formatted_kernel.astype(int)

    print("Formatted gaussian filter")
    print(formatted_kernel)
    
    return (kernel, formatted_kernel)

def generateMeanKernel(rows = 3, cols = 3):
    rows = rows | 1
    cols = cols | 1
    
    formatted_kernel = np.zeros( (rows, cols) )

    for x in range(0, rows):
        for y in range(0, cols):
            formatted_kernel[x,y] = 1.0

    kernel = formatted_kernel / (rows * cols)
    return (kernel, formatted_kernel)

def generateLaplacianKernel( negCenter = True ):
    n = 3    
    other_val = 1 if negCenter else -1
    
    kernel = np.array([
        [0,1,0],
        [1,-4,1],
        [0,1,0]
    ])
    
    # kernel = other_val * np.ones( (n, n) )
    # center = n // 2
    
    # kernel[center, center] = - other_val * ( n*n - 1 )
    
    kernel = other_val * kernel
    
    #print(kernel)
    return (kernel,kernel)

def generateLogKernel(sigma, MUL = 7):
    n = int(sigma * MUL)
    n = n | 1
    
    kernel = np.zeros( (n,n) )

    center = n // 2
    part1 = -1 / (np.pi * sigma**4)
    
    for x in range(n):
        for y in range(n):
            dx = abs(x - center)
            dy = abs(y - center)
            
            part2 = (dx**2 + dy**2) / (2 * sigma**2)
            
            kernel[x][y] =  part1 * (1 - part2) * np.exp(-part2)
    
    #print("Formatted LoG kernel")
    
    mn = np.min(np.abs(kernel))
    formatted_kernel = (kernel / mn).astype(int)
    
    return (kernel, formatted_kernel)

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

    return sobel_x if horiz else sobel_y

def testKernel():
    #kernel, formatted_kernel = generateGaussianKernel( sigmaX = 1, sigmaY = 1, MUL = 5)
    #print(formatted_kernel)
    
    # kernel,format = generateMeanKernel( rows = 5, cols = 5 )
    
    # kernel,format = generateLaplacianKernel( negCenter = False )
    
    kernel,format = generateLogKernel(1.4)
    #print(kernel)
    
    #kernel = generateSobelKernel( horiz = False )
    #print(kernel)
    
    print(kernel)
    print(format)
    
    print("-")



#testKernel()