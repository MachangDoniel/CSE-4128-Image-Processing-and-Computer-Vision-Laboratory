import os
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt




# output

def show(name, image):
    print(f"{name} is generated")
    cv2.imshow({name}, normalize(image))

#  clear terminal
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("<-- Welcome! -->\n\n")


#  Convolution

def find_difference(image1, image2):
    image1 = cv2.resize(image1, (image2.shape[1], image2.shape[0]))
    difference = cv2.absdiff(image1, image2)
    difference = normalize(difference)
    
    return difference

def normalize(image):
    copied = image.copy()
    cv2.normalize(copied,copied,0,255,cv2.NORM_MINMAX)
    return np.round(copied).astype(np.uint8)

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def convolve(image, kernel, kernel_center = (-1,-1)):
    image = image.copy()
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    # if kernel center is not defined, then it will use the center symmetric center
    if kernel_center[0] == -1:
        kernel_center = ( kernel_height // 2, kernel_width // 2 )
    
    # pad the input image based on kernel and center
    padded_image = pad_image(image = image,  kernel_height = kernel_height, kernel_width = kernel_width, kernel_center = kernel_center)

    # generating output with dummy zeros(0)
    output = np.zeros_like(padded_image, dtype='float32')
    
    #print("Padded image")
    #print(padded_image)
    
    # xx = 1
    # yy = 2
    # print(f"Value at ({xx},{yy}) is {padded_image[xx,yy]}")
    
    # padded image height, width
    padded_height, padded_width = padded_image.shape

    kcx = kernel_center[0]
    kcy = kernel_center[1]
    
    # iterating through height. For (1,1) kernel, it iterates from 1 to (h - 1)
    for x in range( kcx, padded_height - ( kernel_height - (kcx+1)) ):
        # iterate through width. For (1,1) kernel, it iterates from 1 to (w - 1)
        for y in range( kcy, padded_width - ( kernel_width - (kcy + 1)) ):
            image_start_x = x - kcx
            image_start_y = y - kcy
            
            sum = 0
            NX = kernel_height // 2
            NY = kernel_width // 2
            for kx in range( -NX, NX+1):
                for ky in range( -NY, NY+1 ):
            # for kx in range(0, kernel_height):
            #     for ky in range(0, kernel_width):
                    rel_pos_in_kernel_x = kx + NX # x-i
                    rel_pos_in_kernel_y = ky + NY # y-j
                    
                    rel_pos_in_image_x = NX - kx # 2
                    rel_pos_in_image_y = NY - ky # 2
                    
                    act_pos_in_image_x = rel_pos_in_image_x + image_start_x # 2 + 2 = 4
                    act_pos_in_image_y = rel_pos_in_image_y + image_start_y # 3 + 2 = 5
                    
                    k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
                    i_val = padded_image[ act_pos_in_image_x ][ act_pos_in_image_y ]
                    # k_val = kernel[ kx ][ ky ]
                    # i_val = padded_image[ kx+image_start_x ][ ky+image_start_y ]
                    
                    sum +=  k_val * i_val
            output[x,y] = sum
    
    out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]
    return out


# Kernel_generator


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

#  Three_Hys

def perform_threshold(image, threes):
    highThreshold = threes * 0.5
    lowThreshold = highThreshold * 0.5
    
    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(75)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    weak_i, weak_j = np.where( np.logical_and( (image <= highThreshold), (image >= lowThreshold) ) )
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def plot_histogram(img):
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()

def perform_hysteresis(image, weak, strong = 255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == weak):
                if np.any( image[i-1:i+2, j-1:j+2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out