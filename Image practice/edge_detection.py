# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


def find_next_threeshold(image, t = -1):
    total1 = 0
    total2 = 0
    c1 = 0
    c2 = 0
    
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x][y]
            if px > t:
                total2 += px
                c2 += 1
            else:
                total1 += px
                c1 += 1
    mu1 = total1 / c1
    mu2 = total2 / c2
    
    return (mu1 + mu2) / 2


def make_binary(t, image, low = 0, high = 255):
    out = image.copy()
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x,y]
            out[x,y] = high if v > t else low
    return out


def find_threeshold(image):
    total = 0
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            px = image[x,y]
            total += px
    oldT = total / (h * w)
    
    newT = find_next_threeshold(image=image,t=oldT)
    while( abs(newT - oldT) > 0.1 ** 6 ) :
        oldT = newT
        newT = find_next_threeshold(image=image,t=oldT)
        print(f"Old: {oldT}, New: {newT}")

    return newT

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def normalize(image):
    copied = image.copy()
    cv2.normalize(copied,copied,0,255,cv2.NORM_MINMAX)
    return np.round(copied).astype(np.uint8)


def merge(image_horiz, image_vert):
    height, width = image_horiz.shape
    out = np.zeros_like(image_horiz, dtype='float32')
        
    for x in range(0, height):
        for y in range(0, width):
            dx = image_horiz[x,y]
            dy = image_vert[x,y]
                
            res = math.sqrt( dx**2 + dy**2 )
            out[x,y] = res

    #out = normalize(out)
    return out

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
    
    return (kernel)

def get_kernel():
    sigma=0.7
    kernel= generateGaussianKernel(sigmaX=sigma, sigmaY=sigma, MUL=7)
    print("Kernel:\n",kernel)
    h= len(kernel)
    kernel_x=np.zeros((h,h))
    kernel_y=np.zeros((h,h))
    
    min1= min2= 100

    cx= h//2
    for x in range(h):
        for y in range(h):
            act_x=x-cx
            act_y=y-cx
            
            kernel_x[x,y]=(-act_x/sigma**2)*kernel[x,y]
            kernel_y[x,y]=(-act_y/sigma**2)*kernel[x,y]
            
            if kernel_x[x,y] != 0:
                min1 = min(abs(kernel_x[x,y]),min1)
            
            if kernel_y[x,y] != 0:
                min2 = min(abs(kernel_y[x,y]),min2)
                
    dr1 = (kernel_x / min1).astype(int)
    dr2 = (kernel_y / min2).astype(int)
    
    print(dr1)
    print(dr2)
    
    print("Kernel_x\n",kernel_x)
    print("Kernel_y\n",kernel_y)
    return (kernel_y,kernel_x)


def start():
    image_path='line5.jpg'
    image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    
    cv2.imshow("Input",image)
    
    cv2.waitKey(0)
    
    kernel_x,kernel_y= get_kernel()
    
    conv_x = convolve(image=image, kernel=kernel_x)
    conv_y = convolve(image=image, kernel=kernel_y)
    
    kernel = generateGaussianKernel(sigmaX=sigma,sigmaY=sigma,MUL=7)
    conv_x = convolve(image=conv_x,kernel=kernel)
    conv_y = convolve(image=conv_y,kernel=kernel)
    
    out = merge(conv_x, conv_y)
    
    out_nor = normalize(out)

    cv2.imshow("X derivative", normalize(conv_x))
    cv2.imshow("Y derivative", normalize(conv_y))
    cv2.imshow("Merged", out_nor)
    
    cv2.waitKey(0)

    #plot_historgram(out)
    t = find_threeshold(image=out_nor)
    print(f"Threeshold {t}")
    final_out = make_binary(t=t*0.8,image=out_nor)
    cv2.imshow("Threesholded", final_out)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

sigma=0.7    
# start()