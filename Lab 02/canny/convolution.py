import cv2
import numpy as np
from enum import Enum

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


# image = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# kernel = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
# ])

# kernel = np.array([
#     [0,0,0],
#     [0,1,0],
#     [0,0,0]
# ])

#out = convolve(image=image, kernel=kernel,kernel_center=(-1,-1))
#print(out)