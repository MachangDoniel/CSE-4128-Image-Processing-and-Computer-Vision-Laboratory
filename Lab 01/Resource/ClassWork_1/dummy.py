import numpy as np
import cv2
from kernal_generator import *

def pad_image(image, kernel_height, kernel_width, kernel_center):
    pad_top = kernel_center[0]
    pad_bottom = kernel_height - kernel_center[0] - 1
    pad_left = kernel_center[1]
    pad_right = kernel_width - kernel_center[1] - 1
    
    padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values = 0)
    return padded_image

def performConvol(imagePath, kernel, kernel_center = (-1,-1)): # kernel_center = (y,x)
    
    image = cv2.imread( imagePath, cv2.IMREAD_GRAYSCALE )
    
    cv2.imshow('input image in grayscale',image)
    cv2.waitKey(0)
    
    kernel_height, kernel_width = len(kernel), len(kernel[0])
    
    # if kernel center is not defined, then it will use the center symmetric center
    if kernel_center[0] == -1:
        kernel_center = ( kernel_height // 2, kernel_width // 2 )
    
    # pad the input image based on kernel and center
    padded_image = pad_image(image = image,  kernel_height = kernel_height, kernel_width = kernel_width, kernel_center = kernel_center)

    output = np.zeros_like(padded_image, dtype='float32')

    padded_height, padded_width = padded_image.shape

    kcy = kernel_center[0]
    kcx = kernel_center[1]
    
    for y in range( kcy, padded_height - ( kernel_height - (kcy+1)) ):
        for x in range( kcx, padded_width - ( kernel_width - (kcx + 1)) ):
            
            image_start_x = x - kernel_center[1]
            image_start_y = y - kernel_center[0]
            
            sum = 0
            N = kernel_width // 2
            for kx in range( -N, N+1):
                for ky in range( -N, N+1 ):
                    rel_pos_in_kernel_x = kx + N
                    rel_pos_in_kernel_y = ky + N
                    
                    rel_pos_in_image_x = N - kx
                    rel_pos_in_image_y = N - ky
                    
                    act_pos_in_image_x = rel_pos_in_image_x + image_start_x
                    act_pos_in_image_y = rel_pos_in_image_y + image_start_y
                    
                    k_val = kernel[ rel_pos_in_kernel_x ][ rel_pos_in_kernel_y ]
                    i_val = padded_image[ act_pos_in_image_y ][ act_pos_in_image_x ]
                    
                    sum +=  k_val * i_val

                output[y,x] = sum

    #out = output[kernel_center[0]:-kernel_height + kernel_center[0] + 1, kernel_center[1]:-kernel_width + kernel_center[1] + 1]

    cv2.normalize(output,output,0,255,cv2.NORM_MINMAX)
    output = np.round(output).astype(np.uint8)
    
    cv2.imshow('Output image', output)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Finally, explain the `cv2.filter2D` function in detail with the following:
# - Overview of how the function actually work internally.
# - Each of the parameter with its functionalities.

#kernel = generateLaplacianKernel(negCenter=False)

kernel_conv = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

flipped_kernel = np.flip(kernel_conv, axis=(0, 1))
print(flipped_kernel)

image = cv2.imread( 'ClassWork\\image_girl.jpg', cv2.IMREAD_GRAYSCALE )

#out = cv2.filter2D(src=image, ddepth=-1, kernel = flipped_kernel,borderType = cv2.BORDER_CONSTANT)
out = cv2.filter2D(image, -1, flipped_kernel, borderType=cv2.BORDER_REPLICATE)
cv2.normalize(out,out,0,255,cv2.NORM_MINMAX)

cv2.imshow("Actual", out)

performConvol('ClassWork\\image_girl.jpg', kernel = kernel_conv)

cv2.waitKey(0)
cv2.destroyAllWindows()
