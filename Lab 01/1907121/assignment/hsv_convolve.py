# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 00:42:05 2024

@author: Doniel
"""

import cv2
import numpy as np
from kernel_generator import *

from extractor_merger import extract_hsv
from extractor_merger import merge_hsv
from extractor_merger import hsv_to_rgb
from extractor_merger import rgb_to_hsv

from convolution import normalize
from convolution import convolve

def convolve_hsv(image, kernel, kernel_center=(-1,-1)):
    image = rgb_to_hsv(image)
    hue, sat, val = extract_hsv(image=image)

    hue_conv = convolve(image=hue, kernel=kernel, kernel_center=kernel_center)
    hue_nor = normalize(hue_conv)
    
    sat_conv = convolve(image=sat, kernel=kernel, kernel_center=kernel_center)
    sat_nor = normalize(sat_conv)
    
    val_conv = convolve(image=val, kernel=kernel, kernel_center=kernel_center)
    val_nor = normalize(val_conv)
    
    cv2.imshow("Extracted Hue", hue_nor)
    cv2.imshow("Extracted Sat", sat_nor)
    cv2.imshow("Extracted Val", val_nor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    merged_hsv = merge_hsv(h=hue_nor, s=sat_nor, v=val_nor)
    #merged_rgb = hsv_to_rgb(merge_hsv)
    
    #orignial_rgb = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    merged_rgb = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("Original(HSV) image", image)
    cv2.imshow("Merged(HSV) image", merged_hsv)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    return merged_rgb


# image_path = '.\images\\shape.jpg'
# image = cv2.imread( image_path )

# kernel = np.array([
#         [0, 0, 0],
#         [0, 1, 0],
#         [0, 0, 0]
#     ])
# kernel = np.array([
#     [0.00291502, 0.0130642,  0.02153923, 0.0130642,  0.00291502],
#     [0.0130642,  0.05854969, 0.09653213, 0.05854969, 0.0130642 ],
#     [0.02153923, 0.09653213, 0.15915457, 0.09653213, 0.02153923],
#     [0.0130642,  0.05854969, 0.09653213, 0.05854969, 0.0130642 ],
#     [0.00291502, 0.0130642,  0.02153923, 0.0130642,  0.00291502]
# ])

# convolve_hsv(image=image, kernel=kernel, kernel_center=(-1,-1))
