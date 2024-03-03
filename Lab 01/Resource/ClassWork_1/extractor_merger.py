import cv2
import numpy as np


def extract_rgb(image):
    blue_channel, green_channel, red_channel = cv2.split(image)
    return (red_channel, green_channel, blue_channel)

def merge_rgb( red, green, blue ):
    return cv2.merge( [blue, green, red] )

def extract_hsv(image):
    h_channel, s_channel, v_channel = cv2.split(image)
    return (h_channel, s_channel, v_channel)

def merge_hsv(h, s, v):
    return cv2.merge([h, s, v])

def hsv_to_rgb(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def rgb_to_hsv(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)

# image_path = 'ClassWork/image_girl.jpg'
# image = cv2.imread(image_path)

# (red, green, blue) = extract_rgb(image)

# cv2.imshow('Red Channel', red)
# cv2.imshow('Green Channel', green)
# cv2.imshow('Blue Channel', blue)

# cv2.waitKey(0)

# merged_image = merge_rgb(red, green, blue)

# cv2.imshow("Actual", image)
# cv2.imshow("Merged", merged_image)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
