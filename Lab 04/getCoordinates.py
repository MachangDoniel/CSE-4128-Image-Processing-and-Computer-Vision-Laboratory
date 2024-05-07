# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 23:33:14 2024

@author: USER
"""
import cv2

points=[]
 
def Capture_Event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN or event==cv2.EVENT_RBUTTONDOWN:
        points.append((y,x))
        print(x,y)
        
        
        
img = cv2.imread('input.jpg', 0)
cv2.imshow('image', img)
# Set the Mouse Callback function, and call the Capture_Event function.
cv2.setMouseCallback('image', Capture_Event)
 
cv2.waitKey(0)
print(points)


cv2.destroyAllWindows()