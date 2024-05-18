import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

from convolution import normalize
from convolution import convolve
from kernel_generator import *



def get_kernel():
    sigma = 0.7
    kernel,formatted_kernel = generateGaussianKernel(sigmaX=sigma, sigmaY=sigma, MUL=7)
    
    h = len(kernel)
    
    kernel_x = np.zeros((h,h))
    kernel_y = np.zeros((h,h))
    
    mn1 = 100
    mn2 = 100
    
    cx = h//2
    for x in range(h):
        for y in range(h):
            
            act_x = (x-cx)
            act_y = (y-cx)
            
            c1 = -act_x / (sigma ** 2)
            c2 = -act_y / (sigma ** 2)
            
            kernel_x[x,y] = c1 * kernel[x,y]
            kernel_y[x,y] = c2 * kernel[x,y]
            
            if kernel_x[x,y] != 0:
                mn1 = min(abs(kernel_x[x,y]),mn1)
            
            if kernel_y[x,y] != 0:
                mn2 = min(abs(kernel_y[x,y]),mn2)

    dr1 = (kernel_x / mn1).astype(int)
    dr2 = (kernel_y / mn2).astype(int)
    
    # print("x & y")
    print(dr1)
    print(dr2)
    print("Kernel_x\n",kernel_x)
    print("Kernel_y\n",kernel_y)
    return (kernel_y,kernel_x)

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

def make_binary(t, image, low = 0, high = 255):
    out = image.copy()
    h,w = image.shape
    for x in range(h):
        for y in range(w):
            v = image[x,y]
            out[x,y] = high if v > t else low
    return out


def start():
    # image_path = '.\images\\lena.jpg'
    
    image_path = '.\images\\line5.jpg'
    
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Input", image)
    
    # image = cv2.GaussianBlur(image, (3,3),0)
    # cv2.imshow("Blurred", image)

    kernel_x, kernel_y = get_kernel()

    conv_x = convolve(image=image, kernel=kernel_x)
    conv_y = convolve(image=image, kernel=kernel_y)
    
    kernel, _ = generateGaussianKernel(sigmaX=.7,sigmaY=.7,MUL=7)
    conv_x = convolve(image=conv_x,kernel=kernel)
    conv_y = convolve(image=conv_y,kernel=kernel)
    out = merge(conv_x, conv_y)
    
    out_nor = normalize(out)

    cv2.imshow("X derivative", normalize(conv_x))
    cv2.imshow("Y derivative", normalize(conv_y))
    cv2.imshow("Merged", out_nor)

    #plot_historgram(out)
    t = find_threeshold(image=out_nor)
    print(f"Threeshold {t}")
    final_out = make_binary(t=t*0.8,image=out_nor)
    cv2.imshow("Threesholded", final_out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def doc_code():
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    
    image_path = '.\images\\lena.jpg'    
    src = cv2.imread( image_path, cv2.IMREAD_COLOR)
    
    src = cv2.GaussianBlur(src, (3, 3), 0)
    
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow("Blurred gray", gray)
    
    
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_x_nor = normalize(grad_x)
    cv2.imshow("X derivative", grad_x_nor)
    
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y_nor = normalize(grad_y)
    cv2.imshow("Y derivative", grad_y_nor)
    
    
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    
    cv2.imshow("Output", grad)
    cv2.waitKey(0)
    
    return 0

start()
#test_convolve()#
#doc_code()