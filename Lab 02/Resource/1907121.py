import cv2
import numpy as np
import math

def generateGaussianKernel(sigmaX, sigmaY, MUL = 7):
    w = int(sigmaX * MUL)
    h = int(sigmaY * MUL) 

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

def gradient_magnitude(dx, dy):
    return np.sqrt(dx**2 + dy**2)

def calculate_threshold(image):
    # Initialize threshold with average pixel intensity
    T = np.mean(image)
    T_infinity = 0.1

    while True:
        first_part = image < T
        second_part = image >= T
        mu1 = np.mean(image[first_part])
        mu2 = np.mean(image[second_part])
        new_T = (mu1 + mu2) / 2
        if abs(T - new_T) < T_infinity:
            break
        T = new_T
    return T

def x_derivative(sigma,size):
    kernel=np.zeros(size,size)

def y_derivative(sigma,size):
    return -y/sigma**2*gaussian_kernel

def start():
    image = cv2.imread('line5.jpg', cv2.IMREAD_GRAYSCALE)

    # Gaussian kernel parameters default definition sigma = 0.7, kernel_size = 5
    sigma = 0.7
    kernel_size = 5

    

    gradient_mag = gradient_magnitude(dx, dy)

    # Calculate the threshold value using the basic global thresholding method
    threshold = calculate_threshold(gradient_mag)

    # Apply global thresholding
    edges = np.where(gradient_mag > threshold, 0, 255).astype(np.uint8)

    cv2.imshow('Image', image)
    cv2.imshow('Edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()