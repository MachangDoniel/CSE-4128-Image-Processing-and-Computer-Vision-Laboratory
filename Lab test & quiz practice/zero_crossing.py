import cv2 as cv
import numpy as np
#import edgeDetectionZeroCrossing as jo

img = cv.imread('Lena.jpg', cv.IMREAD_GRAYSCALE)

def logKernel(sigma, MUL=7):
    n = int(sigma * MUL)
    n = n | 1
    kernel = np.zeros((n, n), dtype=np.float32)
    center = n // 2
    part1 = -1 / (np.pi * sigma ** 4)

    for x in range(n):
        for y in range(n):
            dx = x - center
            dy = y - center
            part2 = (dx * dx + dy * dy) / (2 * sigma * sigma)
            kernel[x, y] = part1 * (1 - part2) * np.exp(-part2)

    return kernel

def zero_crossing(image, threshold, local_variance):
    row, col = image.shape
    zero_crossing_image = np.zeros((row, col), dtype=np.uint8)

    for i in range(1, row-1):
        for j in range(1, col-1):
            neighbours = [image[i-1, j], image[i+1, j], image[i, j-1], image[i, j+1]]
            if any(np.sign(image[i, j]) != np.sign(neighbour) for neighbour in neighbours):
                if local_variance[i, j] > threshold:
                    zero_crossing_image[i, j] = 255
                else:
                    zero_crossing_image[i, j] = 0

    return zero_crossing_image

def estimate_local_variance(image, kernel_size):
    pad = kernel_size // 2
    row, col = image.shape
    variance_image = np.zeros((row, col), dtype=np.float32)
    
    for i in range(pad, row-pad):
        for j in range(pad, col-pad):
            local_region = image[i-pad:i+pad+1, j-pad:j+pad+1]
            local_stddev = np.std(local_region)
            variance_image[i, j] = local_stddev ** 2
    
    return variance_image

# Define parameters
sigma = 1
kernel = logKernel(sigma, 7)
threshold = 60  # You can adjust this threshold based on your requirements
kernel_size = kernel.shape[0]

# Apply LoG operator
convoluted = cv.filter2D(img, -1, kernel)

# Estimate local variance
local_variance = estimate_local_variance(img, kernel_size)

# Apply zero-crossing with local variance thresholding
zero_crossing_image = zero_crossing(convoluted, threshold, local_variance)

# Display the result
cv.imshow("output_image", zero_crossing_image)
cv.waitKey(0)
cv.destroyAllWindows()