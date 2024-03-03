import cv2
import numpy as np

# Function to perform convolution
def convolve(image, kernel):
    # Get dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding size for borders
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Create an empty image to store the result
    convolved_image = np.zeros_like(image)
    
    # Pad the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=0)
    
    # Perform convolution
    for y in range(image_height):
        for x in range(image_width):
            convolved_image[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)
    
    return convolved_image

# Function to define Laplacian kernel
def laplacian_kernel():
    return np.array([[0, -1, 0],
                     [-1, 4, -1],
                     [0, -1, 0]], dtype=np.float32)

# Function to define Gaussian kernel
def gaussian_kernel(size, sigma):
    center = (size // 2, size // 2)  # User defines the center of the kernel
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-center[0])**2 + (y-center[1])**2)/(2*sigma**2)), (size, size))
    return kernel / np.sum(kernel)

# Function to define Mean kernel
def mean_kernel(size):
    return np.ones((size, size), dtype=np.float32) / (size * size)

# Function to define Sobel kernels along x-axis and y-axis
def sobel_kernels():
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    
    return sobel_x, sobel_y

# Function to define Laplacian of Gaussian (LoG) filter
def LoG_kernel(size, sigma):
    gaussian_kernel_array = gaussian_kernel(size, sigma)
    laplacian_kernel_array = laplacian_kernel()
    LoG_kernel_array = convolve(gaussian_kernel_array, laplacian_kernel_array)
    return LoG_kernel_array

def start():
    # Read the input image
    image_path = 'images\Lena.jpg'  
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Define center for kernels
    # x = int(input("Enter the x-coordinate for the center: "))
    # y = int(input("Enter the y-coordinate for the center: "))

    x,y = 3,3
    center = (x, y)

    # Apply Gaussian filter
    gaussian_kernel_size = 5
    sigma = 1.5
    gaussian_kernel_array = gaussian_kernel(gaussian_kernel_size, sigma)
    gaussian_smoothed_image = convolve(image, gaussian_kernel_array)

    # Apply Mean filter
    mean_kernel_size = 5
    mean_kernel_array = mean_kernel(mean_kernel_size)
    mean_smoothed_image = convolve(image, mean_kernel_array)

    # Apply Laplacian filter
    laplacian_kernel_array = laplacian_kernel()
    laplacian_image = convolve(image, laplacian_kernel_array)

    # Apply Laplacian of Gaussian (LoG) filter
    LoG_kernel_size = 5
    sigma_LoG = 1.5
    LoG_image = LoG_kernel(LoG_kernel_size, sigma_LoG)

    # Apply Sobel filters
    sobel_kernel_x, sobel_kernel_y = sobel_kernels()
    sobel_x_image = convolve(image, sobel_kernel_x)
    sobel_y_image = convolve(image, sobel_kernel_y)

    # Display the results
    cv2.imshow('Original Image', image)
    cv2.imshow('Gaussian Smoothed Image', gaussian_smoothed_image)
    cv2.imshow('Mean Smoothed Image', mean_smoothed_image)
    cv2.imshow('Laplacian Image', laplacian_image)
    cv2.imshow('LoG Image', LoG_image)
    cv2.imshow('Sobel X Image', sobel_x_image)
    cv2.imshow('Sobel Y Image', sobel_y_image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


start()