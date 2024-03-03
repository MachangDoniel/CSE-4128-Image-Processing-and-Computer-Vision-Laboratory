# Gray Scale
import cv2
import numpy as np

def convolve(image, kernel):
    if image.ndim == 3:
        # Apply convolution separately for each channel
        channels = [convolution(image[:, :, i], kernel) for i in range(3)]
        return cv2.merge(channels)
    else:
        return convolution(image, kernel)

def convolution(image, kernel):
    # Get image dimensions and kernel size
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Pad the image
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Apply convolution
    convolved_image = np.zeros_like(image)
    for y in range(image_height):
        for x in range(image_width):
            convolved_image[y, x] = np.sum(padded_image[y:y+kernel_height, x:x+kernel_width] * kernel)

    return convolved_image

def mean_filter_kernel(size):
    return np.ones((size, size), np.float32) / (size * size)

def display_images(image_dict, wait_time=0):
    """Displays multiple images simultaneously."""
    for window_name, image in image_dict.items():
        cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()

def start():
    image_path = 'images\Lena.jpg'  
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to load image at path '{image_path}'.")
        exit()

    # Mean filter kernel size
    kernel_size = 3
    kernel = mean_filter_kernel(kernel_size)

    # Convolve in RGB space
    convolved_rgb = convolve(image, kernel)

    # Convert to HSV, convolve, and convert back to RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    convolved_hsv_temp = convolve(image_hsv, kernel)
    convolved_hsv = cv2.cvtColor(convolved_hsv_temp, cv2.COLOR_HSV2BGR)

    # Compute the absolute difference
    difference = cv2.absdiff(convolved_rgb, convolved_hsv)

    # Displaying the images
    image_dict = {
        'Original Image': image,
        'Convolved RGB Image': convolved_rgb,
        'Convolved HSV Image': convolved_hsv,
        'Difference Between RGB and HSV Convolved Images': difference
    }
    display_images(image_dict)


start()