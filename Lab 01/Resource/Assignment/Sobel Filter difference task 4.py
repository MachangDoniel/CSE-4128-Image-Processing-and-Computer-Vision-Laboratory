import cv2
import numpy as np

def convolve(image, kernel):
    if image.ndim == 3:
        channels = [cv2.filter2D(image[:, :, i], -1, kernel) for i in range(3)]
        return cv2.merge(channels)
    else:
        return cv2.filter2D(image, -1, kernel)

def sobel_filter_x():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

def sobel_filter_y():
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]], dtype=np.float32)

def display_image(window_name, image, wait_time=1000):
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    # cv2.destroyWindow(window_name)

def start():
    # input image
    image_path = 'images\Lena.jpg'  
    image = cv2.imread(image_path)

    #if image is None:
    #    print(f"Error: Unable to load image at path '{image_path}'.")
    #   exit()

    # Sobel filter kernels
    sobel_kernel_x = sobel_filter_x()
    sobel_kernel_y = sobel_filter_y()

    # Convolve in RGB space
    convolved_sobel_x_rgb = convolve(image, sobel_kernel_x)
    convolved_sobel_y_rgb = convolve(image, sobel_kernel_y)

    # Convert to HSV, convolve, and convert back to RGB
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    convolved_sobel_x_hsv_temp = convolve(image_hsv, sobel_kernel_x)
    convolved_sobel_y_hsv_temp = convolve(image_hsv, sobel_kernel_y)
    convolved_sobel_x_hsv = cv2.cvtColor(convolved_sobel_x_hsv_temp, cv2.COLOR_HSV2BGR)
    convolved_sobel_y_hsv = cv2.cvtColor(convolved_sobel_y_hsv_temp, cv2.COLOR_HSV2BGR)

    # Compute the magnitude of gradient
    magnitude_rgb = cv2.magnitude(convolved_sobel_x_rgb.astype(np.float32), convolved_sobel_y_rgb.astype(np.float32))
    magnitude_hsv = cv2.magnitude(convolved_sobel_x_hsv_temp.astype(np.float32), convolved_sobel_y_hsv_temp.astype(np.float32))

    # Displaying the images
    display_image('Original Image', image)
    display_image('Convolved Sobel X RGB', convolved_sobel_x_rgb)
    display_image('Convolved Sobel Y RGB', convolved_sobel_y_rgb)
    display_image('Convolved Sobel X HSV', convolved_sobel_x_hsv)
    display_image('Convolved Sobel Y HSV', convolved_sobel_y_hsv)
    #display_image('Magnitude of Gradient RGB', magnitude_rgb)
    #display_image('Magnitude of Gradient HSV', magnitude_hsv)
    # Compute the absolute difference between convolved images in RGB and HSV spaces
    difference_convolved_x = cv2.absdiff(convolved_sobel_x_rgb, convolved_sobel_x_hsv)

    # Display the difference image
    display_image('Absolute Difference of Convolved Sobel X RGB and HSV', difference_convolved_x)
    # Compute the absolute difference between convolved images in RGB and HSV spaces
    difference_convolved_y = cv2.absdiff(convolved_sobel_y_rgb, convolved_sobel_y_hsv)

    # Display the difference image
    display_image('Absolute Difference of Convolved Sobel y RGB and HSV', difference_convolved_y)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    #destroy window
    cv2.destroyAllWindows()


start()