import numpy as np
import cv2

def correlation(input, kernel):
    input_h, input_w = input.shape[:2]
    kernel_h, kernel_w = kernel.shape
    
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    
    if input.ndim == 2:
        input_padded = np.pad(input, ((pad_h, pad_h), (pad_w, pad_w)))
        output = np.zeros((input_h, input_w))

        print("It's not a colored Image")
        
        for i in range(input_h):
            for j in range(input_w):
                region = input_padded[i:i + kernel_h, j:j + kernel_w]
                output[i, j] = np.sum(region * kernel)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
    elif input.ndim == 3:
        input_padded = np.pad(input, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        output = np.zeros((input_h, input_w, input.shape[2]))
        
        print("It's a colored Image")

        for k in range(input.shape[2]):
            for i in range(input_h):
                for j in range(input_w):
                    region = input_padded[i:i + kernel_h, j:j + kernel_w, k]
                    output[i, j, k] = np.sum(region * kernel)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
    
    return output

# with numpy 2d array
def test():
    input = np.array([
                    [1, 1, 0, 1, 0],     
                    [0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1],     
                    [1, 0, 1, 0, 1],     
                    [0, 1, 1, 1, 0],     
                    ])

    print('Input:')
    print(input)
    kernel = np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                    ])
    # print('Kernel:')
    # print(kernel)
    output = correlation(input, kernel)
    print('Output for numpy 2d array:')
    print(output)

# with grayscale image
def test2():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Image', img)
    
    kernel = (1/273) * np.array([[1, 4, 7, 4, 1],
                                 [4, 16, 26, 16, 4],
                                 [7, 26, 41, 26, 7],
                                 [4, 16, 26, 16, 4],
                                 [1, 4, 7, 4, 1]])
    
    # print('Kernel:')
    # print(kernel)
    
    output = correlation(img, kernel)
    print('Output for grayscale image:')
    print(output)

    
    cv2.imshow('Output for grayscale image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# with rgb image
def test3():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    
    kernel = (1/273) * np.array([[1, 4, 7, 4, 1],
                                 [4, 16, 26, 16, 4],
                                 [7, 26, 41, 26, 7],
                                 [4, 16, 26, 16, 4],
                                 [1, 4, 7, 4, 1]])
    
    # print('Kernel:')
    # print(kernel)
    
    output = correlation(img, kernel)
    print('Output for rgb image:')
    print(output)
    
    cv2.imshow('Output for rgb image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# with hsv image
def test4():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow('Image', img_hsv)
    
    kernel = (1/273) * np.array([[1, 4, 7, 4, 1],
                                 [4, 16, 26, 16, 4],
                                 [7, 26, 41, 26, 7],
                                 [4, 16, 26, 16, 4],
                                 [1, 4, 7, 4, 1]])
    
    # print('Kernel:')
    # print(kernel)
    
    output = correlation(img_hsv, kernel)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    print('Output for hsv image:')
    print(output)
    
    cv2.imshow('Output for hsv image', output_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
    test2()
    test3()
    test4()
