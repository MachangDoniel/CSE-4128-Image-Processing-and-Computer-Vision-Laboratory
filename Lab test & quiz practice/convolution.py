import numpy as np
import cv2
from kernel_generator import generateGuassianKernel, generateMeanKernel, generateLaplacianKernel, generateLogKernel, generateSobelKernel


def getKernelCenter(kernel):
    centerX = kernel.shape[0]//2
    centerY = kernel.shape[1]//2
    while(True):
        print("Enter -1 to use the actual center")
        print("Enter centerX: ", end=' ')
        centerX = int(input())
        if(centerX == -1):
            return kernel.shape[0]//2, kernel.shape[1]//2
        else:
            print("Enter centerY: ", end=' ')
            centerY = int(input())
        
        if(0 <= centerX < kernel.shape[0] and 0 <= centerY < kernel.shape[1]):
            break
        else:
            print("Invalid center. Please enter again")
    
    return centerX, centerY


def convolution(input, kernel, center):

    centerX, centerY = center[0], center[1]

    print("Center: ", centerX, centerY)

    kernel = np.flipud(np.fliplr(kernel))
    
    input_h, input_w = input.shape[:2]
    kernel_h, kernel_w = kernel.shape

    pad_t = centerX
    pad_b = kernel_h - centerX - 1
    pad_l = centerY
    pad_r = kernel_w - centerY - 1
    
    # grayscale
    if input.ndim == 2:
        input_padded = np.pad(input, ((pad_t, pad_b), (pad_l, pad_r)))
        output = np.zeros((input_h, input_w))

        print("It's not a colored Image")
        
        for i in range(input_h):
            for j in range(input_w):
                region = input_padded[i:i + kernel_h, j:j + kernel_w]
                output[i, j] = np.sum(region * kernel)
        
        output = np.clip(output, 0, 255).astype(np.uint8)
    # colored
    elif input.ndim == 3:
        input_padded = np.pad(input, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)))
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

    center = getKernelCenter(kernel)
    output = convolution(input, kernel, center)
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

    center = getKernelCenter(kernel)
    output = convolution(img, kernel, center)
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
    
    center = getKernelCenter(kernel)
    output = convolution(img, kernel, center)
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
    
    center = getKernelCenter(kernel)
    output = convolution(img_hsv, kernel, center)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    print('Output for hsv image:')
    print(output)
    
    cv2.imshow('Output for hsv image', output_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_grayscale():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Image', img)
    kernel = generateGuassianKernel(1, 1)
    # kernel = generateMeanKernel(3)
    # kernel = generateLaplacianKernel(True, 3)
    # kernel = generateLaplacianKernel(False, 3)
    # kernel = generateLogKernel(1.4)
    # kernel = generateSobelKernel(True)
    # kernel = generateSobelKernel(False)

    center = getKernelCenter(kernel)
    output = convolution(img, kernel, center)
    cv2.imshow('Output for grayscale image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def normalize(input):
    return cv2.normalize(input,input,0,255,cv2.NORM_MINMAX)

def convolutionGray(input, kernel):
    center = getKernelCenter(kernel)
    convolutionGray(input, kernel, center)

def convolutionRGB(input, kernel):
    center = getKernelCenter(kernel)
    blue, green, red = cv2.split(input)
    blue = convolution(blue, kernel, center)
    green = convolution(green, kernel, center)
    red = convolution(red, kernel, center)
    merged = cv2.merge((blue, green, red))
    cv2.imshow('Blue', blue)
    cv2.imshow('Green', green)
    cv2.imshow('Red', red)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return merged

def convolutionHSV(input, kernel):
    center = getKernelCenter(kernel)
    hue, saturation, value = cv2.split(input)
    # hue = convolution(hue, kernel, center)
    # saturation = convolution(saturation, kernel, center)
    value = convolution(value, kernel, center)
    merged = cv2.merge((hue, saturation, value))
    cv2.imshow('Hue', hue)
    cv2.imshow('Saturation', saturation)
    cv2.imshow('Value', value)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return merged


def test_rgb():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    # kernel = generateGuassianKernel(1, 1)
    # kernel = generateMeanKernel(3)
    # kernel = generateLaplacianKernel(True, 3)
    # kernel = generateLaplacianKernel(False, 3)
    kernel = generateLogKernel(1.4)
    # kernel = generateSobelKernel(True)
    # kernel = generateSobelKernel(False)
    output = convolutionRGB(img, kernel)
    cv2.imshow('Output for rgb image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_hsv():
    img = cv2.imread('Lena.jpg', cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    kernel = generateGuassianKernel(1, 1)
    # kernel = generateMeanKernel(3)
    # kernel = generateLaplacianKernel(True, 3)
    # kernel = generateLaplacianKernel(False, 3)
    # kernel = generateLogKernel(1.4)
    # kernel = generateSobelKernel(True)
    # kernel = generateSobelKernel(False)
    output = convolutionHSV(img, kernel)
    cv2.imshow('Output for hsv image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test()
    # test2()
    # test3()
    # test4()
    test_grayscale()
    # test_rgb()
    # test_hsv()
