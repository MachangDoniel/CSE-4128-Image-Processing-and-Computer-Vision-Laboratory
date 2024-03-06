import cv2
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt 

from convolution import normalize, convolve
from edge_detection import find_threeshold

def perform_threshold(image, threes):
    highThreshold = threes * 0.5
    lowThreshold = highThreshold * 0.5
    
    M, N = image.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(75)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(image >= highThreshold)
    # zeros_i, zeros_j = np.where(image < lowThreshold)
    weak_i, weak_j = np.where( np.logical_and( (image <= highThreshold), (image >= lowThreshold) ) )
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def plot_histogram(img):
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(histr)
    plt.show()

def perform_hysteresis(image, weak, strong = 255):
    M, N = image.shape
    out = image.copy()

    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == weak):
                if np.any( image[i-1:i+2, j-1:j+2] == strong):
                    out[i, j] = strong
                else:
                    out[i, j] = 0
    return out


def start():
    image_path = 'images\suppressed.jpg'
    #image_path = 'images\lena.jpg'
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    plot_histogram(img=image)
    
    threes_val = find_threeshold(image=image)
    print(f"Threeshold value: {threes_val}")
    
    threes_otsu, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"Threeshold value(OTSU): {threes_otsu}")

    image = normalize(image=image)
    threes_image, weak, strong = perform_threshold(image, threes=threes_val)
    #print(weak,strong)

    cv2.imshow("Threesholded image", normalize(threes_image) )
    cv2.waitKey(0)

    final_output = perform_hysteresis( image=threes_image, weak=weak)
    cv2.imshow("Final Result", normalize(final_output))
    
    #final_output = final_output.astype(np.uint8)

    # image_path = '.\images\\lena.jpg'
    # image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    # image_canny = cv2.Canny(image,50,50)
    # cv2.imshow("Final Edge Library",image_canny)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#start()