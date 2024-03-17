import numpy as np
import matplotlib.pyplot as plt
import cv2

from task import *

def double_gaussian_histogram(size, mu1, sigma1, mu2, sigma2, mul1, mul2):
    x = np.arange(256)
    divisor_part1=1/((2*np.pi*sigma1*sigma1)**0.5)
    gaussian1 = mul1 * divisor_part1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    divisor_part2=1/((2*np.pi*sigma2*sigma2)**0.5)
    gaussian2 = mul2 * divisor_part2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    histogram = gaussian1 + gaussian2
    # histogram /= np.sum(histogram)  # Normalize to ensure sum equals 1
    return histogram

target_histogram = double_gaussian_histogram(256, 30, 8, 165, 20, 1, 1)


def histogram_matching(input_image, target_histogram):
    hist, _ = np.histogram(input_image.flatten(), bins=256, range=[0,256], density=True)
    cdf_input = hist.cumsum()
    cdf_input_normalized = (cdf_input - cdf_input.min()) / (cdf_input.max() - cdf_input.min())

    cdf_target = target_histogram.cumsum()
    cdf_target_normalized = (cdf_target - cdf_target.min()) / (cdf_target.max() - cdf_target.min())

    matched_output = np.interp(input_image.flatten(), np.arange(256), cdf_target_normalized * 255)
    return matched_output.reshape(input_image.shape).astype(np.uint8)

def histo(title,img):
    # histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(img)
    plt.title("Histogram of "+title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()

def start():
    print("<-- Welcome! -->\n\n")

    image_name = 'histogram.jpg'
    image_path = '.\images3\\'+image_name

    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    histo("Target Histogram",target_histogram)

    # cv2.imshow("Input image",input_image)
    # cv2.waitKey(0)
    pdf= generatePDF(input_image)
    cdf= generateCDF(pdf)
    histogram("Input Image", input_image,pdf,cdf)

    
    output_image = histogram_matching(input_image, target_histogram)
    # cv2.imshow("Output image",output_image)
    # cv2.waitKey(0)
    pdf= generatePDF(output_image)
    cdf= generateCDF(pdf)
    histogram("Output Image", output_image,pdf,cdf)




start()