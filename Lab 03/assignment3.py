import numpy as np
import matplotlib.pyplot as plt
import cv2

from task import *


def double_gaussian_histogram(mu1, sigma1, mu2, sigma2, mul1, mul2):
    x = np.arange(256)
    divisor_part1 = 1 / ((2 * np.pi * sigma1 * sigma1) ** 0.5)
    gaussian1 = mul1 * divisor_part1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
    divisor_part2 = 1 / ((2 * np.pi * sigma2 * sigma2) ** 0.5)
    gaussian2 = mul2 * divisor_part2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
    histogram = gaussian1 + gaussian2
    # return histogram / np.sum(histogram)  # Normalize the histogram
    return histogram

def histogram_matching(input_image, target_histogram):
    # pdf, _ = np.histogram(input_image.flatten(), bins=256, range=[0, 256], density=True)
    pdf = generatePDF(input_image)
    cdf = generateCDF(pdf)
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())

    # cdf_target = target_histogram.cumsum()
    cdf_target = generateCDF(target_histogram)
    cdf_target_normalized = (cdf_target - cdf_target.min()) / (cdf_target.max() - cdf_target.min())

    # matched_output = np.interp(cdf_normalized, cdf_target_normalized, np.arange(256))
    matched_output = interp_array(cdf_normalized, cdf_target_normalized, 256)
    return matched_output[input_image].astype(np.uint8)

def interp_array(cdf_normalized, cdf_target_normalized, size):
    matched_output = np.zeros(size)
    for i in range(size):
        idx = (np.abs(cdf_target_normalized - cdf_normalized[i])).argmin()
        matched_output[i] = idx
    return matched_output

def histo(title,img):
    # histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    print(img)
    plt.figure()
    plt.plot(img)
    plt.title("Histogram of "+title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()


def histogram(title, img, pdf, cdf):
    # Display input image
    plt.figure(figsize=(12, 8))
    plt.subplot(221)
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Display histogram
    plt.subplot(222)
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(histr)
    plt.title("Histogram of " + title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")

    # Display PDF
    plt.subplot(223)
    plt.plot(pdf)
    plt.title("Probability Density Function (PDF) of " + title)
    plt.xlabel("Intensity")
    plt.ylabel("Probability")

    # Display CDF
    plt.subplot(224)
    plt.plot(cdf)
    plt.title("Cumulative Distribution Function (CDF) of " + title)
    plt.xlabel("Intensity")
    plt.ylabel("CDF Value")

    plt.tight_layout()
    plt.show()


def start():
    print("<-- Welcome! -->\n\n")

    image_name = 'histogram.jpg'
    image_path = '.\images3\\'+image_name

    input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    target_histogram = double_gaussian_histogram(60, 10, 160, 15, 1, 1)

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