import cv2
import numpy as np
from matplotlib import pyplot as plt

def generatePDF(input):
    h, w = input.shape
    pdf = np.zeros(256)
    for i in range(h):
        for j in range(w):
            pdf[input[i, j]] += 1
    pdf = pdf / (h * w)
    return pdf

def round_value(n):
    base = int(n)
    if n >= base + 0.5:
        return min(255, base + 1)
    else:
        return base

def generateCDF(pdf):
    cdf = 255 * pdf
    cdf[0] = round_value(cdf[0])
    for i in range(1, 256):
        cdf[i] += cdf[i - 1]
        cdf[i] = round_value(cdf[i])
    return cdf

def generateOutput(cdf, input):
    h, w = input.shape
    output = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            output[i, j] = cdf[input[i, j]]
    return output

def equalizeRGB(image):
    channels = cv2.split(image)
    equalized_channels = []
    for channel in channels:
        pdf = generatePDF(channel)
        cdf = generateCDF(pdf)
        equalized_channel = generateOutput(cdf, channel)
        equalized_channels.append(equalized_channel)
    equalized_image = cv2.merge(equalized_channels)
    return equalized_image

def equalizeHSV(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    pdf = generatePDF(v)
    cdf = generateCDF(pdf)
    v_equalized = generateOutput(cdf, v)
    hsv_equalized = cv2.merge([h, s, v_equalized])
    equalized_image = cv2.cvtColor(hsv_equalized, cv2.COLOR_HSV2BGR)
    return equalized_image

def test_Color():
    input_rgb = input_hsv = cv2.imread('Hot.png', cv2.IMREAD_COLOR)
    
    cv2.imshow('Original RGB', input_rgb)
    equalized_rgb = equalizeRGB(input_rgb)
    cv2.imshow('Equalized RGB', equalized_rgb)
    
    cv2.imshow('Original HSV', input_hsv)
    equalized_hsv = equalizeHSV(input_hsv)
    cv2.imshow('Equalized HSV', equalized_hsv)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_Gray():
    input = cv2.imread('histogram.jpg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('Input', input)
    pdf = generatePDF(input)
    cdf = generateCDF(pdf)
    output = generateOutput(cdf, input)
    cv2.imshow('Output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # test_Gray()
    test_Color()

if __name__ == '__main__':
    main()