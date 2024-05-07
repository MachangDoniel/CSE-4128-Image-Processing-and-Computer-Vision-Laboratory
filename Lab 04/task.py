
# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

def generatediff(x1,y1,w,h):
    if(x1>w/2):
        x2=w/2-abs(x1-w/2)
    else:
        x2=(w/2)+abs(x1-w/2)
    if(y1>h/2):
        y2=h/2-abs(y1-h/2)
    else:
        y2=h/2+ abs(y1-h/2)
    return x2,y2

def generateH(img,x1,y1):
    img-img.copy()
    w,h = img.shape
    H = np.zeros((w,h))
    x2,y2 = generatediff(x1,y1,w,h)
    for i in range(0,w):
        for j in range(0,h):
            if((i==x1 and j==y1) or (i==x2 and j==y2)):
                H[i,j]=1
            else:
                H[i,j]=0

    return H
    


def generateNotchFilter(img,x1,y1,x2,y2):
    H1 = generateH(img,x1,y1)
    H2 = generateH(img,x2,y2)
    Hnr = H1+H2
    print(Hnr)
    cv2.imshow("Hnr", Hnr)
    cv2.waitKey(0)
    return Hnr


def start():
    print("<-- Welcome! -->\n\n")

        # take input
    img_input = cv2.imread('two_noise.jpeg', 0)
    img = img_input.copy()
    image_size = img.shape[0] * img.shape[1]

    Hnr = generateNotchFilter(img,261,262,272,256)


    #%%
    # fourier transform
    ft = np.fft.fft2(img)
    ft_shift = np.fft.fftshift(ft)
    #ft_shift = ft
    magnitude_spectrum_ac = np.abs(ft_shift)
    magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 


    magnitude_spectrum=Hnr * magnitude_spectrum


    
    

    ang = np.angle(ft_shift)
    ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U) 
    ## phase add
    final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))

    # inverse fourier
    img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
    img_back_scaled = cv2.normalize(img_back, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    ## plot
    cv2.imshow("input", img_input)
    cv2.imshow("Magnitude Spectrum",magnitude_spectrum)
    cv2.imshow("Phase", ang_)
    cv2.waitKey(0)
    cv2.imshow("Inverse transform",img_back_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


start()