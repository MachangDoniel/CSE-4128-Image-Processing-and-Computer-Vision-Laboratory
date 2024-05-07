import cv2
import matplotlib.pyplot as plt
import numpy as np


img_input = cv2.imread('two_noise.jpeg', 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]
height = img.shape[0]
width = img.shape[1]

H1 = np.ones_like(img)
H2 = np.ones_like(img)



# point_x_H1 = int(input("Enter the x co-ordinate for H1: "))
# point_y_H1 = int(input("Enter the y co-ordinate for H1: "))


# point_x_H2 = int(input("Enter the x co-ordinate for H2: "))
# point_y_H2 = int(input("Enter the y co-ordinate for H2: "))

point_x_H1=261
point_y_H1=262
point_x_H2=272
point_y_H2=256

for i in range(-2, 2):
    for j in range(-2, 2):
        H1[int(point_y_H1) + i, int(point_x_H1) + j] = 0
        H1[(width - point_y_H1) + i,(height - point_x_H1)+ j] = 0

        H2[int(point_y_H2) + i, int(point_x_H2) + j] = 0
        H2[width - int(point_y_H2 )+i, height - int(point_x_H2) + j] = 0


Hnr = H1*H2

H1_show = cv2.normalize(H1, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
H2_show = cv2.normalize(H2, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
Hnr_show = cv2.normalize(Hnr, None, 0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

cv2.imshow('H1',H1_show)

cv2.imshow('H2',H2_show)

cv2.imshow('HNR',Hnr_show)

ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift)+1)
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)

## phase add
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j*ang))
final_result =np.multiply(final_result,Hnr)



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