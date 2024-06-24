import cv2
import numpy as np
from matplotlib import pyplot as plt

# Take input
image_path = 'two_noise.jpeg'
img_input = cv2.imread(image_path, 0)
img = img_input.copy()
image_size = img.shape[0] * img.shape[1]


# Define notch filter
def notch_filter(shape, center, radius):
    rows, cols = shape
    mask = np.ones((rows, cols), dtype=np.uint8)
    cv2.circle(mask, center, radius, 0, -1)  # Create a notch at the specified center and radius
    cv2.circle(mask, (cols - center[0], rows - center[1]), radius, 0, -1)  # Create a corresponding notch at (-u0, -v0)
    return mask

# User input for center coordinates
#x = int(input("Enter x coordinate of the center: "))
#y = int(input("Enter y coordinate of the center: "))
x=300
y=300
center = (x, y)
radius = 5

# Create the notch filter
notch_mask = notch_filter(img.shape, center, radius)

# Plot the notch filter
plt.imshow(notch_mask, cmap='gray')
plt.title('Notch Filter')
plt.colorbar()
plt.show()

# Fourier transform
ft = np.fft.fft2(img)
ft_shift = np.fft.fftshift(ft)
magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 20 * np.log(np.abs(ft_shift) + 1)

magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Multiply magnitude spectrum with notch filter
filtered_magnitude_spectrum = magnitude_spectrum * notch_mask




ang = np.angle(ft_shift)
ang_ = cv2.normalize(ang, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Apply notch filter
filtered_ft_shift = ft_shift * notch_mask

# Inverse Fourier transform
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_ft_shift)))
img_back_scaled = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# Plot results
cv2.imshow("Input", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum)
cv2.imshow("Phase", ang_)
cv2.imshow("Output Image", img_back_scaled)
cv2.imshow("Filtered Magnitude Spectrum", filtered_magnitude_spectrum)
#cv2.imshow("Filtered Magnitude Spectrum", filtered_magnitude_spectrum2)

cv2.waitKey(0)
cv2.destroyAllWindows()