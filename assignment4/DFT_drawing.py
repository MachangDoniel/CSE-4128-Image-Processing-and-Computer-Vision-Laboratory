import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
from tqdm import tqdm
import matplotlib.animation as animation

# Function to generate complex numbers x + iy at given time t
def generate_complex(t, time_values, x_values, y_values):
    return np.interp(t, time_values, x_values + 1j * y_values)

# Reading the image and converting it to grayscale
img = cv2.imread("in.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying Canny edge detection
edges = cv2.Canny(gray_img, 100, 200)

# Finding contours in the edge-detected image
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = np.array(contours[1])

# Splitting the coordinate points of the contour
x_vals, y_vals = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)

# Centering the contour to the origin
x_vals = x_vals - np.mean(x_vals)
y_vals = y_vals - np.mean(y_vals)

# Visualizing the contour
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_vals, y_vals)

# Storing data for fixing the figure size
xlim_data = plt.xlim()
ylim_data = plt.ylim()

plt.show()

# Time data from 0 to 2*PI as x and y are functions of time
time_values = np.linspace(0, tau, len(x_vals))

# Finding Fourier coefficients from -n to n circles
num_coefficients = 100
print("Generating coefficients...")
coefficients = []
progress_bar = tqdm(total=(num_coefficients * 2 + 1))

for n in range(-num_coefficients, num_coefficients + 1):
    integral = 1 / tau * quad_vec(lambda t: generate_complex(t, time_values, x_vals, y_vals) * np.exp(-n * t * 1j), 0, tau, limit=100, full_output=1)[0]
    coefficients.append(integral)
    progress_bar.update(1)

progress_bar.close()
print("Completed generating coefficients.")

# Saving the coefficients for later use
np.save("fourier_coefficients.npy", np.array(coefficients))

## Creating animation with epicycle ##

# Lists to store points of the last circle of epicycle which draws the required figure
draw_x, draw_y = [], []

# Setting up the figure for animation
fig, ax = plt.subplots()

# Different plots to make the epicycle
circles = [ax.plot([], [], 'r-')[0] for _ in range(-num_coefficients, num_coefficients + 1)]
circle_lines = [ax.plot([], [], 'b-')[0] for _ in range(-num_coefficients, num_coefficients + 1)]
drawing, = ax.plot([], [], 'k-', linewidth=2)
orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

# Fixing the size of the figure
ax.set_xlim(xlim_data[0] - 200, xlim_data[1] + 200)
ax.set_ylim(ylim_data[0] - 200, ylim_data[1] + 200)
ax.set_axis_off()
ax.set_aspect('equal')

# Setting up formatting for the video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Your Name'), bitrate=1800)

print("Compiling animation...")
num_frames = 300
progress_bar = tqdm(total=num_frames)

# Function to sort coefficients for making epicycles
def sort_coefficients(coefficients):
    new_coeffs = [coefficients[num_coefficients]]
    for i in range(1, num_coefficients + 1):
        new_coeffs.extend([coefficients[num_coefficients + i], coefficients[num_coefficients - i]])
    return np.array(new_coeffs)

def generate_frame(frame_index, time, coeffs):
    global progress_bar
    t = time[frame_index]
    exp_term = np.array([np.exp(n * t * 1j) for n in range(-num_coefficients, num_coefficients + 1)])
    coeffs = sort_coefficients(coeffs * exp_term)
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)
    center_x, center_y = 0, 0

    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        r = np.linalg.norm([x_coeff, y_coeff])
        theta = np.linspace(0, tau, num=50)
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        circles[i].set_data(x, y)
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        circle_lines[i].set_data(x, y)
        center_x, center_y = center_x + x_coeff, center_y + y_coeff

    draw_x.append(center_x)
    draw_y.append(center_y)
    drawing.set_data(draw_x, draw_y)
    orig_drawing.set_data(x_vals, y_vals)
    progress_bar.update(1)

time = np.linspace(0, tau, num=num_frames)
anim = animation.FuncAnimation(fig, generate_frame, frames=num_frames, fargs=(time, coefficients), interval=5)
anim.save('output_animation.mp4', writer=writer)
progress_bar.close()
print("Completed: output_animation.mp4")
