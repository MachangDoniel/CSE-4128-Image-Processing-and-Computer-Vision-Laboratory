import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
from tqdm import tqdm
import matplotlib.animation as animation

Img_name = 'bangladesh.png'
X_list = None
Y_list = None
Ylim_data = None
Circles = None
Circle_lines = None
Drawing = None
Orig_drawing = None
Writer = None
Draw_x = None
Draw_y = None
Fig = None
order = 100
frames = 300
Imgs = []
Imgs_names = []

def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

def performCanny(img_gray):
    img_gray = cv2.Canny(img_gray, 50, 150)
    cv2.imshow('After performing Canny',img_gray)
    Imgs.append(img_gray)
    Imgs_names.append("After performing Canny")
    cv2.waitKey(0)
    # return img_gray

def getGrayImg():
    image_path = "image/" + Img_name
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    Imgs.append(img_gray)
    Imgs_names.append("Input Image")
    cv2.imshow("Input Image", img_gray)
    cv2.waitKey(0)
    # img_gray = performCanny(img_gray)
    performCanny(img_gray)
    return img_gray

def printContours(contours):
    for contour in contours:
        # split the co-ordinate points of the contour
        # we reshape this to make it 1D array
        x_list, y_list = contour[:, :, 0].reshape(-1,), -contour[:, :, 1].reshape(-1,)

        # center the contour to origin
        x_list = x_list - np.mean(x_list)
        y_list = y_list - np.mean(y_list)

        # visualize the contour
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x_list, y_list)

        plt.show()

def printContoursInSubplot(contours):
    num_contours = len(contours)
    num_rows = int(np.ceil(num_contours / 3))  # Assuming 3 columns
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))

    # Reshape axes if it's a 1D array
    if axes.ndim == 1:
        axes = axes.reshape((num_rows, 3))

    for idx, contour in enumerate(contours):
        # split the co-ordinate points of the contour
        # we reshape this to make it 1D array
        x_list, y_list = contour[:, :, 0].reshape(-1,), -contour[:, :, 1].reshape(-1,)

        # center the contour to origin
        x_list = x_list - np.mean(x_list)
        y_list = y_list - np.mean(y_list)

        # Plot contour in separate subplot
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]  # This assumes axes is a 2D array
        ax.plot(x_list, y_list)
        ax.set_title(f'Contour {idx+1}')

    # Remove empty subplots
    for i in range(num_contours, num_rows*3):
        row = i // 3
        col = i % 3
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()



def findContours(img_gray):
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # printContours(contours)
    printContoursInSubplot(contours)

    return contours

def selectContour(contours):

    print("Which Contour you want to animate?")
    contour_no = int(input("->"))
    contours = np.array(contours[contour_no-1]) # contour at index 1 is the one we are looking for
    print("On process...")
    # split the co-ordinate points of the contour
    # we reshape this to make it 1D array
    x_list, y_list = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)

    # center the contour to origin
    x_list = x_list - np.mean(x_list)
    y_list = y_list - np.mean(y_list)

    # visualize the contour
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(x_list, y_list)

    # later we will need these data to fix the size of figure
    # x_limit=(min(x_list),max(x_list))
    # y_limit=(min(y_list),max(y_list))
    # print(x_limit)
    # print(y_limit)
    xlim_data = plt.xlim() 
    ylim_data = plt.ylim()

    plt.show()
    return x_list,y_list,xlim_data,ylim_data


def generateCoefficient(x_list, y_list):
    t_list = np.linspace(0, tau, len(x_list))
    print(t_list)
    order = 100
    c = []
    for n in range(-order, order+1):
        coef = 1/tau * quad_vec(lambda t: f(t, t_list, x_list, y_list) * np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
        c.append(coef)
    c = np.array(c)
    return c

def makeAnimation(xlim_data, ylim_data):
    global Circles, Circle_lines, Drawing, Orig_drawing, Writer, Draw_x, Draw_y, Fig
    draw_x, draw_y = [], []
    fig, ax = plt.subplots()
    circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order+1)]
    circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order+1)]
    drawing, = ax.plot([], [], 'k-', linewidth=2)
    orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)
    ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
    ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)
    ax.set_axis_off()
    ax.set_aspect('equal')
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Machang'), bitrate=1800)
    print("compiling animation ...")
    Circles = circles
    Circle_lines = circle_lines
    Drawing = drawing
    Orig_drawing = orig_drawing
    Writer = writer
    Draw_x = draw_x
    Draw_y = draw_y
    Fig = fig

def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order+1):
        new_coeffs.extend([coeffs[order+i],coeffs[order-i]])
    return np.array(new_coeffs)

def make_frame(i, time, coeffs):
    t = time[i]
    exp_term = np.array([np.exp(n*t*1j) for n in range(-order, order+1)])
    coeffs = sort_coeff(coeffs*exp_term)
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)
    center_x, center_y = 0, 0
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        r = np.linalg.norm([x_coeff, y_coeff])
        theta = np.linspace(0, tau, num=50)
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        Circles[i].set_data(x, y)
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        Circle_lines[i].set_data(x, y)
        center_x, center_y = center_x + x_coeff, center_y + y_coeff
    Draw_x.append(center_x)
    Draw_y.append(center_y)
    Drawing.set_data(Draw_x, Draw_y)
    Orig_drawing.set_data(X_list, Y_list)

def printXY(x_list, y_list, xlim_data, ylim_data):
    print(x_list)
    print(y_list)
    print(xlim_data)
    print(ylim_data)

def main():
    img_gray = getGrayImg()
    contours = findContours(img_gray)
    x_list, y_list, xlim_data, ylim_data = selectContour(contours)
    print(x_list, y_list, xlim_data, ylim_data)
    c = generateCoefficient(x_list, y_list)
    print(c)
    makeAnimation(xlim_data, ylim_data)
    X_list, Y_list, Ylim_data, Ylim_data = x_list, y_list, xlim_data, ylim_data
    time = np.linspace(0, tau, num=frames)
    anim = animation.FuncAnimation(Fig, make_frame, frames=frames, fargs=(time, c), interval=5)
    file_name_part = Img_name.split('.')
    output_file_name = file_name_part[0] + ".mp4"
    output_image_path = "videos/" + output_file_name
    anim.save(output_image_path, writer=Writer)
    print("completed successfully!")


main()
