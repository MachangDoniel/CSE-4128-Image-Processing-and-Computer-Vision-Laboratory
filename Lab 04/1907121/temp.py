import cv2 # for finding contours to extract points of figure
import matplotlib.pyplot as plt # for plotting and creating figures
import numpy as np # for easy and fast number calculation
from math import tau # tau is constant number = 2*PI
from scipy.integrate import quad_vec # for calculating definite integral 
from tqdm import tqdm # for progress bar
import matplotlib.animation as animation # for compiling animation and exporting video 

# Define global variable
Img_name='mine.jpg'
X_list=None
Y_list=None
Ylim_data=None
Ylim_data=None
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



# function to generate x+iy at given time t
def f(t, t_list, x_list, y_list):
    return np.interp(t, t_list, x_list + 1j*y_list)

def getGrayImg():
    # reading the image and convert to greyscale mode
    # ensure that you use image with black image with white background
    image_path="image/"+Img_name
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

def printContours(contours):

    for i in range(len(contours)):
        print(contours[i])

def findContours(img_gray):
    # find the contours in the image
    img_gray=cv2.Canny(img_gray)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0) # making pure black and white image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) # finding available contours i.e closed loop objects
    
    # printContours(contours)
    
    return contours

def selectContour(contours):

    contours = np.array(contours[1]) # contour at index 1 is the one we are looking for
    
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
    xlim_data = plt.xlim() 
    ylim_data = plt.ylim()

    plt.show()
    return x_list,y_list,xlim_data,ylim_data




def generateCoefficient(x_list, y_list):

    # time data from 0 to 2*PI as x,y is the function of time.
    t_list = np.linspace(0, tau, len(x_list)) # now we can relate f(t) -> x,y

    # Now find fourier coefficient from -n to n circles
    # ..., c-3, c-2, c-1, c0, c1, c2, c3, ...
    order = 100 # -order to order i.e -100 to 100
    # you can change the order to get proper figure
    # too much is also not good, and too less will not produce good result

    print("generating coefficients ...")
    # lets compute fourier coefficients from -order to order
    c = []
    # we need to calculate the coefficients from -order to order
    for n in range(-order, order+1):
        # calculate definite integration from 0 to 2*PI
        # formula is given in readme
        coef = 1/tau*quad_vec(lambda t: f(t, t_list, x_list, y_list)*np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
        c.append(coef)
    print("completed generating coefficients.")

    # converting list into numpy array
    c = np.array(c)

    return c


def makeAnimation(xlim_data,ylim_data):

    global Circles, Circle_lines, Drawing, Orig_drawing, Writer, Draw_x, Draw_y, Fig

    ## -- now to make animation with epicycle -- ##

    # this is to store the points of last circle of epicycle which draws the required figure
    draw_x, draw_y = [], []

    # make figure for animation
    fig, ax = plt.subplots()

    # different plots to make epicycle
    # there are -order to order numbers of circles
    circles = [ax.plot([], [], 'r-')[0] for i in range(-order, order+1)]
    # circle_lines are radius of each circles
    circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-order, order+1)]
    # drawing is plot of final drawing
    drawing, = ax.plot([], [], 'k-', linewidth=2)

    # original drawing
    orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)

    # to fix the size of figure so that the figure does not get cropped/trimmed
    ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
    ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)

    # hide axes
    ax.set_axis_off()

    # to have symmetric axes
    ax.set_aspect('equal')

    # Set up formatting for the video file
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=30, metadata=dict(artist='Amrit Aryal'), bitrate=1800)

    print("compiling animation ...")
    
    Circles = circles
    Circle_lines = circle_lines
    Drawing = drawing
    Orig_drawing = orig_drawing
    Writer = writer
    Draw_x = draw_x
    Draw_y = draw_y
    Fig = fig
    # return circles,circle_lines,drawing,orig_drawing,writer,draw_x,draw_y,fig

# save the coefficients in order 0, 1, -1, 2, -2, ...
# it is necessary to make epicycles
def sort_coeff(coeffs):
    new_coeffs = []
    new_coeffs.append(coeffs[order])
    for i in range(1, order+1):
        new_coeffs.extend([coeffs[order+i],coeffs[order-i]])
    return np.array(new_coeffs)

# make frame at time t
# t goes from 0 to 2*PI for complete cycle
def make_frame(i, time, coeffs):
    # get t from time
    t = time[i]

    # exponential term to be multiplied with coefficient 
    # this is responsible for making rotation of circle
    exp_term = np.array([np.exp(n*t*1j) for n in range(-order, order+1)])

    # sort the terms of Fourier expression
    coeffs = sort_coeff(coeffs*exp_term)

    # split into x and y coefficients
    x_coeffs = np.real(coeffs)
    y_coeffs = np.imag(coeffs)

    # center points for first circle
    center_x, center_y = 0, 0

    # make all circles i.e epicycle
    for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
        # calculate radius of current circle
        r = np.linalg.norm([x_coeff, y_coeff])

        # draw circle with given radius at given center points of circle
        theta = np.linspace(0, tau, num=50)
        x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
        Circles[i].set_data(x, y)

        # draw a line to indicate the direction of circle
        x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
        Circle_lines[i].set_data(x, y)

        # calculate center for next circle
        center_x, center_y = center_x + x_coeff, center_y + y_coeff
    
    # center points now are points from last circle
    Draw_x.append(center_x)
    Draw_y.append(center_y)

    # draw the curve from last point
    Drawing.set_data(Draw_x, Draw_y)

    # draw the real curve
    Orig_drawing.set_data(X_list, Y_list)





def main():
    img_gray = getGrayImg()
    contours = findContours(img_gray)
    x_list,y_list,xlim_data,ylim_data = selectContour(contours)
    c = generateCoefficient(x_list, y_list)
    # Circles, Circle_lines, Drawing, Orig_drawing, Writer, Draw_x, Draw_y, Fig = makeAnimation(xlim_data, ylim_data)
    makeAnimation(xlim_data, ylim_data)
    X_list,Y_list,Ylim_data,Ylim_data=x_list,y_list,xlim_data,ylim_data
    # make animation
    # time is array from 0 to tau 
    
    time = np.linspace(0, tau, num=frames)
    
    anim = animation.FuncAnimation(Fig, make_frame, frames=frames, fargs=(time, c), interval=5)
    
    file_name_part = Img_name.split('.')
    output_file_name = file_name_part[0]+".mp4"
    output_image_path = "videos/"+output_file_name
    
    anim.save(output_image_path, writer=Writer)
    
    print("completed successfully!")



main()
