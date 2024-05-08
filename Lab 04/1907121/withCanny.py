import cv2
import matplotlib.pyplot as plt
import numpy as np
from math import tau
from scipy.integrate import quad_vec
import matplotlib.animation as animation
from tkinter import filedialog
import tkinter as tk

class ContourAnimationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Contour Animation App")
        self.root.geometry("400x200")

        self.img_gray = None
        self.contours = None
        self.x_list = None
        self.y_list = None
        self.xlim_data = None
        self.ylim_data = None
        self.circles = None
        self.circle_lines = None
        self.drawing = None
        self.orig_drawing = None
        self.writer = None
        self.draw_x = None
        self.draw_y = None
        self.fig = None
        self.order = 100
        self.frames = 300
        self.img_path = None

        self.create_widgets()

    def create_widgets(self):
        btn_load_img = tk.Button(self.root, text="Load Image", command=self.load_image)
        btn_load_img.pack(pady=10)

        btn_run_animation = tk.Button(self.root, text="Run Animation", command=self.run_animation)
        btn_run_animation.pack(pady=10)

    def load_image(self):
        self.img_path = filedialog.askopenfilename(initialdir="./", title="Select Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        self.img_gray = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)

    def perform_canny(self):
        img_canny = cv2.Canny(self.img_gray, 50, 150)
        return img_canny

    def find_contours(self):
        ret, thresh = cv2.threshold(self.img_gray, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        return contours

    def select_contour(self, contour_no):
        contour = np.array(self.contours[contour_no - 1])
        x_list, y_list = contour[:, :, 0].reshape(-1,), -contour[:, :, 1].reshape(-1,)
        x_list = x_list - np.mean(x_list)
        y_list = y_list - np.mean(y_list)
        return x_list, y_list

    def generate_coefficient(self, x_list, y_list):
        t_list = np.linspace(0, tau, len(x_list))
        c = []
        for n in range(-self.order, self.order+1):
            coef = 1/tau * quad_vec(lambda t: self.f(t, t_list, x_list, y_list) * np.exp(-n*t*1j), 0, tau, limit=100, full_output=1)[0]
            c.append(coef)
        c = np.array(c)
        return c

    def make_animation(self):
        draw_x, draw_y = [], []
        fig, ax = plt.subplots()
        circles = [ax.plot([], [], 'r-')[0] for i in range(-self.order, self.order+1)]
        circle_lines = [ax.plot([], [], 'b-')[0] for i in range(-self.order, self.order+1)]
        drawing, = ax.plot([], [], 'k-', linewidth=2)
        orig_drawing, = ax.plot([], [], 'g-', linewidth=0.5)
        ax.set_xlim(self.xlim_data[0]-200, self.xlim_data[1]+200)
        ax.set_ylim(self.ylim_data[0]-200, self.ylim_data[1]+200)
        ax.set_axis_off()
        ax.set_aspect('equal')
        writer = animation.writers['ffmpeg'](fps=30, metadata=dict(artist='Machang'), bitrate=1800)
        print("compiling animation ...")
        self.circles = circles
        self.circle_lines = circle_lines
        self.drawing = drawing
        self.orig_drawing = orig_drawing
        self.writer = writer
        self.draw_x = draw_x
        self.draw_y = draw_y
        self.fig = fig

    def f(self, t, t_list, x_list, y_list):
        return np.interp(t, t_list, x_list + 1j*y_list)

    def sort_coeff(self, coeffs):
        new_coeffs = []
        new_coeffs.append(coeffs[self.order])
        for i in range(1, self.order+1):
            new_coeffs.extend([coeffs[self.order+i], coeffs[self.order-i]])
        return np.array(new_coeffs)

    def make_frame(self, i, time, coeffs):
        t = time[i]
        exp_term = np.array([np.exp(n*t*1j) for n in range(-self.order, self.order+1)])
        coeffs = self.sort_coeff(coeffs*exp_term)
        x_coeffs = np.real(coeffs)
        y_coeffs = np.imag(coeffs)
        center_x, center_y = 0, 0
        for i, (x_coeff, y_coeff) in enumerate(zip(x_coeffs, y_coeffs)):
            r = np.linalg.norm([x_coeff, y_coeff])
            theta = np.linspace(0, tau, num=50)
            x, y = center_x + r * np.cos(theta), center_y + r * np.sin(theta)
            self.circles[i].set_data(x, y)
            x, y = [center_x, center_x + x_coeff], [center_y, center_y + y_coeff]
            self.circle_lines[i].set_data(x, y)
            center_x, center_y = center_x + x_coeff, center_y + y_coeff
        self.draw_x.append(center_x)
        self.draw_y.append(center_y)
        self.drawing.set_data(self.draw_x, self.draw_y)
        self.orig_drawing.set_data(self.x_list, self.y_list)

    def run_animation(self):
        if self.img_path:
            self.img_gray = self.perform_canny()
            self.contours = self.find_contours()
            self.x_list, self.y_list = self.select_contour(1)  # Select first contour by default
            self.xlim_data = plt.xlim()
            self.ylim_data = plt.ylim()
            coeffs = self.generate_coefficient(self.x_list, self.y_list)
            self.make_animation()
            time = np.linspace(0, tau, num=self.frames)
            anim = animation.FuncAnimation(self.fig, self.make_frame, frames=self.frames, fargs=(time, coeffs), interval=5)
            output_file_name = self.img_path.split('.')[0] + ".mp4"
            output_image_path = "videos/" + output_file_name
            anim.save(output_image_path, writer=self.writer)
            print("completed successfully!")
        else:
            print("Please select an image first.")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ContourAnimationApp()
    app.run()
