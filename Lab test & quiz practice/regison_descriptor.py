import numpy as np
import cv2
from tabulate import tabulate


image_name = ['c1.jpg','t1.jpg','p1.png','c2.jpg','t2.jpg','p2.png', 'st.jpg']

train = []

test = []

dist = []

def find_max_diameter(img):
    h, w = img.shape
    min_x = h
    min_y = w
    max_x = 0
    max_y = 0
    for i in range(h):
        for j in range(w):
            if img[i, j] != 0:
                min_x = min(min_x, i)
                min_y = min(min_y, j)
                max_x = max(max_x, i)
                max_y = max(max_y, j)
    diameter = max(max_x - min_x, max_y - min_y)
    return diameter


def calculate_region_descriptor(img, i):
    kernel = np.ones((3,3))
    eroded = cv2.erode(img, kernel)
    border_img = img - eroded
    # cv2.imshow('Original Image', img)
    # cv2.imshow('Eroded Image', eroded)
    # cv2.imshow('Border Image', border_img)

    area = np.count_nonzero(img)
    perimeter = np.count_nonzero(border_img)
    max_diameter = find_max_diameter(img)

    print("Area, Permimeter, Max_Diameter: ",area, perimeter, max_diameter)

    compactness = perimeter**2 / area
    print("Compactness: ",compactness)
    form_factor = 4 * np.pi * area / (perimeter**2)
    print("Form Factor: ",form_factor)
    roundness = 4 * area / (np.pi * max_diameter**2)
    print("Roundness: ",roundness)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return compactness, form_factor, roundness

def show(distances_matrix):
    row_headers = ['c2.jpg','t2.jpg','p2.png', 'st.jpg']
    col_headers = ['c1.jpg','t1.jpg','p1.png']

    distances_matrix = np.array(distances_matrix)
    print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))

def write_to_file(im_title, main_data):
    file_path = 'output.txt'
    with open(file_path, 'w') as file:
        file.write('\t'.join(map(str, [' ', 'form_factor', 'roundness','compactness'])) + '\n')
        file.write('-' * 50 + '\n')
        i=0
        for row in main_data:
            file.write(im_title[i]+'\t\t')
            line = '\t\t'.join(map(str, row ))
            i=i+1
            file.write(line + '\n')
            file.write('-' * 50 + '\n')


def find_distance(train, test):
    diff_c = abs(train[0] - test[0])
    diff_f = abs(train[1] - test[1])
    diff_r = abs(train[2] - test[2])

    return np.sqrt(diff_c**2 + diff_f**2 + diff_r**2)

for i in range(3):
    root = 'region_images/'
    image_path = root + image_name[i]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Image not found')
        continue
    else:
        print('Image found')
        train.append(calculate_region_descriptor(img, i))

for i in range(3, len(image_name)):
    root = 'region_images/'
    image_path = root + image_name[i]
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print('Image not found')
        continue
    else:
        print('Image found')
        c_f_r = calculate_region_descriptor(img, i)
        test.append(c_f_r)

        my_d = []
        for j in range(3):
            my_d.append(find_distance(train[j], c_f_r))
        dist.append(my_d)

print(dist)

for item in test:
    train.append(item)

write_to_file(image_name, train)
show(dist)

cv2.waitKey(0)
cv2.destroyAllWindows()
