import cv2
import numpy as np

image_path='images/'
image_name = ['c1.jpg','st.jpg','p1.png','p2.png','p3.jpg','t1.jpg','t2.jpg']
temp_image_name = ['c1.jpg','st.jpg']
form_factor = []
round_ness = []
compact_ness = []

perimeter = None
Area = None
MaxDiameter = None

def calculate_descriptors(image,i):
    #_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((3,3), np.uint8)
    eroded = cv2.erode(img, kernel, iterations=1)
    border = img - eroded
    # area,perimeter,
    max_diameter = 0
    area = np.count_nonzero(img)
    border_image = area - eroded
    perimeter = np.count_nonzero(border_image)
    width, height = img.shape[1],img.shape[0]
    arr = []
    for i in range(height):
        for j in range(width):
            if(border_image[i][j]!=0):
                arr.append((i,j))

    # print(arr)
    for pair in arr:
        # print(pair)
        i,j = pair
        for xy in arr:
            # print(xy)
            x,y=xy
            # print(i,j,x,y)
            max_diameter = max(max(i,x)-min(i,j),max(j,y)-min(j,y))
    # print(arr)
    print(area, perimeter, max_diameter)
    # print(img)
    # print(width, height)




    cv2.imshow('Border'+str(i), border)
    cv2.imshow('Input image'+str(i), img)

for i in range(len(temp_image_name)):
    img = cv2.imread(image_path+temp_image_name[i], 0)
    calculate_descriptors(img,i)

cv2.waitKey(0)
cv2.destroyAllWindows()  




# def main():
#     area,perimeter,max_diameter = getValues()
      