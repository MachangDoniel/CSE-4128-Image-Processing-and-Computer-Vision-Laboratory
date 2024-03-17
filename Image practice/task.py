import cv2
import numpy as np




def generatePDF(image_path):
    image= cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    image=image.copy()

    h,w= image.shape
    print("Size of the image: ",w,h)

    output=np.zeros((w+1,h+1))
    pdf=np.zeros(256)

    for i in range(h):
        for j in range(w):
            intensity=image[i][j]
            pdf[intensity]+=1
    
    pdf=pdf/(h*w)
    print(pdf)
    return pdf

def round(n):
    base=int(n)
    if(n>base+.5):
        return min(255,base+1)
    else:
        return base

def generateCDF(pdf):
    cdf=pdf.copy()
    cdf*=255
    for i in range(1,len(cdf)):
        cdf[i]+=cdf[i-1]
        cdf[i]=round(cdf[i])

    print(cdf)
    return cdf

def generateOutput(image_path,cdf):
    image= cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    image=image.copy()

    h,w= image.shape
    for i in range(h):
        for j in range(w):
            intensity=image[i][j]
            image[i][j]=cdf[intensity]

    cv2.imshow("Output image",image)
    cv2.waitKey(0)

    return image

def histogram(img):
    #cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
    histr = cv2.calcHist([img],[0],None,[256],[0,256])
    plt.figure(1)
    plt.plot(histr)
    plt.show()

    #plt.figure(figsize=(10, 4))

    #plt.subplot(1, 2, 1)
    plt.figure(2)
    plt.title("Input Image Histogram")
    plt.hist(img.ravel(),256,[0,255])
    plt.show()

    img2 = cv2.equalizeHist(img)

    #plt.subplot(1, 2, 2)
    plt.figure(3)
    plt.title("output Image Histogram")
    plt.hist(img2.ravel(),256,[0,255])
    plt.show()

    cv2.imshow("output",img2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def start():
    print("<-- Welcome! -->\n\n")
    image_name = 'histogram.jpg'

    image_path = '.\images3\\'+image_name
        
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Input image",image)
    cv2.waitKey(0)


    print(image)
    pdf= generatePDF(image_path)
    cdf= generateCDF(pdf)
    output=generateOutput(image_path,cdf)
    histogram(output)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

start()