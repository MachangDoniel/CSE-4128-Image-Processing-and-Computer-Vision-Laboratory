import cv2
import numpy as np

from matplotlib import pyplot as plt




def generatePDF(image):
    image=image.copy()

    h,w= image.shape
    print("Size of the image: ",w,h)

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
        return base+1
    else:
        return base

def generateCDF(pdf):
    cdf=pdf.copy()
    cdf*=255
    cdf[0]=round(cdf[0])
    for i in range(1,len(cdf)):
        cdf[i]+=cdf[i-1]
        cdf[i]=round(cdf[i])

    # Normalize the CDF to ensure the sum equals 255
    cdf = np.round(cdf / cdf[-1] * 255).astype(np.uint8)
    
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

def histogram(title, img, pdf, cdf):
    # Display input image
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

    #display histogram
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(histr)
    plt.title("Histogram of "+title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.show()

    # Display PDF
    plt.figure()
    plt.plot(pdf)
    plt.title("Probability Density Function (PDF) of "+title)
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.show()

    # Display CDF
    plt.figure()
    plt.plot(cdf)
    plt.title("Cumulative Distribution Function (CDF) of "+title)
    plt.xlabel("Intensity")
    plt.ylabel("CDF Value")
    plt.show()



def start():
    print("<-- Welcome! -->\n\n")
    image_name = 'histogram.jpg'

    image_path = '.\images3\\'+image_name
        
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    # cv2.imshow("Input image",image)
    # cv2.waitKey(0)


    print(image)

    pdf= generatePDF(image)
    cdf= generateCDF(pdf)
    histogram("Input Image",image,pdf,cdf)
    output=generateOutput(image_path,cdf)
    pdf= generatePDF(output)
    cdf= generateCDF(pdf)
    histogram("Output Image",output,pdf,cdf)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

# start()