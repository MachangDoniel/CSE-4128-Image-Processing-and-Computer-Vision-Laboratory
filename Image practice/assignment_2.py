import cv2
import numpy as np


from doniel_library import *
from edge_detection import *

def perform_edge_detection(image):
    kernel_x, kernel_y = get_kernel()

    conv_x = convolve(image=image, kernel=kernel_x)
    conv_y = convolve(image=image, kernel=kernel_y)
    
    kernel= generateGaussianKernel(sigmaX=1,sigmaY=1,MUL=5)
    conv_x = convolve(image=conv_x,kernel=kernel)
    conv_y = convolve(image=conv_y,kernel=kernel)
    
    merged_image = merge(conv_x, conv_y)
    
    # theta = np.arctan2(conv_x, conv_y) # it's changing the value of conv_x & conv_y also
    theta = np.arctan2( conv_y.copy(), conv_x.copy() )

    normalized_merged_image = normalize(merged_image)

    # t = find_threeshold(image=normalized_merged_image)
    # print(f"Threeshold {t}")
    # final_out = make_binary(t=t,image=normalized_merged_image, low = 0, high = 100)
    
    cv2.imshow("X derivative", normalize(conv_x))
    cv2.imshow("Y derivative", normalize(conv_y))
    
    cv2.imshow("Merged", normalized_merged_image)
    # cv2.imshow("Threesholded", final_out)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    #return final_out, theta
    return merged_image, theta

def perform_non_maximum_suppression(image, theta):
    image = image.copy()
    
    image = image / image.max() * 255   # increaisng the range og the image to 255
    
    M, N = image.shape
    out = np.zeros((M,N), dtype=np.uint8)
    
    angle = theta * 180. / np.pi        # max -> 180, min -> -180 # converting the theta to radian

    c1 = c2 = c3 = c4 = 0
    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 0
            r = 0
            
            ang = angle[i,j]
            
            if ( -22.5 <= ang < 22.5) or ( 157.5 <= ang <= 180) or (-180 <= ang <= -157.5):
                r = image[i, j-1]
                q = image[i, j+1]
                c1 += 1

            elif (  -67.5 <= ang <= -22.5 ) or ( 112.5 <= ang <= 157.5):
                r = image[i-1, j+1]
                q = image[i+1, j-1]
                c2 += 1

            elif ( 67.5 <= ang <= 112.5) or ( -112.5 <= ang <= -67.5 ):
                r = image[i-1, j]
                q = image[i+1, j]
                c3 += 1

            elif ( 22.5 <= ang < 67.5 ) or ( -167.5 <= ang <= -112.5 ):
                r = image[i+1, j+1]
                q = image[i-1, j-1]
                c4 += 1

            if (image[i,j] >= q) and (image[i,j] >= r):
                out[i,j] = image[i,j]
            else:
                out[i,j] = 0
    print(f"{c1},{c2},{c3},{c4}")
    return out

def perform_canny(image_path, sigma):
    
    # Gray Scale Coversion
    image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
    main_image = image.copy()
    
    # Perform Gaussian Blurr
    kernel= generateGaussianKernel(sigmaX=sigma,sigmaY=sigma,MUL=7)
    image = convolve(image=image, kernel=kernel)
    
    cv2.imshow("Blurred Input Image", normalize(image))
    cv2.waitKey(0)
    
    # Gradient Calculation
    merged_image, theta = perform_edge_detection(image) 
    
    # Non Maximum Suppression
    suppressed = perform_non_maximum_suppression(image=merged_image,theta=theta)
    
    # Threesholding and hysteresis
    threes = find_threeshold(image=suppressed)
    print(f"Threeshold: ${threes}")
    
    threes_image, weak, strong = perform_threshold(image=suppressed,threes=threes)
    final_output = perform_hysteresis( image=threes_image, weak=weak, strong=strong )
        
    cv2.imshow("Merged image", normalize(merged_image) )
    cv2.imshow("Non maximum suppression", normalize(suppressed) )
    cv2.imshow("Threesholded", normalize(threes_image) )
    cv2.imshow("Final", normalize(final_output))

    
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    

def choose(list, message = "--> ", error_message="Invalid index. Restarting..\n"):
    for i in range( len(list) ):
        print(f"{i}. {list[i]}")
    print()
    
    print(message, end='')
    index = int(input())
    
    if( index >= len(list) ):
        if error_message != None:
            print(error_message)
        return -1
    
    val = list[index]
    return index


def start():
    print("<-- Welcome! -->\n\n")
    options = ['start', 'exit']
    images = ['line3.jpg','line5.jpg','building.jpg', 'edge.png', 'lena.jpg']
    
    while( True ):
        clear_terminal()
        print("Choose your option!")
        index = choose(options)
        if index != 0:
            break

        clear_terminal()
        print("Choose your image!")
        index = choose(images, message="--> ")
        if index == -1:
            continue
        image_name = images[index]
        image_path = '.\images\\'+image_name
        
        image = cv2.imread( image_path, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("Input image",image)
        cv2.waitKey(0)

        clear_terminal()
        print("Enter the value of sigma: ", end=' ')
        sigma = float( input() )
        
        perform_canny(image_path=image_path, sigma=sigma)
        print("Completed")

# start()