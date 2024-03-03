kernel =(1/273) * np.array([[1, 4, 7, 4,1],
                            [4,16,26,16,4],
                            [7,26,41,26,7],
                            [4,16,26,16,4],
                            [1, 4, 7, 4,1]])

print( img.shape[0] )
print( img.shape[1] )
out=img.copy()

n = int( kernel.shape[0]/2 )

for x in range( n, img.shape[0]-n ):
    for y in range( n, img.shape[1]-n ):

        res = 0
        for j in range( -n, n+1 ):
            for i in range( -n, n+1 ):
                f = kernel.item(i,j)
                ii = img.item(x-i,y-j)
                
                res += (f * ii)
        
        out[x,y] = res
        #out.itemset((i,j),a+122) #255-a)

print(out)
cv2.normalize(out,out, 0, 255, cv2.NORM_MINMAX)
out = np.round(out).astype(np.uint8)

print(out)
cv2.imshow('normalised output image',out)

cv2.waitKey(0)
cv2.destroyAllWindows()