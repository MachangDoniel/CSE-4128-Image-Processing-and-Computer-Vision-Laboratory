import cv2
import numpy as np
import math
from tabulate import tabulate
import random

#Table Creation
distances_matrix = [[1,2,3],[4,5,6],[7,8,9]]
row_headers = [f'Test {i + 1}' for i in range(3)]
col_headers = [f'GT {i + 1}' for i in range(3)]

distances_matrix = np.array(distances_matrix)
# Display the distance matrix as a table
print(tabulate(distances_matrix, headers=col_headers, showindex=row_headers, tablefmt='grid'))


#File Creation
im_title = ['c1','t1','s1','c2','t2','s2']
file_path = 'output2.txt'
with open(file_path, 'w') as file:
    # Write the column names
    file.write('\t'.join(map(str, [' ', 'form_factor', 'roundness','compactness'])) + '\n')
    # Write horizontal line
    file.write('-' * 50 + '\n')
    i=0
    for row in distances_matrix:
        file.write(im_title[i]+'\t\t')
        line = '\t\t'.join(map(str, row ))
        i=i+1
        # Write the line to the file
        file.write(line + '\n')
        # Write horizontal line
        file.write('-' * 50 + '\n')
cv2.waitKey(0)
cv2.destroyAllWindows()