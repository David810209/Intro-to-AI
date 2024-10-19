import cv2
import numpy as np

image = cv2.imread('data/image.png')

bbox_file_path = 'data/bounding_box.txt'

with open(bbox_file_path,'r') as file:
    lines = file.readlines()
    
for line in lines:
    x1,y1,x2,y2 = map(int,line.split())
    cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('hw0_111550076_1.png',image)

cv2.destroyAllWindows()