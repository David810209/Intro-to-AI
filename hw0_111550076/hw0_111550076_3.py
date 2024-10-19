import cv2
import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas


image = cv2.imread('data/image.png')

#translate
M = np.float32([[1,0,50],[0,1,75]])
translated_image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0]))    

#rotation
center = (image.shape[1] // 2, image.shape[0] // 2)
angle = 45
scale = 1.0
M = cv2.getRotationMatrix2D(center,angle,scale)
rotated_image = cv2.warpAffine(image,M,(image.shape[1],image.shape[0])) 

#flipping
flipped_image = cv2.flip(image,1)

#scaling
scaled_image = cv2.resize(image,None,fx = 0.5,fy = 0.5,interpolation = cv2.INTER_LINEAR)

#cropping
cropped_image = image[:image.shape[0] // 2, :image.shape[1] // 2]


cv2.imwrite('translated.png', translated_image)
cv2.imwrite('rotation.png', rotated_image)
cv2.imwrite('flipped.png', flipped_image)
cv2.imwrite('scaled.png', scaled_image)
cv2.imwrite('cropped.png', cropped_image)

image_files = ['translated.png','rotation.png','flipped.png','scaled.png','cropped.png']
c = canvas.Canvas('hw0_111550076_3.pdf')
for i in image_files:
    img = Image.open(i)
    
    img_width,img_height = img.size
    c.setPageSize((img_width,img_height))
    c.drawInlineImage(i,0,0,width =img_width,height = img_height)
    c.showPage()
    
c.save()

cv2.destroyAllWindows()