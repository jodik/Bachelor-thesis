import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def createImageWithContrast(image, contrast, brightness):
    means = [0,0,0]
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                            means[k] += image[i][j][k]
    for k in range(len(image[0][0])):
        means[k] = float(means[k]/float(len(image[0])*len(image)))
    for i in range(len(image)):
        for j in range(len(image[i])):
            for k in range(len(image[i][j])):
                            image[i][j][k] = int(max(min(contrast*(image[i][j][k]-means[k]) + means[k], 255),0))
                            print(means[k])
    
    return image

SOURCE_FOLDER_NAME = "../../../Images/Original images/Dataset_400_300/";


img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)

with tf.Session() as sess:
      img1 = sess.run(tf.image.adjust_brightness(img1, 0.2))
cv2.imshow('saturation', img1)      

img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)
with tf.Session() as sess:
      img1 = sess.run(tf.image.adjust_contrast(img1, 1.3))
cv2.imshow('contrast', img1)            

img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)

cv2.imshow('asd', createImageWithContrast(img1,1, 0))
cv2.waitKey(0)
img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)

cv2.imshow('asd', createImageWithContrast(img1,1.1, 0))
cv2.waitKey(0)
img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)

cv2.imshow('asd', createImageWithContrast(img1,1.5, 40))
cv2.waitKey(0)
img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)

cv2.imshow('asd', createImageWithContrast(img1,0.6, 20))
cv2.waitKey(0)