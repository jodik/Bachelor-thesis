import numpy as np
import cv2

SOURCE_FOLDER_NAME = "../../Images/Dataset_400_300/";
img1 = cv2.imread(SOURCE_FOLDER_NAME + 'Blue/Normal/IMG_0817.JPG', cv2.IMREAD_COLOR)
img1 = cv2.resize(img1, (200,300))
cv2.imshow('asd', img1)
cv2.waitKey(0)