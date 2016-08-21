import cv2
import os 
import numpy as np
from os import walk

SOURCE_FOLDER_PATH = '../../../Images/Cropped images/'
SOURCE_FOLDER_NAME = "Dataset"

for SCALE in range(1):
    WIDTH = 16 * (SCALE + 10)
    HEIGHT = 12 * (SCALE + 10)
    NEW_NAME_OF_DIR = "Dataset_"+str(WIDTH)+"_"+str(HEIGHT)
    f = []
    for (dirpath, dirnames, filenames) in walk(SOURCE_FOLDER_PATH+SOURCE_FOLDER_NAME+'/'):
        new_dirpath = dirpath.replace(SOURCE_FOLDER_NAME, NEW_NAME_OF_DIR)
        #os.mkdir(new_dirpath)
        if len(dirnames) == 0:
            f.append((dirpath, filenames))
    
    for (dirpath, filenames) in f:
        new_dirpath = dirpath.replace(SOURCE_FOLDER_NAME, NEW_NAME_OF_DIR)
        for filename in filenames:
            if(filename != '.DS_Store'):
                print filename
                print new_dirpath
                img = cv2.imread(dirpath + '/' + filename, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (img.shape[1]/10,img.shape[0]/10), interpolation=cv2.INTER_AREA)
                rows,cols,_ = img.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
                gray_buf = np.zeros((rows + 100*2, cols + 100*2, 3));
                cv2.imshow('before', img)
                #img = cv2.warpAffine(img,M,(cols,rows))
                gray_buf = cv2.copyMakeBorder(img, 100,100,100,100, cv2.BORDER_CONSTANT)
                cv2.imshow('after', gray_buf)
                cv2.waitKey(0)
                cv2.imwrite(new_dirpath + '/' + filename, img)