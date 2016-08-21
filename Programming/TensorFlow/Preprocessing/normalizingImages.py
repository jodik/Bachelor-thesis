import cv2
import os 
import numpy as np
from os import walk
import random

SOURCE_FOLDER_PATH = '../../../Images/Cropped images/'
SOURCE_FOLDER_NAME = "Dataset"

for SCALE in range(1):
    SIZE = 16 * (SCALE + 2)
    NEW_NAME_OF_DIR = "Dataset_"+str(SIZE)+"_"+str(SIZE)
    f = []
    for (dirpath, dirnames, filenames) in walk(SOURCE_FOLDER_PATH+SOURCE_FOLDER_NAME+'/'):
        new_dirpath = dirpath.replace(SOURCE_FOLDER_NAME, NEW_NAME_OF_DIR)
        os.mkdir(new_dirpath)
        if len(dirnames) == 0:
            f.append((dirpath, filenames))
    
    for (dirpath, filenames) in f:
        new_dirpath = dirpath.replace(SOURCE_FOLDER_NAME, NEW_NAME_OF_DIR)
        for filename in filenames:
            if(filename != '.DS_Store'):
                print filename
                print new_dirpath
                img = cv2.imread(dirpath + '/' + filename, cv2.IMREAD_COLOR)
                if(img.shape[0] * img.shape[1] == 3000*4000 and 'Nothing' not in new_dirpath):
                    print(img.shape)
                    raise ValueError('Something went wrong, uncropped image')
                if 'Nothing' in new_dirpath:
                    temp = 500 + random.randint(0, 500)
                    img = img[0:4000, (temp):(3000-temp)]
    
                ratio = float(SIZE)/max(img.shape)
                img = cv2.resize(img, (int(img.shape[1] * ratio),int(ratio * img.shape[0])), interpolation=cv2.INTER_AREA)
                needToBeRotated = SIZE == img.shape[1]
                borderSize = (SIZE - min(img.shape[0], img.shape[1]))/2
                if needToBeRotated:
                    img = cv2.copyMakeBorder(img, borderSize, borderSize, 0, 0, cv2.BORDER_CONSTANT)
                    M = cv2.getRotationMatrix2D((SIZE/2,SIZE/2),270,1)
                    img = cv2.warpAffine(img,M,(SIZE,SIZE))
                else:
                    
                    img = cv2.copyMakeBorder(img, 0, 0, borderSize, borderSize, cv2.BORDER_CONSTANT)
                img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                    
                cv2.imwrite(new_dirpath + '/' + filename, img)