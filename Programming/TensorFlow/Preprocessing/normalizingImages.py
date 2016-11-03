import cv2
import os 
import numpy as np
from os import walk
import random

SOURCE_FOLDER_PATH = '../../../Images/Cropped images/'
SOURCE_FOLDER_NAME = "Dataset"
BLACK = False
OUTPUT_FOLDER = 'Bordered with black color/' if BLACK else 'Extended with itself/'
OUTPUT_FOLDER = SOURCE_FOLDER_PATH + OUTPUT_FOLDER


for SCALE in range(1):
    SIZE = 16 * (SCALE + 2)
    NEW_NAME_OF_DIR = "Dataset_"+str(SIZE)+"_"+str(SIZE)
    f = []
    for (dirpath, dirnames, filenames) in walk(SOURCE_FOLDER_PATH+SOURCE_FOLDER_NAME+'/'):
        new_dirpath = OUTPUT_FOLDER + NEW_NAME_OF_DIR + dirpath.split(SOURCE_FOLDER_NAME)[1]
        if not os.path.isdir(new_dirpath):
            os.mkdir(new_dirpath)
        if len(dirnames) == 0:
            f.append((dirpath, filenames, new_dirpath))
    
    for (dirpath, filenames, new_dirpath) in f:
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
                img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)

                needToBeRotated = SIZE == img.shape[1]
                borderSize = (SIZE - min(img.shape[0], img.shape[1]))/2
                if needToBeRotated:
                    M = cv2.getRotationMatrix2D((SIZE/2,SIZE/2),270,1)
                    M = np.float32([[0, 1, 0], [1, 0, 0]])
                    img = cv2.warpAffine(img,M,(img.shape[0],img.shape[1]))

                if BLACK:
                    tt = np.zeros((SIZE, (SIZE - img.shape[1])/2, 3), dtype=np.uint8)
                    img = np.append(tt, img, axis=1)
                    tt = np.zeros((SIZE, SIZE - img.shape[1], 3), dtype=np.uint8)
                    img = np.append(img, tt, axis = 1)
                else:
                    while img.shape[1] < SIZE:
                        img = np.append(img,img,axis = 1)
                    img = img[:,:34]
                #cv2.imshow("as", img)
                #cv2.waitKey(0)
                cv2.imwrite(new_dirpath + '/' + filename, img)