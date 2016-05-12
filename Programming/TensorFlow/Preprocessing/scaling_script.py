import cv2
import os 
from os import walk


SOURCE_FOLDER_NAME = "Dataset"

for SCALE in range(10):
    WIDTH = 16 * (SCALE + 1)
    HEIGHT = 12 * (SCALE + 1)
    NEW_NAME_OF_DIR = "Dataset_"+str(WIDTH)+"_"+str(HEIGHT)
    f = []
    for (dirpath, dirnames, filenames) in walk('../../../Images/'+SOURCE_FOLDER_NAME+'/'):
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
                img = cv2.resize(img, (WIDTH,HEIGHT), interpolation=cv2.INTER_AREA)
                cv2.imwrite(new_dirpath + '/' + filename, img)