import cv2
import os 
from cobs import cobs
import array
from os import walk
import numpy as np


SOURCE_FOLDER_NAME = "Dataset"
NAME_OF_DATASET_BYTE_FILE = "data.byte"

def getLabel(path):
     DATA_TYPES = ['Blue','Green', 'White', 'Box', 'Can', 'Chemical', 'Colorful', 'Multiple Objects', 'Nothing']
     DATA_TYPES = sorted(DATA_TYPES)
     for i in range(len(DATA_TYPES)):
         if DATA_TYPES[i] in path:
             return i
     raise ValueError('Bad path.')

def isHard(path):
    if 'Hard' in path:
        return 1
    else:
        return 0
    
for SCALE in range(10):
    WIDTH = 16 * (SCALE + 1)
    HEIGHT = 12 * (SCALE + 1)
    NEW_NAME_OF_DIR = "Dataset_"+str(WIDTH)+"_"+str(HEIGHT)
    f = []
    for (dirpath, dirnames, filenames) in walk('../../../Images/'+SOURCE_FOLDER_NAME+'/'):
        new_dirpath = dirpath.replace(SOURCE_FOLDER_NAME, NEW_NAME_OF_DIR)
        print(dirpath)
        if len(dirnames) == 0:
            f.append((new_dirpath, filenames))
    
    images = []
    byte_data = []
    labels = []
    is_hard = []
    names = ''
    for (dirpath, filenames) in f:
        for filename in filenames:
            if(filename != '.DS_Store' and filename!=NAME_OF_DATASET_BYTE_FILE):
                img = cv2.imread(dirpath + '/' + filename, cv2.IMREAD_COLOR)
                images.append(img)
                labels.append(getLabel(dirpath))
                is_hard.append(isHard(dirpath))
                names+=filename
                byte_data.extend(img.flatten())
    os.mkdir('../../../Datasets/'+NEW_NAME_OF_DIR)
    with open ('../../../Datasets/'+NEW_NAME_OF_DIR+'/' + NAME_OF_DATASET_BYTE_FILE, "wb") as compdata:
        bytes = bytearray(cobs.encode(array.array('B',byte_data).tostring()))
        compdata.write(bytes)
        compdata.close()
    with open ('../../../Datasets/'+NEW_NAME_OF_DIR+'/' + 'labels.byte', "wb") as compdata:
        bytes = bytearray(cobs.encode(array.array('B',labels).tostring()))
        compdata.write(bytes)
        compdata.close()
    with open ('../../../Datasets/'+NEW_NAME_OF_DIR+'/' + 'ishard.byte', "wb") as compdata:
        bytes = bytearray(cobs.encode(array.array('B',is_hard).tostring()))
        compdata.write(bytes)
        compdata.close()
    with open ('../../../Datasets/'+NEW_NAME_OF_DIR+'/' + 'names.byte', "wb") as compdata:
        bytes = bytearray(cobs.encode(names))
        compdata.write(bytes)
        compdata.close()
    with open ('../../../Datasets/'+NEW_NAME_OF_DIR+'/' + NAME_OF_DATASET_BYTE_FILE, "rb") as readdata:
        data = cobs.decode(readdata.read())
        result = array.array('B',data)
        num_of_images =  len(result)/(WIDTH*HEIGHT*3)
        result = np.asarray(result, dtype=np.uint8)
        result = result.reshape(num_of_images, HEIGHT, WIDTH, 3)
        print (images[0])
        print ('test')
        print (result[0])
            