import cv2
import os 
from cobs import cobs
import array
from os import walk
import numpy as np
import tensorflow as tf
import time

TYPE_DIRECTORY = 'Cropped' # or Original
BLACK_BORDER = False
SUB_FOLDER = 'Bordered with black color/' if BLACK_BORDER else 'Extended with itself/'
IMAGES_FOLDER_NAME = '../../../Images/'+ TYPE_DIRECTORY + ' images/' + SUB_FOLDER
DATASETS_FOLDER = '../../../Datasets/' + TYPE_DIRECTORY + ' datasets/' + SUB_FOLDER

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

def writeToFile(file, bytes):
    with open (file, "wb") as compdata:
        bytes = bytearray(cobs.encode(bytes))
        compdata.write(bytes)
        compdata.close()

def saveToFile(byte_data, names, labels, ishard, path):
    if(os.path.isdir(path) == False):
        os.makedirs(path)
    writeToFile(path + 'labels.byte', array.array('B',labels).tostring())
    writeToFile(path + 'data.byte', array.array('B',byte_data).tostring())
    writeToFile(path + 'ishard.byte', array.array('B',ishard).tostring())
    writeToFile(path + 'names.byte', ''.join(map(str, names)))

def extendListEightTimes(l):
    l.extend(l)
    l.extend(l)
    l.extend(l)
    return l

def enlargeDataset(images, byte_data, names, labels, is_hard):
    labels = extendListEightTimes(labels)
    names = extendListEightTimes(names)
    is_hard = extendListEightTimes(is_hard)
    with tf.Session() as sess:
       tf.initialize_all_variables().run()
       l = len(images)
       for j in range(7): 
        print(l)
        train_data2 = []
        start = time.time()
        for i in range(l):
            imageTensor = tf.image.random_contrast(images[i],0.2,1.8)
            imageTensor = tf.image.random_flip_left_right(imageTensor)
            imageTensor = tf.image.random_flip_up_down(imageTensor)
            imageTensor = tf.image.random_brightness(imageTensor, max_delta = 50/255.0)
            imageTensor = tf.image.random_saturation(imageTensor, 0.2, 1.8)
            train_data2.append(imageTensor)
        print(time.time() - start)
        start = time.time()
        train_data2 = sess.run(train_data2)
        print(type(train_data2))
        print('time2:', time.time() - start)
        print train_data2[0][16]
        for i in range(l):
             byte_data.extend(train_data2[i].flatten())
    return (byte_data, names, labels, is_hard)     
        
        
for SCALE in range(1):
    WIDTH = 16 * (SCALE + 2)
    HEIGHT = 16 * (SCALE + 2)
    MAIN_FOLDER_NAME = "Dataset_"+str(WIDTH)+"_"+str(HEIGHT) + '/'
    f = []
    for (dirpath, dirnames, filenames) in walk(IMAGES_FOLDER_NAME + MAIN_FOLDER_NAME):
        print(dirpath)
        if len(dirnames) == 0:
            f.append((dirpath, filenames))
    images = []
    byte_data = []
    labels = []
    is_hard = []
    names = []
    for (dirpath, filenames) in f:
        for filename in filenames:
            if(filename != '.DS_Store' and filename!='data.byte'):
                img = cv2.imread(dirpath + '/' + filename, cv2.IMREAD_COLOR)
                images.append(img)
                labels.append(getLabel(dirpath))
                is_hard.append(isHard(dirpath))
                names.append(filename)
                byte_data.extend(img.flatten())
    byte_data, names, labels, is_hard = enlargeDataset(images, byte_data, names, labels, is_hard)
    path = DATASETS_FOLDER + MAIN_FOLDER_NAME
    saveToFile(byte_data, names, labels, is_hard, path)
    with open (path + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        result = array.array('B',data)
        num_of_images =  len(result)/(WIDTH*HEIGHT*3)
        result = np.asarray(result, dtype=np.uint8)
        result = result.reshape(num_of_images, HEIGHT, WIDTH, 3)
        print (images[0])
        print ('test')
        print (result[0])
                