import array
import os
import time

import cv2
import numpy as np
import tensorflow as tf
from cobs import cobs

import Programming.configuration as conf
from Programming.HelperScripts import helper
from Programming.Preprocessing import general


def isHard(path):
    if 'Hard' in path:
        return 1
    else:
        return 0


def saveToFile(byte_data, names, labels, ishard, path):

    general.write_to_file(path + 'labels.byte', array.array('B', labels).tostring())
    general.write_to_file(path + 'data.byte', array.array('B', byte_data).tostring())
    general.write_to_file(path + 'ishard.byte', array.array('B', ishard).tostring())
    general.write_to_file(path + 'names.byte', ''.join(map(str, names)))


def extendListEightTimes(l):
    l.extend(l)
    l.extend(l)
    l.extend(l)
    return l


def enlargeDataset(images, byte_data, names, labels, is_hard):
    extendListEightTimes(labels)
    extendListEightTimes(names)
    extendListEightTimes(is_hard)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        l = len(images)
        for j in range(7):
            print(l)
            train_data2 = []
            start = time.time()
            for i in range(l):
                imageTensor = tf.image.random_contrast(images[i], 0.2, 1.8)
                imageTensor = tf.image.random_flip_left_right(imageTensor)
                imageTensor = tf.image.random_flip_up_down(imageTensor)
                imageTensor = tf.image.random_brightness(imageTensor, max_delta=50 / 255.0)
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
    return byte_data, names, labels, is_hard


for SCALE in range(1):
    f = general.list_images()
    images = []
    byte_data = []
    labels = []
    is_hard = []
    names = []
    for (img, filename, dir_path) in f:
        images.append(img)
        labels.append(helper.get_label_index(dir_path, conf.ALL_DATA_TYPES))
        is_hard.append(isHard(dir_path))
        names.append(filename)
        byte_data.extend(img.flatten())
    byte_data, names, labels, is_hard = enlargeDataset(images, byte_data, names, labels, is_hard)
    saveToFile(byte_data, names, labels, is_hard, general.DATASET_FOLDER)
    with open(general.DATASET_FOLDER + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        result = array.array('B', data)
        num_of_images = len(result) / (WIDTH * HEIGHT * 3)
        result = np.asarray(result, dtype=np.uint8)
        result = result.reshape(num_of_images, HEIGHT, WIDTH, 3)
        print (images[0])
        print ('test')
        print (result[0])
