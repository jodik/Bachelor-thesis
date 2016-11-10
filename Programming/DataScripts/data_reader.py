import array

import numpy as np
from cobs import cobs

import Programming.configuration as conf
import data_process
from Programming.HelperScripts import helper
from data import FullData


def extract_data():
    correct_vals = np.zeros((0), dtype=np.uint8)
    with open(conf.SOURCE_FOLDER_NAME + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        all_images = array.array('B', data)
        num_of_images = int(len(all_images) / (conf.IMAGE_WIDTH * conf.IMAGE_HEIGHT * conf.NUM_CHANNELS))
        all_images = np.asarray(all_images, dtype=np.uint8)
        all_images = all_images.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS)
        all_images = all_images / conf.PIXEL_DEPTH - 0.5
    with open(conf.SOURCE_FOLDER_NAME + 'labels.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        labels = array.array('B', data)
        labels = np.asarray(labels, dtype=np.uint8)

    with open(conf.SOURCE_FOLDER_NAME + 'ishard.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        is_hard_all = array.array('B', data)
        is_hard_all = np.asarray(is_hard_all, dtype=np.uint8)
    names = []
    with open(conf.SOURCE_FOLDER_NAME + 'names.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        l = len("IMG_0519.JPG")
        for i in range(num_of_images):
            names.append(data[i * l:(i + 1) * l])
    images = np.zeros((num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS))
    is_hard = np.zeros(num_of_images, dtype=np.uint8)
    size = 0
    names_chosen = []
    for i in range(num_of_images):
        label_word = conf.ALL_DATA_TYPES[labels[i]]
        if label_word in conf.DATA_TYPES_USED and (is_hard_all[i] == 0 or conf.HARD_DIFFICULTY):
            category = helper.getLabelIndex(label_word, conf.DATA_TYPES_USED)
            correct_vals = np.append(correct_vals, [category])
            images[size] = all_images[i]
            is_hard[size] = is_hard_all[i]
            names_chosen.append(names[i])
            size += 1
    images = images[0:size]
    is_hard = is_hard[0:size]
    return FullData(images, correct_vals, np.array(names_chosen), is_hard)


def read_datasets(permutation_index):
    np.random.seed(conf.SEED)

    data = extract_data()
    if not conf.EXTENDED_DATASET:
        data = data.create_data(0, data.num_examples / 8)
    return data_process.process(data, permutation_index)
