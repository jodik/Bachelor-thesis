import numpy as np
from cobs import cobs
import array
import data_process
from Programming.HelperScripts import helper
import Programming.TensorFlow.configuration as conf
from data import Data


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
        ishard = array.array('B', data)
        ishard = np.asarray(ishard, dtype=np.uint8)
    names = []
    with open(conf.SOURCE_FOLDER_NAME + 'names.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        l = len("IMG_0519.JPG")
        for i in range(num_of_images):
            names.append(data[i * l:(i + 1) * l])
    images = np.zeros((num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS))
    size = 0
    names_chosen = []
    for i in range(num_of_images):
        label_word = conf.ALL_DATA_TYPES[labels[i]]
        if label_word in conf.DATA_TYPES_USED and (ishard[i] == 0 or conf.HARD_DIFFICULTY):
            category = helper.getLabelIndex(label_word, conf.DATA_TYPES_USED)
            correct_vals = np.append(correct_vals, [category])
            images[size] = all_images[i]
            names_chosen.append(names[i])
            size += 1
    images = images[0:size]
    return Data(images, correct_vals, np.array(names_chosen), ishard)


def read_datasets(permutation_index):
    np.random.seed(conf.SEED)

    data = extract_data()
    if not conf.EXTENDED_DATASET:
        data = data.createData(0, data.size() / 8)
    return data_process.process(data, permutation_index)
