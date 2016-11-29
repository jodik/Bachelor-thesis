import array
import numpy as np
from cobs import cobs
import Programming.configuration as conf
from Programming.HelperScripts import helper
from data_set import FullData
import cv2


def read_basic(path):
    with open(path, "rb") as read_data:
        data = cobs.decode(read_data.read())
        data = array.array('B', data)
        data = np.asarray(data, dtype=np.uint8)
    return data


def read_images():
    all_images = read_basic(conf.SOURCE_FOLDER_NAME + 'data.byte')
    num_of_images = int(len(all_images) / (conf.IMAGE_WIDTH * conf.IMAGE_HEIGHT * conf.NUM_CHANNELS))
    all_images = all_images.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS)
    all_images = all_images / conf.PIXEL_DEPTH - 0.5
    return all_images


def read_labels():
    labels = read_basic(conf.SOURCE_FOLDER_NAME + 'labels.byte')
    return labels


def read_is_hard():
    is_hard_all = read_basic(conf.SOURCE_FOLDER_NAME + 'ishard.byte')
    return is_hard_all


def read_edge_descriptors():
    edge_descriptors_all = read_basic(conf.SOURCE_FOLDER_NAME2 + 'edges.byte')
    #num_of_images = int(len(edge_descriptors_all) / (conf.IMAGE_WIDTH * conf.IMAGE_HEIGHT))
    num_of_images = int(len(edge_descriptors_all) / (160 * 160))
    #edge_descriptors_all = edge_descriptors_all.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH)
    edge_descriptors_all = edge_descriptors_all.reshape(num_of_images, 160, 160, 1)
    edge_descriptors_all = np.concatenate((edge_descriptors_all, edge_descriptors_all))
    edge_descriptors_all = np.concatenate((edge_descriptors_all, edge_descriptors_all))
    edge_descriptors_all = np.concatenate((edge_descriptors_all, edge_descriptors_all))
    print(edge_descriptors_all.shape)
    #cv2.imshow("as3", edge_descriptors_all[0])
    #cv2.waitKey(0)
    edge_descriptors_all = edge_descriptors_all / conf.PIXEL_DEPTH - 0.5
    #cv2.imshow("as3", edge_descriptors_all[0])
    #cv2.waitKey(0)
    return edge_descriptors_all


def read_names(num_of_images):
    names = []
    example = "IMG_0519.JPG"
    with open(conf.SOURCE_FOLDER_NAME + 'names.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        l = len(example)
        for i in range(num_of_images):
            names.append(data[i * l:(i + 1) * l])
    return names


def filter_unselected_categories(all_images, num_of_all_images, all_labels, all_names, all_is_hard, all_edge_descriptors):
    labels = np.zeros(num_of_all_images, dtype=np.uint8)
    images = np.zeros((num_of_all_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS))
    #edge_descriptors = np.zeros((num_of_all_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH))
    edge_descriptors = np.zeros((num_of_all_images, 160, 160, 1))
    is_hard = np.zeros(num_of_all_images, dtype=np.uint8)
    num_selected = 0
    names_chosen = []
    for i in range(num_of_all_images):
        label_word = conf.ALL_DATA_TYPES[all_labels[i]]
        if label_word in conf.DATA_TYPES_USED and (all_is_hard[i] == 0 or conf.HARD_DIFFICULTY):
            category = helper.get_label_index(label_word, conf.DATA_TYPES_USED)
            labels[num_selected] = category
            images[num_selected] = all_images[i]
            is_hard[num_selected] = all_is_hard[i]
            edge_descriptors[num_selected] = all_edge_descriptors[i]
            names_chosen.append(all_names[i])
            num_selected += 1
    images = images[0:num_selected]
    is_hard = is_hard[0:num_selected]
    labels = labels[0:num_selected]
    edge_descriptors = edge_descriptors[0:num_selected]
    return images, labels, np.array(names_chosen), is_hard, edge_descriptors


def extract_data():
    all_images = read_images()
    num_of_all_images = all_images.shape[0]

    all_labels = read_labels()

    print (num_of_all_images, all_labels.shape[0])
    all_names = read_names(num_of_all_images)
    all_is_hard = read_is_hard()
    all_edge_descriptors = read_edge_descriptors()

    images, labels, names, is_hard, edge_descriptors = filter_unselected_categories(all_images, num_of_all_images, all_labels, all_names, all_is_hard, all_edge_descriptors)
    return FullData(images, labels, names, is_hard, edge_descriptors)


def read_data():
    full_data_set = extract_data()
    if not conf.EXTENDED_DATASET:
        full_data_set = full_data_set.create_data_set(0, full_data_set.num_examples / 8)
    return full_data_set
