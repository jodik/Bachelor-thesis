import numpy as np
from cobs import cobs
import array
from sets import Set
from Programming.HelperScripts import helper
import Programming.TensorFlow.configuration as conf


def extract_images():
    correct_vals = np.zeros((0), dtype=np.uint8)
    with open(conf.SOURCE_FOLDER_NAME + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        all_images = array.array('B', data)
        num_of_images = int(len(all_images) / (conf.IMAGE_WIDTH * conf.IMAGE_HEIGHT * conf.NUM_CHANNELS))
        all_images = np.asarray(all_images, dtype=np.uint8)
        all_images = all_images.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS)
        print(all_images[2][1])
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
    print('test3')
    print(num_of_images)
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
    print('test4')
    # for i in range(count):
    #   images[i] = images[i].reshape(width, height, channels)
    return images, correct_vals, np.array(names_chosen)


def filterAndCreateTrainSet(validation_names, test_names, images, labels, names):
    size = 0
    print('LEN: '+str(len(Set(names))))
    for i in range(images.shape[0]):
        if (names[i] not in validation_names) and (names[i] not in test_names):
            images[size] = images[i]
            labels[size] = labels[i]
            names[size] = names[i]
            size += 1
    perm = np.arange(size)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    names = names[perm]
    print('LEN: '+str(len(Set(names))))
    return images, labels, names


def getPermutation(permutation_index, labels, validation_size, test_size):
    num_of_images_total = len(labels)
    if permutation_index >= 10 or permutation_index < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    perm = np.arange(num_of_images_total)
    for i in range(permutation_index + 1):
        np.random.shuffle(perm)

    percentage = validation_size / float(num_of_images_total)
    counts = np.array(percentage * np.bincount(np.array(labels, dtype=int)), dtype=int)
    to_add = validation_size - sum(counts)
    for i in np.arange(counts.shape[0])[:to_add]:
        counts[i] += 1
    a = np.zeros(0, dtype=int)
    b = np.zeros(0, dtype=int)
    c = np.zeros(0, dtype=int)
    d = np.zeros(0, dtype=int)
    counts_tmp = np.copy(counts)
    for t in perm:
        if counts[labels[t]] > 0:
            counts[labels[t]] -= 1
            a = np.append(a, t)
        else:
            b = np.append(b, t)
    counts = counts_tmp
    print(counts)
    for t in b:
        if counts[labels[t]] > 0:
            counts[labels[t]] -= 1
            c = np.append(c, t)
        else:
            d = np.append(d, t)

    print (np.append(a, np.append(c, d)))
    return np.append(a, np.append(c, d))


def read_datasets(permutation_index):
    np.random.seed(conf.SEED)

    images, labels, names = extract_images()
    TEST_SIZE = int(images.shape[0] / 8 * (conf.TEST_PERCENTAGE / 100.0))
    VALIDATION_SIZE = int(images.shape[0] / 8 * (conf.VALIDATION_PERCENTAGE / 100.0))

    original_set_size = int(images.shape[0] / 8)
    perm2 = getPermutation(permutation_index, labels[:original_set_size], VALIDATION_SIZE, TEST_SIZE)
    print(images.shape[0])
    print(labels.shape)
    print(names.shape)
    images_original_set = images[perm2]
    labels_original_set = labels[perm2]
    names_original_set = names[perm2]

    test_images = images_original_set[:TEST_SIZE, ...]
    test_labels = labels_original_set[:TEST_SIZE]
    test_names = names_original_set[:TEST_SIZE]
    images_original_set = images_original_set[TEST_SIZE:, ...]
    labels_original_set = labels_original_set[TEST_SIZE:]
    names_original_set = names_original_set[TEST_SIZE:]

    validation_images = images_original_set[:VALIDATION_SIZE, ...]
    validation_labels = labels_original_set[:VALIDATION_SIZE]
    validation_names = names_original_set[:VALIDATION_SIZE]
    images_original_set = images_original_set[VALIDATION_SIZE:, ...]
    labels_original_set = labels_original_set[VALIDATION_SIZE:]
    names_original_set = names_original_set[VALIDATION_SIZE:]

    if conf.EXTENDED_DATASET:
        train_images, train_labels, train_names = filterAndCreateTrainSet(validation_names, test_names, images, labels,
                                                                          names)
    else:
        train_images, train_labels, train_names = images_original_set, labels_original_set, names_original_set

    return train_images, train_labels, validation_images, validation_labels, test_images, test_labels