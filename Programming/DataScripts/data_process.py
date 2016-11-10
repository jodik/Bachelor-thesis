import numpy as np
import copy
import Programming.TensorFlow.configuration as conf
from dataset import DataSet


def filterAndCreateTrainSet(validation_names, test_names, full_data):
    chosen = []
    for i in range(full_data.size()):
        if (full_data.names[i] not in validation_names) and (full_data.names[i] not in test_names):
            chosen.append(i)
    train_data = copy.deepcopy(full_data).applyPermutation(chosen)
    perm = np.arange(train_data.size())
    np.random.shuffle(perm)
    train_data.applyPermutation(perm)
    return train_data


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


def getOriginalDatasetSize(data):
    divide_by = 8 if conf.EXTENDED_DATASET else 1
    original_dataset_size = data.size() / divide_by
    return original_dataset_size


def process(full_data, permutation_index):
    original_set_size = getOriginalDatasetSize(full_data)
    TEST_SIZE = int(original_set_size * (conf.TEST_PERCENTAGE / 100.0))
    VALIDATION_SIZE = int(original_set_size * (conf.VALIDATION_PERCENTAGE / 100.0))

    perm = getPermutation(permutation_index, full_data.labels[:original_set_size], VALIDATION_SIZE, TEST_SIZE)

    original_data = copy.deepcopy(full_data).applyPermutation(perm)
    test_data = original_data.createData(0, TEST_SIZE)
    validation_data = original_data.createData(TEST_SIZE, TEST_SIZE + VALIDATION_SIZE)
    train_data = filterAndCreateTrainSet(validation_data.names, test_data.names, full_data)

    data_set = DataSet(train_data, validation_data, test_data)

    return data_set
