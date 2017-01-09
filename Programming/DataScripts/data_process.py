import copy

import numpy as np

import Programming.configuration as conf
from data_sets import DataSets


def filterAndCreateTrainSet(validation_names, test_names, full_data):
    chosen = []
    for i in range(full_data.num_examples):
        if (full_data.names[i] not in validation_names) and (full_data.names[i] not in test_names):
            chosen.append(i)
    train_data = copy.deepcopy(full_data).apply_permutation(chosen)
    perm = np.arange(train_data.num_examples)
    np.random.shuffle(perm)
    train_data.apply_permutation(perm)
    return train_data


def choose_subset(to_skip, to_choose, total_count, perm, labels):
    for i in range(len(to_skip)):
        s = to_skip[i] + to_choose[i]
        if s > total_count[i]:
            d = total_count[i] - s
            to_skip[i] -= d
    subset = np.zeros(0, dtype=int)
    rest = np.zeros(0, dtype=int)
    for t in perm:
        if to_choose[labels[t]] > 0 and to_skip[labels[t]] == 0:
            label = labels[t]
            to_choose[label] -= 1
            total_count[label] -= 1
            subset = np.append(subset, t)
        else:
            to_skip[labels[t]] -= 1
            rest = np.append(rest, t)
    return subset, rest


def getPermutation(permutation_index, labels, test_size):
    num_of_images_total = len(labels)
    if permutation_index >= 10 or permutation_index < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    perm = np.arange(num_of_images_total)
    np.random.shuffle(perm)

    total_count = np.bincount(np.array(labels, dtype=int))
    percentage = test_size / float(num_of_images_total)
    counts = np.array(percentage * total_count, dtype=int)
    to_add = test_size - sum(counts)
    for i in np.arange(counts.shape[0])[:to_add]:
        counts[i] += 1

    print(total_count)
    test_set, perm = choose_subset(np.zeros(len(counts)), np.copy(counts), total_count, perm, labels)

    validation_percentage = (float(1)/conf.CROSS_VALIDATION_ITERATIONS)
    counts = np.array(validation_percentage * total_count, dtype=int)
    validation_set, perm = choose_subset(np.copy(counts) * permutation_index, np.copy(counts), total_count, perm, labels)

    return perm, validation_set, test_set


def process(full_data, permutation_index):
    original_set_size = full_data.get_original_data_set_size()
    TEST_SIZE = int(original_set_size * (conf.TEST_PERCENTAGE / 100.0))

    train_perm, val_perm, test_perm = getPermutation(permutation_index, full_data.labels[:original_set_size], TEST_SIZE)

    test_data = copy.deepcopy(full_data).apply_permutation(test_perm)
    validation_data = copy.deepcopy(full_data).apply_permutation(val_perm)
    train_data = filterAndCreateTrainSet(validation_data.names, test_data.names, full_data)

    data_sets = DataSets(train_data, validation_data, test_data)

    return data_sets
