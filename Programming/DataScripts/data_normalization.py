import numpy as np


def normalize_data(data):
    num_of_pixels = 0
    channels = data[0][0, 0, 0].shape[0]
    sum_rgb = np.sum(np.sum(s, axis=(0, 1, 2)) for s in data)

    for image_set in data:
        num_of_pixels += image_set.size / channels

    sum_rgb /= num_of_pixels
    for image_set in data:
        image_set -= sum_rgb

    sum_rgb_pow = np.zeros(channels, dtype=np.float64)
    for image_set in data:
        tmp = np.apply_along_axis(lambda x: np.power(x, 2), 2, image_set)
        sum_rgb_pow += np.sum(tmp, axis=(0, 1, 2))

    c = num_of_pixels / sum_rgb_pow
    c = np.sqrt(c)
    for image_set in data:
        image_set *= c


def equal_counts_perms(labels):
    counts = np.bincount(labels)
    counts = np.full(len(counts), max(counts), dtype=np.uint32)
    res = np.zeros(0, dtype=np.uint8)
    while max(counts) > 0:
        for i in range(len(labels)):
            label = labels[i]
            if counts[label] > 0:
                res = np.append(res, i)
                counts[label] -= 1
    np.random.shuffle(res)
    return res


def make_equal_count_per_category(data_sets):
    train_perm, validation_perm, test_perm = map(lambda x: equal_counts_perms(x),
                                                 data_sets.get_label_sets())
    data_sets.train.apply_permutation(train_perm)
    data_sets.validation.apply_permutation(validation_perm)
    data_sets.test.apply_permutation(test_perm)
    return data_sets


def normalize_data_sets(data_sets):
    normalize_data(data_sets.get_image_sets())
    normalize_data(data_sets.get_edge_sets())
    data_sets = make_equal_count_per_category(data_sets)
    return data_sets
