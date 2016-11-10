import Programming.configuration as conf


class Data(object):
    def __init__(self, images, labels, names, ishard):
        assert len(images) == len(labels) == len(names) == len(ishard)
        self._data = (images, labels, names, ishard)
        self._images = images
        self._labels = labels
        self._names = names
        self._ishard = ishard
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def apply_permutation(self, permutation):
        new_data = map(lambda x: x[permutation], self._data)
        self.__init__(*new_data)
        return self

    def create_data(self, from_i, to_i):
        new_data = map(lambda x: x[from_i:to_i], self._data)
        return Data(*new_data)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def names(self):
        return self._names

    @property
    def num_examples(self):
        return self._images.shape[0]

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


class FullData(Data):
    def getOriginalDatasetSize(self):
        divide_by = 8 if conf.EXTENDED_DATASET else 1
        original_dataset_size = self.num_examples / divide_by
        return original_dataset_size

    def create_data(self, from_i, to_i):
        new_data = map(lambda x: x[from_i:to_i], self._data)
        return FullData(*new_data)
