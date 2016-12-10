class DataSets(object):
    def __init__(self, train_data, validation_data, test_data):
        self._data = (train_data, validation_data, test_data)
        self._train = train_data
        self._validation = validation_data
        self._test = test_data

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    def size(self):
        size = self.train.num_examples + self.validation.num_examples + self.test.num_examples
        return size

    def get_image_sets(self):
        return self.train.images, self.validation.images, self.test.images

    def get_edge_sets(self):
        return self.train.edge_descriptors, self.validation.edge_descriptors, self.test.edge_descriptors

    def get_label_sets(self):
        return self.train.labels, self.validation.labels, self.test.labels

    def flatten(self):
        for x in self._data:
            x.flatten()
