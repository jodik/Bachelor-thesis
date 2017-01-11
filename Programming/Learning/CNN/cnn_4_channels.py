import numpy as np
from Programming.Learning.CNN import configuration_4_channels
from Programming.Learning.CNN.cnn_default import CNNDefault


def concatante(data_set):
    sh = data_set.images.shape
    sh = sh[0], sh[1], sh[2], 1
    t = np.zeros(sh)
    return np.concatenate([data_set.images, data_set.edge_descriptors], axis=3)


class CNN4Channels(CNNDefault):
    def __init__(self, data_sets):
        CNNDefault.__init__(self, data_sets)

    def init_name(self):
        self.name = "CNN 4 Channels"

    def init_data(self, data_sets):
        self.test_data = concatante(data_sets.test)
        self.validation_data = concatante(data_sets.validation)
        self.train_data = concatante(data_sets.train)

    def init_configuration(self):
        self.conf_s = configuration_4_channels