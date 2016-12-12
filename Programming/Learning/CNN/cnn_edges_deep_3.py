from Programming.Learning.CNN import configuration_edges_deep_3
from Programming.Learning.CNN.cnn_default_deep_3 import CNNDefaultDeep3


class CNNEdgesDeep3(CNNDefaultDeep3):
    def __init__(self, data_sets):
        CNNDefaultDeep3.__init__(self, data_sets)

    def init_name(self):
        self.name = "CNN Edges Deep 3"

    def init_data(self, data_sets):
        self.test_data = data_sets.test.edge_descriptors
        self.validation_data = data_sets.validation.edge_descriptors
        self.train_data = data_sets.train.edge_descriptors

    def init_configuration(self):
        self.conf_s = configuration_edges_deep_3
