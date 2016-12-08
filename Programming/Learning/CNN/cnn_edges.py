from Programming.Learning.CNN import configuration_edges
from Programming.Learning.CNN.cnn_default import CNNDefault


class CNNEdges(CNNDefault):
    def __init__(self, data_sets):
        CNNDefault.__init__(self, data_sets)

    def init_name(self):
        self.name = "CNN Edges"

    def init_data(self, data_sets):
        self.test_data = data_sets.test.edge_descriptors
        self.validation_data = data_sets.validation.edge_descriptors
        self.train_data = data_sets.train.edge_descriptors

    def init_configuration(self):
        self.conf_s = configuration_edges