from Programming.Learning.CNN import configuration_edges
from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.model import Model


class CNNEdges(CNNDefault):
    def __init__(self, data_sets):
        self.test_data = data_sets.test.edge_descriptors
        self.test_labels = data_sets.test.labels
        self.validation_data = data_sets.validation.edge_descriptors
        self.validation_labels = data_sets.validation.labels
        self.train_data = data_sets.train.edge_descriptors
        self.train_labels = data_sets.train.labels
        self.past_validation_errors = []
        self.past_test_results = []
        self.train_size = self.train_labels.shape[0]
        self.model = Model(configuration_edges)
        self.model.init(self.train_size)
