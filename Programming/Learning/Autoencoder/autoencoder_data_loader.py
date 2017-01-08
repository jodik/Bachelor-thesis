import numpy as np

from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator


class AutoencoderDataLoader(object):
    def __init__(self, data_sets):
        self.time_logger = TimeCalculator(self.name)
        self.init_labels(data_sets)
        self.init_data(data_sets)

    def init_data(self, data_sets):
        self.test_data = data_sets.test.original_images
        self.test_data = self.test_data.reshape((len(self.test_data), np.prod(self.test_data.shape[1:])))
        self.validation_data = data_sets.validation.original_images
        self.validation_data = self.validation_data.reshape((len(self.validation_data), np.prod(self.validation_data.shape[1:])))
        self.train_data = data_sets.train.original_images
        self.train_data = self.train_data.reshape((len(self.train_data), np.prod(self.train_data.shape[1:])))

    def init_labels(self, data_sets):
        self.test_labels = data_sets.test.labels
        self.validation_labels = data_sets.validation.labels
        self.train_labels = data_sets.train.labels
        self.train_size = self.train_labels.shape[0]

    @staticmethod
    def write_losses(train_loss_history, val_loss_history):
        helper.write_line()
        epoch = 1
        for x in val_loss_history:
            print ("Train and Validation loss epoch %d: %.5f %.5f" % (epoch, train_loss_history[epoch - 1], x))
            epoch = epoch + 1