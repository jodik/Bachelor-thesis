from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import keras
import sys
import logging
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.Autoencoder import configuration_simple

class SimpleAutoEncoder(object):
    def __init__(self, data_sets):
        self.init_name()
        self.time_logger = TimeCalculator(self.name)
        self.init_labels(data_sets)
        self.init_data(data_sets)
        self.init_configuration()

    def init_name(self):
        self.name = "SimpleAutoEncoder"

    def init_configuration(self):
        self.conf_s = configuration_simple

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

    def model(self):
        # this is our input placeholder
        input_img = Input(shape=(32*32*3,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(self.conf_s.FEATURES, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(32*32*3, activation='sigmoid')(encoded)

        return Model(input=input_img, output=decoded)

    def write_losses(self, loss_history):
        helper.write_line()
        epoch = 1
        for x in loss_history:
            print ("Validation loss epoch %d: %.5f" % (epoch, x))
            epoch = epoch + 1

    def run(self):
        keras.callbacks.BaseLogger()
        self.time_logger.show("Start learning")
        autoencoder = self.model()
        autoencoder.compile(self.conf_s.OPTIMIZER, loss='mean_squared_error')
        logging.basicConfig(level=logging.ERROR)

        history_callback = autoencoder.fit(self.train_data, self.train_data,
                        nb_epoch=self.conf_s.NUMBER_OF_EPOCHS,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(self.validation_data, self.validation_data),
                        verbose=0)

        self.write_losses(history_callback.history["val_loss"])
        if self.conf_s.SAVE_MODEL:
            autoencoder.save('Learning/Autoencoder/models/simple.h5')
        self.time_logger.show("Finished learning")

