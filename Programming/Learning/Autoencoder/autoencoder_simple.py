from keras.layers import Input, Dense
from keras import regularizers
from keras.models import Model
import numpy as np
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.Autoencoder import configuration_simple
import Programming.configuration as conf
from Programming.Learning.Autoencoder.autoencoder_data_loader import AutoencoderDataLoader


class SimpleAutoEncoder(AutoencoderDataLoader):
    def __init__(self, data_sets):
        self.init_name()
        AutoencoderDataLoader.__init__(self, data_sets)
        self.time_logger = TimeCalculator(self.name)
        self.init_configuration()

    def init_name(self):
        self.name = "SimpleAutoEncoder"

    def init_configuration(self):
        self.conf_s = configuration_simple

    def model(self):
        # this is our input placeholder
        input_img = Input(shape=(32*32*3,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(self.conf_s.FEATURES, activation='relu')(input_img)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(32*32*3, activation='sigmoid')(encoded)

        return Model(input=input_img, output=decoded)

    def save_model(self, model):
        model.save('Learning/Autoencoder/models/simple44.h5')

    def run(self):
        self.time_logger.show("Start learning")
        autoencoder = self.model()
        autoencoder.compile(self.conf_s.OPTIMIZER, loss='mean_squared_error')

        history_callback = autoencoder.fit(self.train_data, self.train_data,
                        nb_epoch=self.conf_s.NUMBER_OF_EPOCHS,
                        batch_size=256,
                        shuffle=True,
                        validation_data=(self.validation_data, self.validation_data),
                        verbose=0 if conf.WRITE_TO_FILE else 1)

        AutoencoderDataLoader.write_losses(history_callback.history["loss"], history_callback.history["val_loss"])
        if self.conf_s.SAVE_MODEL:
            self.save_model(autoencoder)
        self.time_logger.show("Finished learning")

