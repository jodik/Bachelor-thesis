from keras.engine import Input
import numpy as np
import tensorflow as tf
from keras.engine import Model
from keras.layers import Dense
from keras.models import load_model

from Programming.HelperScripts import helper
from Programming.Learning.Autoencoder import configuration_classifier_simple
import Programming.configuration as conf
from Programming.Learning.Autoencoder.autoencoder_data_loader import AutoencoderDataLoader
from keras.utils.np_utils import to_categorical
from keras.metrics import categorical_accuracy


class ClassifierSimpleAutoEncoder(AutoencoderDataLoader):
    def __init__(self, data_sets):
        self.init_name()
        AutoencoderDataLoader.__init__(self, data_sets)
        self.init_configuration()
        self.transform_input_data()
        self.transform_input_labels()

    def init_name(self):
        self.name = "ClassifierSimpleAutoEncoder"

    def init_configuration(self):
        self.conf_s = configuration_classifier_simple

    def retrieve_encoder(self):
        img_input = Input(shape=(32 * 32 * 3,))
        autoencoder = load_model(configuration_classifier_simple.AUTOENCODER_MODEL)
        encoder = Model(input=img_input, output=autoencoder.layers[-2](img_input))
        return encoder

    def transform_input_data(self):
        encoder = self.retrieve_encoder()
        self.train_data = encoder.predict(self.train_data)
        self.validation_data = encoder.predict(self.validation_data)
        self.test_data = encoder.predict(self.test_data)

    def transform_input_labels(self):
        self.train_labels = to_categorical(self.train_labels, nb_classes=conf.NUM_LABELS)
        self.validation_labels_o = self.validation_labels
        self.validation_labels = to_categorical(self.validation_labels, nb_classes=conf.NUM_LABELS)
        self.test_labels_o = self.test_labels
        self.test_labels = to_categorical(self.test_labels, nb_classes=conf.NUM_LABELS)

    def model(self):
        encoded_input = Input(shape=(128,))
        hidden_layer = Dense(self.conf_s.NUMBER_OF_NEURONS_IN_HIDDEN_LAYER, activation='sigmoid')(encoded_input)
        output_layer = Dense(conf.NUM_LABELS, activation='sigmoid')(hidden_layer)
        classifier = Model(input=encoded_input, output=output_layer)
        classifier.compile(self.conf_s.OPTIMIZER, loss='categorical_crossentropy')
        return classifier

    def run(self):
        self.time_logger.show("Start learning")

        classifier = self.model()
        history_callback = classifier.fit(self.train_data, self.train_labels,
                                           nb_epoch=self.conf_s.NUMBER_OF_EPOCHS,
                                           batch_size=256,
                                           shuffle=True,
                                           validation_data=(self.validation_data, self.validation_labels),
                                           verbose=0 if conf.WRITE_TO_FILE else 1)

        AutoencoderDataLoader.write_losses(history_callback.history["loss"], history_callback.history["val_loss"])

        val_predcs = classifier.predict(self.validation_data)
        accuracy, confusion_matrix = helper.error_rate(val_predcs, self.validation_labels_o)
        helper.write_eval_stats(confusion_matrix, accuracy, self.conf_s.USE_TEST_DATA)
        self.time_logger.show("Finished learning")
        return accuracy, confusion_matrix
