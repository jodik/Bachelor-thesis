from keras.utils.np_utils import to_categorical
import keras
import Programming.configuration as conf

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Convolution2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K


# Helper to build a conv -> BN -> relu block
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.Autoencoder.autoencoder_data_loader import AutoencoderDataLoader
from Programming.Learning.CNN import configuration_res


def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[ROW_AXIS] // residual._keras_shape[ROW_AXIS]
    stride_height = input._keras_shape[COL_AXIS] // residual._keras_shape[COL_AXIS]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 2, 2, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 2, 2)(conv1)
        return _shortcut(input, residual)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


def handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
        :param num_outputs: The number of outputs at final softmax layer
        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50
        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved
        :return: The keras model.
        """
        handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=8, nb_row=3, nb_col=3, subsample=(1, 1))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filters = 8
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0)(block)
            nb_filters *= 2
            #nb_filters = min(nb_filters, 10)

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[ROW_AXIS],
                                            block._keras_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        block = keras.layers.core.Dropout(0.75)(block)
        flatten1 = Flatten()(block)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax", W_regularizer=keras.regularizers.l2(0.02))(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1, 1])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


class CNNRes(object):
    def __init__(self, data_sets):
        self.init_name()
        self.time_logger = TimeCalculator(self.name)
        self.init_labels(data_sets)
        self.init_data(data_sets)
        self.init_configuration()
        self.init_model()

    def init_name(self):
        self.name = "CNN Residual"

    def init_data(self, data_sets):
        self.test_data = data_sets.test.images
        self.validation_data = data_sets.validation.images
        self.train_data = data_sets.train.images

    def init_labels(self, data_sets):
        self.test_labels = data_sets.test.labels
        self.validation_labels = data_sets.validation.labels
        self.train_labels = data_sets.train.labels
        self.train_labels = to_categorical(self.train_labels, nb_classes=conf.NUM_LABELS)
        self.validation_labels_o = self.validation_labels
        self.validation_labels = to_categorical(self.validation_labels, nb_classes=conf.NUM_LABELS)
        self.test_labels_o = self.test_labels
        self.test_labels = to_categorical(self.test_labels, nb_classes=conf.NUM_LABELS)

    def init_configuration(self):
        self.conf_s = configuration_res

    def init_model(self):
        if self.conf_s.LOAD_MODEL:
            self.model = keras.models.load_model('Learning/CNN/models/res.h5')
        else:
            self.model = ResnetBuilder.build_resnet_18((3, 32, 32), conf.NUM_LABELS)
            self.model.compile(loss="categorical_crossentropy", optimizer="sgd")
        self.model.summary()

    def save_model(self):
        self.model.save('Learning/CNN/models/res.h5')

    def run(self):
        self.time_logger.show("Start learning")
        history_callback = self.model.fit(self.train_data, self.train_labels,
                                           nb_epoch=self.conf_s.NUMBER_OF_EPOCHS,
                                           batch_size=256,
                                           shuffle=True,
                                           validation_data=(self.validation_data, self.validation_labels),
                                           verbose=0 if conf.WRITE_TO_FILE else 1)
        AutoencoderDataLoader.write_losses(history_callback.history["loss"], history_callback.history["val_loss"])
        if self.conf_s.SAVE_MODEL:
            self.save_model()
        val_predcs = self.model.predict(self.validation_data)
        accuracy, confusion_matrix = helper.error_rate(val_predcs, self.validation_labels_o)
        helper.write_eval_stats(confusion_matrix, accuracy, self.conf_s.USE_TEST_DATA)
        self.time_logger.show("Finished learning")
        return accuracy, confusion_matrix


"""
def main():
    model = ResnetBuilder.build_resnet_18((3, 32, 32), 1000)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()


if __name__ == '__main__':
    main()"""