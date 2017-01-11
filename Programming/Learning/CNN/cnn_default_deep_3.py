from Programming.Learning.CNN import configuration_default_deep_3
from Programming.Learning.CNN.cnn_default import CNNDefault
from Programming.Learning.CNN.model import Model
import tensorflow as tf
import Programming.configuration as conf_global


class MyModel(Model):
    def __init__(self, configuration_specific):
        Model.__init__(self, configuration_specific)

    def init_convolutional_layers(self):
        Model.init_convolutional_layers(self)
        self.conv3_weights = tf.Variable(
            tf.truncated_normal(
                [self.configuration_specific.CONV_THIRD_FILTER_SIZE,
                 self.configuration_specific.CONV_THIRD_FILTER_SIZE, self.configuration_specific.CONV_SECOND_DEPTH,
                 self.configuration_specific.CONV_THIRD_DEPTH],
                stddev=0.1,
                seed=conf_global.SEED))
        self.conv3_biases = tf.Variable(tf.constant(0.1, shape=[self.configuration_specific.CONV_THIRD_DEPTH]))

    def count_fc1_input_size(self):
        res = int((
                     self.configuration_specific.IMAGE_HEIGHT * self.configuration_specific.IMAGE_WIDTH * self.configuration_specific.CONV_THIRD_DEPTH) / (
                         pow(self.configuration_specific.CON_FIRST_STRIDE, 2) * pow(
                             self.configuration_specific.POOL_SEC_SIZE, 2) * pow(
                             self.configuration_specific.POOL_FIRST_SIZE, 2) * pow(
                             self.configuration_specific.POOL_THIRD_SIZE, 2)))
        return res

    def connect_conv_layers(self, input_node):
        pool = Model.connect_conv_layers(self, input_node)
        conv = tf.nn.conv2d(pool,
                            self.conv3_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv3_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, self.configuration_specific.POOL_THIRD_SIZE,
                                     self.configuration_specific.POOL_THIRD_SIZE, 1],
                              strides=[1, self.configuration_specific.POOL_THIRD_SIZE,
                                       self.configuration_specific.POOL_THIRD_SIZE, 1],
                              padding='SAME')
        return pool


class CNNDefaultDeep3(CNNDefault):
    def __init__(self, data_sets):
        CNNDefault.__init__(self, data_sets)

    def init_name(self):
        self.name = "CNN Default Deep 3"

    def init_data(self, data_sets):
        self.test_data = data_sets.test.images
        self.validation_data = data_sets.validation.images
        self.train_data = data_sets.train.images

    def init_configuration(self):
        self.conf_s = configuration_default_deep_3

    def init_model(self, validation_size, test_size):
        self.model = MyModel(self.conf_s)
        self.model.init(self.train_size, validation_size, test_size)
        self.time_logger.show("Model creation")