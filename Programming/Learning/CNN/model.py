import tensorflow as tf
import Programming.configuration as conf_global


class Model(object):
    def __init__(self, configuration_specific):
        self.configuration_specific = configuration_specific

    def init(self, train_size, eval_size):
        self.init_input_nodes(eval_size, train_size)
        self.init_convolutional_layers()
        self.init_normal_layers()
        self.init_predictions()
        self.init_optimizer(train_size)

    def init_input_nodes(self, eval_size, train_size):
        self.train_data_node = tf.placeholder(
            tf.float32,
            shape=(self.configuration_specific.BATCH_SIZE, self.configuration_specific.IMAGE_HEIGHT,
                   self.configuration_specific.IMAGE_WIDTH, self.configuration_specific.NUM_CHANNELS))
        self.train_labels_node = tf.placeholder(tf.int64, shape=(self.configuration_specific.BATCH_SIZE,))
        self.eval_data_node = tf.placeholder(
            tf.float32,
            shape=(eval_size, self.configuration_specific.IMAGE_HEIGHT,
                   self.configuration_specific.IMAGE_WIDTH, self.configuration_specific.NUM_CHANNELS))
        self.train_eval_data_node = tf.placeholder(
            tf.float32,
            shape=(train_size, self.configuration_specific.IMAGE_HEIGHT,
                   self.configuration_specific.IMAGE_WIDTH, self.configuration_specific.NUM_CHANNELS))

    def init_convolutional_layers(self):
        self.conv1_weights = tf.Variable(
            tf.truncated_normal(
                [self.configuration_specific.CONV_FIRST_FILTER_SIZE, self.configuration_specific.CONV_FIRST_FILTER_SIZE, self.configuration_specific.NUM_CHANNELS, self.configuration_specific.CONV_FIRST_DEPTH],
                # 5x5 filter, depth 32.
                stddev=0.1,
                seed=conf_global.SEED))
        self.conv1_biases = tf.Variable(tf.zeros([self.configuration_specific.CONV_FIRST_DEPTH]))
        self.conv2_weights = tf.Variable(
            tf.truncated_normal(
                [self.configuration_specific.CONV_SECOND_FILTER_SIZE, self.configuration_specific.CONV_SECOND_FILTER_SIZE, self.configuration_specific.CONV_FIRST_DEPTH, self.configuration_specific.CONV_SECOND_DEPTH],
                stddev=0.1,
                seed=conf_global.SEED))
        self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[self.configuration_specific.CONV_SECOND_DEPTH]))

    def count_fc1_input_size(self):
        res = int((
                     self.configuration_specific.IMAGE_HEIGHT * self.configuration_specific.IMAGE_WIDTH * self.configuration_specific.CONV_SECOND_DEPTH) / (
                         pow(self.configuration_specific.CON_FIRST_STRIDE, 2) * pow(
                             self.configuration_specific.POOL_SEC_SIZE, 2) * pow(
                             self.configuration_specific.POOL_FIRST_SIZE, 2)))
        return res

    def init_normal_layers(self):
        self.fc1_weights = tf.Variable(  # fully connected, depth 512.
            tf.truncated_normal(
                [self.count_fc1_input_size(),
                 self.configuration_specific.FC1_FEATURES],
                stddev=0.1,
                seed=conf_global.SEED))
        self.fc1_biases = tf.Variable(tf.constant(0.1, shape=[self.configuration_specific.FC1_FEATURES]))
        self.fc2_weights = tf.Variable(
            tf.truncated_normal([self.configuration_specific.FC1_FEATURES, conf_global.NUM_LABELS],
                                stddev=0.1,
                                seed=conf_global.SEED))
        self.fc2_biases = tf.Variable(tf.constant(0.1, shape=[conf_global.NUM_LABELS]))



    """Connect layers"""

    def connect_conv_layers(self, input_node):
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(input_node,
                            self.conv1_weights,
                            strides=[1, self.configuration_specific.CON_FIRST_STRIDE,
                                     self.configuration_specific.CON_FIRST_STRIDE, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, self.configuration_specific.POOL_FIRST_SIZE,
                                     self.configuration_specific.POOL_FIRST_SIZE, 1],
                              strides=[1, self.configuration_specific.POOL_FIRST_SIZE,
                                       self.configuration_specific.POOL_FIRST_SIZE, 1],
                              padding='SAME')
        conv = tf.nn.conv2d(pool,
                            self.conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, self.conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, self.configuration_specific.POOL_SEC_SIZE,
                                     self.configuration_specific.POOL_SEC_SIZE, 1],
                              strides=[1, self.configuration_specific.POOL_SEC_SIZE,
                                       self.configuration_specific.POOL_SEC_SIZE, 1],
                              padding='SAME')
        return pool

    def create_model(self, input_node, train=False):
        """The Model definition."""
        pool = self.connect_conv_layers(input_node)
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, self.fc1_weights) + self.fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, 1 - self.configuration_specific.DROPOUT_PROBABILITY, seed=conf_global.SEED)
        return tf.matmul(hidden, self.fc2_weights) + self.fc2_biases

    def create_train_model(self):
        return self.create_model(self.train_data_node, True)

    def create_eval_model(self):
        return self.create_model(self.eval_data_node)

    def create_train_eval_model(self):
        return self.create_model(self.train_eval_data_node)

    def init_predictions(self):
        # Training computation: logits + cross-entropy loss.
        self.train_logits = self.create_train_model()
        # Predictions for the current training minibatch.
        self.train_prediction = tf.nn.softmax(self.train_logits)
        # Predictions for the test and validation, which we'll compute less often.
        self.eval_prediction = tf.nn.softmax(self.create_eval_model())
        self.train_eval_prediction = tf.nn.softmax(self.create_train_eval_model())

    def init_optimizer(self, train_size):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            self.train_logits, self.train_labels_node))

        # L2 regularization for the fully connected parameters.
        regularizers = (tf.nn.l2_loss(self.fc1_weights) + tf.nn.l2_loss(self.fc1_biases) +
                        tf.nn.l2_loss(self.fc2_weights) + tf.nn.l2_loss(self.fc2_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * regularizers

        # Optimizer: set up a variable that's incremented once per batch and
        # controls the learning rate decay.
        batch = tf.Variable(0)
        # Decay once per epoch, using an exponential schedule starting at 0.01.
        self.learning_rate = tf.train.exponential_decay(
            self.configuration_specific.BASE_LEARNING_RATE,  # Base learning rate.
            batch * self.configuration_specific.BATCH_SIZE,  # Current index into the dataset.
            self.configuration_specific.DECAY_STEP_X_TIMES_TRAIN_SIZE * train_size,  # Decay step.
            self.configuration_specific.DECAY_RATE,  # Decay rate.
            staircase=False)
        # Use simple momentum for the optimization.
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    self.configuration_specific.MOMENTUM).minimize(self.loss,
                                                                                                   global_step=batch)
