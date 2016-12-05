# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import time

import numpy
import tensorflow as tf
from six.moves import xrange

import Programming.Learning.CNN.configuration_edges as confs
import Programming.configuration as conf
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'

FLAGS = tf.app.flags.FLAGS


def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    error_rate = 100.0 - (100.0 *
                          numpy.sum(numpy.argmax(predictions, 1) == labels) /
                          predictions.shape[0])
    num_cat = max(labels) + 1
    num_cat = len(conf.DATA_TYPES_USED)
    correct = numpy.zeros((num_cat, num_cat), dtype=int)
    for prediction, label in zip(predictions, labels):
        correct[int(label), numpy.argmax(prediction)] += 1
    return (error_rate, correct)


def compute(datasets):
    test_data = datasets.test.edge_descriptors
    test_labels = datasets.test.labels
    validation_data = datasets.validation.edge_descriptors
    validation_labels = datasets.validation.labels
    train_data = datasets.train.edge_descriptors
    train_labels = datasets.train.labels

    train_size = train_labels.shape[0]

    image = tf.placeholder(
        tf.float32,
        shape=(confs.IMAGE_HEIGHT, confs.IMAGE_WIDTH, confs.NUM_CHANNELS))
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        tf.float32,
        shape=(conf.BATCH_SIZE, confs.IMAGE_HEIGHT, confs.IMAGE_WIDTH, confs.NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(conf.BATCH_SIZE,))
    eval_data = tf.placeholder(
        tf.float32,
        shape=(conf.EVAL_BATCH_SIZE, confs.IMAGE_HEIGHT, confs.IMAGE_WIDTH, confs.NUM_CHANNELS))

    # The variables below hold all the trainable weights. They are passed an
    # initial value which will be assigned when we call:
    # {tf.initialize_all_variables().run()}
    conv1_weights = tf.Variable(
        tf.truncated_normal([15, 15, confs.NUM_CHANNELS, confs.CONV_FIRST_DEPTH],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=conf.SEED))
    conv1_biases = tf.Variable(tf.zeros([confs.CONV_FIRST_DEPTH]))
    conv2_weights = tf.Variable(
        tf.truncated_normal([5, 5, confs.CONV_FIRST_DEPTH, confs.CONV_SECOND_DEPTH],
                            stddev=0.1,
                            seed=conf.SEED))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[confs.CONV_SECOND_DEPTH]))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal(
            [int((confs.IMAGE_HEIGHT * confs.IMAGE_WIDTH * confs.CONV_SECOND_DEPTH) / (
            pow(confs.CON_FIRST_STRIDE, 2) * pow(confs.POOL_SEC_SIZE, 2) * pow(confs.POOL_FIRST_SIZE, 2))),
             confs.FC1_FEATURES],
            stddev=0.1,
            seed=conf.SEED))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[confs.FC1_FEATURES]))
    fc2_weights = tf.Variable(
        tf.truncated_normal([confs.FC1_FEATURES, conf.NUM_LABELS],
                            stddev=0.1,
                            seed=conf.SEED))
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[conf.NUM_LABELS]))

    flip = tf.image.random_flip_left_right(image, conf.SEED)

    # We will replicate the model structure for the training subgraph, as well
    # as the evaluation subgraphs, while sharing the trainable parameters.
    def model(data, train=False):
        """The Model definition."""
        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matches the data layout: [image index, y, x, depth].
        conv = tf.nn.conv2d(data,
                            conv1_weights,
                            strides=[1, confs.CON_FIRST_STRIDE, confs.CON_FIRST_STRIDE, 1],
                            padding='SAME')
        # Bias and rectified linear non-linearity.
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
        # Max pooling. The kernel size spec {ksize} also follows the layout of
        # the data. Here we have a pooling window of 2, and a stride of 2.
        pool = tf.nn.max_pool(relu,
                              ksize=[1, confs.POOL_FIRST_SIZE, confs.POOL_FIRST_SIZE, 1],
                              strides=[1, confs.POOL_FIRST_SIZE, confs.POOL_FIRST_SIZE, 1],
                              padding='SAME')
        conv_shape = pool.get_shape().as_list()
        print('pool shape', conv_shape)
        conv = tf.nn.conv2d(pool,
                            conv2_weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME')
        relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
        pool = tf.nn.max_pool(relu,
                              ksize=[1, confs.POOL_SEC_SIZE, confs.POOL_SEC_SIZE, 1],
                              strides=[1, confs.POOL_SEC_SIZE, conf.POOL_SEC_SIZE, 1],
                              padding='SAME')
        conv_shape = pool.get_shape().as_list()
        print('pool shape', conv_shape)
        # Reshape the feature map cuboid into a 2D matrix to feed it to the
        # fully connected layers.
        pool_shape = pool.get_shape().as_list()
        reshape = tf.reshape(
            pool,
            [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
        # Add a 50% dropout during training only. Dropout also scales
        # activations such that no rescaling is needed at evaluation time.
        if train:
            hidden = tf.nn.dropout(hidden, confs.DROPOUT_PROBABILITY, seed=conf.SEED)
        return tf.matmul(hidden, fc2_weights) + fc2_biases

    # Training computation: logits + cross-entropy loss.
    logits = model(train_data_node, True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, train_labels_node))

    # L2 regularization for the fully connected parameters.
    regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0)
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        conf.BASE_LEARNING_RATE,  # Base learning rate.
        batch * conf.BATCH_SIZE,  # Current index into the dataset.
        conf.DECAY_STEP_X_TIMES_TRAIN_SIZE * train_size,  # Decay step.
        conf.DECAY_RATE,  # Decay rate.
        staircase=False)
    # Use simple momentum for the optimization.
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           conf.MOMENTUM).minimize(loss,
                                                                   global_step=batch)

    # Predictions for the current training minibatch.
    train_prediction = tf.nn.softmax(logits)

    # Predictions for the test and validation, which we'll compute less often.
    eval_prediction = tf.nn.softmax(model(eval_data))

    def shouldContinueTraining(validation_errors):
        if (len(validation_errors) == 0):
            return True
        best_index = validation_errors.index(min(validation_errors))
        return best_index + conf.TRAIN_VALIDATION_CONDINATION >= len(validation_errors)

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < conf.EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, conf.NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, conf.EVAL_BATCH_SIZE):
            end = begin + conf.EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data: data[-conf.EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    # Create a local session to run the training.
    time_logger = TimeCalculator()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        print('Initialized!')
        past_validation_errors = []
        past_test_results = []
        step = 0
        # Loop through training steps.
        start_time = time.time()
        while shouldContinueTraining(past_validation_errors):
            # Compute the offset of the current minibatch in the data.
            # Note that we could use better randomization across epochs.
            offset = (step * conf.BATCH_SIZE) % (train_size - conf.BATCH_SIZE)
            batch_data = train_data[offset:(offset + conf.BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + conf.BATCH_SIZE)]
            # This dictionary maps the batch data (as a numpy array) to the
            # node in the graph it should be fed to.

            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [optimizer, loss, learning_rate, train_prediction],
                feed_dict=feed_dict)
            if step % conf.EVAL_FREQUENCY == 0:
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('\nStep %d' % step)
                validation_error, validation_confusion_matrix = error_rate(eval_in_batches(validation_data, sess),
                                                                           validation_labels)
                # helper.writeConfusionMatrix(validation_confusion_matrix)
                print('Step %d (epoch %.2f), %.1f ms' %
                      (step, float(step) * conf.BATCH_SIZE / train_size,
                       1000 * elapsed_time / conf.EVAL_FREQUENCY))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels)[0])
                print('Validation error: %.1f%%' % validation_error)
                sys.stdout.flush()
                past_validation_errors.append(validation_error)
                past_test_results.append(error_rate(eval_in_batches(test_data, sess), test_labels))
            step += 1
        min_validation_error = min(past_validation_errors)
        best_validation_error_index = past_validation_errors.index(min_validation_error)
        # Finally print the result!
        num_of_epochs = (step * conf.BATCH_SIZE) / float(train_size)
        print('')
        time_logger.show("Training")
        print('Time per epoch: %.2f secs' % (time_logger.getTotalTime() / num_of_epochs))
        print('')
        print('Number of epochs: %.1f' % num_of_epochs)
        print('Min validation error: %.1f%%' % min_validation_error)
        test_error, test_confusion_matrix = past_test_results[best_validation_error_index]
        helper.write_test_stats(test_confusion_matrix, test_error)

        return test_error, test_confusion_matrix




