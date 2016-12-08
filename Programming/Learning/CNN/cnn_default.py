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

import Programming.configuration as conf
from Programming.Learning.CNN import configuration_edges
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.CNN.model import Model


class CNNDefault(object):
    def __init__(self, data_sets):
        self.test_data = data_sets.test.images
        self.test_labels = data_sets.test.labels
        self.validation_data = data_sets.validation.images
        self.validation_labels = data_sets.validation.labels
        self.train_data = data_sets.train.images
        self.train_labels = data_sets.train.labels
        self.past_validation_errors = []
        self.past_test_results = []
        self.train_size = self.train_labels.shape[0]
        self.model = Model(conf)
        self.model.init(self.train_size)

    # Small utility function to evaluate a dataset by feeding batches of data to
    # {eval_data} and pulling the results from {eval_predictions}.
    # Saves memory and enables this to run on smaller GPUs.
    def eval_in_batches(self, data, sess):
        """Get all predictions for a dataset by running it in small batches."""
        size = data.shape[0]
        if size < conf.EVAL_BATCH_SIZE:
            raise ValueError("batch size for evals larger than dataset: %d" % size)
        predictions = numpy.ndarray(shape=(size, conf.NUM_LABELS), dtype=numpy.float32)
        for begin in xrange(0, size, conf.EVAL_BATCH_SIZE):
            end = begin + conf.EVAL_BATCH_SIZE
            if end <= size:
                predictions[begin:end, :] = sess.run(
                    self.model.eval_prediction,
                    feed_dict={self.model.eval_data_node: data[begin:end, ...]})
            else:
                batch_predictions = sess.run(
                    self.model.eval_prediction,
                    feed_dict={self.model.eval_data_node: data[-conf.EVAL_BATCH_SIZE:, ...]})
                predictions[begin:, :] = batch_predictions[begin - size:, :]
        return predictions

    def error_rate(self, predictions, labels):
        """Return the error rate based on dense predictions and sparse labels."""
        error_rate = 100.0 - (100.0 *
                              numpy.sum(numpy.argmax(predictions, 1) == labels) /
                              predictions.shape[0])
        num_cat = max(labels) + 1
        num_cat = len(conf.DATA_TYPES_USED)
        correct = numpy.zeros((num_cat, num_cat), dtype=int)
        for prediction, label in zip(predictions, labels):
            correct[int(label), numpy.argmax(prediction)] += 1
        return error_rate, correct

    def should_continue_training(self):
        if len(self.past_validation_errors) == 0:
            return True
        best_index = self.past_validation_errors.index(min(self.past_validation_errors))
        return best_index + conf.TRAIN_VALIDATION_CONDINATION >= len(self.past_validation_errors)

    def run(self):
        # Create a local session to run the training.
        time_logger = TimeCalculator("CNN run")
        time_logger.show("Start")
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
            time_logger.show("Initialized!")
            self.past_validation_errors = []
            self.past_test_results = []
            step = 0
            # Loop through training steps.
            start_time = time.time()
            while self.should_continue_training():
                # Compute the offset of the current minibatch in the data.
                # Note that we could use better randomization across epochs.
                offset = (step * conf.BATCH_SIZE) % (self.train_size - conf.BATCH_SIZE)
                batch_data = self.train_data[offset:(offset + conf.BATCH_SIZE), ...]
                batch_labels = self.train_labels[offset:(offset + conf.BATCH_SIZE)]
                # This dictionary maps the batch data (as a numpy array) to the
                # node in the graph it should be fed to.

                feed_dict = {self.model.train_data_node: batch_data,
                             self.model.train_labels_node: batch_labels}
                # Run the graph and fetch some of the nodes.
                _, l, lr, predictions = sess.run(
                    [self.model.optimizer, self.model.loss, self.model.learning_rate, self.model.train_prediction],
                    feed_dict=feed_dict)
                if step % conf.EVAL_FREQUENCY == 0:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print('\nStep %d' % step)
                    validation_error, validation_confusion_matrix = self.error_rate(self.eval_in_batches(self.validation_data, sess),
                                                                               self.validation_labels)
                    # helper.writeConfusionMatrix(validation_confusion_matrix)
                    print('Step %d (epoch %.2f), %.1f ms' %
                          (step, float(step) * conf.BATCH_SIZE / self.train_size,
                           1000 * elapsed_time / conf.EVAL_FREQUENCY))
                    print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                    print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels)[0])
                    print('Validation error: %.1f%%' % validation_error)
                    sys.stdout.flush()
                    self.past_validation_errors.append(validation_error)
                    self.past_test_results.append(self.error_rate(self.eval_in_batches(self.test_data, sess), self.test_labels))
                step += 1
            min_validation_error = min(self.past_validation_errors)
            best_validation_error_index = self.past_validation_errors.index(min_validation_error)
            # Finally print the result!
            num_of_epochs = (step * conf.BATCH_SIZE) / float(self.train_size)
            print('')
            time_logger.show("Training")
            print('Time per epoch: %.2f secs' % (time_logger.getTotalTime() / num_of_epochs))
            print('')
            print('Number of epochs: %.1f' % num_of_epochs)
            print('Min validation error: %.1f%%' % min_validation_error)
            test_error, test_confusion_matrix = self.past_test_results[best_validation_error_index]
            helper.write_test_stats(test_confusion_matrix, test_error)

            return test_error, test_confusion_matrix




