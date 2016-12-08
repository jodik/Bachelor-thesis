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

import numpy
import tensorflow as tf

import Programming.configuration as conf
from Programming.Learning.CNN import configuration_default
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.CNN.model import Model


class CNNDefault(object):
    def __init__(self, data_sets):
        self.init_name()
        self.time_logger = TimeCalculator(self.name)
        self.clear()
        self.init_labels(data_sets)
        self.init_data(data_sets)
        self.init_configuration()
        self.init_model(data_sets.validation.labels.shape[0])

    def init_name(self):
        self.name = "CNN Default"

    def clear(self):
        self.past_validation_errors = []
        self.past_eval_results = []

    def init_data(self, data_sets):
        self.test_data = data_sets.test.images
        self.validation_data = data_sets.validation.images
        self.train_data = data_sets.train.images

    def init_labels(self, data_sets):
        self.test_labels = data_sets.test.labels
        self.validation_labels = data_sets.validation.labels
        self.train_labels = data_sets.train.labels
        self.train_size = self.train_labels.shape[0]

    def init_configuration(self):
        self.conf_s = configuration_default

    def init_model(self, eval_size):
        self.model = Model(self.conf_s)
        self.model.init(self.train_size, eval_size)
        self.time_logger.show("Model creation")

    def eval(self, data, sess):
        predictions = sess.run(self.model.eval_prediction, feed_dict={self.model.eval_data_node: data})
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
        return best_index + self.conf_s.TRAIN_VALIDATION_CONDITION >= len(self.past_validation_errors)

    def retrieve_batch(self, step):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * self.conf_s.BATCH_SIZE) % (self.train_size - self.conf_s.BATCH_SIZE)
        batch_data = self.train_data[offset:(offset + self.conf_s.BATCH_SIZE), ...]
        batch_labels = self.train_labels[offset:(offset + self.conf_s.BATCH_SIZE)]
        return batch_data, batch_labels

    def num_of_epochs(self, steps):
        return float(steps) * self.conf_s.BATCH_SIZE / self.train_size

    def log_results(self, step, sess, l, lr, predictions, batch_labels):
        self.time_logger.show('Step %d (epoch %.2f)' % (step, self.num_of_epochs(step)))
        validation_error, _ = self.error_rate(self.eval(self.validation_data, sess), self.validation_labels)
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % self.error_rate(predictions, batch_labels)[0])
        print('Validation error: %.1f%%' % validation_error)

    def retrieve_eval_data(self):
        if self.conf_s.USE_TEST_DATA:
            return self.test_data, self.test_labels
        else:
            return self.validation_data, self.validation_labels

    def validate(self, sess):
        validation_error, _ = self.error_rate(
            self.eval(self.validation_data, sess),
            self.validation_labels)
        self.past_validation_errors.append(validation_error)
        eval_data, eval_labels = self.retrieve_eval_data()
        self.past_eval_results.append(self.error_rate(self.eval(eval_data, sess), eval_labels))

    def train_in_batches(self, sess):
        step = 0
        # Loop through training steps.
        while self.should_continue_training():
            batch_data, batch_labels = self.retrieve_batch(step)
            feed_dict = {self.model.train_data_node: batch_data,
                         self.model.train_labels_node: batch_labels}
            # Run the graph and fetch some of the nodes.
            _, l, lr, predictions = sess.run(
                [self.model.optimizer, self.model.loss, self.model.learning_rate, self.model.train_prediction],
                feed_dict=feed_dict)

            if step % self.conf_s.VALIDATION_FREQUENCY == 0:
                self.validate(sess)
            if step % self.conf_s.EVAL_FREQUENCY == 0:
                self.log_results(step, sess, l, lr, predictions, batch_labels)
            step += 1
        return step

    def final_results(self, steps):
        num_of_epochs = self.num_of_epochs(steps)
        print('Time per epoch: %.2f secs' % (self.time_logger.getTotalTime() / num_of_epochs))
        print('Number of epochs: %.1f' % num_of_epochs)

        min_validation_error = min(self.past_validation_errors)
        print('Min validation error: %.1f%%' % min_validation_error)

        best_validation_error_index = self.past_validation_errors.index(min_validation_error)
        eval_error, eval_confusion_matrix = self.past_eval_results[best_validation_error_index]
        helper.write_eval_stats(eval_confusion_matrix, eval_error, self.conf_s.USE_TEST_DATA)
        return eval_error, eval_confusion_matrix

    def run(self):
        self.clear()
        self.time_logger.show("Start")
        # Create a local session to run the training.
        with tf.Session() as sess:
            # Run all the initializers to prepare the trainable parameters.
            tf.initialize_all_variables().run()
            self.time_logger.show("Variable initialization")

            steps = self.train_in_batches(sess)
            self.time_logger.show("Training the model")

            eval_error, eval_confusion_matrix = self.final_results(steps)
            return eval_error, eval_confusion_matrix
