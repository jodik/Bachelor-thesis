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
from Programming.HelperScripts import helper
from Programming.HelperScripts.time_calculator import TimeCalculator
from Programming.Learning.CNN.model import Model


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  error_rate = 100.0 - (100.0 * 
      numpy.sum(numpy.argmax(predictions, 1) == labels) / 
      predictions.shape[0])
  num_cat = max(labels) + 1
  correct = numpy.zeros((num_cat, num_cat), dtype=int)
  for  prediction, label in zip(predictions, labels):
      correct[int(label), numpy.argmax(prediction)]+= 1
  return(error_rate, correct)

   
def compute(datasets):

  test_data = datasets.test.images
  test_labels = datasets.test.labels
  validation_data = datasets.validation.images
  validation_labels = datasets.validation.labels
  train_data = datasets.train.images
  train_labels = datasets.train.labels
    
  train_size = train_labels.shape[0]

  model = Model(conf)
  model.init(train_size)

  def shouldContinueTraining(validation_errors):
       if(len(validation_errors) == 0):
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
            model.eval_prediction,
            feed_dict={model.eval_data_node: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            model.eval_prediction,
            feed_dict={model.eval_data_node: data[-conf.EVAL_BATCH_SIZE:, ...]})
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
      
      feed_dict = {model.train_data_node: batch_data,
                   model.train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
          [model.optimizer, model.loss, model.learning_rate, model.train_prediction],
          feed_dict=feed_dict)
      if step % conf.EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('\nStep %d' % step)
        validation_error, validation_confusion_matrix = error_rate(eval_in_batches(validation_data, sess),
                                                                   validation_labels)
        #helper.writeConfusionMatrix(validation_confusion_matrix)
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
    num_of_epochs = (step * conf.BATCH_SIZE)/float(train_size)
    print('')
    time_logger.show("Training")
    print('Time per epoch: %.2f secs' % (time_logger.getTotalTime() / num_of_epochs))
    print('')
    print('Number of epochs: %.1f' % num_of_epochs)
    print('Min validation error: %.1f%%' % min_validation_error)
    test_error, test_confusion_matrix = past_test_results[best_validation_error_index]
    helper.write_test_stats(test_confusion_matrix, test_error)

    return test_error, test_confusion_matrix


    

