"""Functions for downloading and reading MNIST data."""
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
from PIL import Image
from os import walk
from cobs import cobs
import array
from collections import Counter

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import sys
sys.path.append("../")
import configuration

def count_images():
    return extract_images()[0].shape[0]
    
def getLabel(label, data_types):
     data_types = sorted(data_types)
     for i in range(len(data_types)):
         if data_types[i] in label:
             return i
     raise ValueError('Bad path.')
    
def extract_images():
    images = np.zeros((0, configuration.IMAGE_HEIGHT, configuration.IMAGE_WIDTH , configuration.NUM_CHANNELS))
    correct_vals = np.zeros((0))
    
    with open (configuration.SOURCE_FOLDER_NAME + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        all_images = array.array('B',data)
        num_of_images = int(len(all_images)/(configuration.IMAGE_WIDTH*configuration.IMAGE_HEIGHT*configuration.NUM_CHANNELS))
        all_images = np.asarray(all_images, dtype=np.uint8)
        all_images = all_images.reshape(num_of_images, configuration.IMAGE_HEIGHT, configuration.IMAGE_WIDTH, configuration.NUM_CHANNELS)
        all_images = all_images / configuration.PIXEL_DEPTH
     
    with open (configuration.SOURCE_FOLDER_NAME + 'labels.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        labels = array.array('B',data)
        labels = np.asarray(labels, dtype=np.uint8)
        
    with open (configuration.SOURCE_FOLDER_NAME + 'ishard.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        ishard = array.array('B',data)
        ishard = np.asarray(ishard, dtype=np.uint8)
        
    for i in range(num_of_images):
        label = configuration.ALL_DATA_TYPES[labels[i]]
        if label in configuration.DATA_TYPES_USED and (ishard[i] == 0 or configuration.HARD_DIFFICULTY):
            category = getLabel(label, configuration.DATA_TYPES_USED)
            correct_vals = np.append(correct_vals, [category])
            images = np.append(images, [all_images[i]], axis = 0)
    #for i in range(count):
     #   images[i] = images[i].reshape(width, height, channels)
    return (images, correct_vals)

class MyDataSet(object):

  def __init__(self, images, labels, one_hot=True,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    assert len(images) == len(labels), (
          'images.shape: %s labels.shape: %s' % (len(images),
                                                 len(labels)))
    self._num_examples = len(images)
    
    
    if dtype == tf.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      labels = labels.astype(np.float32)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def getPermutation():
    if configuration.PERMUTATION_INDEX >= 10 or configuration.PERMUTATION_INDEX < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    folder = configuration.PERMUTATION_FOLDER_NAME + configuration.DATA_TYPES_USED[0]
    for i in range(configuration.NUM_LABELS - 1):
        folder+='_' + configuration.DATA_TYPES_USED[i+1]
    if configuration.HARD_DIFFICULTY:
        folder+='_Hard'
    else:
        folder+='_Normal'
    if not os.path.exists(folder):
        os.makedirs(folder)
        num_of_images = count_images()
        perm = np.arange(num_of_images)
        for i in range(10):
            fo = open(folder+"/permutation"+str(i)+".txt", "wb")
            np.random.shuffle(perm)
            for item in perm:
                fo.write("%s\n" % item)
    print(folder+"/permutation"+str(configuration.PERMUTATION_INDEX)+".txt")
    with open(folder+"/permutation"+str(configuration.PERMUTATION_INDEX)+".txt", 'r') as f:
        lines = f.read().splitlines()
        perm = []
        for i in range(len(lines)):
            perm.append(int(lines[i]))
        return perm



def read_data_sets(one_hot=True, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images_and_labels = extract_images()
  images = images_and_labels[0]
  labels = images_and_labels[1]
  
  perm2 = getPermutation()
  print(len(perm2))
  print(images.shape[0])
  images = images[perm2]
  labels = labels[perm2]
  print(labels)
  
  TEST_SIZE = int(len(images) * (configuration.TEST_PERCENTAGE/100.0))
  TEST_SIZE = int(len(images) * (configuration.VALIDATION_PERCENTAGE/100.0))

  test_images = images[:TEST_SIZE, ...]
  test_labels = labels[:TEST_SIZE]
  train_images = images[TEST_SIZE:, ...]
  train_labels = labels[TEST_SIZE:]

  validation_images = train_images[:TEST_SIZE, ...]
  validation_labels = train_labels[:TEST_SIZE]
  train_images = train_images[TEST_SIZE:, ...]
  train_labels = train_labels[TEST_SIZE:]

  data_sets.train = MyDataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = MyDataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = MyDataSet(test_images, test_labels, dtype=dtype)

  return data_sets

