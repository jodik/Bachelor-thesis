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
from os import walk

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def count_images(SOURCE_FOLDER_NAME, DATA_TYPES, hard_difficulty = False):
    count = 0
    DIFICULTIES_INCLUDED = ['Normal']
    if (hard_difficulty):
        DIFICULTIES_INCLUDED.append('Hard')
    for type in DATA_TYPES:
        for difficulty in DIFICULTIES_INCLUDED:
            for (dirpath, dirnames, filenames) in walk(SOURCE_FOLDER_NAME+type+'/'+difficulty+'/'):
                for filename in filenames:
                    if(filename != '.DS_Store'):
                        count+=1
    return count
    
    
    
def extract_images(SOURCE_FOLDER_NAME, DATA_TYPES, hard_difficulty = False):
    width = 6
    height = 4
    size = width*height*3;
    count = count_images(SOURCE_FOLDER_NAME, DATA_TYPES, hard_difficulty)
    images = np.zeros((count,size))
    DIFICULTIES_INCLUDED = ['Normal']
    if (hard_difficulty):
        DIFICULTIES_INCLUDED.append('Hard')
    # and the correct values
    categories = len(DATA_TYPES)
    correct_vals = np.zeros((count,categories))
    i = 0
    for type_index in range(categories):
        for difficulty in DIFICULTIES_INCLUDED:
            for (dirpath, dirnames, filenames) in walk(SOURCE_FOLDER_NAME+DATA_TYPES[type_index]+'/'+difficulty+'/'):
                for filename in filenames:
                    if(filename != '.DS_Store'):
                        img = cv2.imread(dirpath + '/' + filename, cv2.IMREAD_COLOR)    
                        img = cv2.resize(img, (width,height))  
                        img = img.flatten()    
                        images[i] = img / 255.0
                        correct_label = np.zeros((categories))
                        correct_label[type_index] = 1
                        correct_vals[i] = correct_label
                        i+=1
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
    print(images)
    print(labels)
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
    
def getPermutation(data_types, permutation_index = 0, hard_difficulty = False):
    if permutation_index >= 10 or permutation_index < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    PERMUTATION_FOLDER_NAME = "../../../Programming/Permutations/";
    SOURCE_FOLDER_NAME = "../../../Images/Dataset_400_300/";
    data_types = sorted(data_types)
    folder = PERMUTATION_FOLDER_NAME + data_types[0]
    for i in range(len(data_types) - 1):
        folder+='_' + data_types[i+1]
    if hard_difficulty:
        folder+='_Hard'
    else:
        folder+='_Normal'
    if not os.path.exists(folder):
        os.makedirs(folder)
        num_of_images = count_images(SOURCE_FOLDER_NAME, data_types, hard_difficulty)
        perm = np.arange(num_of_images)
        for i in range(10):
            fo = open(folder+"/permutation"+str(i)+".txt", "wb")
            np.random.shuffle(perm)
            for item in perm:
                fo.write("%s\n" % item)
    print(folder+"/permutation"+str(permutation_index)+".txt")
    with open(folder+"/permutation"+str(permutation_index)+".txt", 'r') as f:
        lines = f.read().splitlines()
        perm = []
        for i in range(len(lines)):
            perm.append(int(lines[i]))
        return perm    

def read_data_sets(data_permutation = 0, permutation_index = 3, hard_difficulty = False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()

  SOURCE_FOLDER_NAME = "../../../Images/Dataset_400_300/";
  DATA_TYPES = ['Blue','Green', 'White', 'Box', 'Can', 'Chemical', 'Colorful', 'Nothing']
  TEST_PERCENTAGE = 20
  VALIDATION_PERCENTAGE = 20

  images_and_labels = extract_images(SOURCE_FOLDER_NAME, DATA_TYPES, hard_difficulty)
  images = images_and_labels[0]
  labels = images_and_labels[1]
  
  perm = getPermutation(DATA_TYPES, permutation_index, False)
  images = images[perm]
  labels = labels[perm]
  
  TEST_SIZE = int(len(images) * (TEST_PERCENTAGE/100.0))
  TEST_SIZE = int(len(images) * (VALIDATION_PERCENTAGE/100.0))

  test_images = images[:TEST_SIZE]
  test_labels = labels[:TEST_SIZE]
  train_images = images[TEST_SIZE:]
  train_labels = labels[TEST_SIZE:]

  validation_images = train_images[:TEST_SIZE]
  validation_labels = train_labels[:TEST_SIZE]
  train_images = train_images[TEST_SIZE:]
  train_labels = train_labels[TEST_SIZE:]

  data_sets.train = MyDataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = MyDataSet(validation_images, validation_labels, dtype=dtype)
  data_sets.test = MyDataSet(test_images, test_labels, dtype=dtype)

  return data_sets


def load_mnist():
    return read_data_sets("MNIST_data")

