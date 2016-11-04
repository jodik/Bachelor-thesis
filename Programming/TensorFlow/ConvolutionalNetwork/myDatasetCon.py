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
from cobs import cobs
import array
from sets import Set

import numpy as np
import tensorflow as tf
import Programming.TensorFlow.configuration as conf
from Programming.HelperScripts import helper

def count_images():
    return extract_images()[0].shape[0]
    
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


g = 0


def white_or_black(x):
    global g
    g += 1
    if g%2 == 1:
        return np.zeros(3) + 255
    return x

def extract_images():
    correct_vals = np.zeros((0))
    with open (conf.SOURCE_FOLDER_NAME + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        all_images = array.array('B',data)
        num_of_images = int(len(all_images)/(conf.IMAGE_WIDTH*conf.IMAGE_HEIGHT*conf.NUM_CHANNELS))
        all_images = np.asarray(all_images, dtype=np.uint8)
        all_images = all_images.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS)
       # all_images[np.sum(all_images, axis = 3) < 10] = np.zeros(3)
        #all_images = np.apply_along_axis(white_or_black, axis = 3, arr=all_images)
        print(all_images[2][1])
        all_images = all_images / conf.PIXEL_DEPTH - 0.5
    with open (conf.SOURCE_FOLDER_NAME + 'labels.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        labels = array.array('B',data)
        labels = np.asarray(labels, dtype=np.uint8)
        
    with open (conf.SOURCE_FOLDER_NAME + 'ishard.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        ishard = array.array('B',data)
        ishard = np.asarray(ishard, dtype=np.uint8)
    names = []
    with open (conf.SOURCE_FOLDER_NAME + 'names.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        l = len("IMG_0519.JPG")
        for i in range(num_of_images):
            names.append(data[i*l:(i+1)*l])
    print('test3') 
    print(num_of_images) 
    images = np.zeros((num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH , conf.NUM_CHANNELS))
    size = 0
    names_chosen = []
    for i in range(num_of_images):
        label_word = conf.ALL_DATA_TYPES[labels[i]]
        if label_word in conf.DATA_TYPES_USED and (ishard[i] == 0 or conf.HARD_DIFFICULTY):
            category = helper.getLabelIndex(label_word, conf.DATA_TYPES_USED)
            correct_vals = np.append(correct_vals, [category])
            images[size] = all_images[i]
            names_chosen.append(names[i])
            size = size + 1
    images=images[0:size]
    print('test4')
    #for i in range(count):
     #   images[i] = images[i].reshape(width, height, channels)
    return (images, correct_vals, np.array(names_chosen))



def getPermutation(permutation_index, labels, validation_size, test_size):
    num_of_images_total = len(labels)
    if permutation_index >= 10 or permutation_index < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    perm = np.arange(num_of_images_total)
    for i in range(permutation_index + 1):
        np.random.shuffle(perm)

    percentage = validation_size / float(num_of_images_total)
    counts = np.array(percentage * np.bincount(np.array(labels, dtype=int)), dtype=int)
    to_add = validation_size - sum(counts)
    for i in np.arange(counts.shape[0])[:to_add]:
        counts[i] += 1
    a = np.zeros(0, dtype=int)
    b = np.zeros(0, dtype=int)
    c = np.zeros(0, dtype=int)
    d = np.zeros(0, dtype=int)
    counts_tmp = np.copy(counts)
    for t in perm:
        if counts[labels[t]] > 0:
            counts[labels[t]]-=1
            a = np.append(a, t)
        else:
            b = np.append(b, t)
    counts = counts_tmp
    print(counts)
    for t in b:
        if counts[labels[t]] > 0:
            counts[labels[t]]-=1
            c = np.append(c, t)
        else:
            d = np.append(d, t)

    print (np.append(a, np.append(c, d)))
    return np.append(a, np.append(c, d))

def filterAndCreateTrainSet(validation_names, test_names, images, labels, names):
    size = 0
    print('LEN: '+str(len(Set(names))))
    for i in range(images.shape[0]):
        if (names[i] not in validation_names) and (names[i] not in test_names):
            images[size] = images[i]
            labels[size] = labels[i]
            names[size] = names[i]
            size+=1
    perm = np.arange(size)
    np.random.shuffle(perm)
    images = images[perm]
    labels = labels[perm]
    names = names[perm]
    print('LEN: '+str(len(Set(names))))
    return (images, labels, names)

def normalize(train_images, validation_images, test_images):
    images = (train_images, validation_images, test_images)
    n = 0
    sum_rgb = np.zeros(3, dtype=np.float64)

    for image_set in images:
        n += image_set.size/3
        for image in image_set:
            for col_i in image:
                for pixel in col_i:
                    sum_rgb += pixel

    sum_rgb /= n
    #sum_rgb = (math.sqrt(n)/np.sqrt(sum_rgb))
    train_images -= sum_rgb
    test_images -= sum_rgb
    validation_images -= sum_rgb

    sum_rgb_pow = np.zeros(3, dtype=np.float64)
    images = (train_images, validation_images, test_images)
    for image_set in images:
        n += image_set.size/3
        for image in image_set:
            for col_i in image:
                for pixel in col_i:
                    color_pow = np.multiply(pixel, pixel)
                    sum_rgb_pow += color_pow

    c = n/sum_rgb_pow
    c = np.sqrt(c)
    print('c')
    print(c)
    validation_images *= c
    train_images *= c
    test_images *= c

    print (validation_images[0, 15])
    return train_images, validation_images, test_images

def read_data_sets(permutation_index = 0,one_hot=True, dtype=tf.float32):
  class DataSets(object):
    pass

  np.random.seed(conf.SEED)
  data_sets = DataSets()

  
  images, labels, names = extract_images()
  TEST_SIZE = int(images.shape[0]/8 * (conf.TEST_PERCENTAGE/100.0))
  VALIDATION_SIZE = int(images.shape[0]/8 * (conf.VALIDATION_PERCENTAGE/100.0))

  original_set_size = int(images.shape[0]/8)
  perm2 = getPermutation(permutation_index, labels[:original_set_size], VALIDATION_SIZE, TEST_SIZE)
  print(images.shape[0])
  print(labels.shape)
  print(names.shape)
  images_original_set = images[perm2]
  labels_original_set = labels[perm2]
  names_original_set = names[perm2]
  

  test_images = images_original_set[:TEST_SIZE, ...]
  test_labels = labels_original_set[:TEST_SIZE]
  test_names = names_original_set[:TEST_SIZE]
  images_original_set = images_original_set[TEST_SIZE:, ...]
  labels_original_set = labels_original_set[TEST_SIZE:]
  names_original_set = names_original_set[TEST_SIZE:]
  
  validation_images = images_original_set[:VALIDATION_SIZE, ...]
  validation_labels = labels_original_set[:VALIDATION_SIZE]
  validation_names = names_original_set[:VALIDATION_SIZE]
  images_original_set = images_original_set[VALIDATION_SIZE:, ...]
  labels_original_set = labels_original_set[VALIDATION_SIZE:]
  names_original_set = names_original_set[VALIDATION_SIZE:]
  
  
  if conf.EXTENDED_DATASET:
      train_images, train_labels, train_names = filterAndCreateTrainSet(validation_names, test_names, images, labels, names)
  else:
      train_images, train_labels, train_names = images_original_set, labels_original_set, names_original_set
  
  # Should be fixed, rozptyl 1, stred 0, somtehing went wrong
  train_images, validation_images, test_images = normalize(train_images, validation_images, test_images)
  #print(validation_images[0, 16:, 15])
  print (labels.shape)
  
  print(train_images.shape)
  print(validation_images.shape)
  print(test_images.shape)
  data_sets.train = MyDataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = MyDataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = MyDataSet(test_images, test_labels, dtype=dtype)

  return data_sets

