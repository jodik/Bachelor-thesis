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
from cobs import cobs
import array
from sets import Set
import math

import numpy as np
import tensorflow as tf
import sys
sys.path.append("../")
import configuration as conf

def count_images():
    return extract_images()[0].shape[0]
    
def getLabel(label, data_types):
     data_types = sorted(data_types)
     for i in range(len(data_types)):
         if data_types[i] in label:
             return i
     raise ValueError('Bad path.')

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
    
def extract_images():
    correct_vals = np.zeros((0))
    print('test')
    with open (conf.SOURCE_FOLDER_NAME + 'data.byte', "rb") as readdata:
        data = cobs.decode(readdata.read())
        all_images = array.array('B',data)
        num_of_images = int(len(all_images)/(conf.IMAGE_WIDTH*conf.IMAGE_HEIGHT*conf.NUM_CHANNELS))
        all_images = np.asarray(all_images, dtype=np.uint8)
        all_images = all_images.reshape(num_of_images, conf.IMAGE_HEIGHT, conf.IMAGE_WIDTH, conf.NUM_CHANNELS)
        all_images = all_images / conf.PIXEL_DEPTH - 0.5
    print('test2') 
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
        label = conf.ALL_DATA_TYPES[labels[i]]
        if label in conf.DATA_TYPES_USED and (ishard[i] == 0 or conf.HARD_DIFFICULTY):
            category = getLabel(label, conf.DATA_TYPES_USED)
            correct_vals = np.append(correct_vals, [category])
            images[size] = all_images[i]
            names_chosen.append(names[i])
            size = size + 1
    images=images[0:size]
    print('test4')
    #for i in range(count):
     #   images[i] = images[i].reshape(width, height, channels)
    return (images, correct_vals, np.array(names_chosen))



def getPermutation(permutation_index, num_of_images_total):
    if permutation_index >= 10 or permutation_index < 0:
        raise ValueError('Permutation index should not be larger than 9 and lower than 0')
    folder = conf.PERMUTATION_FOLDER_NAME + conf.DATA_TYPES_USED[0]
    for i in range(conf.NUM_LABELS - 1):
        folder+='_' + conf.DATA_TYPES_USED[i+1]
    if conf.HARD_DIFFICULTY:
        folder+='_Hard'
    else:
        folder+='_Normal'
    if not os.path.exists(folder):
        os.makedirs(folder)
        
        perm = np.arange(num_of_images_total)
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
  data_sets = DataSets()

  
  images, labels, names = extract_images()
  TEST_SIZE = int(images.shape[0]/8 * (conf.TEST_PERCENTAGE/100.0))
  VALIDATION_SIZE = int(images.shape[0]/8 * (conf.VALIDATION_PERCENTAGE/100.0))
  
  perm2 = getPermutation(permutation_index, int(images.shape[0]/8))
  print(len(perm2))
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
  
  
  data_sets.train = MyDataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = MyDataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = MyDataSet(test_images, test_labels, dtype=dtype)

  return data_sets

