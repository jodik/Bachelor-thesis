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

from Programming.DataScripts import data_reader
import numpy as np
import tensorflow as tf

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


def normalize(data_set):
    num_of_pixels = 0
    image_sets = data_set.getImageSets()
    sum_rgb = np.sum(np.sum(s, axis=(0, 1, 2)) for s in image_sets)

    for image_set in image_sets:
        num_of_pixels += image_set.size/3

    sum_rgb /= num_of_pixels
    for image_set in image_sets:
        image_set -= sum_rgb

    sum_rgb_pow = np.zeros(3, dtype=np.float64)
    for image_set in image_sets:
        tmp = np.apply_along_axis(lambda x: np.power(x, 2), 2, image_set)
        sum_rgb_pow += np.sum(tmp, axis=(0, 1, 2))

    c = num_of_pixels/sum_rgb_pow
    c = np.sqrt(c)
    for image_set in image_sets:
        image_set *= c

    return image_sets


def equalCountsPerms(labels):
    counts = np.bincount(labels)
    counts = np.full(len(counts), max(counts), dtype=np.uint32)
    res = np.zeros(0, dtype=np.uint8)
    while max(counts) > 0:
        for i in range(len(labels)):
            label = labels[i]
            if counts[label] > 0:
                res = np.append(res, i)
                counts[label] -= 1
    np.random.shuffle(res)
    return res


def read_data_sets(permutation_index, dtype=tf.float32):
  class DataSets(object):
    pass

  data_sets = DataSets()

  data_set = data_reader.read_datasets(permutation_index)

  train_images, validation_images, test_images = normalize(data_set)
  train_labels, validation_labels, test_labels = data_set.train.labels, data_set.validation.labels, data_set.test.labels

  train_perm, validation_perm, test_perm = map(lambda x: equalCountsPerms(x), (train_labels, validation_labels, test_labels))
  train_images = train_images[train_perm]
  train_labels = train_labels[train_perm]
  validation_images = validation_images[validation_perm]
  validation_labels = validation_labels[validation_perm]
  test_images = test_images[test_perm]
  test_labels = test_labels[test_perm]


  data_sets.train = MyDataSet(train_images, train_labels, dtype=dtype)
  data_sets.validation = MyDataSet(validation_images, validation_labels,
                                 dtype=dtype)
  data_sets.test = MyDataSet(test_images, test_labels, dtype=dtype)

  return data_sets

