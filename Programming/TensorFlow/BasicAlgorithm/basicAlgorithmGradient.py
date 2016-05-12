import tensorflow as tf
import numpy as np
import cv2
from os import walk
import myDataset as mds

BATCH_SIZE = 476


i = 0

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

myset = mds.read_data_sets()
print myset.train.num_examples
print myset.test.num_examples
image_size = myset.train.images[0].shape[0]
categories = myset.train.labels[0].shape[0]
print categories

x = tf.placeholder(tf.float32, [None, image_size])

W = tf.Variable(tf.zeros([image_size, categories]))
b = tf.Variable(tf.zeros([categories]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, categories])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y + 1e-20), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.03).minimize(cross_entropy)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


tazkost = np.zeros((categories))
for label in myset.test.labels:
    for i in range(label.shape[0]):
        tazkost[i]+=label[i]
    
print(tazkost)

# Fit the line.
for step in range(10000):
  batch_xs, batch_ys = myset.train.next_batch(BATCH_SIZE)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if step % 200 == 0:
    print('Step %d (epoch %.2f)' %
              (step, float(step) * BATCH_SIZE / myset.train.num_examples))
    print('Validation error: %.1f%%' % (100 - float(str(sess.run(accuracy, feed_dict={x: myset.validation.images, y_: myset.validation.labels}))) * 100))
    #print(sess.run(y, feed_dict={x: myset.test.images, y_: myset.test.labels}))
# Finally print the result!
test_error = (1 - sess.run(accuracy, feed_dict={x: myset.test.images, y_: myset.test.labels})) * 100
print('Test error: %.1f%%' % test_error)