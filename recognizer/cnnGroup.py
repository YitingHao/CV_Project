
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('test_data_path', sys.argv[1], \
                           """file path of test data """)
tf.app.flags.DEFINE_string('modelPath', sys.argv[2], """file path of model """)
tf.app.flags.DEFINE_string('saveFile', sys.argv[3], """file path to save prediction results """)

IMAGE_SIZE = 32
NUM_CHANNELS = 1
NUM_LABELS = 2
VALIDATION_SIZE = 1400  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 1

def cnn(argv=None):

  test_data = numpy.load(FLAGS.test_data_path)

  eval_data = tf.placeholder(
      tf.float32,
      shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

  # The variables below hold all the trainable weights.
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=0.01,
                          seed=SEED, dtype=tf.float32))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
  conv2_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=0.01,
      seed=SEED, dtype=tf.float32))
  conv2_biases = tf.Variable(tf.zeros([64], dtype=tf.float32))
  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                          stddev=0.01,
                          seed=SEED,
                          dtype=tf.float32))
  fc1_biases = tf.Variable(tf.zeros([512], dtype=tf.float32))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.01,
                                                seed=SEED,
                                                dtype=tf.float32))
  fc2_biases = tf.Variable(tf.zeros([NUM_LABELS], dtype=tf.float32))


  def model(data, train=False):
    """The Model definition."""
    # 2D convolution, with 'SAME' padding
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    # Max pooling
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    # Add a 50% dropout during training only. 
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and sparse labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])

  # Create a local session to run the training.
  with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.modelPath)
    predictions = eval_in_batches(test_data, sess)
    # print(predictions.shape)
    # print(type(predictions))
    numpy.save(FLAGS.saveFile, predictions)

tf.app.run(main=cnn)


# def cnn_predict(test_data_path, modelPath, saveFile, numLabels):
#   tf.app.flags.DEFINE_string('test_data_path', test_data_path, \
#                              """file path of test data """)
#   tf.app.flags.DEFINE_string('modelPath', modelPath, """file path of model """)
#   tf.app.flags.DEFINE_string('saveFile', saveFile, """file path to save prediction results """)
#   tf.app.flags.DEFINE_integer('numLabels', numLabels, """number of classes """)
#   tf.app.run(main=cnn)
