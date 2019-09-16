import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import conv2d_transpose, sigmoid, sparse_softmax_cross_entropy_with_logits
import time

start = time.time()
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
CHANNELS = 1

batch_size = 8
num_labels = 2

in_data = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
labels = tf.placeholder(tf.int32, shape=(batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 1))

# Variables
w0 = tf.Variable(tf.truncated_normal([3, 3, CHANNELS, 32]))
b0 = tf.Variable(tf.zeros([32]))

# Down sample
conv_0 = tf.nn.relu(tf.nn.conv2d(in_data, w0, [1, 2, 2, 1], padding='SAME') + b0)
print("Convolution 0:", conv_0)


# Up sample 1. Upscale to 128 x 128 x 24
wt1 = tf.Variable(tf.truncated_normal([3, 3, 24, 32]))
convt_1 = sigmoid(
          conv2d_transpose(conv_0,
                                 filter=wt1,
                                 output_shape=[batch_size, 128, 128, 24],
                                 strides=[1, 1, 1, 1]))
print("Deconvolution 1:", convt_1)


# Up sample 2. Upscale to 256 x 256 x 2
wt2 = tf.Variable(tf.truncated_normal([3, 3, 2, 24]))
convt_2 = sigmoid(
          conv2d_transpose(convt_1,
                                 filter=wt2,
                                 output_shape=[batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 2],
                                 strides=[1, 2, 2, 1]))
print("Deconvolution 2:", convt_2)
end1 = time.time()
print("Checkpoint model", end1-start)

# Loss computation
logits = tf.reshape(convt_2, [-1, num_labels])
reshaped_labels = tf.reshape(labels, [-1])
cross_entropy = sparse_softmax_cross_entropy_with_logits(logits=logits, labels=reshaped_labels)
loss = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
end2 = time.time()
print("Checkpoint eval", end2-end1)