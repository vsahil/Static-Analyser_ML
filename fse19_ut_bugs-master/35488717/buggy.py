import tensorflow as tf
import time, sys
sys.path.append("../")

start_time = time.time()
sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 8, 8, 128]
strides = [1, 2, 2, 1]

l = tf.constant(0.1, shape=[batch_size, 32, 32, 4])
w = tf.constant(0.1, shape=[7, 7, 128, 4])

import check
h1 = tf.nn.conv2d_transpose(l, w, output_shape=output_shape, strides=strides, padding='SAME')
print(sess.run(h1))
