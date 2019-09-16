import tensorflow as tf
import numpy as np
import time, sys
sys.path.append("../")

start_time = time.time()
data = np.array([[0.1], [0.2]])
x = tf.placeholder(tf.float32, shape=[2])
T1 = tf.Variable(tf.ones([2, 2]))
import check
l1 = tf.matmul(T1, x)
init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    sess.run(l1, feed_dict={x: data})
