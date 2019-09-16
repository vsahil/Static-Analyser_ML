import tensorflow as tf
import numpy as np
import time

start = time.time()
data = np.array([[0.1], [0.2]])
x = tf.placeholder(tf.float32, shape=[2, 1])
T1 = tf.Variable(tf.ones([2, 2]))
l1 = tf.matmul(T1, x)
init = tf.initialize_all_variables()
end1 = time.time()
print("Checkpoint model", end1-start)
with tf.Session() as sess:
    sess.run(init)
    sess.run(l1, feed_dict={x: data})
end2 = time.time()
print("Checkpoint eval", end2-end1)