import tensorflow as tf
import numpy as np
import time

start = time.time()
weight_tensor = tf.truncated_normal([227,227],**{'stddev':0.1,'mean':0.0})
weight_var = tf.Variable(weight_tensor)
batch_size = 1
prev_net_2d = tf.placeholder(tf.float32, shape=(batch_size, 227, 227))
my = tf.expand_dims(weight_var, axis=0)
weight_var_batch_size = tf.tile(my, [batch_size, 1, 1])
matrix = tf.matmul(prev_net_2d,weight_var_batch_size)
end1 = time.time()
print("Checkpoint model", end1-start)