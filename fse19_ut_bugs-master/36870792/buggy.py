import tensorflow as tf
import numpy as np
import time, sys
sys.path.append("../")

start_time = time.time()
weight_tensor = tf.truncated_normal([227,227],**{'stddev':0.1,'mean':0.0})
weight_var = tf.Variable(weight_tensor)
prev_net_2d = tf.placeholder(tf.float32, shape=(None, 227, 227))
import check
matrix = tf.matmul(prev_net_2d,weight_var)

