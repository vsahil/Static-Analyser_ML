import tensorflow as tf
import numpy as np
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import array_ops

with tf.Graph().as_default(), tf.Session() as sess:
  initial_m = tf.Variable(0.0, name='m')

  #The code no longer works after I change shape=(4) to shape=(None)
  inputs = tf.placeholder(dtype='float32', shape=(None))

  time_steps = tf.shape(inputs)[0]

  initial_outputs = tf.TensorArray(dtype=tf.float32, size=time_steps)
  initial_t = tf.constant(0, dtype='int32')

  def should_continue(t, *args):
    return t < time_steps

  def iteration(t, m, outputs_):
    cur = tf.gather(inputs, t)
    m  = m * 0.5 + cur * 0.5
    outputs_ = outputs_.write(t, m)
    return t + 1, m, outputs_

  t, m, outputs = tf.while_loop(
    should_continue, iteration,
    [initial_t, initial_m, initial_outputs])

  outputs = outputs.stack()
  init = tf.global_variables_initializer()
  sess.run([init])
  print(sess.run([outputs], feed_dict={inputs: np.asarray([1,1,1,1])}))
