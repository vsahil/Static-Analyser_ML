import tensorflow as tf
input = tf.placeholder(tf.float32, shape=(None,1))

# `tf.shape(input)` takes the dynamic shape of `input`.
t = tf.fill(tf.shape(input), 0.5)
