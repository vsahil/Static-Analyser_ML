import tensorflow as tf
import tensorflow.contrib.opt as opt

X = tf.Variable([1.0, 2.0])

part_X = tf.scatter_nd([[0]], [X[0]], [2])

X_2 = part_X + tf.stop_gradient(-part_X + X)

Y = tf.constant([2.0, -3.0])

loss = tf.reduce_sum(tf.squared_difference(X_2, Y))

opt = opt.ScipyOptimizerInterface(loss, [X])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    opt.minimize(sess)
    print("X: {}".format(X.eval()))
