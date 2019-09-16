import tensorflow as tf
import tensorflow.contrib.opt as opt

X = tf.Variable([1.0, 2.0])
X0 = tf.Variable([3.0])

Y = tf.constant([2.0, -3.0])

scatter = tf.scatter_update(X, [0], X0)

with tf.control_dependencies([scatter]):
    loss = tf.reduce_sum(tf.squared_difference(X, Y))

opt = opt.ScipyOptimizerInterface(loss, [X0])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    opt.minimize(sess)

    print("X: {}".format(X.eval()))
    print("X0: {}".format(X0.eval()))
