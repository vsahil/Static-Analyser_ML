import start as our
# import tensorflow as tf

# assert tf.__version__ == "1.8.0"

# tf.set_random_seed(20180130)

# 

y = our.rand_normal_([2, 2], stddev=0.35)

x = our.var(y)

# print(x, y)

# init_op = tf.initialize_all_variables()

# sess = tf.Session(init_op)

# print(sess.run(x))
