import tensorflow as tf
import time

start = time.time()
sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 8, 8, 128]
strides = [1, 2, 2, 1]

l = tf.constant(0.1, shape=[batch_size, 4, 4, 4])
w = tf.constant(0.1, shape=[7, 7, 128, 4])

h1 = tf.nn.conv2d_transpose(l, w, output_shape=output_shape, strides=strides, padding='SAME')
end1 = time.time()
print("Checkpoint model", end1-start)
print(sess.run(h1))
end2 = time.time()
print("Checkpoint eval", end2-end1)
