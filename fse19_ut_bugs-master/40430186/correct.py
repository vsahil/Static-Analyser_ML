import tensorflow as tf
import math, time
import numpy as np
from PIL import Image
from numpy import array

start = time.time()
# image parameters
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
NUM_CLASSES = 2

def main():
    image = np.zeros((64, 64, 3))
    img = Image.open('./IMG_0849.JPG')

    img = img.resize((64, 64))
    image = array(img).reshape(1, 64, 64, 3)

    k = int(math.ceil(IMAGE_SIZE / 2.0 / 2.0 / 2.0 / 2.0))
    # Store weights for our convolution and fully-connected layers
    with tf.name_scope('weights'):
        weights = {
            # 5x5 conv, 3 input channel, 32 outputs each
            'wc1': tf.Variable(tf.random_normal([5, 5, 1 * IMAGE_CHANNELS, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # 5x5 conv, 64 inputs, 128 outputs
            'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
            # 5x5 conv, 128 inputs, 256 outputs
            'wc4': tf.Variable(tf.random_normal([5, 5, 128, 256])),
            # fully connected, k * k * 256 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([k * k * 256, 1024])),
            # 1024 inputs, 2 class labels (prediction)
            'out': tf.Variable(tf.random_normal([1024, NUM_CLASSES]))
        }

    # Store biases for our convolution and fully-connected layers
    with tf.name_scope('biases'):
        biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bc3': tf.Variable(tf.random_normal([128])),
            'bc4': tf.Variable(tf.random_normal([256])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([NUM_CLASSES]))
        }
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #saver.restore(sess, "./model.ckpt")
        #print "...Model Loaded..."
        x_ = tf.placeholder(tf.int32, shape=[None, IMAGE_SIZE , IMAGE_SIZE , IMAGE_CHANNELS])
        y_ = tf.placeholder(tf.int32, shape=[None, NUM_CLASSES])
        keep_prob = tf.placeholder(tf.float32)

        init = tf.initialize_all_variables()

        sess.run(init)
        my_classification = sess.run(x_, feed_dict={x_:image})
        print('Neural Network predicted', my_classification[0], "for your image")


if __name__ == '__main__':
     main()

end2 = time.time()
print("Checkpoint model", end2-start)