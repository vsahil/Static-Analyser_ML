import tensorflow as tf
import numpy as np

from tqdm import tqdm

# The width and height of the input images that we will be classifying.
WIDTH = 224
HEIGHT = 224

N_INPUT = WIDTH * HEIGHT * 3
N_CLASSES = 10
P_DROPOUT = 0.75

LEARNING_RATE = 0.001
TRAINING_ITERS = 1000
BATCH_SIZE = 1

# tf Graph input
x = tf.placeholder(tf.float32, [None, N_INPUT])
y = tf.placeholder(tf.float32, [None, N_CLASSES])
keepProb = tf.placeholder(tf.float32) # dropout (keep probability)

def conv2d(img, w, b, s) :
    '''
    Create a convolutional layer with weights @w and biases @b
    '''
    conv = tf.nn.conv2d(img, w, strides=[1, s, s, 1], padding='SAME')
    bias = tf.nn.bias_add(conv, b)
    relu = tf.nn.relu(bias)

    return relu

def maxPool(img, k, s) :
    '''
    Create a max-pooling layer
    '''
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding='SAME')

def norm(layer, lsize) :
    return tf.nn.lrn(layer, lsize, bias=2.0, alpha=0.001 / 9.0, beta=0.75)

def convolutionalNet(img, _weights, _biases, _dropout) :
    # First, reshape the input layer to match the dimensions of the images we
    # plan to classify.
    _X = tf.reshape(img, shape=[-1, WIDTH, HEIGHT, 3])

    # 1. Convolution layer, max-pooling, then dropout.
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'], 4)
    print "conv1.shape:", conv1.get_shape()
    pool1 = maxPool(conv1, 3, 2)
    print "pool1.shape:", pool1.get_shape()
    norm1 = norm(pool1, 4)
    print "norm1.shape:", norm1.get_shape()
    dropout1 = tf.nn.dropout(norm1, _dropout) # Is dropout necessary here?
    print "dropout1.shape:", dropout1.get_shape()

    # 2. Second conv layer, setup identical
    conv2 = conv2d(dropout1, _weights['wc2'], _biases['bc2'], 1)
    print "conv2.shape:", conv2.get_shape()
    pool2 = maxPool(conv2, 3, 2)
    print "pool2.shape:", pool2.get_shape()
    norm2 = norm(pool2, 4)
    print "norm2.shape:", norm2.get_shape()
    dropout2 = tf.nn.dropout(norm2, _dropout) # Is dropout necessary here?
    print "dropout2.shape:", dropout2.get_shape()

    # 3. Third conv layer
    conv3 = conv2d(dropout2, _weights['wc3'], _biases['bc3'], 1)
    print "conv3.shape:", conv3.get_shape()

    # 4. Fourth conv layer
    conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'], 1)
    print "conv4.shape:", conv4.get_shape()

    # 5. Fifth conv layer
    conv5 = conv2d(conv4, _weights['wc5'], _biases['bc5'], 1)
    print "conv5.shape:", conv5.get_shape()
    pool5 = maxPool(conv5, 3, 2)
    print "pool5.shape:", pool5.get_shape()

    # 6. Fully connected layer 1
    wd1Shape = _weights['wd1'].get_shape().as_list()
    print wd1Shape
    fc1 = tf.reshape(pool5, [-1, wd1Shape[0]])
    print "fc1.shape:", fc1.get_shape()

    mul1 = tf.matmul(fc1, _weights['wd1']) + _biases['bd1']
    print "mul1.shape:", mul1.get_shape()
    relu1 = tf.nn.relu(mul1)
    print "relu1.shape:", relu1.get_shape()

    # 7. Fully connected layer 2
    mul2 = tf.matmul(relu1, _weights['wd2']) + _biases['bd2']
    print "mul2.shape:", mul2.get_shape()
    relu2 = tf.nn.relu(mul2)
    print "relu2.shape:", relu2.get_shape()

    # Output layer
    out = tf.matmul(relu2, _weights['out']) + _biases['out']
    print "out.shape:", out.get_shape()
    print "out: ", out

    return out

def buildModel() :
    # Store layers weight & bias
    weights = { # AKA Kernels
        'wc1': tf.Variable(tf.random_normal([11, 11, 3, 96])),
        'wc2': tf.Variable(tf.random_normal([5, 5, 96, 192])),
        'wc3': tf.Variable(tf.random_normal([3, 3, 192, 384])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384])),
        'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256])),

        'wd1': tf.Variable(tf.random_normal([256, 4096])),
        'wd2': tf.Variable(tf.random_normal([4096, 1024])),

        'out': tf.Variable(tf.random_normal([1024, N_CLASSES]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([96])),
        'bc2': tf.Variable(tf.random_normal([192])),
        'bc3': tf.Variable(tf.random_normal([384])),
        'bc4': tf.Variable(tf.random_normal([384])),
        'bc5': tf.Variable(tf.random_normal([256])),

        'bd1': tf.Variable(tf.random_normal([4096])),
        'bd2': tf.Variable(tf.random_normal([1024])),

        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    result = convolutionalNet(x, weights, biases, keepProb)

    return result

def genData() :
    # Generate dummy data for standalone test-case
    labels = []
    images = []
    for i in tqdm(range(TRAINING_ITERS)) :
        images.append(np.ndarray(WIDTH * HEIGHT * 3, dtype='float32'))

        onehot = np.zeros(N_CLASSES)
        onehot[i % N_CLASSES] = 1.0
        labels.append(onehot)

    return zip(labels, images)

def batchSequence(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))

def main() :
    # Build the model
    pred = buildModel()

    # Define loss function and optimizer
    softmax = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
    cost = tf.reduce_mean(softmax)

    print "cost:", cost

    print "pred:", pred
    print "y:", y
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    print "optimizer:", optimizer

    # XXX Model evaluation

    init = tf.initialize_all_variables()

    data = genData()
    batches = batchSequence(data, BATCH_SIZE)

    with tf.Session() as sess :
        sess.run(init)

        iterations = 0
        for batch in batches :
            labels, images = [ list(l) for l in zip(*batch) ]

            batchLabels = np.ndarray([ len(labels), N_CLASSES ])
            for i, l in enumerate(labels) :
                batchLabels[i] = l

            batchImages = np.ndarray([ len(images), WIDTH * HEIGHT * 3 ])
            for i, img in enumerate(images) :
                batchImages[i] = img

            sess.run(optimizer, feed_dict={x: batchImages, y: batchLabels, keepProb: P_DROPOUT})
            print iterations
            print "x.shape", x.shape,  "y.shape" ,y.shape

            iterations += len(labels)
            if iterations >= TRAINING_ITERS :
                break

if __name__ == '__main__' :
    main()
