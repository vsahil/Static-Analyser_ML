#! /usr/bin/env python

"""A simple MNIST classifier which displays summaries in TensorBoard.
Access Tensorboard (use the log directory):
tensorboard --logdir=/tmp/tensorflow/mnist/logs --port 6006
This is an unimpressive MNIST model, but it is a good example of using
tf.name_scope to make a graph legible in the TensorBoard graph explorer, and of
naming summary tags so that they are grouped meaningfully in TensorBoard.
It demonstrates the functionality of every TensorBoard dashboard.
"""
# Compatibility to python 2
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
#from builtins import *

import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


class MnistIterator(object):
    """ A simple data iterator which loads the mnist dataset into ram and provides
    methods for batch training and test evaluation. """
    mnist = None

    def __init__(self, batch_size=100, data_dir='/tmp/tensorflow/mnist/input_data'):
        """ Invokes the setup method which triggers dataset downloads

        :param batch_size: batch size used during training
        :param data_dir: directory to store the data
        """
        self.batch_size = batch_size
        self.data_dir = data_dir

        self._setup()

    def _setup(self):
        """ Import data, downloads if not existent yet. """
        mnist = input_data.read_data_sets(self.data_dir, one_hot=True)
        self.mnist = mnist

    def train_batch(self):
        """ proved the next training batch as an infinite iterator. """
        assert self.mnist is not None

        while True:
            xs, ys = self.mnist.train.next_batch(batch_size=self.batch_size)

            yield xs, ys

    def test_data(self):
        """ Provides the training data as a single data chunk. For larger
        datasets this should be an iterator too. """
        xs, ys = self.mnist.test.images, self.mnist.test.labels

        return xs, ys


class MnistNet(object):
    """ Simple fully connected network for classification. Utilizes dropout. """

    def __init__(self, layers, keep_proba=1., input_dim=784, output_dim=10):
        """ Define internal parameters and default parameters

        :param layers: list or tuple of layer units, for each entry a layer is created
        :param keep_proba: in [0,1] defining the probability of a node dropout
        :param input_dim: number of input features (default are mnist values)
        :param output_dim: number of output classes (default are mnist values)
        """
        assert isinstance(layers, list) or isinstance(layers, tuple)

        self.layers = layers
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.keep_probability = keep_proba
        # define the learning rate as a tensorflow variable (current graph)
        self.learning_rate = tf.Variable(learning_rate,
                                         dtype=tf.float32,
                                         name='learning_rate',
                                         trainable=False)

    #### custom building elements
    def weight_variable(self, shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """Create a bias variable with appropriate initialization."""
        # We can't initialize these variables to 0 - the network will get stuck.
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def nn_layer(self, input_tensor, output_dim,
                 layer_name, keep_prob=None, act=tf.nn.relu):
        """ Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to
        nonlinearize. It also sets up name scoping so that the resultant
        graph is easy to read, and adds a number of summary ops.

        :param input_tensor: previous layer (tensor)
        :param output_dim: number of units in this layer
        :param layer_name: name of this layer for name scoping
        :param keep_prob: keep probability during dropout, if None, no dropout is added
        :param act: tensorflow activation function
        :return: output tensor after all operations are applied
        """
        input_shape = input_tensor.get_shape().as_list()
        shape_fcn = [np.prod(input_shape[1:]), output_dim]
        print("shape_fcn:", shape_fcn)
        # Adding a name scope ensures logical grouping of the layers in the
        # graph.
        with tf.name_scope(layer_name):
            # This Variable will hold the state of the weights for the layer
            with tf.name_scope('weights'):
                weights = self.weight_variable(shape_fcn)
                # self.variable_summaries(weights)
            with tf.name_scope('biases'):
                biases = self.bias_variable([output_dim])
                # self.variable_summaries(biases)
                print(input_tensor, weights, "HELLO")
                preactivate = tf.matmul(input_tensor, weights) + biases
                tf.summary.histogram('pre_activations', preactivate)

            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)

            if keep_prob is not None:
                with tf.name_scope('dropout'):
                    activations = tf.nn.dropout(activations, keep_prob)

        return activations

    #### building the network
    def build_graph(self):
        """ Build the computation graph. This covers input placeholders, the network
        architecture as well as loss function, training operation and summaries. """

        # Input placeholders
        with tf.name_scope('input'):
            keep_prob_default = tf.Variable(self.keep_probability,
                                            name='keep_prob_default',
                                            dtype=tf.float32,  # default dropout param
                                            trainable=False)

            x = tf.placeholder(tf.float32, [None, self.input_dim], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, self.output_dim], name='y-input')
            keep_prob = tf.placeholder_with_default(keep_prob_default, shape=[], name='keep_prob')

        # build layers
        layer = x
        for i, la in enumerate(self.layers):
            layer = self.nn_layer(input_tensor=layer,
                                  output_dim=la,
                                  layer_name='layer%d' % i,
                                  keep_prob=keep_prob,
                                  act=tf.nn.relu)

        # last layer. Do not apply softmax activation yet, see below.
        y = self.nn_layer(input_tensor=layer,
                          output_dim=self.output_dim,
                          layer_name='last_layer',
                          keep_prob=None,  # no dropout here
                          act=tf.identity)
        exit()
        # loss function
        with tf.name_scope('cross_entropy'):
            # The raw formulation of cross-entropy, can be numerically unstable.
            #
            # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
            #                               reduction_indices=[1]))
            #
            # So here we use tf.nn.softmax_cross_entropy_with_logits on the
            # raw outputs of the nn_layer above, and then average across
            # the batch.
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)  # single scalar loss value
                tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar('accuracy', accuracy)

        # summaries
        with tf.name_scope('input_reshape'):
            image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])  # single channel input images for visualization
            tf.summary.image('input', image_shaped_input, 10)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged_summary = tf.summary.merge_all()

        # store local variables
        self.x = x
        self.y_ = y_
        self.keep_prob = keep_prob
        self.train_op = train_op
        self.y = y
        self.accuracy = accuracy
        self.merged_summary = merged_summary

    def train(self, session, data_iterator, log_dir, max_steps=1000, learning_rate=0.001):
        """ Method for training the network with specific training parameters

        :param session: currently open session with loaded graph
        :param data_iterator: iterator which provides training and test data
        :param log_dir: folder to store log files, train and test subfolders are created
        :param max_steps: number of training iterations
        :param learning_rate: learning rate for the optimizer, will be set at training start
        """
        # Train the model, and also write summaries.
        # Every 10th step, measure test-set accuracy, and write test summaries

        # define summary writers
        train_writer = tf.summary.FileWriter(log_dir + '/train', session.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

        # set learning rate
        _ = session.run(tf.assign(self.learning_rate, learning_rate))

        # train loop
        for i, (xs, ys) in enumerate(data_iterator.train_batch()):
            # exit training loop?
            if i > max_steps:
                break

            # for dropout, the default value is automatically used
            feed_dict = {self.x: xs, self.y_: ys}
            summary, _ = session.run([self.merged_summary, self.train_op],
                                     feed_dict=feed_dict)
            train_writer.add_summary(summary, i)

            # Test: Record summaries and test-set accuracy
            if i % 10 == 0:
                xs_test, ys_test = data_iterator.test_data()
                feed_dict = {self.x: xs_test,
                             self.y_: ys_test,
                             self.keep_prob: 1.0}  # no dropout during testing
                summary, acc = session.run([self.merged_summary, self.accuracy],
                                           feed_dict=feed_dict)
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %s' % (i, acc))

        train_writer.close()
        test_writer.close()


def main(log_dir, data_dir, keep_proba, layers, learning_rate=0.001, max_steps=1000, batch_size=100):
    """ This method initializes the tensorflow graph, invokes network construction and
    performs the training loop

    :param data_dir: directory to store data (mnist data)
    :param log_dir: folder to store log files, train and test subfolders are created
    :param keep_proba: in [0,1] defining the probability of a node dropout
    :param layers: list or tuple of layer units, for each entry a layer is created
    :param learning_rate: learning rate for the optimizer, will be set at training start
    :param max_steps: number of training iterations
    :param batch_size: number of samples per batch training
    """
    # reset log directory
    if tf.gfile.Exists(log_dir):
        tf.gfile.DeleteRecursively(log_dir)
    tf.gfile.MakeDirs(log_dir)

    # create data iterators
    data_iterator = MnistIterator(data_dir=data_dir, batch_size=batch_size)

    # create tensorflow session to build the graph in
    # sess = tf.InteractiveSession()
    with tf.Session() as sess:
        # Create a multilayer model
        model = MnistNet(keep_proba=keep_proba, layers=layers)
        
        model.build_graph()
        # assert False
        # initialize variables
        tf.global_variables_initializer().run()

        # start training loop
        model.train(session=sess,
                    data_iterator=data_iterator,
                    log_dir=log_dir,
                    learning_rate=learning_rate,
                    max_steps=max_steps)


if __name__ == '__main__':
    #### MNIST Stats ####
    # train set size: 60k
    # test set size: 10k
    # Best Accuracy (http://yann.lecun.com/exdb/mnist/):
    # SVM: 0.9944
    # FCN: 0.9965
    # CNN: 0.9977
    #####################

    # Number of batches during training
    max_steps = 2
    # Initial learning rate
    learning_rate = 0.001  # 0.1 is not working, 0.0001 @ 1k steps not working
    # Keep probability for training dropout
    dropout = 0.9
    # Directory for storing input data
    data_dir = './data/mnist'
    # Summaries log directory
    log_dir = '/tmp/null'
    # Specify hidden units in different layers
    layers = [500]
    # Define the batch size during training
    batch_size = 100

    main(log_dir=log_dir,
         data_dir=data_dir,
         keep_proba=dropout,
         learning_rate=learning_rate,
         max_steps=max_steps,
         layers=layers,
         batch_size=100)
