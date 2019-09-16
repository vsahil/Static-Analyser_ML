import tensorflow as tf
"""Example for learning a regression."""


import tensorflow as tf
import numpy

# Parameters
learning_rate = 0.01
training_epochs = 1
display_step = 50

# Generate training data
train_X = []
train_Y = []
f = lambda x: x**2
for x in range(-20, 20):
    train_X.append(float(x))
    train_Y.append(f(x))
train_X = numpy.asarray(train_X)
train_Y = numpy.asarray(train_Y)
n_samples = train_X.shape[0]

# Graph input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Create Model
W1 = tf.Variable(tf.truncated_normal([1, 10], stddev=0.1), name="weight")
b1 = tf.Variable(tf.constant(0.1, shape=[1, 10]), name="bias")
mul = X * W1
print(mul, X, W1, "SEE")
# h1 = tf.nn.sigmoid(mul) + b1
# W2 = tf.Variable(tf.truncated_normal([10, 1], stddev=0.1), name="weight")
# b2 = tf.Variable(tf.constant(0.1, shape=[1]), name="bias")
# activation = tf.nn.sigmoid(tf.matmul(h1, W2) + b2)

# # Minimize the squared errors
# l2_loss = tf.reduce_sum(tf.pow(activation-Y, 2))/(2*n_samples)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(l2_loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    # Fit all training data
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):
        #     sess.run(optimizer, feed_dict={X: int(x), Y: int(y)})

        # Display logs per epoch step
        if epoch % display_step == 0:
            print(train_X.shape)
            print("cost=", sess.run(mul, feed_dict={X: train_X, Y: train_Y}), "W1=", sess.run(W1)); exit()

    print("Optimization Finished!")
    print("cost=", sess.run(l2_loss, feed_dict={X: train_X, Y: train_Y}),
          "W1=", sess.run(W1), )  # "b2=", sess.run(b2)
