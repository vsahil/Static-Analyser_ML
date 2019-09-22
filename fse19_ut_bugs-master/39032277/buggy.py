import tensorflow as tf
import random, time, sys
import cv2
sys.path.append("../")

from tensorflow.examples.tutorials.mnist import input_data
if sys.argv[1] == "1":
    mnist_file = "MNIST_data/"
elif sys.argv[1] == "2":
    mnist_file = "/home/saverma/TensorFlow-Program-Bugs-master/dummy_mnist"

start = time.time()
mnist = input_data.read_data_sets(mnist_file, one_hot=True, validation_size=0)

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

learning_rate = 0.01
training_epochs = 1
batch_size = 100
display_step = 1

### modeling ###

activation = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
end1 = time.time()
print("Checkpoint model", end1-start)
### training ###

for epoch in range(training_epochs) :

    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch) :

        batch_xs, batch_ys =mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
        avg_cost += sess.run(cross_entropy, feed_dict = {x: batch_xs, y: batch_ys}) / total_batch

    if epoch % display_step == 0 :
        print("Epoch : ", "%04d" % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

print("Optimization Finished")
start_time = time.time()
print("Checkpoint train", start_time-end1)
import check
### predict number ###

r = random.randint(0, mnist.test.num_examples - 1)
print("Prediction: ", sess.run(tf.argmax(activation,1), feed_dict={x: mnist.test.images[r:r+1]}))

image = cv2.imread("img_easy.jpg")
resized_image = cv2.resize(image, (784,1))
print("Prediction: ", sess.run(tf.argmax(activation,1), feed_dict={x: resized_image}))
print("Correct Answer: 9")
