from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import msgpack
import zmq
import uuid
import time

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
saver = tf.train.Saver()

def runLogisticMnistWithAccuracy(epochs=1000):
    with tf.Session() as sess:

        saver.restore(sess, "./model.ckpt")
        print("Model restored.")

        # sess.run(init)

        for i in range(epochs):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


runLogisticMnistWithAccuracy(epochs=300)
