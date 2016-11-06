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

def runLogisticMnistWithAccuracy(epochs=1000):
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def pixmap(ls):
    def go(p):
        '''
        map entries of weight matrix to rgb
        '''
        if p == 0:
            return [0,0,0]
        elif p > 0:
            return [0,0,255]
        else:
            return [255,0,0]
    asRGB = map(go, ls)
    return [item for sublist in asRGB for item in sublist]

def runLogisticMnistShowWeights(epochs=1000):
    '''
    map zero weights to black 0,0,0
    map negative weights to red 255,0,0
    map positive weights to blue 0,0,255
    '''
    ctx = zmq.Context()

    outChan = 'tcp://172.17.0.2:8888'
    outSock = ctx.socket(zmq.PUSH)
    outSock.bind(outChan)


    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        Wobj = sess.run(W)
        epochId = str(uuid.uuid4())
        for i in range(10):
            vizobj = {
                'epochId': epochId,
                'wIdx': i,
                'data': pixmap(Wobj[:,i].tolist())
            }
            outSock.send(msgpack.packb(vizobj))
        time.sleep(1)


def inspectWOutput():
    init = tf.initialize_all_variables()
    sess = tf.InteractiveSession()
    sess.run(init)

    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(W).shape)

runLogisticMnistShowWeights(epochs=200)
