import input_data
import tensorflow as tf
import numpy as np
#读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#创建模型
x = tf.placeholder(dtype=np.float32, shape=[None, 784], name="x")
y = tf.placeholder(dtype=np.float32, shape=[None, 10], name="y")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
pre = tf.nn.softmax(tf.matmul(x, W) + b)

#设计损失函数
cross_entropy = -tf.reduce_sum(y * tf.log(pre))

#设计优化器
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs, batch_ys = np.float32(np.array(batch_xs)), np.float32(np.array(batch_ys))
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        if i % 20 == 0:
            print("第{}次迭代，loss为{}".format(i, sess.run(cross_entropy, feed_dict={x: batch_xs, y: batch_ys})))
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(pre,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    print(sess.run(accuracy,feed_dict={x: mnist.test.images, y: mnist.test.labels}))