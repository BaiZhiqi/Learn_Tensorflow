import tensorflow as tf
import numpy as np
import input_data

# 读取数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# 初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 自定义卷积层和池化层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


# 构建模型
def build_models():
    x = tf.placeholder("float", [None, 784])
    y = tf.placeholder("float", [None, 10])
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    conv1 = conv2d(x_image, W_conv1) + b_conv1
    # 第一层激活层
    r_conv1 = tf.nn.relu(conv1)
    # 第一层池化层
    m_pool1 = max_pool_2x2(r_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    conv2 = conv2d(m_pool1, W_conv2) + b_conv2
    # 第二层激活层
    r_conv2 = tf.nn.relu(conv2)
    # 第二层池化层
    m_pool2 = max_pool_2x2(r_conv2)

    # 密集连接层Dense
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(m_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 随机丢弃
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc1 = weight_variable([1024, 10])
    b_fc1 = bias_variable([10])
    pre = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc1) + b_fc1)

    #损失函数
    cross_entropy = -tf.reduce_sum(y*tf.log(pre))

    #优化器
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #准确率
    correct_prediction = tf.equal(tf.argmax(pre, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #训练
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 0.5})
            if i%100==0:
                #以下两种方式结果一致，但是下面那种方式可以同时获取多个变量的值
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch[0], y: batch[1], keep_prob: 1.0})
                #train_accuracy = sess.run(accuracy,feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))
        print("test accuracy{}".format( accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})))
if __name__ == '__main__':
    build_models()