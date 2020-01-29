import tensorflow as tf
import numpy as np

def my_softmax(x):
    x = tf.exp(x)/tf.reduce_sum(tf.exp(x))
    return x
#tf.placeholder(x)表示的输入的变量，类似于input,其中参数有形状和类型。
# 因为你对同一个图可以每次的输入不一样，这时就需要一个变量来保存，这就是tf.placeholder()
input1 = tf.placeholder(np.float32)
input2 = tf.placeholder(np.float32)
sig = tf.nn.softmax(input1)
sig1 = my_softmax(input1)
add = tf.multiply(input1,input2)
exp = tf.exp(input1)
with tf.Session() as sess:
    # result = sess.run([add],feed_dict={input1:[7.],input2:[2.]})
    result = sess.run([sig,sig1],feed_dict={input1:[0.6,-1.0,1.0]})
    print(result)
    result = sess.run(exp, feed_dict={input1: [0.6, -1.0, 1.0]})
    print(result)