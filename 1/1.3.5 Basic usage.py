import tensorflow as tf
import numpy as np
#tf.placeholder(x)表示的输入的变量，类似于input,其中参数有形状和类型。
# 因为你对同一个图可以每次的输入不一样，这时就需要一个变量来保存，这就是tf.placeholder()
input1 = tf.placeholder(np.float32)
input2 = tf.placeholder(np.float32)

add = tf.multiply(input1,input2)

with tf.Session() as sess:
    result = sess.run([add],feed_dict={input1:[7.],input2:[2.]})
    print(result)