import tensorflow as tf
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
mul = tf.multiply(add,input1)
#可以在一次运行操作中，一次性获取多个tensor的值（而不是逐个去获取 tensor）
with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)
