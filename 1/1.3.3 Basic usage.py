import tensorflow as tf


state = tf.Variable(0,name="count")

one = tf.constant(1)
new_value = tf.add(one,state)
#要通过tf.assign()对state进行更新，而不能使用state = tf.add(one,state)的形式。
update = tf.assign(state,new_value)

#对全局进行初始化
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print(sess.run(state))
