import tensorflow as tf

sess = tf.InteractiveSession()
#这里创建的是一个变量，其中的数组为其初始化的内容而不是shape
x = tf.Variable([1.0,2.0])
a = tf.constant([3.0,3.0])
#这里因为x为变量，所以需要初始化，如果变量太多的话，可以通过全局初始化进行
x.initializer.run()
r = x - a
#这里是使用了.eval的形式代替sess.run,这两种形式的结果是一样的
print(r.eval())
print(sess.run(r))