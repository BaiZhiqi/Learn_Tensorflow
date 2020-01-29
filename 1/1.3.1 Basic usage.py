import tensorflow as tf
# 创建一个源 op,源 op 不需要任何输入,例如常量 产生一个 1x2 矩阵. 这个 op 被作为一个节点
#加到默认图中
# 构造器的返回值代表该常量 op 的返回值.
#通过tf.constant(x)创建一个op
matrix1 = tf.constant([[3.,3.]])
matrix2 = tf.constant([[2.],[2.]])
res = tf.matmul(matrix1,matrix2)

#这时已经创建了3个op，其中包括两个常量，和一个矩阵乘法的操作
sess = tf.Session()
#这里就会有人要问，为什么没有全局初始化啊？
#这是因为当前图没有变量
result = sess.run(res)
print(result)
sess.close()


