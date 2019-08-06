import tensorflow as tf
import numpy as np
"""
该代码是取自于Tensorflow官网，是为了新手入门而写，代码中会加入一些我的理解。
每个函数我都给出了相应的中文解释
"""

##生成数据 因为需要float32的 而np.rand是生成float64的
x_data = np.float32(np.random.rand(2,100))
y_data = np.dot([0.100,0.200],x_data)+0.300

#构造一个线性模型
#tf.Variable(initializer,name),参数initializer是初始化参数，name是可自定义的变量名称
#tf.random_uniform((6, 6), minval=low,maxval=high,dtype=tf.float32)))返回6*6的矩阵，产生于low和high之间，产生的值是均匀分布的。
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))

#.tf.matmul（）将矩阵a乘以矩阵b，生成a * b。而与之相对应的tf.multiply（）两个矩阵中对应元素各自相乘
y_pre = tf.matmul(W,x_data)+b

##最小化方差
#tf.square(x)对x内的所有元素进行平方操作
#tf.reduce_mean(x) 对x内的元素求均值
loss = tf.reduce_mean(tf.square(y_pre - y_data))
#使用随机梯度下降算法，使参数沿着 梯度的反方向，即总损失减小的方向移动，实现更新参数
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

#因为tensorflow的定义和初始化是很开的，所以要全局初始化
init = tf.global_variables_initializer()

#启动图，所有的语句都要通过sess.run执行
sess = tf.Session()
sess.run(init)

#拟合平面
for step  in range(1,1000):
    sess.run(train)
    if step%20 == 0:
        print("第{}步，当前Loss为{}，拟合的曲线为y ={}x+{}".format(step,sess.run(loss),sess.run(W), sess.run(b)))