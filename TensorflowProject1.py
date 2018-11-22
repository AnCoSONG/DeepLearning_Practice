import numpy as np
import tensorflow as tf

#生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2

#创建一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k*x_data + b

#方差函数(二次代价函数)
loss = tf.reduce_mean(tf.square(y_data-y))

#定义梯度下降优化方法

optimizer = tf.train.GradientDescentOptimizer(0.2)

#使方差最小(最小化代价函数)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1000):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))

