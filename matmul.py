import tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3])

sub = tf.subtract(x,a)

add = tf.add(x,sub)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(sub))
    print(sess.run(add))
