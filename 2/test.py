import tensorflow as tf
import numpy as np
if __name__ == '__main__':
    sess = tf.InteractiveSession()
    Y= np.mgrid[0:1:0.1]
    xs = tf.constant(Y)
    xs = tf.Variable(xs)
    tf.initialize_all_variables().run()
    zs = xs * xs
    step = xs.assign(zs)
    print(sess.run(zs))
    print(sess.run(xs))
    print(sess.run(step))
    print(sess.run(xs))