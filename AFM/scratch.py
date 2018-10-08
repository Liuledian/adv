from datasets import iPinYou
import numpy as np
import tensorflow as tf
a=tf.Variable(0)
update_a=a.assign(1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(update_a)
    print(sess.run(a))
