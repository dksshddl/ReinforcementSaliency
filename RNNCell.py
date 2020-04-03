# import tensorflow as tf
# import numpy as np
#
# ph = tf.placeholder(dtype=tf.float32, shape=[None, None, 10, 10, 3])
#
# a = np.zeros([3, 10, 10, 3])
# b = np.zeros([5, 10, 10, 3])
#
# c = np.array([a, b])
#
# inp = tf.keras.layers.Input(shape=[None, 10, 10, 3])
#
# dense = tf.keras.layers.Dense(50)(inp)
#
# model = tf.keras.Model(inputs=inp, outputs=dense)
#
# model.summary()
#
# model.predict(c)

