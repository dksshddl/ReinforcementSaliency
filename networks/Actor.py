import tensorflow as tf
import numpy as np

timestep = None
batch_size = None

steps = None


class Actor:
    def __init__(self, sess, state_dim, action_dim, scope):
        self.sess = sess
        self.state_dim = list(state_dim)
        self.action_dim = action_dim

        self.action_gradient_ph = tf.placeholder(tf.float32, [batch_size] + [action_dim])

        with tf.variable_scope(scope):
            self.model = self.create_network()

        self.gradient_param = tf.gradients(self.model.output, self.model.trainable_weights, -self.action_gradient_ph)
        self.optimize = tf.train.AdamOptimizer().apply_gradients(zip(self.gradient_param, self.model.trainable_weights))

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)

        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)  # 2048
        # feature = tf.keras.applications.MobileNetV2(include_top=False, weights=None)  # 1280

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())

        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        lstm = tf.keras.layers.LSTM(256)(x)
        out = tf.keras.layers.Dense(self.action_dim, activation="tanh")(lstm)
        model = tf.keras.models.Model(inputs=state_in, outputs=out)

        model.summary()
        return model

    def train(self, history, gradients):

        feed_dict = {self.model.inputs[0]: history, self.action_gradient_ph: gradients}
        self.sess.run(self.optimize, feed_dict=feed_dict)

    def get_action(self, history):
        return self.sess.run(self.model.output, feed_dict={self.model.input: history})

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


if __name__ == '__main__':
    sess = tf.Session()
    a = Actor(sess, (224, 224, 3), 2)
    sess.run(tf.global_variables_initializer())
    hist = np.zeros([1, 4, 224, 224, 3])
    print(a.get_action(hist))
