import tensorflow as tf
import numpy as np

timestep = 5
batch_size = None


# Q function
class Critic:
    def __init__(self, sess, state_dim, action_dim, scope):
        self.sess = sess
        self.state_dim = list(state_dim)
        self.history_dim = [None] + list(state_dim)
        self.action_dim = action_dim

        self.q_value_ph = tf.placeholder(tf.float32, shape=[batch_size] + [1])

        with tf.variable_scope(scope):
            self.model = self.create_network()

        self.gradient_q = tf.gradients(self.model.output, self.model.inputs[1])

        self.critic_loss = tf.reduce_mean((self.model.output - self.q_value_ph))
        self.optimize = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(self.critic_loss)
        # self.optimize = tf.keras.optimizers.AdamOptimizer().minimize(self.critic_loss)

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)  # history_in
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + [self.action_dim])

        weights = bias = tf.keras.initializers.RandomUniform(-3 * 10e-4, 3 * 10e-4)
        fan_in_conv = tf.keras.initializers.RandomUniform(-1 / pow(64, 0.5), 1 / pow(64, 0.5))
        fan_in_dense = tf.keras.initializers.RandomUniform(-1 / pow(200, 0.5), 1 / pow(200, 0.5))

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
                                           bias_initializer=fan_in_conv))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
                                           bias_initializer=fan_in_conv))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
                                           bias_initializer=fan_in_conv))
        feature.add(tf.keras.layers.BatchNormalization())

        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.Flatten()(x)

        y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
                                  bias_initializer=fan_in_dense)(action_in)
        y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
                                  bias_initializer=fan_in_dense)(y)
        concat = tf.keras.layers.concatenate([x, y])

        out = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=weights, bias_initializer=bias)(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)
        model.summary()
        return model

    def train(self, history, action, q_value):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: action, self.q_value_ph: q_value}
        self.sess.run(self.optimize, feed_dict=feed_dict)

    def predict_q(self, data_gen):
        return self.model.predict_generator(data_gen)

    def get_q_value(self, history, actions):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: actions}
        return self.sess.run(self.model.output, feed_dict)

    def get_q_gradient(self, history, actions):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: actions}
        return self.sess.run(self.gradient_q, feed_dict)

    def gradient(self, history, actions):
        grads = []
        for hist, acs in zip(history, actions):
            grads.append(self.get_q_gradient(np.array([hist]), np.array([acs])))
        return grads

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


if __name__ == '__main__':
    sess = tf.Session()
    a = Critic(sess, (84, 84, 3), 2)
    sess.run(tf.global_variables_initializer())
    hist = np.zeros([1, 3, 84, 84, 3])
    acs = np.zeros([1, 2])
    print(a.get_q_value(hist, acs))
