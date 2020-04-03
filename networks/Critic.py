import tensorflow as tf
import numpy as np

timestep = None
batch_size = None


# Q function
class Critic:
    def __init__(self, sess, state_dim, action_dim, scope):
        self.sess = sess
        self.state_dim = list(state_dim)
        self.history_dim = [None] + list(state_dim)
        self.action_dim = action_dim

        self.q_value_ph = tf.placeholder(tf.float32, shape=[None] + [self.action_dim])

        with tf.variable_scope(scope):
            self.model = self.create_network()

        self.gradient_q = tf.gradients(self.model.output, self.model.inputs[1])
        self.critic_loss = tf.reduce_mean((self.model.output - self.q_value_ph))
        self.optimize = tf.train.AdamOptimizer().minimize(self.critic_loss)

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)  # history_in
        action_in = tf.keras.layers.Input(shape=[self.action_dim])

        feature = tf.keras.applications.ResNet50(include_top=False, weights=None)  # 2048
        # feature = tf.keras.applications.MobileNetV2(include_top=False, weights=None)  # 1280
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(256)(x)

        y = tf.keras.layers.Dense(128)(action_in)
        y = tf.keras.layers.Dense(64)(y)

        concat = tf.keras.layers.concatenate([x, y])

        out = tf.keras.layers.Dense(1, activation="linear")(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)

        model.summary()
        return model

    def train(self, history, action, q_value):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: action, self.q_value_ph: q_value}
        self.sess.run(self.optimize, feed_dict=feed_dict)

    def get_q_value(self, history, actions):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: actions}
        return self.sess.run(self.model.output, feed_dict)

    def get_q_gradient(self, history, actions):
        feed_dict = {self.model.inputs[0]: history, self.model.inputs[2]: actions}
        return self.sess.run(self.gradient_q, feed_dict)

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
