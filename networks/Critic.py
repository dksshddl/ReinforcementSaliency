import os

import tensorflow as tf
import numpy as np

timestep = None
batch_size = None

# Q function
class Critic:
    def __init__(self, state_dim, action_dim, scope):
        self.state_dim = list(state_dim)
        self.history_dim = [None] + list(state_dim)
        self.action_dim = action_dim
        self.scope = scope
        # self.q_value_ph = tf.placeholder(tf.float32, shape=[batch_size] + [1])

        self.model = self.create_network()

        # self.gradient_q = tf.gradients(self.model.output, self.model.inputs[1])
        # self.critic_loss = tf.reduce_mean((self.model.output - self.q_value_ph))
        # self.opt = tf.train.AdamOptimizer(learning_rate=5e-5)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

        # self.optimize = tf.train.AdamOptimizer(learning_rate=5e-5).minimize(self.critic_loss)
        # self.optimize = tf.keras.optimizers.AdamOptimizer().minimize(self.critic_loss)

    def save(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def restore(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)  # history_in
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + [self.action_dim])

        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)  # 2048
        # feature = tf.keras.applications.MobileNetV2(include_top=False, weights=None)  # 1280

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        # feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        # feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        # feature.add(tf.keras.layers.BatchNormalization())

        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        lstm = tf.keras.layers.LSTM(256, return_sequences=False)(x)
        # x = tf.keras.layers.LSTM(256, return_sequences=True)(x)

        # y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu))(action_in)
        # y = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu))(y)
        # y = tf.keras.layers.LSTM(256, return_sequences=True)(y)
        # concat = tf.keras.layers.concatenate([x, y])

        # x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(state_in)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        y = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)(action_in)
        y = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)(y)
        concat = tf.keras.layers.concatenate([lstm, y])

        out = tf.keras.layers.Dense(1, activation="linear")(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)
        model.summary()
        return model

    # def train(self, history, action, q_value):
    #     feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: action, self.q_value_ph: q_value}
    #     self.sess.run(self.optimize, feed_dict=feed_dict)

    def predict_q(self, data_gen):
        return self.model.predict_generator(data_gen)

    # def get_q_value(self, history, actions):
    #     feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: actions}
    #     return self.sess.run(self.model.output, feed_dict)
    #
    # def get_q_gradient(self, history, actions):
    #     feed_dict = {self.model.inputs[0]: history, self.model.inputs[1]: actions}
    #     return self.sess.run(self.gradient_q, feed_dict)

    # def gradient(self, history, actions):
    #     grads = []
    #     for hist, acs in zip(history, actions):
    #         grads.append(self.get_q_gradient(np.array([hist]), np.array([acs])))
    #     return grads

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)


# if __name__ == '__main__':
#     sess = tf.Session()
#     a = Critic(sess, (84, 84, 3), 2)
#     sess.run(tf.global_variables_initializer())
#     hist = np.zeros([1, 3, 84, 84, 3])
#     acs = np.zeros([1, 2])
#     print(a.get_q_value(hist, acs))
