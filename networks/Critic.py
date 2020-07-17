import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Reshape, Dense, ConvLSTM2D, TimeDistributed, Input, Flatten, concatenate, LSTM, Bidirectional
batch_size = 1
timestep = 5


# Q function
class Critic:
    def __init__(self, state_dim, action_dim, scope, feature=None):
        self.state_dim = list(state_dim)
        self.history_dim = [None] + list(state_dim)
        self.action_dim = action_dim
        self.scope = scope

        if feature is None:
            self.model = self.create_network()
        else:
            self.model = self.shared_network(feature)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    def save(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.model.save_weights(os.path.join(checkpoint_directory, "critic.h5"), overwrite=True)

        # self.checkpoint.save(file_prefix=checkpoint_prefix)

    def restore(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    def shared_network(self, feature):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + [self.action_dim])

        weights = bias = tf.keras.initializers.RandomUniform(-3 * 10e-4, 3 * 10e-4)
        fan_in_dense = tf.keras.initializers.RandomUniform(-1 / pow(200, 0.5), 1 / pow(200, 0.5))

        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(256, return_sequences=False, stateful=True)(x)
        # out = tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.tanh, kernel_initializer=weights, bias_initializer=bias)(lstm)
        # model = tf.keras.models.Model(inputs=state_in, outputs=out)

        y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
                                  bias_initializer=fan_in_dense)(action_in)
        y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
                                  bias_initializer=fan_in_dense)(y)
        concat = tf.keras.layers.concatenate([x, y])

        out = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=weights, bias_initializer=bias)(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)
        model.summary()
        return model

    def reset_state(self):
        self.model.reset_states()

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)  # history_in
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + [1, self.action_dim])

        weights = bias = tf.keras.initializers.RandomUniform(-3 * 10e-4, 3 * 10e-4)
        # fan_in_conv = tf.keras.initializers.RandomUniform(-1 / pow(64, 0.5), 1 / pow(64, 0.5))
        # fan_in_dense = tf.keras.initializers.RandomUniform(-1 / pow(200, 0.5), 1 / pow(200, 0.5))

        # input_shape = (1, 5, 224, 224, 3)
        # state_in = Input(batch_shape=input_shape)
        # actio_in = Input(batch_shape=(1, 1, 2))
        x = ConvLSTM2D(64, 3, 3, padding='same', return_sequences=True, stateful=True)(state_in)
        x = ConvLSTM2D(64, 3, 3, padding='same', return_sequences=True, stateful=True)(x)
        x = ConvLSTM2D(64, 3, 3, padding='same', stateful=True)(x)
        x = Flatten()(x)
        x = Reshape([1, -1])(x)
        concat = concatenate([x, action_in])
        x = Bidirectional(LSTM(128, return_sequences=True, stateful=True, activation="relu"))(concat)
        out = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=weights, bias_initializer=bias)(x)

        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)
        model.summary()
        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)  # 2048
        # feature = tf.keras.applications.MobileNetV2(include_top=False, weights=None)  # 1280

        # feature = tf.keras.Sequential()
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # feature.add(tf.keras.layers.BatchNormalization())
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # feature.add(tf.keras.layers.BatchNormalization())
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # feature.add(tf.keras.layers.BatchNormalization())

        # x = tf.keras.layers.TimeDistributed(feature)(state_in)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        # x = tf.keras.layers.LSTM(256, return_sequences=False)(x)
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
        #
        # y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
        #                           bias_initializer=fan_in_dense)(action_in)
        # y = tf.keras.layers.Dense(200, activation=tf.nn.leaky_relu, kernel_initializer=fan_in_dense,
        #                           bias_initializer=fan_in_dense)(y)
        # concat = tf.keras.layers.concatenate([x, y])
        #
        # out = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=weights, bias_initializer=bias)(concat)
        # model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=out)
        # model.summary()
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
