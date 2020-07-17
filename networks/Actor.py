import os

import tensorflow as tf
from tensorflow.keras.layers import Reshape, Dense, ConvLSTM2D, TimeDistributed, Input, Flatten, concatenate, LSTM, Bidirectional
import numpy as np

batch_size = 1
timestep = 5


class Actor:
    def __init__(self, state_dim, action_dim, scope, feature=None):
        self.state_dim = list(state_dim)
        self.action_dim = action_dim
        self.scope = scope

        if feature is None:
            self.model = self.create_network()
        else:
            self.model = self.shared_network(feature)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)

    def save(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        # checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        # self.checkpoint.save(file_prefix=checkpoint_prefix)
        self.model.save_weights(os.path.join(checkpoint_directory, "actor.h5"), overwrite=True)


    def restore(self, path):
        checkpoint_directory = f"weights/{path}/{self.scope}"
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

    def shared_network(self, feature):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)

        weights = bias = tf.keras.initializers.RandomUniform(-3 * 10e-4, 3 * 10e-4)

        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(128, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.LSTM(64, return_sequences=True, stateful=True)(x)
        lstm = tf.keras.layers.LSTM(2, return_sequences=False, stateful=True)(x)

        out = tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.tanh, kernel_initializer=weights,
                                    bias_initializer=bias)(lstm)
        model = tf.keras.models.Model(inputs=state_in, outputs=lstm)

        model.summary()
        return model

    def create_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [timestep] + self.state_dim)
        weights = bias = tf.keras.initializers.RandomUniform(-3 * 10e-4, 3 * 10e-4)

        x = ConvLSTM2D(64, 3, 3, padding='same', return_sequences=True, stateful=True)(state_in)
        x = ConvLSTM2D(64, 3, 3, padding='same', return_sequences=True, stateful=True)(x)
        x = ConvLSTM2D(64, 3, 3, padding='same', stateful=True)(x)
        x = Flatten()(x)
        out = tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.tanh, kernel_initializer=weights,
                                    bias_initializer=bias)(x)
        # fan_in_conv = tf.keras.initializers.RandomUniform(-1 / pow(64, 0.5), 1 / pow(64, 0.5))
        #
        # feature = tf.keras.Sequential()
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # # feature.add(tf.keras.layers.BatchNormalization())
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # # feature.add(tf.keras.layers.BatchNormalization())
        # feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu", kernel_initializer=fan_in_conv,
        #                                    bias_initializer=fan_in_conv))
        # feature.add(tf.keras.layers.BatchNormalization())
        #
        # x = tf.keras.layers.TimeDistributed(feature)(state_in)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        # lstm = tf.keras.layers.LSTM(256, return_sequences=False)(x)
        # out = tf.keras.layers.Dense(self.action_dim, activation=tf.keras.activations.tanh, kernel_initializer=weights,
        #                             bias_initializer=bias)(lstm)
        model = tf.keras.models.Model(inputs=state_in, outputs=out)
        #
        model.summary()
        return model

    # def train(self, history, gradients):
    #     feed_dict = {self.model.inputs[0]: history, self.action_gradient_ph: gradients}
    #     self.sess.run(self.optimize, feed_dict=feed_dict)

    def predict_action(self, history_gen):
        return self.model.predict_generator(history_gen)

    def reset_state(self):
        self.model.reset_states()

    def custom(self, history, gradient):

        def loss(y_true, y_pred):
            pass

    def get_action(self, history):

        return self.model(history)
        # return self.sess.run(self.model.output, feed_dict={self.model.input: history})

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

# if __name__ == '__main__':
#     sess = tf.Session()
#     a = Actor(sess, (224, 224, 3), 2)
#     sess.run(tf.global_variables_initializer())
#     hist = np.zeros([1, 4, 224, 224, 3])
#     print(a.get_action(hist))
#
