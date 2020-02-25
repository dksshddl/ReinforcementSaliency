import os

import tensorflow as tf
import numpy as np


class Resnet:
    def __init__(self, state_dim, action_dim):
        self.state_dim = [8] + state_dim
        self.action_dim = action_dim

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.state_dim)
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None] + self.action_dim)
        # self.model = self.network()
        self.model = self.conv_network()
        self.model.summary()

    def conv_network(self):
        state_in = tf.keras.layers.Input(batch_shape=[1] + self.state_dim)
        mask = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(mask_value=256.))(state_in)
        x = tf.keras.layers.ConvLSTM2D(40, 3, 3, return_sequences=True, name="conv1")(inputs=mask)
        x = tf.keras.layers.BatchNormalization(name="batch1")(x)
        x = tf.keras.layers.ConvLSTM2D(40, 3, 3, return_sequences=True, name="conv2")(inputs=x)
        x = tf.keras.layers.BatchNormalization(name="batch2")(x)
        x = tf.keras.layers.ConvLSTM2D(40, 3, 3, return_sequences=False, name="conv3")(inputs=x)
        x = tf.keras.layers.BatchNormalization(name="batch3")(x)
        x = tf.keras.layers.Flatten(name="flatten")(x)
        x = tf.keras.layers.Dense(2, name="action")(x)
        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        model.compile(optimizer="adam", loss=tf.keras.losses.mean_squared_error)
        return model

    def reset_state(self):
        self.model.reset_states()

    def network(self):
        feature = tf.keras.applications.ResNet50(include_top=False, weights=None)
        state_in = tf.keras.layers.Input(batch_shape=[None] + self.state_dim)
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.LSTM(1, stateful=True)(x)
        x = tf.keras.layers.Dense(2, activation="linear")(x)
        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        return model

    def train_on_batch(self, state, action):
        loss = self.model.train_on_batch(state, action)
        # print("train loss: ", loss)
        # if loss <= 0.001:
        #     pred = self.model.predict(state)
        #     print("pred, true: {}, {}".format(pred, action))
        return loss

    def save(self, step):
        self.model.save_weights(os.path.join("weights", "resnet", "model_convlstem2_" + str(step) + ".h5"))
