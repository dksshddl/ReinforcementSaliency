import os

import tensorflow as tf
from utils.config import *

writer_path = os.path.join(log_path, "supervised")

timestep = 8
batch_size = None


class Resnet:
    def __init__(self, state_dim, session):
        self.state_dim = state_dim
        self.session = session

        self.y_true = tf.placeholder(tf.float32, [None, 2])

        self.construct()
        pass

    def construct(self):
        state_in = tf.keras.layers.Input(batch_shape=[1] + [timestep] + list(self.state_dim))

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(256))(state_in)
        x = tf.keras.layers.ConvLSTM2D(128, 5, 1, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(128, 5, 1, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 5, 1, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 5, 1, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(32, 5, 1, return_sequences=True, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(32, 5, 1, return_sequences=False, stateful=True)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Flatten()(x)

        x = tf.keras.layers.Dense(2, activation="tanh")(x)

        self.model = tf.keras.models.Model(inputs=state_in, outputs=x)
        self.model.compile(optimizer="adam", loss="mse")
        self.model.summary()

        self.loss = tf.keras.losses.mean_squared_error(self.y_true, self.model.output)
        self.opt = tf.keras.optimizers.Adam().minimize(self.loss)
        tf.summary.scalar()

        if not os.path.exists(writer_path):
            os.mkdir(writer_path)

        self.writer = tf.compat.v1.summary.FileWriter(writer_path, tf.get_default_graph())
        self.loss_summary = tf.compat.v1.summary.scalar("loss", self.loss)

    def optimize(self, inputs, true, step):
        loss, summary = tf.get_default_session().run([self.opt, self.loss_summary],
                                                     feed_dict={self.model.input: inputs,
                                                                self.y_true: true})
        self.writer.add_summary(summary, step)

        return loss

    def reset_state(self):
        self.model.reset_states()

    def save(self, save_path=None, step=None):
        if save_path is None:
            save_path = os.path.join(weight_path, "supervised")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        if step is None:
            model_name = "model.ckpt"
        else:
            model_name = f"model_{step}.ckpt"
        self.model.save_weights(os.path.join(save_path, model_name))


if __name__ == '__main__':
    a = Resnet((224, 224, 3))
