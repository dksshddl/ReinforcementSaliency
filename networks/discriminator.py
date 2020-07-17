import os

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM

batch_size = None
n_samples = None


def discriminator_loss(y_true, y_pred):
    loss_expert = tf.reduce_mean(tf.math.log(tf.clip_by_value(y_true, 0.01, 1)))
    loss_agent = tf.reduce_mean(tf.math.log(tf.clip_by_value(1 - y_pred, 0.01, 1)))
    l = loss_expert + loss_agent
    return -l


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """

        state = Input(dtype=tf.float32, batch_shape=[batch_size] + [n_samples] + list(env.observation_space.shape))
        action = Input(dtype=tf.float32, batch_shape=[batch_size] + list(env.action_space.shape))

        self.model = self.construct_network(state, action)
        self.optimizer = tf.optimizers.Adam(1e-4)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.step = 0
    def construct_network(self, input_s, input_a):
        feature = tf.keras.applications.mobilenet_v2.MobileNetV2((160, 160, 3), include_top=False, pooling="avg")

        x = tf.keras.layers.TimeDistributed(feature)(input_s)
        x = tf.keras.layers.LSTM(128)(x)

        y = tf.keras.layers.Dense(128, activation="relu")(input_a)
        y = tf.keras.layers.Dense(128, activation="relu")(y)
        concat = tf.keras.layers.concatenate([x, y])
        prob = tf.keras.layers.Dense(1, activation="sigmoid")(concat)
        model = tf.keras.models.Model(inputs=[input_s, input_a], outputs=prob)
        return model

    def reset_state(self):
        self.model.reset_states()

    def train(self, expert_s, expert_a, agent_s, agent_a, epochs=2):
        for _ in range(epochs):
            self.step += 1
            batch_grad = []
            for ex_s, ex_a, ag_s, ag_a in zip(expert_s, expert_a, agent_s, agent_a):
                episode_grad = []
                for s1, a1, s2, a2 in zip(ex_s, ex_a, ag_s, ag_a):
                    with tf.GradientTape() as tape:
                        expert_prob = self.model([s1, a1], training=True)
                        agent_prob = self.model([s2, a2], training=True)
                        loss = discriminator_loss(expert_prob, agent_prob)
                    grads = tape.gradient(loss, self.get_trainable_variables())

                    episode_grad.append(grads)
                episode_grad = np.transpose(episode_grad)
                episode_grad = episode_grad.tolist()
                episode_grad = [tf.reduce_mean(i, axis=0) for i in episode_grad]
                batch_grad.append(episode_grad)
            batch_grad = np.transpose(batch_grad)
            batch_grad = batch_grad.tolist()
            batch_grad = [tf.reduce_mean(i, axis=0) for i in batch_grad]
            self.optimizer.apply_gradients(zip(batch_grad, self.get_trainable_variables()))
            tf.summary.scalar('discriminator', float(loss), self.step)

    def save(self):
        checkpoint_directory = f"weights/gail/discriminator"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)

    def get_rewards(self, agent_s, agent_a):
        reward = self.model([agent_s, agent_a])
        return tf.math.log(tf.clip_by_value(reward, 1e-10, 1))

    def get_trainable_variables(self):
        return self.model.trainable_variables
