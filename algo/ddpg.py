# coding=utf8
import os

import numpy as np
import tensorflow as tf

from utils import config
from utils.config import log_path, weight_path

#### HYPER PARAMETERS ####
gamma = 0.99  # reward discount factor

lr_critic = 3e-3  # learning rate for the critic
lr_actor = 1e-3  # learning rate for the actor
batch_size = None


# tau = 1e-2  # soft target update rate


class DDPG:
    def __init__(self, sess, state_dim, action_dim, action_max, action_min, load=None):
        self.sess = sess
        n_samples = 6
        self.state_dim = [n_samples] + state_dim  # [6, 224, 224, 3]
        self.action_dim = action_dim  # [2]
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + self.state_dim)
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + self.state_dim)
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + self.action_dim)
        self.action_grad_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size] + self.action_dim)
        self.done_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])
        self.qvalue_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 1])

        with tf.variable_scope('actor'):
            self.actor = self.generate_resnet_actor(trainable=True)
            # self.actor = self.generate_actor_network(trainable=True)
        with tf.variable_scope('target_actor'):
            self.target_actor = self.generate_resnet_actor(trainable=False)
            # self.target_actor = self.generate_actor_network(trainable=False)
        with tf.variable_scope('critic'):
            self.critic = self.generate_resnet_critic(trainable=True)
            # self.critic = self.generate_critic_network(trainable=True)
        with tf.variable_scope('target_critic'):
            self.target_critic = self.generate_resnet_critic(trainable=False)
            # self.target_critic = self.generate_critic_network(trainable=False)

        self.actor.summary()
        self.critic.summary()

        self.action_grad = tf.gradients(self.critic.output, self.critic.input[1])
        self.params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -self.action_grad_ph)
        self.critic_loss = tf.losses.mean_squared_error(self.qvalue_ph, self.critic.output)

        self.critic_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.critic_loss)
        self.optimize = tf.train.AdamOptimizer(learning_rate=0.001) \
            .apply_gradients(zip(self.params_grad, self.actor.trainable_weights))

        self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(os.path.join(log_path, config.DDPG_MOBILE), self.sess.graph)
        self.sess.run(tf.global_variables_initializer())
        self.critic_loss_summary = tf.summary.scalar("critic_loss", self.critic_loss)
        if load is not None:
            self.load(load)

    def gradients(self, states, actions):
        feed_dict = {self.critic.inputs[0]: states, self.critic.inputs[1]: actions}
        return self.sess.run(self.action_grad, feed_dict=feed_dict)[0]

    def target_critic_train(self, tau=0.01):
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = tau * critic_weights[i] + (1 - tau) * critic_target_weights[i]
        self.target_critic.set_weights(critic_target_weights)

    def target_actor_train(self, tau=0.01):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

    def train_actor(self, state, action_grads, step=None):
        feed_dict = {self.actor.inputs[0]: state, self.action_grad_ph: action_grads}
        self.sess.run(self.optimize, feed_dict=feed_dict)

    def train_critic(self, state, action, q_value, step=None):
        feed_dict = {self.critic.inputs[0]: state, self.critic.inputs[1]: action, self.qvalue_ph: q_value}
        _, summary = self.sess.run([self.critic_opt, self.critic_loss_summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, step)

    def predict(self, state):
        return self.actor.predict([state])

    def save(self, epochs):
        self.saver.save(self.sess, os.path.join(weight_path, config.DDPG_MOBILE, 'model_ddpg_' + str(epochs) + '.ckpt'))

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    # policy
    def generate_critic_network(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.action_dim)
        # x = tf.keras.layers.TimeDistributed(tf.keras.applications.ResNet50)
        # mask = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(mask_value=256., trainable=trainable), trainable=trainable)(state_in)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(state_in)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(inputs=x)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False, stateful=True, trainable=trainable)(inputs=x)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.Flatten(trainable=trainable)(x)
        concat = tf.keras.layers.concatenate([x, action_in])
        x = tf.keras.layers.Dense(1, trainable=trainable)(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mean_squared_error)
        return model

    # action-value
    def generate_actor_network(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)
        # mask = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(mask_value=256., trainable=trainable), trainable=trainable)(state_in)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(state_in)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(inputs=x)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False, stateful=True, trainable=trainable)(inputs=x)
        x = tf.keras.layers.BatchNormalization(trainable=trainable)(x)
        x = tf.keras.layers.Flatten(trainable=trainable)(x)
        x = tf.keras.layers.Dense(2, trainable=trainable)(x)
        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        return model

    def generate_resnet_critic(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.action_dim)
        feature = tf.keras.applications.MobileNet(include_top=False, weights=None)
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, stateful=True))(x)
        concat = tf.keras.layers.concatenate([x, action_in])
        x = tf.keras.layers.Dropout(0.5)(concat)
        x = tf.keras.layers.Dense(1)(x)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.losses.mean_squared_error)
        return model

    def generate_resnet_actor(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)
        feature = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Flatten()(x)
        # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, stateful=True))(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(2)(x)
        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        return model

    def save_actor_weights(self, path):
        self.actor.save_weights(path)

    def save_critic_weights(self, path):
        self.critic.save_weights(path)

    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def reset_state(self):
        self.actor.reset_states()
        self.critic.reset_states()
        self.target_actor.reset_states()
        self.target_critic.reset_states()
