# coding=utf8
import os

import numpy as np
import tensorflow as tf

from utils import config
from utils.config import log_path, weight_path

#### HYPER PARAMETERS ####
gamma = 0.99  # reward discount factor

lr_critic = 1e-3  # learning rate for the critic
lr_actor = 1e-4  # learning rate for the actor

batch_size = 1
n_samples = 30


# tau = 1e-2  # soft target update rate


class DDPG:
    def __init__(self, sess, state_dim, action_dim, action_max, action_min, load=None):
        self.sess = sess

        self.state_dim = [n_samples] + state_dim  # [6, 224, 224, 3]
        self.action_dim = action_dim  # [2]
        self.action_max = float(action_max)
        self.action_min = float(action_min)

        self.state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size] + self.state_dim)
        self.reward_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 1])
        self.next_state_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size] + self.state_dim)
        self.action_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size] + self.action_dim)
        self.action_grad_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size] + self.action_dim)
        self.done_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 1])
        self.qvalue_ph = tf.compat.v1.placeholder(dtype=tf.float32, shape=[batch_size, 1])

        # self.feature = tf.keras.applications.ResNet50(include_top=False, weights=None)
        # self.feature_tareget = tf.keras.applications.ResNet50(include_top=False, weights=None)

        with tf.compat.v1.variable_scope('actor'):
            # self.actor = self.generate_resnet_actor(trainable=True)
            self.actor = self.generate_actor_network(trainable=True)
        with tf.compat.v1.variable_scope('target_actor'):
            # self.target_actor = self.generate_resnet_actor(trainable=False)
            self.target_actor = self.generate_actor_network(trainable=False)
        with tf.compat.v1.variable_scope('critic'):
            # self.critic = self.generate_resnet_critic(trainable=True)
            self.critic = self.generate_critic_network(trainable=True)
        with tf.compat.v1.variable_scope('target_critic'):
            # self.target_critic = self.generate_resnet_critic(trainable=False)
            self.target_critic = self.generate_critic_network(trainable=False)

        self.actor.summary()
        self.critic.summary()

        self.action_grad = tf.gradients(self.critic.output, self.critic.input[1])  # q_value, action
        self.params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -self.action_grad_ph)
        self.critic_loss = tf.compat.v1.losses.mean_squared_error(self.qvalue_ph, self.critic.output)

        self.critic_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_critic).minimize(self.critic_loss)
        self.actor_opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_actor) \
            .apply_gradients(zip(self.params_grad, self.actor.trainable_weights))

        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        writer_path = os.path.join(log_path, config.DDPG_RESNET)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.compat.v1.summary.FileWriter(writer_path, self.sess.graph)
        self.critic_loss_summary = tf.compat.v1.summary.scalar("critic_loss", self.critic_loss)

        if load is not None:
            self.load(load)

    def gradients(self, states, actions):
        feed_dict = {self.critic.inputs[0]: states, self.critic.inputs[1]: actions}
        return self.sess.run(self.action_grad, feed_dict=feed_dict)[0]

    def target_critic_train(self, tau=0.001):
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = tau * critic_weights[i] + (1 - tau) * critic_target_weights[i]
        self.target_critic.set_weights(critic_target_weights)

    def target_actor_train(self, tau=0.001):
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = tau * actor_weights[i] + (1 - tau) * actor_target_weights[i]
        self.target_actor.set_weights(actor_target_weights)

    def train_actor(self, state, action_grads, step=None):
        feed_dict = {self.actor.inputs[0]: state, self.action_grad_ph: action_grads}
        self.sess.run(self.actor_opt, feed_dict=feed_dict)

    def train_critic(self, state, action, q_value, step=None):
        feed_dict = {self.critic.inputs[0]: state, self.critic.inputs[1]: action, self.qvalue_ph: q_value}
        _, summary = self.sess.run([self.critic_opt, self.critic_loss_summary], feed_dict=feed_dict)
        self.writer.add_summary(summary, step)

    def predict(self, state):
        return self.actor.predict([state])

    def save(self, epochs=None):
        model_save_path = os.path.join(weight_path, config.DDPG_RESNET)
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if epochs is None:
            self.saver.save(self.sess, os.path.join(model_save_path, 'model_ddpg' + '.ckpt'))
        else:
            self.saver.save(self.sess, os.path.join(model_save_path, 'model_ddpg_' + str(epochs) + '.ckpt'))

    def load(self, load_path):
        self.saver.restore(self.sess, load_path)

    # policy
    def generate_critic_network(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)
        action_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.action_dim)

        weight_decay = tf.keras.regularizers.l2(0.001)
        final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(256))(state_in)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Flatten(trainable=trainable)(x)

        y = tf.keras.layers.Dense(128, activation="relu")(action_in)
        y = tf.keras.layers.Dense(64, activation="relu")(y)
        concat = tf.keras.layers.concatenate([x, y])

        x = tf.keras.layers.Dense(1, activation="linear", activity_regularizer=weight_decay,
                                  kernel_initializer=final_initializer, bias_initializer=final_initializer)(concat)

        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.compat.v1.losses.mean_squared_error)
        return model

    # action-value
    def generate_actor_network(self, trainable):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + self.state_dim)

        final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(256))(state_in)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False, stateful=True, trainable=trainable)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Flatten(trainable=trainable)(x)

        x = tf.keras.layers.Dense(2, activation="tanh", bias_initializer=final_initializer,
                                  kernel_initializer=final_initializer)(x)

        x = tf.keras.layers.Dense(2, trainable=trainable)(x)

        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        return model

    def generate_resnet_critic(self, trainable):
        state_in = tf.keras.layers.Input(shape=self.state_dim)
        action_in = tf.keras.layers.Input(shape=self.action_dim)

        weight_decay = tf.keras.regularizers.l2(0.001)
        conv_initializer1 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer2 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer3 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        lstm_initailizer = tf.keras.initializers.RandomUniform(-1 / 256, 1 / 256)
        dense_initializer = tf.keras.initializers.RandomUniform(-1 / 200, 1 / 200)
        final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

        x = tf.keras.models.Sequential()
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer1,
                                     bias_initializer=conv_initializer1))
        x.add(tf.keras.layers.BatchNormalization())
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer2,
                                     bias_initializer=conv_initializer2))
        x.add(tf.keras.layers.BatchNormalization())
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer3,
                                     bias_initializer=conv_initializer3))
        x.add(tf.keras.layers.BatchNormalization())

        x = tf.keras.layers.TimeDistributed(x)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, kernel_initializer=lstm_initailizer, bias_initializer=lstm_initailizer,
                                 recurrent_initializer=lstm_initailizer))(x)
        y = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=dense_initializer,
                                  bias_initializer=dense_initializer)(action_in)
        y = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=dense_initializer,
                                  bias_initializer=dense_initializer)(y)
        concat = tf.keras.layers.concatenate([x, y])
        # x = tf.keras.layers.Dropout(0.5)(concat)
        x = tf.keras.layers.Dense(1, activation="linear", activity_regularizer=weight_decay,
                                  kernel_initializer=final_initializer, bias_initializer=final_initializer)(concat)
        model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.compat.v1.losses.mean_squared_error)
        return model

    def generate_resnet_actor(self, trainable):
        state_in = tf.keras.layers.Input(shape=self.state_dim)

        conv_initializer1 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer2 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer3 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)
        lstm_initailizer = tf.keras.initializers.RandomUniform(-1 / 256, 1 / 256)
        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)
        x = tf.keras.models.Sequential()
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer1,
                                     bias_initializer=conv_initializer1))
        x.add(tf.keras.layers.BatchNormalization())
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer2,
                                     bias_initializer=conv_initializer2))
        x.add(tf.keras.layers.BatchNormalization())
        x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer3,
                                     bias_initializer=conv_initializer3))
        x.add(tf.keras.layers.BatchNormalization())

        x = tf.keras.layers.TimeDistributed(x)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, kernel_initializer=lstm_initailizer, bias_initializer=lstm_initailizer,
                                 recurrent_initializer=lstm_initailizer))(x)
        # x = tf.keras.layers.Dropout(0.5)(x)
        x = tf.keras.layers.Dense(2, activation="tanh", bias_initializer=final_initializer,
                                  kernel_initializer=final_initializer)(x)
        model = tf.keras.models.Model(inputs=state_in, outputs=x)
        return model

    def save_actor_weights(self, path):
        self.actor.save_weights(path)

    def save_critic_weights(self, path):
        self.critic.save_weights(path)

    def load_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)

    def load_all(self, path):
        self.saver.restore(self.sess, path)

    def reset_state(self):
        self.actor.reset_states()
        self.critic.reset_states()
        self.target_actor.reset_states()
        self.target_critic.reset_states()
