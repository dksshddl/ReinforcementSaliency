import tensorflow as tf
import numpy as np


eps = 1e-8
entropy_beta = 1e-2  # policy entropy weight


class A3CNetwork:
    def __init__(self, state_dim, action_dim, scope):
        self.state_ph = tf.placeholder(tf.float32, shape=[None] + list(state_dim))
        self.action_ph = tf.placeholder(tf.float32, shape=[None] + list(action_dim))
        self.target_q_ph = tf.placeholder(tf.float32, shpae=[None, ])

        self.scope = scope

        self.construct_network(state_dim, action_dim)

    def construct_network(self, state_dim, action_dim):
        state_in = tf.keras.layers.Input(shape=state_dim)
        action_in = tf.keras.layers.Input(shape=action_dim)

        weight_decay = tf.keras.regularizers.l2(0.001)
        conv_initializer1 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer2 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        conv_initializer3 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
        final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)
        lstm_initailizer = tf.keras.initializers.RandomUniform(-1 / 256, 1 / 256)
        # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)

        # shared network
        share = tf.keras.models.Sequential()
        share.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer1,
                                         bias_initializer=conv_initializer1))
        share.add(tf.keras.layers.BatchNormalization())
        share.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer2,
                                         bias_initializer=conv_initializer2))
        share.add(tf.keras.layers.BatchNormalization())
        share.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer3,
                                         bias_initializer=conv_initializer3))
        share.add(tf.keras.layers.BatchNormalization())

        share = tf.keras.layers.TimeDistributed(share)(state_in)
        share = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(share)
        share = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, kernel_initializer=lstm_initailizer, bias_initializer=lstm_initailizer,
                                 recurrent_initializer=lstm_initailizer))(share)
        self.shared_network = tf.keras.models.Model(inputs=state_in, outputs=share)
        self.shared_network.summary()
        # policy
        with tf.variable_scope("{}_policy".format(self.scope)):
            x = tf.keras.layers.Dense(200, activation='relu')(share)
            x = tf.keras.layers.Dense(200, activation='relu')(x)
            policy_out = tf.keras.layers.Dense(2, activation="tanh", bias_initializer=final_initializer,
                                                    kernel_initializer=final_initializer)(x)
            self.policy_network = tf.keras.models.Model(inputs=share, outputs=policy_out)
            self.policy_network.summary()
        # value
        with tf.variable_scope("{}_value".format(self.scope)):
            y = tf.keras.layers.Dense(200, activation='relu')(share)
            y = tf.keras.layers.Dense(200, activation='relu')(y)

            y_a = tf.keras.layers.Dense(200, activation='relu')(action_in)
            y_a = tf.keras.layers.Dense(200, activation='relu')(y_a)

            y_concat = tf.keras.layers.concatenate([y, y_a])
            value_out = tf.keras.layers.Dense(1, activation="linear", activity_regularizer=weight_decay,
                                                   kernel_initializer=final_initializer,
                                                   bias_initializer=final_initializer)(y_concat)

            self.value_network = tf.keras.models.Model(inputs=[share, action_in], outputs=value_out)
            self.value_network.summary()

        with tf.op_scope([policy_out, value_out], "{}_loss".format(self.scope)):
            self.entropy = - tf.reduce_mean(policy_out * tf.log(policy_out + eps))
            time_diff = self.target_q_ph - value_out
            policy_prob = tf.log(tf.reduce_sum(tf.multiply(policy_out, self.action_ph), reduction_indices=1))
            self.policy_loss = - tf.reduce_sum(policy_prob * time_diff)
            self.value_loss = tf.reduce_sum(tf.square(time_diff))
            self.total_loss = self.policy_loss + self.value_loss * 0.5 + self.entropy * entropy_beta
            self.shared_loss = self.policy_loss + self.value_loss * 0.5

            self.policy_grads = tf.gradients(self.policy_loss, self.policy_network.get_weights())
            self.value_grads = tf.gradients(self.value_loss, self.value_network.get_weights())
            self.shared_grads = tf.gradients(self.shared_loss, self.shared_network.get_weights())
            # self.shared_out_grad = tf.gradients(self.policy_loss, [shared_out])[0] + \
            #                        tf.gradients(self.value_loss, [shared_out])[0] * 0.5
            # self.shared_grads = tf.gradients(shared_out,
            #                                  [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3],
            #                                  grad_ys=self.shared_out_grad)

    def get_vars(self):
        value_weight = self.value_network.get_weights()
        policy_weight = self.policy_network.get_weights()
        shared_weight = self.shared_network.get_weights()

        return np.concatenate([value_weight, policy_weight, shared_weight])


if __name__ == '__main__':
    a = A3CNetwork((224, 224, 3), 2, "test")
