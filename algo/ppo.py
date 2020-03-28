import copy

import tensorflow as tf
import numpy as np

from networks.policy_net import Policy_net


class PPOTrain:
    def __init__(self, Policy: Policy_net, Old_Policy: Policy_net, gamma=0.95, clip_value=0.2, c_1=1, c_2=0.01):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')  # value
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')  # advantage

        log_prob = self.Policy.dist.log_prob(self.actions)
        old_log_prob = self.Old_Policy.dist.log_prob(self.actions)
        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            ratios = tf.exp(log_prob - old_log_prob)
            ratios = tf.reduce_mean(ratios, axis=1, keep_dims=True)

            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value,
                                              clip_value_max=1 + clip_value)  # clip value - epsilon
            surr1 = ratios * self.gaes
            surr2 = clipped_ratios * self.gaes
            surr = tf.minimum(surr1, surr2)
            policy_loss = tf.reduce_mean(surr)
            tf.summary.scalar('policy_loss', policy_loss)

            # construct computation graph for loss of entropy bonus
            entropy = tf.reduce_mean(self.Policy.dist.entropy())
            tf.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            value_loss = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)  # value loss
            value_loss = tf.reduce_mean(value_loss)
            tf.summary.scalar('value_loss', value_loss)

            # construct computation graph for loss
            loss = policy_loss - c_1 * value_loss + c_2 * entropy  # c1, c2 is weight

            # minimize -loss == maximize loss
            loss = -loss
            tf.summary.scalar('total', loss)

        self.merged = tf.summary.merge_all()
        optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})
