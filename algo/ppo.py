import copy
import os

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, LSTM, TimeDistributed, Flatten
import numpy as np

from networks.policy_net import Policy_net

gamma = 0.95
clip_value = 0.2
c_1 = 1
c_2 = 0.01


def gail_loss(v_pred, v_pred_next, rewards, gaes, entropy):
    def loss(log_prob, old_log_prob):
        ratios = tf.exp(log_prob - old_log_prob)
        ratios = tf.reduce_mean(ratios, axis=1, keep_dims=True)
        clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value,
                                          clip_value_max=1 + clip_value)  # clip value - epsilon
        surr1 = ratios * gaes
        surr2 = clipped_ratios * gaes
        surr = tf.minimum(surr1, surr2)
        policy_loss = tf.reduce_mean(surr)
        # entropy = tf.reduce_mean(self.Policy.dist.entropy())

        loss_vf = tf.math.squared_difference(rewards + gamma * v_pred_next, v_pred)
        loss_vf = tf.reduce_mean(loss_vf)

        total_loss = -(policy_loss - c_1 * loss_vf + c_2 * entropy)  # c1, c2 is weight

        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('entropy', entropy)
        tf.summary.scalar('total', total_loss)

        return total_loss
    return loss

class PPOTrain:
    def __init__(self, Policy: Policy_net, Old_Policy: Policy_net):
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

        self.Old_Policy.set_trainable_variables(self.Policy.get_trainable_variables())
        self.optimizer = tf.optimizers.Adam(learning_rate=5e-5, epsilon=1e-5)
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.Policy.policy_model)


    def train(self, obs, actions, gaes, rewards, v_preds_next):
        with tf.GradientTape() as tape:
            v_preds = self.Policy.get_preds(obs, training=True)
            loss_fn = gail_loss(v_preds, v_preds_next, rewards, gaes)
            mean, std = self.Policy.mean_std(obs, training=True)
            old_mean, old_std = self.Old_Policy.mean_std(obs, training=True)
            log_prob = self.Policy.log_prob(mean, std, actions)
            old_log_prob = self.Old_Policy.log_prob(old_mean, old_std, actions)
            loss = loss_fn(log_prob, old_log_prob)
        grads = tape.gradient(loss, self.Policy.get_trainable_variables())
        self.optimizer.apply_gradients(zip(grads, self.Policy.get_trainable_variables()))

    def train(self, obs, actions, gaes, rewards, v_preds_next):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes})

    def assign_policy_parameters(self):
        self.Old_Policy.set_trainable_variables(self.Policy.get_trainable_variables())

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + gamma * gaes[t + 1]
        return gaes

    def reset_states(self):
        self.Policy.reset_states()
        self.Old_Policy.reset_states()

    def save(self):
        checkpoint_directory = f"weights/gail/policy"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)