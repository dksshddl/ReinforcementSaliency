import threading
import random

import numpy as np
import tensorflow as tf

from custom_env.envs import CustomEnv
from networks.a3c_network import A3CNetwork

grad_clip = 40  # gradient clipping cut-off
learning_rate = 5e-4
gamma = 0.95  # discount ratio


class A3CAgent(threading.Thread):
    def __init__(self, thread_id, master):
        super().__init__(name=f"thread_{thread_id}")

        self.thread_id = thread_id
        self.master = master
        self.local_net = A3CNetwork()

        self.env = CustomEnv()

        self.sync = self.sync_network(master.shared_net)
        self.accum_grads = self.create_accumulate_gradients()
        self.do_accum_grads_ops = self.do_accum_grads_ops()
        self.reset_accum_grads_ops = self.reset_accum_grads_ops()

        clip_accum_grads = [tf.clip_by_value(grad, grad_clip, grad_clip) for grad in self.accum_grads]
        self.apply_gradients = master.shared_opt.apply_gradients(
            zip(clip_accum_grads, master.shared_net.get_vars()), global_step=master.global_step)
        self.summary_op = tf.merge_summary(summaries)

    def sync_network(self, source_net):
        sync_ops = []
        with tf.op_scope([], name=f"sync_ops_{self.thread_id}"):
            for (target_var, source_var) in zip(self.local_net.get_vars(), source_net.get_vars()):
                ops = tf.assign(target_var, source_var)
                sync_ops.append(ops)
            return tf.group(*sync_ops, name=f"sync_group_{self.thread_id}")

    def create_accumulate_gradients(self):
        accum_grads = []
        with tf.op_scope([self.local_net], name=f"create_accum_{self.thread_id}"):
            for var in self.local_net.get_vars():
                zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                name = var.name.replace(":", "_") + "_accum_grad"
                accum_grad = tf.Variable(zero, name=name, trainable=False)
                accum_grads.append(accum_grad.ref())
            return accum_grads

    def do_accumulate_gradients(self):
        net = self.local_net
        accum_grad_ops = []
        with tf.op_scope([], name=f"accum_ops_{self.thread_id}"):
            grads = net.shared_grads + net.policy_grads + net.value_grads
            for (grad, var, accum_grad) in zip(grads, net.get_vars(), self.accum_grads):
                name = var.name.replace(":", "_") + "_accum_grad_ops"
                accum_ops = tf.assign_add(accum_grad, grad, name=name)
                accum_grad_ops.append(accum_ops)
            return tf.group(*accum_grad_ops, name=f"accum_group_{self.thread_id}")

    def reset_accumulate_gradients(self):
        net = self.local_net
        reset_grad_ops = []
        with tf.op_scope([net], name="reset_grad_ops_%d" % self.thread_id):
            for var, accum_grad in zip(net.get_vars(), self.accum_grads):
                zero = tf.zeros(var.get_shape().as_list(), dtype=var.dtype)
                name = var.name.replace(":", "_") + "_reset_grad_ops"
                reset_ops = tf.assign(accum_grad, zero, name=name)
                reset_grad_ops.append(reset_ops)
            return tf.group(*reset_grad_ops, name="reset_accum_group_%d" % self.thread_id)

    def weighted_choose_action(self, pi_probs):
        r = random.uniform(0, sum(pi_probs))
        upto = 0
        for idx, prob in enumerate(pi_probs):
            if upto + prob >= r:
                return idx
            upto += prob
        return len(pi_probs) - 1

    def forward_explore(self, train_step):
        terminal = False
        t_start = train_step
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}
        while not terminal and (train_step - t_start <= flags.t_max):
            pi_probs = self.local_net.get_policy(self.master.sess, self.env.state)
            if random.random() < 0.8:
                action = self.weighted_choose_action(pi_probs)
            else:
                action = np.random.uniform(-1, 1, self.env.action_space)
            _, reward, terminal = self.env.forward_action(action)
            train_step += 1
            rollout_path["state"].append(self.env.state)
            one_hot_action = np.zeros(self.env.action_dim)
            one_hot_action[action] = 1
            rollout_path["action"].append(one_hot_action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(terminal)
        return train_step, rollout_path

    def run(self):
        sess = self.master.sess
        self.env.reset_env()
        loop = 0
        while flags.train_step <= flags.t_train:
            train_step = 0
            loop += 1
            # reset gradients
            sess.run(self.reset_accum_grads_ops)

            # sync variables
            sess.run(self.sync)

            # forward explore
            train_step, rollout_path = self.forward_explore(train_step)

            # rollout for discounted R values
            if rollout_path["done"][-1]:
                rollout_path["rewards"][-1] = 0
                self.env.reset()
                self.local_net.reset_lstm_state()
            else:
                rollout_path["rewards"][-1] = self.local_net.get_value(sess, rollout_path["state"][-1])
            rollout_path["returns"] = self.discount(rollout_path["rewards"])
            # accumulate gradients
            lc_net = self.local_net
            fetches = [self.do_accum_grads_ops, self.master.global_step]
            if loop % 10 == 0:
                fetches.append(self.summary_op)
            if flags.use_lstm:
                res = sess.run(fetches, feed_dict={lc_net.state_ph: rollout_path["state"],
                                                   lc_net.action_ph: rollout_path["action"],
                                                   lc_net.target_q_ph: rollout_path["returns"],
                                                   lc_net.initial_lstm_state: lc_net.lstm_state_out,
                                                   lc_net.sequence_length: [1]})
            else:
                res = sess.run(fetches, feed_dict={lc_net.state_ph: rollout_path["state"],
                                                   lc_net.action_ph: rollout_path["action"],
                                                   lc_net.target_q_ph: rollout_path["returns"]})
            if loop % 10 == 0:
                global_step, summary_str = res[1:3]
                self.master.summary_writer.add_summary(summary_str, global_step=global_step)
                self.master.global_step_val = int(global_step)
            # async update grads to global network
            sess.run(self.apply_gradients)
            flags.train_step += train_step
            # evaluate
            if loop % 10 == 0 and self.thread_id == 1:
                self.test_phase()
            if loop % 1000 == 0 and self.thread_id == 1:
                save_model(self.master.sess, flags.train_dir, self.master.saver, "a3c_model",
                           global_step=self.master.global_step_val)
