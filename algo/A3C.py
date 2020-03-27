import threading
import signal
import sys

import tensorflow as tf

from a3c_agent import A3CAgent
from networks.a3c_network import A3CNetwork

jobs = 8  # parallel running thread number



class A3CMaster:
    def __init__(self):
        self.envs = []
        self.graph = tf.get_default_graph()

        self.shared_net = A3CNetwork()

        # shared optimizer
        self.shared_opt, self.global_step, self.summary_writer = self.shared_optimizer()

        self.jobs = []

        for thread_id in range(jobs):
            job = A3CAgent(thread_id, self)
            self.jobs.append(job)

        # session set
        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

        self.saver = tf.train.Saver()
        restore_model(self.sess, flags.train_dir, self.saver)
        self.global_step_val = 0

    def train(self):
        train_step = 0
        signal.signal(signal.SIGINT, lambda _: sys.exit(0))
        for job in self.jobs:
            job.start()
        for job in self.jobs:
            job.join()

    def shared_optimizer(self):
        with tf.device("/gpu:%d" % flags.gpu):
            # optimizer
            optimizer = tf.train.AdamOptimizer(flags.learn_rate, name="global_optimizer")
            global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)
            summary_writer = tf.train.SummaryWriter(flags.train_dir, graph_def=self.graph)
        return optimizer, global_step, summary_writer

    def test_sync(self):
        self.env.reset_env()
        done = False
        map(lambda job: self.sess.run(job.sync), self.jobs)
        step = 0
        while not done:
            step += 1
            action = random.choice(range(self.env.action_dim))
            for job in self.jobs:
                pi = job.local_net.get_policy(self.sess, self.env.state)
                val = job.local_net.get_value(self.sess, self.env.state)
                _, _, done = self.env.forward_action(action)
                print("step:", step, ", job:", job.thread_id, ", policy:", pi, ", value:", val)
            print()
        print("done!")