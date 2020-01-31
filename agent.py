import tensorflow as tf

from custom_env.envs import CustomEnv
from algo.ddpg import DDPG


class Agent:
    def __init__(self, env):
        if env is None:
            self.env = CustomEnv()
        else:
            self.env = env

        self.state_dim = self.env.observation_space
        self.action_dim = self.env.action_space
        self.action_max = self.action_dim.high
        self.action_min = self.action_dim.low
        self.sess = tf.Session()
        self.model = DDPG(self.sess, self.state_dim, self.action_dim, self.action_max, self.action_min)

        self.sess.run(tf.global_variables_initializer())

    def train(self, max_epochs):
        epoch = 0
        while epoch < max_epochs:
            ob, ac, target_video = self.env.reset()

            while True:
                next_ob, done, reward, next_ac = self.env.step(ac)

                if done:
                    break
                else:
                    # TODO replay buffer
                    self.model.update()
                    ob = next_ob
                    ac = next_ac
                    pass

            if epoch is not 0 and epoch % 10 is 0:
                self.model.save()

            epoch += 1
        pass

    def test(self, max_epochs):
        epoch = 0
        while epoch < max_epochs:
            acs, pred_acs = [], []
            ob, ac, target_video = self.env.reset("test")
            pred_ac = self.model.predict(ob)
            acs.append(ac)
            pred_acs.append(pred_ac)
            while True:
                next_ob, done, reward, next_ac = self.env.step(ac)
                if done:
                    break
                else:
                    pred_ac = self.model.predict(next_ob)



                    ob = next_ob
                    ac = next_ac
        pass
