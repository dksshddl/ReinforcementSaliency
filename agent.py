import numpy as np
import tensorflow as tf

from tensorflow.core.protobuf import rewriter_config_pb2

from custom_env.envs import CustomEnv
from algo.ddpg import DDPG
from utils.replay import ReplayBuffer


class Agent:
    def __init__(self, env, sess=None):
        if env is None:
            self.env = CustomEnv()
        else:
            self.env = env

        state_dim = list(self.env.observation_space.shape)  # [224,224,3]
        action_dim = list(self.env.action_space.shape)  # [2]
        action_max = env.action_space.high
        action_min = env.action_space.low
        if sess is None:
            self.sess = tf.Session()
        else:
            self.sess = sess
        self.model = DDPG(self.sess, state_dim, action_dim, action_max[0], action_min[0])
        self.replay_buffer = ReplayBuffer(10_000)
        self.sess.run(tf.global_variables_initializer())

    def train(self, max_epochs):
        epoch = 0
        global_step = 0
        train_indicator = None
        while epoch < max_epochs:
            ob, _, target_video = self.env.reset()

            while True:
                # TODO OU noise
                ac = self.model.actor.predict(np.array([ob]))
                next_ob, reward, done, _ = self.env.step(ac)
                global_step += 1
                # TODO replay buffer
                # transition = (ob, ac, reward, next_ob, done)
                # self.replay_buffer.append(transition)
                # batch = self.replay_buffer.get_batch(30) --> [30, :, :, :, :]

                # obs, acs, rewards, next_obs, dones = map(np.array, zip(*batch))
                # y_t = rewards.copy()
                # obs = np.asarray([e[0] for e in batch])
                # acs = np.asarray([e[1] for e in batch])
                # rewards = np.asarray([e[2] for e in batch])
                # next_obs = np.asarray([e[3] for e in batch])
                # dones = np.asarray([e[4] for e in batch])
                # y_t = np.asarray([e[2] for e in batch])


                # for k in range(len(batch)):
                #     if dones[k]:
                #         y_t[k] = rewards[k]
                #     else:
                #         y_t[k] = rewards[k] + 0.99 * target_q_value[k]
                # loss = 0


                # self.model.update()

                if done:
                    break
                else:
                    next_acs = self.model.target_actor.predict(np.array([next_ob]))
                    target_q_value = self.model.target_critic.predict([np.array([next_ob]), np.array(next_acs)])
                    y = reward + 0.99 * np.array([target_q_value])
                    self.model.train_critic(np.array([ob]), np.array(ac), np.array(y[0]), global_step)
                    a_for_grad = self.model.actor.predict(np.array([ob]))
                    grads = self.model.gradients(np.array([ob]), np.array(a_for_grad))
                    self.model.train_actor(np.array([ob]), np.array(grads))
                    self.model.target_actor_train()
                    self.model.target_critic_train()
                    ob = next_ob

            self.model.reset_state()
            if epoch is not 0 and epoch % 50 is 0:
                self.model.save(epoch)

            epoch += 1

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
                    pred_ac = self.model.predict(ob)
                    acs.append(ac)
                    pred_acs.append(pred_ac)

                    ob = next_ob
                    ac = next_ac


if __name__ == '__main__':
    tf.keras.backend.clear_session()  # For easy reset of notebook state.

    config_proto = tf.ConfigProto()
    off = rewriter_config_pb2.RewriterConfig.OFF
    config_proto.graph_options.rewrite_options.arithmetic_optimization = off
    session = tf.Session(config=config_proto)

    my_env = CustomEnv()
    a = Agent(my_env, session)
    a.train(5000)
