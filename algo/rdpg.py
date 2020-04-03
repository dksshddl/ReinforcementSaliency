import tensorflow as tf
import numpy as np

from networks.Actor import Actor
from networks.Critic import Critic
from utils.config import *
from custom_env.envs import CustomEnv

# Recurrent Deterministic Policy Gradient
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

gamma = 0.95


class RDPG:
    def __init__(self, obs_dim, acs_dim):
        self.sess = tf.Session()
        self.obs_dim = list(obs_dim)
        self.acs_dim = [acs_dim]

        self.actor = Actor(self.sess, self.obs_dim, self.acs_dim, "actor")
        self.actor_target = Actor(self.sess, self.obs_dim, self.acs_dim, "actor_target")

        self.critic = Critic(self.sess, self.obs_dim, self.acs_dim, "critic")
        self.critic_target = Critic(self.sess, self.obs_dim, self.acs_dim, "critic_target")

        self.sess.run(tf.global_variables_initializer())

        writer_path = os.path.join(log_path, RDPG)
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)
        self.writer = tf.compat.v1.summary.FileWriter(writer_path, self.sess.graph)

        self.soft_target_update(tau=1)

        self.env = CustomEnv()

        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(5000)

    def train(self, max_epochs):
        epoch = 0
        global_step = 0
        w = open(os.path.join("log", DDPG_RESNET, "reward.txt"), "a")
        while epoch < max_epochs:
            ob, _, target_video = self.env.reset(trajectory=False, fx=1, fy=1)
            history = {"observations": [], "next_observations": [], "actions": [], "rewards": []}

            ep_reward = 0
            while True:

                pred_ac = self.actor.get_action()
                noise_ac = np.squeeze(pred_ac) + self.noise.noise()
                next_ob, reward, done, next_ac = self.env.step(noise_ac)

                reward = reward * 10

                if done:
                    history["observations"].append(next_ob)  # 마지막 처리
                    break

                history["observations"].append(ob)
                history["actions"].append(noise_ac)
                history["rewards"].append(reward)

                ep_reward += reward
                ob = next_ob

            print(f"episode {target_video}, reward {reward}")
            reward_summary = tf.Summary(value=[tf.Summary.Value(tag=target_video, simple_value=ep_reward)])
            global_step += 1
            self.writer.add_summary(reward_summary, global_step)
            self.learn(history)

    def soft_target_update(self, tau=0.001):
        critic_weight = self.critic.get_weights()
        critic_target_weight = self.critic_target.get_weights()

        actor_weight = self.actor.get_weights()
        actor_target_weight = self.actor_target.get_weights()

        for i in range(len(critic_weight)):
            critic_target_weight[i] = tau * critic_weight[i] + (1 - tau) * critic_target_weight[i]
        for i in range(len(actor_weight)):
            actor_target_weight[i] = tau * actor_weight[i] + (1 - tau) * actor_target_weight[i]

        self.critic_target.set_weights(critic_target_weight)
        self.actor_target.set_weights(actor_target_weight)

    def learn(self, buf_data):
        self.replay_buffer.append(buf_data)

        if len(self.replay_buffer) >= 1_000:
            batches = self.replay_buffer.get_batch(32)  # --> [40, :, :, :, :]
            print("train start ", np.shape(batches))

            omega = []
            theta = []
            for history in batches:  # batch --> [5, 5, 5, 5, 5]

                observations = history["observations"]
                actions = history["actions"]
                rewards = history["rewards"]
                target_value = []  # target_value vector

                # compute target_value
                h_t1 = observations[0]
                for t in range(len(rewards)):
                    h_t1 = np.concatenate([h_t1, observations[t+1]])

                    a_t1 = self.actor_target.get_action(h_t1)
                    q_t1 = self.critic_target.get_q_value(h_t1, a_t1)
                    y_t = rewards[t] + gamma * q_t1.item()

                    q = self.critic.get_q_value(h_t1, actions[t])
                    a_for_grad = self.actor.get_action(h_t1)

                    omega.append()
                    theta.append()

