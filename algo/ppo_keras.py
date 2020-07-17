import os
import copy
import gc

import numpy as np
import gym

from tensorflow import keras
import tensorflow as tf

# from keras.models import Model
# from keras.layers import Input, Dense
# from keras import backend as K
# from keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import numba as nb

from custom_env.envs import CustomEnv

# from tensorboardX import SummaryWriter
from utils.ou_noise import OUNoise
from utils.replay import ReplayBuffer

ENV = 'LunarLander-v2'
CONTINUOUS = True

EPISODES = 100000

LOSS_CLIPPING = 0.2  # Only implemented clipping for the surrogate loss, paper said it was best
EPOCHS = 10
NOISE = 1.0  # Exploration noise

GAMMA = 0.99

BUFFER_SIZE = 2048
BATCH_SIZE = 4
NUM_ACTIONS = 2
NUM_STATE = [1, 5, 160, 320, 3]
HIDDEN_SIZE = 128
NUM_LAYERS = 2
ENTROPY_LOSS = 5e-3
LR = 1e-4  # Lower lr stabilises training greatly

DUMMY_ACTION, DUMMY_VALUE = np.zeros((1, NUM_ACTIONS)), np.zeros((1, 1))


@nb.jit
def exponential_average(old, new, b1):
    return old * b1 + (1 - b1) * new


def proximal_policy_optimization_loss(advantage, old_prediction):
    def loss(y_true, y_pred):
        prob = K.sum(y_true * y_pred, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)
        r = prob / (old_prob + 1e-10)
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING,
                                                       max_value=1 + LOSS_CLIPPING) * advantage) + ENTROPY_LOSS * -(
                prob * K.log(prob + 1e-10)))

    return loss


def proximal_policy_optimization_loss_continuous(advantage, old_prediction):
    def loss(y_true, y_pred):
        var = K.square(NOISE)
        pi = 3.1415926
        denom = K.sqrt(2 * pi * var)
        prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

        prob = prob_num / denom
        old_prob = old_prob_num / denom
        r = prob / (old_prob + 1e-10)

        return -K.mean(
            K.minimum(r * advantage, K.clip(r, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantage))

    return loss


class Agent:
    def __init__(self):

        # feature = tf.keras.applications.V2(include_top=False, pooling="avg", input_shape=(160, 160, 3))  # 2048
        feature = None

        # feature = tf.keras.Sequential()
        # feature.add(Conv2D(64, 3, 3, padding="same"))
        # feature.add(Conv2D(64, 3, 3, padding="same"))
        # feature.add(Conv2D(128, 3, 3, padding="same"))
        # feature.add(Conv2D(128, 3, 3, padding="same"))
        # feature.add(Conv2D(256, 3, 3, padding="same"))
        # feature.add(Conv2D(256, 3, 3, padding="same"))
        # feature.add(GlobalAveragePooling2D())
        self.critic = self.build_critic(feature)
        if CONTINUOUS is False:
            self.actor = self.build_actor()
        else:
            self.actor = self.build_actor_continuous(feature)

        self.checkpoint_actor = tf.train.Checkpoint(optimizer=self.actor.optimizer, model=self.actor)
        self.checkpoint_critic = tf.train.Checkpoint(optimizer=self.critic.optimizer, model=self.critic)

        self.env = CustomEnv()
        self.noise = OUNoise(2)
        self.replay_buffer = ReplayBuffer(20_000)

        print(self.env.action_space, 'action_space', self.env.observation_space, 'observation_space')
        self.episode = 0
        self.reset_env()
        self.val = False
        self.reward = []
        self.reward_over_time = []

        writer_path = os.path.join("log", "PPO")
        if not os.path.exists(writer_path):
            os.mkdir(writer_path)

        self.writer = tf.summary.create_file_writer(writer_path)
        self.gradient_steps = 0

    def build_actor(self):
        state_input = Input(shape=(NUM_STATE,))
        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(NUM_ACTIONS,))

        x = Dense(HIDDEN_SIZE, activation='tanh')(state_input)
        for _ in range(NUM_LAYERS - 1):
            x = Dense(HIDDEN_SIZE, activation='tanh')(x)

        out_actions = Dense(NUM_ACTIONS, activation='softmax', name='output')(x)

        model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_actor_continuous(self, feature):
        state_input = Input(batch_shape=NUM_STATE)
        advantage = Input(batch_shape=(None, 1))
        old_prediction = Input(batch_shape=(None, NUM_ACTIONS))

        # feature = tf.keras.Sequential()
        # feature.add(Conv2D(64, 7, 2))
        # feature.add(tf.keras.layers.MaxPool2D((3, 3), 2))
        # feature.add(Conv2D(64, 3, 1))
        # feature.add(Conv2D(64, 3, 1))
        # feature.add(Conv2D(128, 3, 2))
        # feature.add(Conv2D(128, 3, 1))
        # feature.add(Conv2D(256, 3, 2))
        # feature.add(Conv2D(256, 3, 1))
        # feature.add(Conv2D(512, 3, 2))
        # feature.add(Conv2D(512, 3, 1))
        # feature.add(GlobalAveragePooling2D())

        feature = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg", input_shape=(160, 320, 3), weights=None)  # 2048

        x = tf.keras.layers.TimeDistributed(feature)(state_input)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, stateful=True))(x)

        out = tf.keras.layers.Dense(NUM_ACTIONS, activation=tf.keras.activations.tanh)(x)
        model = tf.keras.models.Model(inputs=[state_input, advantage, old_prediction], outputs=out)
        model.compile(optimizer=Adam(lr=LR),
                      loss=[proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        model.summary()

        return model

    def build_critic(self, feature):
        state_input = Input(batch_shape=NUM_STATE)

        feature = tf.keras.applications.MobileNetV2(include_top=False, pooling="avg", input_shape=(160, 320, 3), weights=None)  # 2048

        x = tf.keras.layers.TimeDistributed(feature)(state_input)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False, stateful=True))(x)
        out = tf.keras.layers.Dense(1)(x)
        model = tf.keras.models.Model(inputs=state_input, outputs=out)

        model.compile(optimizer=Adam(lr=LR), loss='mse')

        model.summary()

        return model

    def reset_env(self):
        self.episode += 1
        if self.episode % 100 == 0:
            self.val = True
        else:
            self.val = False
        ob, ac, self.target_video = self.env.reset(trajectory=False, fx=0.3, fy=0.3, saliency=True, inference=False, target_video="09_MattSwift.mp4")
        # self.observation = np.array(ob, dtype=np.float32) / 255.  # 10, 160, 320, 3
        self.observation = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(ob))
        self.reward = []

    def get_action(self):
        p = self.actor.predict([self.observation.reshape(1, NUM_STATE), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            action = np.random.choice(NUM_ACTIONS, p=np.nan_to_num(p[0]))
        else:
            action = np.argmax(p[0])
        action_matrix = np.zeros(NUM_ACTIONS)
        action_matrix[action] = 1
        return action, action_matrix, p

    def get_action_continuous(self):
        p = self.actor([np.array([self.observation]), DUMMY_VALUE, DUMMY_ACTION])
        if self.val is False:
            # action = action_matrix = p[0] + np.random.normal(loc=0, scale=NOISE, size=p[0].shape)  # Gaussian noise
            action = action_matrix = p[0] + self.noise.noise()  # ou noise
        else:
            action = action_matrix = p[0]
        return action, action_matrix, p

    def transform_reward(self):
        with self.writer.as_default():
            if self.val is True:
                tf.summary.scalar("Val episode reward: " + self.target_video, np.sum(self.reward), self.episode)
            else:
                tf.summary.scalar("Episode reward: " + self.target_video, np.sum(self.reward), self.episode)
        for j in range(len(self.reward) - 2, -1, -1):
            self.reward[j] += self.reward[j + 1] * GAMMA

    def get_batch(self):
        batch = [[], [], [], []]
        tmp_batch = [[], [], []]

        while len(batch[0]) < BUFFER_SIZE:
            if CONTINUOUS is False:
                action, action_matrix, predicted_action = self.get_action()
            else:
                action, action_matrix, predicted_action = self.get_action_continuous()
            observation, reward, done, info = self.env.step(action)
            self.reward.append(reward)

            tmp_batch[0].append(self.observation)
            tmp_batch[1].append(action_matrix)
            tmp_batch[2].append(predicted_action)
            self.observation = observation

            if done:
                self.transform_reward()
                if self.val is False:
                    for i in range(len(tmp_batch[0])):
                        obs, action, pred = tmp_batch[0][i], tmp_batch[1][i], tmp_batch[2][i]
                        r = self.reward[i]
                        batch[0].append(obs)
                        batch[1].append(action)
                        batch[2].append(pred)
                        batch[3].append(r)
                tmp_batch = [[], [], []]
                self.reset_env()

        obs, action, pred, reward = np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), np.reshape(
            np.array(batch[3]), (len(batch[3]), 1))
        pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
        return obs, action, pred, reward

    def run(self):
        while self.episode < EPISODES:
            history = [[], [], [], []]
            while True:
                if CONTINUOUS:
                    action, action_matrix, predicted_action = self.get_action_continuous()
                else:
                    action, action_matrix, predicted_action = self.get_action()
                observation, reward, done, info = self.env.step(action)

                self.reward.append(reward)
                history[0].append(self.observation)  # observation
                history[1].append(action)  # action
                history[2].append(predicted_action[0])  # pred
                # history[3].append(reward)  # reward
                self.observation = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(observation))

                if done:
                    print(f"episode {self.target_video}, reward {np.sum(self.reward)}, step {self.episode}")
                    self.transform_reward()
                    history[3] = copy.deepcopy(self.reward)
                    if self.val is False:
                        self.replay_buffer.append(history)
                    self.actor.reset_states()
                    self.critic.reset_states()
                    self.reset_env()

                    break
            if len(self.replay_buffer) == 10:
                self.train()
            if self.episode % 50 == 0:
                self.save()

    def save(self):
        weight_path = "weights/PPO"
        if not os.path.exists(weight_path):
            os.mkdir(weight_path)

        checkpoint_directory_actor = f"weights/PPO/ACTOR"
        checkpoint_directory_critic = f"weights/PPO/CRITIC"

        if not os.path.exists(checkpoint_directory_actor):
            os.mkdir(checkpoint_directory_actor)
        if not os.path.exists(checkpoint_directory_critic):
            os.mkdir(checkpoint_directory_critic)

        checkpoint_prefix_actor = os.path.join(checkpoint_directory_actor, "ckpt")
        checkpoint_prefix_critic = os.path.join(checkpoint_directory_critic, "ckpt")
        self.checkpoint_actor.save(file_prefix=checkpoint_prefix_actor)
        self.checkpoint_critic.save(file_prefix=checkpoint_prefix_critic)

    def train(self):
        print("train start")
        batches = self.replay_buffer.get_batch(BATCH_SIZE)  # --> [batch_size, action, reward]
        mse = tf.keras.losses.MeanSquaredError()

        batch_actor_grads = []
        batch_critic_grads = []
        batch_actor_loss = []
        batch_critic_loss = []

        for batch in batches:
            obs, action, pred, reward = batch[0], batch[1], batch[2], batch[3]
            reward = np.reshape(reward, [-1, 1])

            a_grads = []
            q_grads = []
            actor_loss = []
            critic_loss = []

            for i in range(len(obs)):
                o, a, p, r = obs[i], action[i], pred[i], reward[i]

                with tf.GradientTape(persistent=True) as tape:
                    pred_values = self.critic(np.array([o]), training=True)
                    actions = self.actor(np.array([o]), training=True)
                    advantage = r - pred_values
                    old_prediction = p
                    loss = proximal_policy_optimization_loss_continuous(advantage, old_prediction)
                    a_loss = loss(a, actions)
                    c_loss = mse(r, pred_values)
                grad_actor = tape.gradient(a_loss, self.actor.trainable_weights)
                grad_critic = tape.gradient(c_loss, self.critic.trainable_weights)
                a_grads.append(grad_actor)
                q_grads.append(grad_critic)
                actor_loss.append(float(a_loss))
                critic_loss.append(float(c_loss))
                del tape
            q_grads = np.transpose(q_grads)
            a_grads = np.transpose(a_grads)
            q_grads = q_grads.tolist()
            a_grads = a_grads.tolist()
            q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
            a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
            batch_actor_grads.append(a_grads)
            batch_critic_grads.append(q_grads)
            batch_critic_loss.append(np.mean(critic_loss))
            batch_actor_loss.append(np.mean(actor_loss))
            self.actor.reset_states()
            self.critic.reset_states()
        q_grads = np.transpose(batch_critic_grads)
        a_grads = np.transpose(batch_actor_grads)
        q_grads = q_grads.tolist()
        a_grads = a_grads.tolist()
        q_grads = [tf.reduce_mean(i, axis=0) for i in q_grads]
        a_grads = [tf.reduce_mean(i, axis=0) for i in a_grads]
        with self.writer.as_default():
            tf.summary.scalar('Actor loss', np.mean(batch_actor_loss), self.gradient_steps)
            tf.summary.scalar('Critic loss', np.mean(batch_critic_loss), self.gradient_steps)
        self.critic.optimizer.apply_gradients(zip(q_grads, self.critic.trainable_weights))
        self.actor.optimizer.apply_gradients(zip(a_grads, self.actor.trainable_weights))

        self.gradient_steps += 1
        self.replay_buffer.clear()


class ActorGenerator(tf.keras.utils.Sequence):
    def __init__(self, obs, adv, pred, acs):
        self.obs, self.adv, self.pred, self.acs = obs, adv, pred, acs

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, item):
        print(np.shape([self.obs[item]]), np.shape([self.adv[item]]), np.shape([self.pred[item]]))
        obs, adv, pred = map(np.array, [[self.obs[item]], [self.adv[item]], [self.pred[item]]])
        return [obs, adv, pred], np.array([self.acs[item]])


class CriticGenerator(tf.keras.utils.Sequence):
    def __init__(self, obs, reward):
        self.obs, self.reward = obs, reward

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, item):
        return np.array([self.obs[item]]), np.array([self.reward[item]])


if __name__ == '__main__':
    ag = Agent()
    ag.run()
