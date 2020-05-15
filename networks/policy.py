import tensorflow as tf
import numpy as np

batch_size = None
n_samples = 5


class ContinuousControlModel:

    def __init__(self, lr, brain, h_size, epsilon, max_step, normalize, num_layers):
        """
        Creates Continuous Control Actor-Critic model.
        :param brain: State-space size
        :param h_size: Hidden layer size
        """
        s_size = [84, 84, 3]
        a_size = 2

        self.normalize = normalize
        self._create_global_steps()
        self._create_reward_encoder()

        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [n_samples] + s_size)

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.Flatten()(x)

        self.batch_size = tf.placeholder(shape=None, dtype=tf.int32, name='batch_size')

        self.mu = tf.layers.dense(x, a_size, activation=None, use_bias=False,
                                  kernel_initializer=c_layers.variance_scaling_initializer(factor=0.01))
        self.log_sigma_sq = tf.get_variable("log_sigma_squared", [a_size], dtype=tf.float32,
                                            initializer=tf.zeros_initializer())
        self.sigma_sq = tf.exp(self.log_sigma_sq)

        self.epsilon = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='epsilon')

        self.output = self.mu + tf.sqrt(self.sigma_sq) * self.epsilon
        self.output_max = tf.identity(self.mu, name='action_max')
        self.output = tf.identity(self.output, name='action')

        a = tf.exp(-1 * tf.pow(tf.stop_gradient(self.output) - self.mu, 2) / (2 * self.sigma_sq))
        b = 1 / tf.sqrt(2 * self.sigma_sq * np.pi)
        self.probs = tf.multiply(a, b, name="action_probs")

        self.entropy = tf.reduce_sum(0.5 * tf.log(2 * np.pi * np.e * self.sigma_sq))

        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [n_samples] + s_size)

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation="relu"))
        feature.add(tf.keras.layers.BatchNormalization())
        x = tf.keras.layers.TimeDistributed(feature)(state_in)

        x = tf.keras.layers.Flatten()(x)

        self.value = tf.layers.dense(x, 1, activation=None, use_bias=False)
        self.value = tf.identity(self.value, name="value_estimate")

        self.old_probs = tf.placeholder(shape=[None, a_size], dtype=tf.float32, name='old_probabilities')

        self._create_ppo_optimizer(self.probs, self.old_probs, self.value, self.entropy, 0.0, epsilon, lr, max_step)
