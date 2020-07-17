import os

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

import numpy as np

# import tensorflow_probability as tfp

batch_size = None
n_samples = None

# class CustomKLDiagNormal(tfp.distributions.MultivariateNormalDiag):
#     """Multivariate Normal with diagonal covariance and our custom KL code."""
#     pass


# @tfp.RegisterKL(CustomKLDiagNormal, CustomKLDiagNormal)
# def _custom_diag_normal_kl(lhs, rhs, name=None):  # pylint: disable=unused-argument
#     """Empirical KL divergence of two normals with diagonal covariance.
#     Args:
#       lhs: Diagonal Normal distribution.
#       rhs: Diagonal Normal distribution.
#       name: Name scope for the op.
#     Returns:
#       KL divergence from lhs to rhs.
#     """
# with tf.name_scope(name or 'kl_divergence'):
#     mean0 = lhs.mean()
#     mean1 = rhs.mean()
#     logstd0 = tf.log(lhs.stddev())
#     logstd1 = tf.log(rhs.stddev())
#     logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
#     return 0.5 * (
#             tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
#             tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
#             tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
#             mean0.shape[-1].value)


init_output_factor = 0.1
init_std = 0.35
# Losses
discount = 0.995
kl_target = 1e-2
kl_cutoff_factor = 2
kl_cutoff_coef = 1000
kl_init_penalty = 1
# Optimization
update_every = 30
update_epochs = 25
learning_rate = 1e-4


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        self.ob_space = env.observation_space.shape
        self.act_space = env.action_space.shape

        # before_softplus_std_initailizer = tf.keras.initializers.constant(np.log(np.exp(init_std) - 1))
        # mean = tf.keras.layers.Dense(x, 2, activation="linear",
        #                                 kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(0.01/)))

        # std = tf.keras.activations.softplus(tf.get_variable("before_softplus_std", mean.shape[2:], tf.float32,
        #                                                     initializer=before_softplus_std_initailizer))
        # std = tf.tile(std[None, None], [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
        #
        # policy = CustomKLDiagNormal(mean, std)
        self.create_actor()
        self.create_critic()

    def create_actor(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [n_samples] + list(self.ob_space))

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)
        mean = tf.keras.layers.Dense(2, activation="tanh")(x)

        self.policy_model = tf.keras.models.Model(inputs=state_in, outputs=mean)
        self.policy_model.summary()

        # logstd = tf.zeros_like(mean)
        # std = tf.exp(logstd)
        # self.dist = tfd.Normal(loc=mean, scale=std).log_prob()

    def mean_std(self, obs, training):
        mean = self.policy_model([obs], training=training)
        std = tf.exp(tf.zeros_like(mean))
        return mean, std

    def log_prob(self, mean, std, acs):
        return tfd.Normal(loc=mean, scale=std).log_prob(acs)

    def get_preds(self, obs, training):
        return self.value_model([obs], training=training)

    def entropy(self, mean, std):
        return tf.reduce_mean(tfd.Normal(loc=mean, scale=std).entropy())

    def create_critic(self):
        state_in = tf.keras.layers.Input(batch_shape=[batch_size] + [n_samples] + list(self.ob_space))

        feature = tf.keras.Sequential()
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        feature.add(tf.keras.layers.Conv2D(64, 3, 3, activation=tf.nn.leaky_relu))
        x = tf.keras.layers.TimeDistributed(feature)(state_in)
        x = tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(x)

        v_preds = tf.keras.layers.Dense(1, activation="linear")(x)
        self.value_model = tf.keras.models.Model(inputs=state_in, outputs=v_preds)
        self.value_model.summary()

    def act(self, obs, training):
        mean = self.policy_model([obs], training=training)
        std = tf.exp(tf.zeros_like(mean))
        action = tf.random.normal(mean.shape, mean=mean, stddev=std)
        v_preds = self.value_model([obs], training=training)
        return action, v_preds
        # return tf.get_default_session().run([self.action, self.v_preds],
        #                                     feed_dict={self.policy_model.inputs: obs, self.value_model.inputs: obs})
    def get_variables(self):
        return self.value_model.get_weights()

    def get_trainable_variables(self):
        return self.value_model.get_weights()

    def set_trainable_variables(self, weight):
        self.value_model.set_weights(weight)

    def reset_states(self):
        self.value_model.reset_states()
        self.policy_model.reset_states()

    def save(self):
        checkpoint_directory = f"weights/gail/discriminator"
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        self.checkpoint.save(file_prefix=checkpoint_prefix)