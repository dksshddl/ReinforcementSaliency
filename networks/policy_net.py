import tensorflow as tf
import numpy as np

from custom_env.envs import CustomEnv

# import tensorflow_probability as tfp

batch_size = 1
n_samples = 8

# class CustomKLDiagNormal(tfp.distributions.MultivariateNormalDiag):
#     """Multivariate Normal with diagonal covariance and our custom KL code."""
#     pass
#
#
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
#     with tf.name_scope(name or 'kl_divergence'):
#         mean0 = lhs.mean()
#         mean1 = rhs.mean()
#         logstd0 = tf.log(lhs.stddev())
#         logstd1 = tf.log(rhs.stddev())
#         logstd0_2, logstd1_2 = 2 * logstd0, 2 * logstd1
#         return 0.5 * (
#                 tf.reduce_sum(tf.exp(logstd0_2 - logstd1_2), -1) +
#                 tf.reduce_sum((mean1 - mean0) ** 2 / tf.exp(logstd1_2), -1) +
#                 tf.reduce_sum(logstd1_2, -1) - tf.reduce_sum(logstd0_2, -1) -
#                 mean0.shape[-1].value)
#

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
optimizer = tf.train.AdamOptimizer
learning_rate = 1e-4


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """
        if env is not None:
            ob_space = env.observation_space
            act_space = env.action_space
        else:
            ob_space = (84, 84, 3)

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[batch_size] + [n_samples] + list(ob_space
                                                                                                ),
                                      name='obs')

            # before_softplus_std_initailizer = tf.keras.initializers.constant(np.log(np.exp(init_std) - 1))
            init_output_weights = tf.initializers.variance_scaling(scale=init_output_factor)

            with tf.variable_scope('policy_net'):
                pstate_in = tf.keras.layers.Input(shape=[n_samples] + list(ob_space), batch_size=batch_size)
                x = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(256))(pstate_in)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=True, stateful=True)(inputs=x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, 3, return_sequences=False, stateful=True)(inputs=x)
                x = tf.keras.layers.Flatten()(x)
                out = tf.keras.layers.Dense(2, activation="tanh", kernel_initializer=init_output_weights)(x)
                # mean = tf.keras.layers.Dense(x, 2, activation="linear",
                #                                 kernel_initializer=tf.random_normal_initializer(stddev=np.sqrt(0.01/)))

                # std = tf.keras.activations.softplus(tf.get_variable("before_softplus_std", mean.shape[2:], tf.float32,

                #                                                     initializer=before_softplus_std_initailizer))
                # std = tf.tile(std[None, None], [tf.shape(mean)[0], tf.shape(mean)[1]] + [1] * (mean.shape.ndims - 2))
                #
                # policy = CustomKLDiagNormal(mean, std)
                self.policy_model = tf.keras.models.Model(inputs=pstate_in, outputs=out)
                mean = self.policy_model.output
                logstd = tf.get_variable(name='logstd', shape=[1, 2],
                                         initializer=tf.zeros_initializer())
                std = tf.zeros_like(mean) + tf.exp(logstd)
                self.dist = tf.distributions.Normal(loc=mean, scale=std)

                print(f"policy is {self.dist}")

            with tf.variable_scope('value_net'):
                vstate_in = tf.keras.layers.Input(shape=[n_samples] + list(ob_space), batch_size=batch_size)
                x = tf.keras.layers.TimeDistributed(tf.keras.layers.Masking(256))(vstate_in)
                x = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True, stateful=True)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=True, stateful=True)(inputs=x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ConvLSTM2D(64, 3, return_sequences=False, stateful=True)(inputs=x)
                x = tf.keras.layers.Flatten()(x)
                self.v_preds = tf.keras.layers.Dense(1, activation="linear")(x)

                self.value_model = tf.keras.models.Model(inputs=vstate_in, outputs=self.v_preds)

            self.action = tf.reshape(self.dist.sample(1), [2])
            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        action = tf.get_default_session().run([self.action], feed_dict={self.policy_model.input: obs})
        v_preds = self.value_model.predict(np.array(obs))
        return action, v_preds

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset_states(self):
        self.value_model.reset_states()
        self.policy_model.reset_states()


if __name__ == '__main__':
    session = tf.Session()
    aa = Policy_net("test", None)
    session.run(tf.global_variables_initializer())
    inp = np.zeros([1, 8, 84, 84, 3])
    a, b = aa.act(inp)
    print(a, b)
