import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = list(env.observation_space.shape)
        act_space = list(env.action_space.shape)

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + ob_space, name='obs')
            self.acs = tf.placeholder(dtype=tf.float32, shape=[None] + act_space, name='acs')
            # actor
            with tf.variable_scope('policy_net'):
                feature = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights=None)
                x = tf.keras.layers.TimeDistributed(feature)(self.obs)
                x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

                self.act_probs = tf.keras.layers.Dense(act_space[0], activation='linear')(x)

            # critic
            with tf.variable_scope('value_net'):
                feature = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=False, weights=None)
                x = tf.keras.layers.TimeDistributed(feature)(self.obs)
                x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

                x = tf.keras.layers.concatenate([x, self.acs])
                self.v_preds = tf.keras.layers.Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.RandomUniform())(x)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, acs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_probs, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_probs, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
