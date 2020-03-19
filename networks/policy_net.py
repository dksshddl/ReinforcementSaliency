import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')

            with tf.variable_scope('policy_net'):
                x = tf.keras.layers.ConvLSTM2D(128, 5, return_sequences=True, stateful=True)(self.obs)
                x = tf.keras.layers.ConvLSTM2D(64, 5, return_sequences=True, stateful=True)(inputs=x)
                x = tf.keras.layers.ConvLSTM2D(32, 5, return_sequences=False, stateful=True)(inputs=x)
                x = tf.keras.layers.Flatten()(x)
                self.act_probs = tf.keras.layers.Dense(act_space.n, activation="sigmoid")(x)

            with tf.variable_scope('value_net'):
                x = tf.keras.layers.ConvLSTM2D(128, 5, return_sequences=True, stateful=True)(self.obs)
                x = tf.keras.layers.ConvLSTM2D(64, 5, return_sequences=True, stateful=True)(inputs=x)
                x = tf.keras.layers.ConvLSTM2D(32, 5, return_sequences=False, stateful=True)(inputs=x)
                x = tf.keras.layers.Flatten()(x)
                self.v_preds = tf.keras.layers.Dense(1, activation="linear")(x)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=1)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)