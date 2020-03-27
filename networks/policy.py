import tensorflow as tf

batch_size = None


def generate_actor_network(state_dim, action_dim, trainable=True):
    state_in = tf.keras.layers.Input(batch_shape=[batch_size] + state_dim)

    final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=True, trainable=trainable)(state_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=True, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=False, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Flatten(trainable=trainable)(x)

    x = tf.keras.layers.Dense(2, activation="tanh", bias_initializer=final_initializer,
                              kernel_initializer=final_initializer)(x)

    x = tf.keras.layers.Dense(2, trainable=trainable)(x)

    model = tf.keras.models.Model(inputs=state_in, outputs=x)
    return model


def generate_resnet_actor(state_dim, action_dim, trainable=True):
    state_in = tf.keras.layers.Input(shape=state_dim)

    conv_initializer1 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    conv_initializer2 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    conv_initializer3 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)
    lstm_initailizer = tf.keras.initializers.RandomUniform(-1 / 256, 1 / 256)
    # feature = tf.keras.applications.ResNet50(include_top=False, weights=None)
    x = tf.keras.models.Sequential()
    x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer1,
                                 bias_initializer=conv_initializer1))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer2,
                                 bias_initializer=conv_initializer2))
    x.add(tf.keras.layers.BatchNormalization())
    x.add(tf.keras.layers.Conv2D(32, 3, 3, activation=tf.nn.leaky_relu, kernel_initializer=conv_initializer3,
                                 bias_initializer=conv_initializer3))
    x.add(tf.keras.layers.BatchNormalization())

    x = tf.keras.layers.TimeDistributed(x)(state_in)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(256, kernel_initializer=lstm_initailizer, bias_initializer=lstm_initailizer,
                             recurrent_initializer=lstm_initailizer))(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(2, activation="tanh", bias_initializer=final_initializer,
                              kernel_initializer=final_initializer)(x)
    model = tf.keras.models.Model(inputs=state_in, outputs=x)
    return model
