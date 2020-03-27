import tensorflow as tf

batch_size = None


def generate_critic_network(state_dim, action_dim, trainable=True):
    state_in = tf.keras.layers.Input(batch_shape=[batch_size] + state_dim)
    action_in = tf.keras.layers.Input(batch_shape=[batch_size] + action_dim)

    weight_decay = tf.keras.regularizers.l2(0.001)
    final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=True, trainable=trainable)(state_in)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=True, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ConvLSTM2D(32, 3, 3, return_sequences=False, trainable=trainable)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    x = tf.keras.layers.Flatten(trainable=trainable)(x)

    y = tf.keras.layers.Dense(128, activation="relu")(action_in)
    y = tf.keras.layers.Dense(64, activation="relu")(y)
    concat = tf.keras.layers.concatenate([x, y])

    x = tf.keras.layers.Dense(1, activation="linear", activity_regularizer=weight_decay,
                              kernel_initializer=final_initializer, bias_initializer=final_initializer)(concat)

    model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.compat.v1.losses.mean_squared_error)
    return model


def generate_resnet_critic(state_dim, action_dim, trainable=True):
    state_in = tf.keras.layers.Input(shape=state_dim)
    action_in = tf.keras.layers.Input(shape=action_dim)

    weight_decay = tf.keras.regularizers.l2(0.001)
    conv_initializer1 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    conv_initializer2 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    conv_initializer3 = tf.keras.initializers.RandomUniform(-1 / 32, 1 / 32)
    lstm_initailizer = tf.keras.initializers.RandomUniform(-1 / 256, 1 / 256)
    dense_initializer = tf.keras.initializers.RandomUniform(-1 / 200, 1 / 200)
    final_initializer = tf.keras.initializers.RandomUniform(-0.0003, 0.0003)

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
    y = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=dense_initializer,
                              bias_initializer=dense_initializer)(action_in)
    y = tf.keras.layers.Dense(200, activation="relu", kernel_initializer=dense_initializer,
                              bias_initializer=dense_initializer)(y)
    concat = tf.keras.layers.concatenate([x, y])
    # x = tf.keras.layers.Dropout(0.5)(concat)
    x = tf.keras.layers.Dense(1, activation="linear", activity_regularizer=weight_decay,
                              kernel_initializer=final_initializer, bias_initializer=final_initializer)(concat)
    model = tf.keras.models.Model(inputs=[state_in, action_in], outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.compat.v1.losses.mean_squared_error)
    return model
