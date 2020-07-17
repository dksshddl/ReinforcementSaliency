import tensorflow as tf


# import tensorlayer as tl
# from tensorlayer.layers import (Input, Conv2d, BatchNorm2d, Elementwise, SubpixelConv2d, Flatten, Dense)
# from tensorlayer.models import Model


def get_G(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    g_init = tf.random_normal_initializer(1., 0.02)

    nin = tf.keras.layers.Input(input_shape)
    n = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), activation=tf.nn.relu, padding='same', kernel_initializer=w_init)(
        nin)
    temp = n

    # B residual blocks
    for i in range(16):
        nn = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same',
                                    kernel_initializer=w_init, bias_initializer=None)(n)
        nn = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(nn)
        nn = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same',
                                    kernel_initializer=w_init, bias_initializer=None)(nn)
        nn = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(nn)
        nn = tf.keras.layers.Add()([n, nn])
        n = nn

    n = tf.keras.layers.Conv2D(64, (3, 3), (1, 1), padding='same', kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=g_init)(n)
    n = tf.keras.layers.Add()([n, temp])
    # B residual blacks end

    n = tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same', kernel_initializer=w_init)(n)
    n = Subpixel(filters=3, kernel_size=3, scale=2, activation="relu", padding="same")(n)

    n = tf.keras.layers.Conv2D(256, (3, 3), (1, 1), activation=None, padding='same', kernel_initializer=w_init)(n)
    n = Subpixel(filters=3, kernel_size=3, scale=2, activation="relu", padding="same")(n)

    nn = tf.keras.layers.Conv2D(3, (1, 1), (1, 1), activation=tf.nn.tanh, padding='same', kernel_initializer=w_init)(n)
    G = tf.keras.Model(inputs=nin, outputs=nn, name="generator")
    G.summary()
    return G


def get_D(input_shape):
    w_init = tf.random_normal_initializer(stddev=0.02)
    gamma_init = tf.random_normal_initializer(1., 0.02)
    df_dim = 64
    lrelu = tf.nn.leaky_relu
    nin = tf.keras.layers.Input(input_shape)
    n = tf.keras.layers.Conv2D(df_dim, (4, 4), (2, 2), activation=lrelu, padding='same', kernel_initializer=w_init)(nin)

    n = tf.keras.layers.Conv2D(df_dim * 2, (4, 4), (2, 2), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 4, (4, 4), (2, 2), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 8, (4, 4), (2, 2), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 16, (4, 4), (2, 2), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 32, (4, 4), (2, 2), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 16, (1, 1), (1, 1), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 8, (1, 1), (1, 1), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    nn = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)

    n = tf.keras.layers.Conv2D(df_dim * 2, (1, 1), (1, 1), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(nn)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 2, (3, 3), (1, 1), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Conv2D(df_dim * 8, (3, 3), (1, 1), padding='same',
                               kernel_initializer=w_init, bias_initializer=None)(n)
    n = tf.keras.layers.BatchNormalization(gamma_initializer=gamma_init)(n)
    n = tf.keras.layers.Add()([n, nn])

    n = tf.keras.layers.Flatten()(n)
    no = tf.keras.layers.Dense(1, bias_initializer=w_init)(n)
    D = tf.keras.Model(inputs=nin, outputs=no, name="discriminator")
    D.summary()
    return D


class Subpixel(tf.keras.layers.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 scale,
                 padding='valid',
                 data_format=None,
                 strides=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Subpixel, self).__init__(
            filters=scale * scale * filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.scale = scale

    def _phase_shift(self, I):
        scale = self.scale
        bsize, a, b, c = I.get_shape().as_list()
        bsize = tf.keras.backend.shape(I)[0]  # Handling Dimension(None) type for undefined batch dim
        X = tf.keras.backend.reshape(I, [bsize, a, b, int(c / (scale * scale)), scale,
                                         scale])  # bsize, a, b, c/(scale*scale), scale, scale

        X = tf.keras.backend.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, scale, scale, c/(scale*scale)
        # Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:, i, :, :, :, :] for i in range(a)]  # a, [bsize, b, scale, scale, c/(scale*scale)
        X = tf.keras.backend.concatenate(X, 2)  # bsize, b, a*scale, scale, c/(scale*scale)
        X = [X[:, i, :, :, :] for i in range(b)]  # b, [bsize, scale, scale, c/(scale*scale)
        X = tf.keras.backend.concatenate(X, 2)  # bsize, a*scale, b*scale, c/(scale*scale)
        return X

    def call(self, inputs):
        return self._phase_shift(super(Subpixel, self).call(inputs))

    def compute_output_shape(self, input_shape):
        unshifted = super(Subpixel, self).compute_output_shape(input_shape)
        return (
            unshifted[0], self.scale * unshifted[1], self.scale * unshifted[2],
            unshifted[3] / (self.scale * self.scale))

    def get_config(self):
        config = super(tf.keras.layers.Conv2D, self).get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.scale * self.scale
        config['scale'] = self.scale
        return config


class SubpixelConv2D(tf.keras.layers.Layer):
    """ Subpixel Conv2D Layer
    upsampling a layer from (h, w, c) to (h*r, w*r, c/(r*r)),
    where r is the scaling factor, default to 4
    # Arguments
    upsampling_factor: the scaling factor
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        the second and the third dimension increased by a factor of
        `upsampling_factor`; the last layer decreased by a factor of
        `upsampling_factor^2`.
    # References
        Real-Time Single Image and Video Super-Resolution Using an Efficient
        Sub-Pixel Convolutional Neural Network Shi et Al. https://arxiv.org/abs/1609.05158
    """

    def __init__(self, upsampling_factor=2, **kwargs):
        super(SubpixelConv2D, self).__init__(**kwargs)
        self.upsampling_factor = upsampling_factor

    def build(self, input_shape):
        last_dim = input_shape[-1]
        factor = self.upsampling_factor * self.upsampling_factor
        if last_dim % factor != 0:
            raise ValueError('Channel ' + str(last_dim) + ' should be of '
                                                          'integer times of upsampling_factor^2: ' +
                             str(factor) + '.')

    def call(self, inputs, **kwargs):
        return tf.depth_to_space(inputs, self.upsampling_factor)

    def get_config(self):
        config = {'upsampling_factor': self.upsampling_factor, }
        base_config = super(SubpixelConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        factor = self.upsampling_factor * self.upsampling_factor
        input_shape_1 = None
        if input_shape[1] is not None:
            input_shape_1 = input_shape[1] * self.upsampling_factor
        input_shape_2 = None
        if input_shape[2] is not None:
            input_shape_2 = input_shape[2] * self.upsampling_factor
        dims = [input_shape[0],
                input_shape_1,
                input_shape_2,
                int(input_shape[3] / factor)
                ]
        return tuple(dims)