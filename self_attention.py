from tensorflow.keras import layers
import keras.backend as K
import tensorflow as tf


class SelfAttention(layers.Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def get_config(self):
        config = super(SelfAttention, self).get_config().copy()
        return config

    def build(self, input_shape):
        self.num_channels = input_shape[-1]
        self.hw = input_shape[1] * input_shape[2]
        # Key
        self.conv_f = tf.keras.layers.Conv2D(self.num_channels // 8, 1)
        # Query
        self.conv_g = tf.keras.layers.Conv2D(self.num_channels // 8, 1)
        # Value
        self.conv_h = tf.keras.layers.Conv2D(self.num_channels // 2, 1)
        # Input feature convolution
        self.conv_o = tf.keras.layers.Conv2D(self.num_channels, 1)

    def call(self, x):
        f = self.conv_f(x)  # [_, h, w, c']
        g = self.conv_g(x)  # [_, h, w, c']
        h = self.conv_h(x)  # [_, h, w, c]

        f = tf.keras.layers.Reshape([self.hw, f.shape[-1]])(f)
        g = tf.keras.layers.Reshape([self.hw, g.shape[-1]])(g)
        h = tf.keras.layers.Reshape([self.hw, h.shape[-1]])(h)
        s = tf.matmul(g, f, transpose_b=True)  # [_, h*w, h*w]
        # Attention Map
        beta = tf.nn.softmax(s)

        o = tf.matmul(beta, h)  # [_, N, C]
        o = tf.keras.layers.Reshape(
            [x.shape[1], x.shape[2], self.num_channels//2])(o)
        o = self.conv_o(o)
        # [_, h, w, C]
        return x + o
