import tensorflow as tf

class GBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size=4, strides=2, padding="same", **kwargs):
        super().__init__(**kwargs)
        self.conv2d_trans = tf.keras.layers.Conv2DTranspose(
            out_channels,
            kernel_size,
            strides,
            padding,
            use_bias=False
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv2d_trans(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


class DBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        out_channels,
        kernel_size=4,
        strides=2,
        padding="same",
        alpha=0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.conv2d = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size,
            strides,
            padding,
            use_bias=False
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.LeakyReLU(alpha)

    def call(self, x):
        x = self.conv2d(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x


def build_generator(latent_dim=100, base_filters=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, 1, latent_dim)),
        GBlock(out_channels=base_filters * 8, strides=1, padding="valid"),
        GBlock(out_channels=base_filters * 4),
        GBlock(out_channels=base_filters * 2),
        GBlock(out_channels=base_filters),
        tf.keras.layers.Conv2DTranspose(
            3,
            kernel_size=4,
            strides=2,
            padding="same",
            use_bias=False,
            activation="tanh"
        )
    ])


def build_discriminator(base_filters=64):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)),
        DBlock(base_filters),
        DBlock(out_channels=base_filters * 2),
        DBlock(out_channels=base_filters * 4),
        DBlock(out_channels=base_filters * 8),
        tf.keras.layers.Conv2D(1, kernel_size=4, use_bias=False)
    ])