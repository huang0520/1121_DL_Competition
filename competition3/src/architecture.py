import math

import tensorflow as tf
from tensorflow import keras


def embedding_layer(embedding_max_freq, embedding_dim):
    def sinusoidal_embedding(x: tf.Tensor):
        min_freq = 1.0
        max_freq = embedding_max_freq
        freqs = tf.exp(
            tf.linspace(
                tf.math.log(min_freq),
                tf.math.log(max_freq),
                num=embedding_dim // 2,
            )
        )
        angular_speed = tf.cast(2 * math.pi * freqs, tf.float32)

        embeddings = tf.concat(
            [tf.sin(x * angular_speed), tf.cos(x * angular_speed)], axis=-1
        )

        return embeddings

    return keras.layers.Lambda(sinusoidal_embedding)


def swish_glu(width):
    def apply(x):
        x = keras.layers.Dense(width * 2)(x)
        a, b = tf.split(x, 2, axis=-1)
        return a * keras.layers.Activation("swish")(b)

    return apply


def residual_block(width):
    def apply(input):
        x, n = input
        residual = keras.layers.Conv2D(width, 1)(x) if x.shape[-1] != width else x

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Conv2D(width, 3, padding="same")(x)

        n = keras.layers.Activation("swish")(n)
        n = keras.layers.Dense(width)(n)

        x = keras.layers.Add()([x, n])

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Activation("swish")(x)
        x = keras.layers.Conv2D(width, 3, padding="same")(x)

        x = keras.layers.Add()([x, residual])

        return x

    return apply


def basic_transformer_block(width, n_head=2):
    norm = keras.layers.GroupNormalization(epsilon=1e-5)
    attn = keras.layers.MultiHeadAttention(num_heads=n_head, key_dim=width // n_head)

    def apply(input):
        x, t = input
        x = keras.layers.Dense(width, use_bias=False)(x)
        t = keras.layers.Dense(width, use_bias=False)(t)

        x = attn(norm(x), norm(x)) + x
        x = attn(norm(x), t) + x
        return keras.layers.Dense(width)(swish_glu(width * 4)(norm(x))) + x

    return apply


def spatial_transformer_block(width, n_head=2):
    def apply(input):
        x, t = input
        _, h, w, c = x.shape
        residual = x

        x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
        x = keras.layers.Conv2D(width, 1)(x)
        x = keras.layers.Reshape((h * w, c))(x)
        x = basic_transformer_block(width, n_head)([x, t])
        x = keras.layers.Reshape((h, w, c))(x)

        return keras.layers.Conv2D(width, 1)(x) + residual

    return apply


def downsampling_block(width):
    def apply(input):
        return keras.layers.Conv2D(width, 3, strides=2, padding="same")(input)

    return apply


def upsampling_block(width):
    def apply(input):
        x = keras.layers.UpSampling2D(2, interpolation="nearest")(input)
        return keras.layers.Conv2D(width, 3, padding="same")(x)

    return apply


def get_network(
    image_size=64,
    image_embedding_dim=64,
    noise_embedding_dim=512,
    text_embedding_shape=(20, 512),
    widths=(64, 96, 128, 160),
    block_depth=2,
    embedding_max_frequency=1000.0,
):
    noisy_image = keras.layers.Input(shape=(image_size, image_size, 3))
    noise_power = keras.layers.Input(shape=(1, 1, 1))
    t_emb = keras.layers.Input(shape=text_embedding_shape)

    n_emb = embedding_layer(embedding_max_frequency, noise_embedding_dim)(noise_power)
    n_emb = keras.layers.Dense(noise_embedding_dim, activation="swish")(n_emb)
    n_emb = keras.layers.Dense(noise_embedding_dim)(n_emb)

    x = keras.layers.Conv2D(image_embedding_dim, 1)(noisy_image)

    skips = []
    x = residual_block(widths[0])([x, n_emb])
    x = residual_block(widths[0])([x, n_emb])
    x = downsampling_block(widths[0])(x)
    skips.append(x)

    for width in widths[1:-1]:
        x = residual_block(width)([x, n_emb])
        x = spatial_transformer_block(width, n_head=2)([x, t_emb])
        skips.append(x)

        x = residual_block(width)([x, n_emb])
        x = spatial_transformer_block(width, n_head=2)([x, t_emb])
        x = downsampling_block(width)(x)
        skips.append(x)

    x = residual_block(widths[-1])([x, n_emb])
    x = residual_block(widths[-1])([x, n_emb])
    skips.append(x)

    x = residual_block(widths[-1])([x, n_emb])
    x = spatial_transformer_block(widths[-1], n_head=2)([x, t_emb])
    x = residual_block(widths[-1])([x, n_emb])

    x = keras.layers.Concatenate()([x, skips.pop()])
    x = residual_block(widths[-1])([x, n_emb])
    x = residual_block(widths[-1])([x, n_emb])

    for width in reversed(widths[1:-1]):
        x = keras.layers.Concatenate()([x, skips.pop()])
        x = upsampling_block(width)(x)
        x = residual_block(width)([x, n_emb])
        x = spatial_transformer_block(width, n_head=2)([x, t_emb])

        x = keras.layers.Concatenate()([x, skips.pop()])
        x = residual_block(width)([x, n_emb])
        x = spatial_transformer_block(width, n_head=2)([x, t_emb])

    x = keras.layers.Concatenate()([x, skips.pop()])
    x = upsampling_block(widths[0])(x)
    x = residual_block(widths[0])([x, n_emb])
    x = residual_block(widths[0])([x, n_emb])

    x = keras.layers.GroupNormalization(epsilon=1e-5)(x)
    x = keras.layers.Activation("swish")(x)
    x = keras.layers.Conv2DTranspose(3, 1)(x)

    return keras.Model(
        inputs=[noisy_image, noise_power, t_emb],
        outputs=x,
        name="noise_predictor",
    )
