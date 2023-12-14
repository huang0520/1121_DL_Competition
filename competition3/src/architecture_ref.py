import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def get_network(
    image_size=64,
    noise_embedding_max_frequency=1000.0,
    noise_embedding_dims=64,
    image_embedding_dims=64,
    text_embedding_dims=64,
    block_depth=2,
    widths=(64, 96, 128, 160),
    attentions=(False, False, False, False),
    patch_size=1,
):
    def EmbeddingLayer(embedding_max_frequency, embedding_dims):
        def sinusoidal_embedding(x):
            embedding_min_frequency = 1.0
            frequencies = tf.exp(
                tf.linspace(
                    tf.math.log(embedding_min_frequency),
                    tf.math.log(embedding_max_frequency),
                    embedding_dims // 2,
                )
            )
            angular_speeds = 2.0 * math.pi * frequencies
            embeddings = tf.concat(
                [
                    tf.sin(angular_speeds * x),
                    tf.cos(angular_speeds * x),
                ],
                axis=3,
            )
            return embeddings

        def forward(x):
            x = layers.Lambda(sinusoidal_embedding)(x)
            return x

        return forward

    def ResidualBlock(width, attention):
        def forward(x):
            x, n = x
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, kernel_size=1)(x)

            n = layers.Dense(width)(n)

            x = layers.GroupNormalization(groups=8)(x)
            x = keras.activations.swish(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

            x = layers.Add()([x, n])

            x = layers.GroupNormalization(groups=8)(x)
            x = keras.activations.swish(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

            x = layers.Add()([residual, x])

            if attention:
                residual = x
                x = layers.GroupNormalization(groups=8, center=False, scale=False)(x)
                x = layers.MultiHeadAttention(
                    num_heads=4, key_dim=width, attention_axes=(1, 2)
                )(x, x)

                x = layers.Add()([residual, x])

            return x

        return forward

    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x, n])
                skips.append(x)
            x = layers.AveragePooling2D(pool_size=2)(x)
            return x

        return forward

    def UpBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width, attention)([x, n])
            return x

        return forward

    images = keras.Input(shape=(image_size, image_size, 3))
    noise_powers = keras.Input(shape=(1, 1, 1))
    text_embeddings = keras.Input(shape=(20, 512))

    x = layers.Conv2D(image_embedding_dims, kernel_size=patch_size, strides=patch_size)(
        images
    )

    n = EmbeddingLayer(noise_embedding_max_frequency, noise_embedding_dims)(
        noise_powers
    )
    n = layers.Dense(noise_embedding_dims, activation=keras.activations.swish)(n)
    n = layers.Dense(noise_embedding_dims, activation=keras.activations.swish)(n)

    t = layers.Flatten()(text_embeddings)
    t = layers.Dense(text_embedding_dims, activation=keras.activations.swish)(t)
    t = layers.Dense(text_embedding_dims, activation=keras.activations.swish)(t)
    t = layers.Reshape((1, 1, text_embedding_dims))(t)

    e = layers.Concatenate()([n, t])

    skips = []
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, e, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x, e])

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, e, skips])

    x = layers.Conv2DTranspose(
        3, kernel_size=patch_size, strides=patch_size, kernel_initializer="zeros"
    )(x)

    return keras.Model([images, noise_powers, text_embeddings], x, name="residual_unet")
