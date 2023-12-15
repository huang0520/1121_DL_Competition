import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def embedding_layer(embedding_max_frequency, embedding_dim):
    def sinusoidal_embedding(x: tf.Tensor):
        """Generate sinusoidal embeddings for step (noise variance)"""
        embedding_min_frequency = 1.0

        frequencies = tf.exp(
            tf.linspace(
                start=tf.math.log(embedding_min_frequency),
                stop=tf.math.log(embedding_max_frequency),
                num=embedding_dim // 2,
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = tf.concat(
            [tf.sin(x * angular_speeds), tf.cos(x * angular_speeds)], -1
        )
        return embeddings

    return sinusoidal_embedding


def residual_block(width):
    def apply(x):
        input_width = x.shape[3]
        residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def get_network(
    image_size=64,
    noise_embedding_dim=512,
    image_embedding_dim=32,
    widths=(32, 64, 96, 128),
    block_depth=2,
    embedding_max_frequency=1000.0,
):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_power = keras.Input(shape=(1, 1))
    text_embs = keras.Input(shape=(20, 512))

    noise_embedding = layers.Lambda(
        embedding_layer(embedding_max_frequency, noise_embedding_dim),
        output_shape=(1, noise_embedding_dim),
    )(noise_power)
    embeddings = layers.Concatenate(axis=1)([text_embs, noise_embedding])
    embeddings = layers.Dense(image_embedding_dim)(embeddings)

    x = layers.Conv2D(image_embedding_dim, kernel_size=1)(noisy_images)
    embeddings = layers.Attention()([x, embeddings])
    x = layers.concatenate([x, embeddings])

    skips = []
    # Downsampling blocks
    for width in widths[:-1]:
        for _ in range(block_depth):
            x = residual_block(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)

    # Bottleneck block
    for _ in range(block_depth):
        x = residual_block(widths[-1])(x)

    # Upsampling blocks
    for width in reversed(widths[:-1]):
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(width)(x)

    noise = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(
        [noisy_images, noise_power, text_embs], noise, name="noise_predictor"
    )
