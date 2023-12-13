import math

import tensorflow as tf
from src.config import ModelConfig
from tensorflow import keras
from tensorflow.keras import layers


def get_network():
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

        return layers.Lambda(sinusoidal_embedding)

    def swish_glu():
        def apply(x):
            x = layers.Dense(x.shape[-1] * 2)(x)
            a, b = tf.split(x, 2, axis=-1)
            return a * layers.Activation("swish")(b)

        return apply

    def residual_block(width, attention: bool):
        def apply(x):
            x, n, t = x
            input_width = x.shape[-1]

            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, 1)(x)

            # Noise variance embedding
            n = layers.Dense(width)(n)
            n = layers.UpSampling2D(x.shape[1], interpolation="bilinear")(n)

            # Text embedding
            t = layers.Dense(width)(t)
            t = tf.transpose(t, perm=[0, 2, 1])
            t = layers.MaxPool1D(width // (width // ModelConfig.widths[0]))(t)
            t = layers.Reshape((1, 1, -1))(t)
            t = layers.UpSampling2D(x.shape[1], interpolation="bilinear")(t)

            x = layers.GroupNormalization(groups=8)(x)
            x = swish_glu()(x)
            x = layers.Conv2D(width, 3, padding="same")(x)
            x = layers.Concatenate()([x, n, t])

            x = layers.GroupNormalization(groups=12)(x)
            x = swish_glu()(x)
            x = layers.Conv2D(width, 3, padding="same")(x)
            x = layers.Concatenate()([x, residual])

            if attention:
                residual = x
                x = layers.GroupNormalization(groups=8, center=False, scale=False)(x)
                x = layers.MultiHeadAttention(
                    num_heads=4, key_dim=width, attention_axes=(1, 2)
                )(x, x)
                x = layers.Concatenate()([x, residual])

            return x

        return apply

    def downsampling_block(block_depth, width, attention: bool):
        def apply(x):
            x, n, t, skip = x
            for _ in range(block_depth):
                x = residual_block(width, attention)([x, n, t])
                skip.append(x)

            x = layers.AveragePooling2D(2)(x)
            return x

        return apply

    def upsampling_block(block_depth, width, attention: bool):
        def apply(x):
            x, n, t, skip = x
            x = layers.UpSampling2D(2)(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skip.pop()])
                x = residual_block(width, attention)([x, n, t])
            return x

        return apply

    noisy_image = layers.Input(
        shape=(ModelConfig.image_size, ModelConfig.image_size, 3)
    )
    noise_variance = layers.Input(shape=(1, 1, 1))
    caption_embedding = layers.Input(
        shape=(ModelConfig.max_seq_len, ModelConfig.embedding_dim)
    )

    x = layers.Conv2D(ModelConfig.widths[0], 1)(noisy_image)

    n = embedding_layer(ModelConfig.embedding_max_freq, ModelConfig.embedding_dim)(
        noise_variance
    )
    n = layers.Dense(ModelConfig.widths[0], "swish")(n)

    t = layers.Dense(ModelConfig.widths[0], "swish")(caption_embedding)

    skips = []
    for width, attention in zip(ModelConfig.widths[:-1], ModelConfig.attentions[:-1]):
        x = downsampling_block(ModelConfig.block_depth, width, attention)(
            [x, n, t, skips]
        )

    for _ in range(ModelConfig.block_depth):
        x = residual_block(ModelConfig.widths[-1], ModelConfig.attentions[-1])(
            [x, n, t]
        )

    for width, attention in zip(
        ModelConfig.widths[-2::-1], ModelConfig.attentions[-2::-1]
    ):
        x = upsampling_block(ModelConfig.block_depth, width, attention)(
            [x, n, t, skips]
        )

    x = layers.Conv2DTranspose(3, 1)(x)

    return keras.Model(
        inputs=[noisy_image, noise_variance, caption_embedding],
        outputs=x,
        name="noise_predictor",
    )
