import tensorflow as tf
from src.config import ModelConfig, TrainConfig
from tensorflow import keras
from tensorflow.keras import layers


class KID(keras.metrics.Metric):
    def __init__(self, name="KID", **kwargs):
        super().__init__(name=name, **kwargs)
        self.kid_record = tf.keras.metrics.Mean(name="kid_record")
        self.encoder = keras.Sequential(
            [
                layers.Input(shape=(ModelConfig.image_size, ModelConfig.image_size, 3)),
                layers.Lambda(self.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(
                        ModelConfig.kid_image_size,
                        ModelConfig.kid_image_size,
                        3,
                    ),
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def preprocess_input(self, images):
        images = tf.image.resize(
            images,
            (ModelConfig.kid_image_size, ModelConfig.kid_image_size),
            method="bicubic",
            antialias=True,
        )
        images = tf.clip_by_value(images, 0.0, 1.0)
        images = keras.applications.inception_v3.preprocess_input(images * 255.0)
        return images

    def polynomial_kernel(self, x, y):
        feature_dim = tf.cast(x.shape[1], tf.float32)
        return (tf.matmul(x, y, transpose_b=True) / feature_dim + 1) ** 3

    def update_state(self, img_real, img_pred, sample_weight=None):
        feature_real = self.encoder(img_real)
        feature_pred = self.encoder(img_pred)

        kernel_real = self.polynomial_kernel(feature_real, feature_real)
        kernel_pred = self.polynomial_kernel(feature_pred, feature_pred)
        kernel_cross = self.polynomial_kernel(feature_real, feature_pred)

        mask = 1.0 - tf.eye(img_real.shape[0])
        batch_size = tf.cast(img_real.shape[0], tf.float32)

        kid = (
            tf.reduce_sum(kernel_real * mask)
            + tf.reduce_sum(kernel_pred * mask)
            - 2 * tf.reduce_sum(kernel_cross * mask)
        ) / (batch_size * (batch_size - 1))

        self.kid_record.update_state(kid)

    def result(self):
        return self.kid_record.result()

    def reset_state(self):
        self.kid_record.reset_state()
