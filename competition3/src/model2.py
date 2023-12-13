import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from icecream import ic
from tensorflow import keras

from .architecture import get_network
from .config import RANDOM_STATE, UNCONDITIONAL_SENT, ModelConfig, TrainConfig


class DiffusionModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.network = get_network()
        self.ema_network = tf.keras.models.clone_model(self.network)

    def compile(self, normalizer, **kwargs):
        super().compile(**kwargs)

        self.normalizer = normalizer
        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.velocity_loss_tracker = keras.metrics.Mean(name="velocity_loss")

    def denomalize(self, image):
        images = image * self.normalizer.variance**0.5 + self.normalizer.mean
        return tf.clip_by_value(images, 0.0, 1.0)

    def get_component(self, noisy_image, predict_velocities, signal_rates, noise_rates):
        pred_velocities = predict_velocities
        pred_images = signal_rates * noisy_image - noise_rates * pred_velocities
        pred_noises = noise_rates * noisy_image + signal_rates * pred_velocities

        pred_images = tf.clip_by_value(
            pred_images,
            0.0 - self.normalizer.mean / self.normalizer.variance**0.5,
            1.0 - self.normalizer.mean / self.normalizer.variance**0.5,
        )

        return pred_noises, pred_images, pred_velocities

    def diffusion_schedule(self, diffustion_times):
        # Signal step linear schedule
        start_snr = tf.exp(ModelConfig.start_log_snr)
        end_snr = tf.exp(ModelConfig.end_log_snr)

        start_noise_power = 1.0 / (1.0 + start_snr)
        end_noise_power = 1.0 / (1.0 + end_snr)

        noise_power = 1.0 - (1.0 - start_noise_power) * (1.0 - end_noise_power) / (
            1.0 - start_noise_power
        ) ** (diffustion_times**2)
        signal_power = 1.0 - noise_power

        signal_rate = tf.math.sqrt(signal_power)
        noise_rate = tf.math.sqrt(noise_power)

        return signal_rate, noise_rate

    def generate(self, num_images, text_embs, diffusion_steps):
        initial_noise = tf.random.normal(
            [num_images, ModelConfig.image_size, ModelConfig.image_size, 3],
            seed=RANDOM_STATE,
        )

        generated_images = self.diffusion_process(
            initial_noise, text_embs, diffusion_steps
        )
        return self.denomalize(generated_images)

    def diffusion_process(self, initial_noise, text_embs, diffusion_steps):
        batch_size = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        noisy_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones([batch_size, 1, 1, 1]) - step_size * step
            signal_rate, noise_rate = self.diffusion_schedule(diffusion_times)

            predictions = self.ema_network(
                [noisy_images, noise_rate**2, text_embs], training=False
            )
            pred_noises, pred_images, _ = self.get_component(
                noisy_images, predictions, signal_rate, noise_rate
            )

            next_signal_rates, next_noise_rates = self.diffusion_schedule(
                diffusion_times - step_size
            )
            noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    @property
    def metrics(self):
        return [
            self.noise_loss_tracker,
            self.image_loss_tracker,
            self.velocity_loss_tracker,
        ]

    def train_step(self, input):
        images, text_embs = input

        images = self.normalizer(images, training=True)
        noises = tf.random.normal(tf.shape(images), seed=RANDOM_STATE)

        noise_powers = tf.random.uniform(
            [images.shape[0], 1, 1, 1], 0.0, 1.0, seed=RANDOM_STATE
        )
        signal_powers = 1.0 - noise_powers
        noise_rates = tf.math.sqrt(noise_powers)
        signal_rates = tf.math.sqrt(signal_powers)

        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        with tf.GradientTape() as tape:
            prediction = self.network(
                [noisy_images, noise_rates**2, text_embs], training=True
            )
            pred_noises, pred_images, pred_velocity = self.get_component(
                noisy_images, prediction, signal_rates, noise_rates
            )

            velocity_loss = self.loss(velocities, pred_velocity)
            image_loss = self.loss(images, pred_images)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(velocity_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(
            self.network.trainable_weights, self.ema_network.trainable_weights
        ):
            ema_weight.assign(
                TrainConfig.ema * ema_weight + (1 - TrainConfig.ema) * weight
            )

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, input):
        images, text_embs = input

        images = self.normalizer(images, training=False)
        noises = tf.random.normal(tf.shape(images), seed=RANDOM_STATE)

        noise_powers = tf.random.uniform(
            [images.shape[0], 1, 1, 1], 0.0, 1.0, seed=RANDOM_STATE
        )
        signal_powers = 1.0 - noise_powers
        noise_rates = tf.math.sqrt(noise_powers)
        signal_rates = tf.math.sqrt(signal_powers)

        noisy_images = signal_rates * images + noise_rates * noises
        velocities = -noise_rates * images + signal_rates * noises

        prediction = self.ema_network(
            [noisy_images, noise_rates**2, text_embs], training=False
        )
        pred_noises, pred_images, pred_velocity = self.get_component(
            noisy_images, prediction, signal_rates, noise_rates
        )

        velocity_loss = self.loss(velocities, pred_velocity)
        image_loss = self.loss(images, pred_images)
        noise_loss = self.loss(noises, pred_noises)

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, text_embs, num_rows=2, num_cols=5, diffusion_steps=50):
        assert num_rows * num_cols == text_embs.shape[0]

        generated_images = self.generate(
            num_rows * num_cols, text_embs, diffusion_steps
        )

        generated_images = tf.reshape(
            generated_images, [num_rows, num_cols, *generated_images.shape[1:]]
        )
        generated_images = tf.transpose(generated_images, [0, 2, 1, 3, 4])
        generated_images = tf.reshape(
            generated_images,
            [
                num_rows * generated_images.shape[1],
                num_cols * generated_images.shape[3],
                3,
            ],
        )

        plt.figure(figsize=(num_cols * 2, num_rows * 2))
        plt.imshow(generated_images.numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()