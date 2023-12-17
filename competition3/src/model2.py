import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from .architecture import get_network
from .config import RANDOM_STATE


class DiffusionModel(keras.Model):
    def __init__(
        self,
        image_size,
        noise_embedding_dim,
        image_embedding_dim,
        text_embedding_shape,
        widths,
        block_depth,
        embedding_max_frequency,
        start_log_snr,
        end_log_snr,
    ):
        super().__init__()

        self.network = get_network(
            image_size,
            noise_embedding_dim,
            image_embedding_dim,
            text_embedding_shape,
            widths,
            block_depth,
            embedding_max_frequency,
        )

        self.image_size = image_size
        self.start_log_snr = start_log_snr
        self.end_log_snr = end_log_snr

    def compile(self, normalizer, prediction_type: str = "velocity", **kwargs):
        super().compile(**kwargs)
        assert prediction_type in ["noise", "image", "velocity"]

        self.prediction_type = prediction_type
        self.normalizer = normalizer
        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
        self.velocity_loss_tracker = keras.metrics.Mean(name="velocity_loss")

    @property
    def metrics(self):
        return [
            self.noise_loss_tracker,
            self.image_loss_tracker,
            self.velocity_loss_tracker,
        ]

    def denomalize(self, image):
        images = image * self.normalizer.variance**0.5 + self.normalizer.mean
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffustion_times):
        start_snr = tf.math.exp(self.start_log_snr)
        end_snr = tf.math.exp(self.end_log_snr)

        start_noise_power = 1.0 / (1.0 + start_snr)
        end_noise_power = 1.0 / (1.0 + end_snr)

        noise_power = start_snr**diffustion_times / (
            start_snr * end_snr**diffustion_times + start_snr**diffustion_times
        )
        signal_power = 1.0 - noise_power

        noise_rate = tf.math.sqrt(noise_power)
        signal_rate = tf.math.sqrt(signal_power)

        return noise_rate, signal_rate

    def get_component(self, noisy_images, predictions, signal_rates, noise_rates):
        if self.prediction_type == "velocity":
            pred_velocities = predictions
            pred_images = signal_rates * noisy_images - noise_rates * pred_velocities
            pred_noises = noise_rates * noisy_images + signal_rates * pred_velocities

        elif self.prediction_type == "image":
            pred_images = predictions
            pred_noises = (noisy_images - signal_rates * pred_images) / noise_rates
            pred_velocities = (signal_rates * noisy_images - pred_images) / noise_rates

        elif self.prediction_type == "noise":
            pred_noises = predictions
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
            pred_velocities = (pred_noises - noise_rates * noisy_images) / signal_rates

        return pred_noises, pred_images, pred_velocities

    def denoise(self, noisy_images, text_embs, noise_rate, signal_rate, training):
        predictions = self.network(
            [noisy_images, noise_rate**2, text_embs], training=training
        )
        pred_noises, pred_images, pred_velocities = self.get_component(
            noisy_images, predictions, signal_rate, noise_rate
        )
        return pred_noises, pred_images, pred_velocities

    def reverse_diffusion(
        self, initial_noise, text_embs, un_text_embs, diffusion_steps, cfg_scale
    ):
        batch_size = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        noisy_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones([batch_size, 1, 1, 1]) - step_size * step
            noise_rate, signal_rate = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images, _ = self.denoise(
                noisy_images, text_embs, noise_rate, signal_rate, training=False
            )
            un_pred_noises, _, _ = self.denoise(
                noisy_images, un_text_embs, noise_rate, signal_rate, training=False
            )

            pred_noises = un_pred_noises + cfg_scale * (pred_noises - un_pred_noises)
            pred_images = (noisy_images - noise_rate * pred_noises) / signal_rate

            next_diffusion_times = diffusion_times - step_size
            next_noise_rate, next_signal_rate = self.diffusion_schedule(
                next_diffusion_times
            )
            noisy_images = (
                next_signal_rate * pred_images + next_noise_rate * pred_noises
            )

        return pred_images

    def generate(self, num_images, text_embs, un_text_embs, diffusion_steps, cfg_scale):
        initial_noise = tf.random.normal(
            (num_images, self.image_size, self.image_size, 3), seed=RANDOM_STATE
        )
        generated_images = self.reverse_diffusion(
            initial_noise, text_embs, un_text_embs, diffusion_steps, cfg_scale
        )
        generated_images = self.denomalize(generated_images)
        return generated_images

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
            pred_noises, pred_images, pred_velocity = self.denoise(
                noisy_images, text_embs, noise_rates, signal_rates, training=True
            )
            velocity_loss = self.loss(velocities, pred_velocity)
            image_loss = self.loss(images, pred_images)
            noise_loss = self.loss(noises, pred_noises)

        if self.prediction_type == "noise":
            loss = noise_loss
        elif self.prediction_type == "image":
            loss = image_loss
        elif self.prediction_type == "velocity":
            loss = velocity_loss

        gradients = tape.gradient(loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

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

        pred_noises, pred_images, pred_velocity = self.denoise(
            noisy_images, text_embs, noise_rates, signal_rates, training=False
        )
        velocity_loss = self.loss(velocities, pred_velocity)
        image_loss = self.loss(images, pred_images)
        noise_loss = self.loss(noises, pred_noises)

        self.velocity_loss_tracker.update_state(velocity_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_image(
        self,
        text_embeddings,
        un_text_embs,
        num_rows,
        num_cols,
        diffusion_steps,
        cfg_scale,
    ):
        generate_images = self.generate(
            num_rows * num_cols,
            text_embeddings,
            un_text_embs,
            diffusion_steps=diffusion_steps,
            cfg_scale=cfg_scale,
        )
        generate_images = tf.reshape(
            generate_images, (num_rows, num_cols, self.image_size, self.image_size, 3)
        )
        generate_images = tf.transpose(generate_images, (0, 2, 1, 3, 4))
        generate_images = tf.reshape(
            generate_images,
            (num_rows * self.image_size, num_cols * self.image_size, 3),
        )

        plt.figure(figsize=(num_cols * 1.5, num_rows * 1.5))
        plt.imshow(generate_images.numpy())
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        return generate_images.numpy()
