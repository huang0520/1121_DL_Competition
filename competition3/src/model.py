import tensorflow as tf
from tensorflow import keras

from .architecture import get_network
from .config import RANDOM_STATE, ModelConfig, TrainConfig


class DiffusionModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.network = get_network()

    def compile(self, normalizer, **kwargs):
        super().compile(**kwargs)

        self.normalizer = normalizer
        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def diffusion_schedule(self, diffusion_times):
        start_angle = tf.math.acos(ModelConfig.max_singal_rate)
        end_angle = tf.math.acos(ModelConfig.min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        image_ratio = tf.math.cos(diffusion_angles)
        noise_ratio = tf.math.sin(diffusion_angles)
        return image_ratio, noise_ratio

    def denoise(self, noisy_image, noise_ratio, image_ratio, text_embeddings, training):
        pred_noises = self.network(
            [noisy_image, noise_ratio**2, text_embeddings], training=training
        )
        pred_images = (noisy_image - noise_ratio * pred_noises) / image_ratio

        return pred_noises, pred_images

    def denormalize(self, image):
        image = self.normalizer.mean + image * self.normalizer.variance**0.5
        return tf.clip_by_value(image, 0.0, 1.0)

    def reverse_diffusion(self, initial_noise, text_embeddings, diffusion_steps):
        num_images = tf.shape(initial_noise)[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_image = initial_noise
        for step in range(diffusion_steps):
            noisy_image = next_noisy_image

            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step_size * step
            noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)

            pred_noises, pred_images = self.denoise(
                noisy_image,
                noise_ratio,
                image_ratio,
                text_embeddings,
                training=False,
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_ratio, next_image_ratio = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_image = (
                next_image_ratio * pred_images + next_noise_ratio * pred_noises
            )

        return pred_images

    def generate(self, num_images, text_embeddings, diffusion_steps):
        initial_noise = tf.random.normal(
            shape=(num_images, ModelConfig.image_size, ModelConfig.image_size, 3),
            seed=RANDOM_STATE,
        )
        generate_images = self.reverse_diffusion(
            initial_noise, text_embeddings, diffusion_steps
        )
        generate_images = self.denormalize(generate_images)
        return generate_images

    @tf.function
    def train_step(self, data):
        images, embeddings = data

        images = self.normalizer(images, training=True)
        noises = tf.random.normal(
            shape=(
                TrainConfig.batch_size,
                ModelConfig.image_size,
                ModelConfig.image_size,
                3,
            ),
            seed=RANDOM_STATE,
        )

        diffusion_times = tf.random.uniform(
            shape=(TrainConfig.batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            seed=RANDOM_STATE,
        )
        noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)
        noisy_images = image_ratio * images + noise_ratio * noises

        with tf.GradientTape() as tape:
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_ratio, image_ratio, embeddings, training=True
            )

            noise_loss = self.loss(noises, pred_noises)
            image_loss = self.loss(images, pred_images)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        images, embeddings = data

        images = self.normalizer(images, training=False)
        noises = tf.random.normal(
            shape=(
                TrainConfig.batch_size,
                ModelConfig.image_size,
                ModelConfig.image_size,
                3,
            ),
            seed=RANDOM_STATE,
        )

        diffusion_times = tf.random.uniform(
            shape=(TrainConfig.batch_size, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            seed=RANDOM_STATE,
        )
        noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)
        pred_noises, pred_images = self.denoise(
            images, noise_ratio, image_ratio, embeddings, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        return {m.name: m.result() for m in self.metrics}

    def plot_image(self, text_embedding):
        generate_images = self.generate(
            num_images=1, text_embeddings=text_embedding, diffusion_steps=100
        )
        generate_images = self.denormalize(generate_images)
        return generate_images[0]
