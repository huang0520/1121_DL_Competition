import matplotlib.pyplot as plt
import tensorflow as tf
from src.config import TrainConfig
from tensorflow import keras


class EMACallback(keras.callbacks.Callback):
    def __init__(self, ema_decay: float):
        super().__init__()
        self.ema_decay = ema_decay

    def on_train_begin(self, logs=None):
        self.ema_weights = self.model.network.get_weights()

    def on_train_batch_begin(self, batch, logs=None):
        for ema_weight, weight in zip(
            self.ema_weights, self.model.network.get_weights()
        ):
            ema_weight *= self.ema_decay
            ema_weight += (1 - self.ema_decay) * weight

    def on_train_end(self, logs=None):
        self.model.network.set_weights(self.ema_weights)

    def on_test_begin(self, logs=None):
        self.backup = self.model.network.get_weights()
        self.model.network.set_weights(self.ema_weights)

    def on_test_end(self, logs=None):
        self.model.network.set_weights(self.backup)


class SamplePlotCallback(keras.callbacks.Callback):
    def __init__(self, sample_embeddings, diffusions_steps, n_row=2, n_col=5):
        super().__init__()
        self.sample_embeddings = sample_embeddings
        self.diffusions_steps = diffusions_steps
        self.n_row = n_row
        self.n_col = n_col
        self.epoch_counter = 0

    def on_test_end(self, logs=None):
        if (self.epoch_counter + 1) % 5 == 0:
            generate_images = self.model.generate(
                num_images=self.n_row * self.n_col,
                text_embeddings=self.sample_embeddings,
                diffusion_steps=self.diffusions_steps,
            )

            plt.figure(figsize=(2 * self.n_col, 2 * self.n_row))
            for i, img in enumerate(generate_images):
                plt.subplot(self.n_row, self.n_col, i + 1)
                plt.imshow(img)
                plt.axis("off")
            plt.tight_layout()
            plt.show()
            plt.close()

        else:
            self.epoch_counter += 1
