from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm, trange

from .metrics import KID


class EMACallback(keras.callbacks.Callback):
    def __init__(self, ema_decay: float):
        super().__init__()
        self.ema_decay = ema_decay

    def on_train_begin(self, logs=None):
        self.ema_weights = self.model.network.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        self.ema_weights = [
            self.ema_decay * ema_weight + (1 - self.ema_decay) * weight
            for ema_weight, weight in zip(
                self.ema_weights, self.model.network.get_weights()
            )
        ]

    def on_test_begin(self, logs=None):
        self.backup = self.model.network.get_weights()
        self.model.network.set_weights(self.ema_weights)

    def on_test_end(self, logs=None):
        self.model.network.set_weights(self.backup)

    def on_train_end(self, logs=None):
        self.model.network.set_weights(self.ema_weights)


class SamplePlotCallback(keras.callbacks.Callback):
    def __init__(
        self,
        sample_embeddings,
        unconditional_sample_embeddings,
        diffusions_steps,
        num_rows,
        num_cols,
        plot_frequency,
        cfg_scale,
        save: bool = False,
        save_path: Path = None,
    ):
        super().__init__()

        assert save_path is not None if save else True

        self.sample_embeddings = sample_embeddings
        self.unconditional_sample_embeddings = unconditional_sample_embeddings
        self.diffusions_steps = diffusions_steps
        self.n_row = num_rows
        self.n_col = num_cols
        self.plot_freq = plot_frequency
        self.cfg_scale = cfg_scale
        self.save = save
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_freq == 0:
            generate_images = self.model.plot_image(
                self.sample_embeddings,
                self.unconditional_sample_embeddings,
                self.n_row,
                self.n_col,
                self.diffusions_steps,
                self.cfg_scale,
            )

        if self.save:
            plt.imsave(self.save_path, generate_images)


class PBarCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs=None):
        assert self.params["verbose"] == 0, "Set verbose=0 when using tqdm pbar"

    def on_epoch_begin(self, epoch, logs=None):
        self.pbar = trange(
            self.params["steps"],
            desc=f"Epoch {epoch + 1}/{self.params['epochs']}",
            colour="green",
            unit="batch",
        )

    def on_batch_end(self, batch, logs=None):
        self.pbar.update(1)
        self.pbar.set_postfix({
            "image_loss": logs["image_loss"],
            "noise_loss": logs["noise_loss"],
            "velocity_loss": logs["velocity_loss"],
        })

    def on_test_begin(self, logs=None):
        self.pbar.colour = "blue"
        self.pbar.refresh()

    def on_epoch_end(self, epoch, logs=None):
        self.pbar.colour = "red"
        self.pbar.set_postfix({
            "val_image_loss": logs["val_image_loss"],
            "val_noise_loss": logs["val_noise_loss"],
            "val_velocity_loss": logs["val_velocity_loss"],
        })
        self.pbar.close()
