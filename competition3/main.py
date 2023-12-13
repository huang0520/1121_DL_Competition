# %% [markdown]
# ## Prepare environment

# %%
import os
import shutil
from dataclasses import fields
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from icecream import ic
from sklearn.model_selection import train_test_split
from src.callback import EMACallback, SamplePlotCallback
from src.config import RANDOM_STATE, DirPath, ModelConfig, TrainConfig
from src.dataset import generate_dataset
from src.embedding import (
    TextEncoder,
    TextTokenizer,
    get_embedding_df,
    remove_embedding_df,
)
from src.model import DiffusionModel
from src.utils import check_gpu
from tensorflow import keras
from tqdm import tqdm

check_gpu()

# %%
for field in fields(DirPath):
    dir = getattr(DirPath, field.name)
    if not dir.exists():
        dir.mkdir(parents=True)

# %% [markdown]
# ## Preprocess text

# %%
df_train, df_test = get_embedding_df()

sample_sents = pd.read_csv(DirPath.data / "sample_sentence.csv")["sentence"].tolist()
token = TextTokenizer(
    sample_sents,
    max_length=ModelConfig.max_seq_len,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
)
sample_embeddings = TextEncoder(**token).last_hidden_state.detach().numpy()

# %% [markdown]
# ## Dataset

# %%
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)

dataset_train = generate_dataset(df_train, "train")
dataset_val = generate_dataset(df_val, "val")
dataset_test = generate_dataset(df_test, "test")

# %% [markdown]
# ## Train

# %%
ckpt_path = DirPath.checkpoint / "diffusion.ckpt"
ckpt_callback = keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="val_kid",
    verbose=0,
)
ema_callback = EMACallback(ema_decay=TrainConfig.ema)
plot_callback = SamplePlotCallback(sample_embeddings, TrainConfig.plot_diffusion_steps)

normalizer = tf.keras.layers.Normalization()
normalizer.adapt(dataset_train.map(lambda image, embedding: image))

model = DiffusionModel()

model.compile(
    normalizer=normalizer,
    optimizer=keras.optimizers.AdamW(
        learning_rate=TrainConfig.lr, weight_decay=TrainConfig.lr_decay
    ),
    loss=keras.losses.mean_absolute_error,
)

ckpt = tf.train.latest_checkpoint(DirPath.checkpoint, "diffusion.ckpt")
if ckpt:
    model.load_weights(ckpt)

history = model.fit(
    dataset_train,
    validation_data=dataset_val,
    epochs=TrainConfig.epochs,
    verbose=1,
    callbacks=[ckpt_callback, plot_callback, ema_callback],
)

# %%
ckpt = tf.train.latest_checkpoint(DirPath.checkpoint)
model.load_weights(ckpt)

for images, embeddings in dataset_train.take(1):
    plt.imshow(images[0])
    plt.show()

    generated_images = model.generate(
        num_images=1,
        text_embeddings=embeddings[0:1],
        diffusion_steps=TrainConfig.plot_diffusion_steps,
    )
    plt.imshow(generated_images[0])
    plt.show()

# %%
model.load_weights(ckpt_path)


if (Path(OUTPUT_DIR) / "inference").exists():
    shutil.rmtree(Path(OUTPUT_DIR) / "inference")

(Path(OUTPUT_DIR) / "inference").mkdir(parents=True)

test_epoch = len(df_test) // BATCH_SIZE + 1
step = 0
for embeddings, id in dataset_test:
    step += 1
    if step > test_epoch:
        break

    generated_images = model.generate(
        num_images=BATCH_SIZE,
        text_embeddings=embeddings,
        diffusion_steps=PLOT_DIFFUSION_STEPS,
    )
    for i, img in enumerate(generated_images):
        plt.imsave(
            Path(OUTPUT_DIR) / f"inference/inference_{id[i]:04d}.jpg",
            img.numpy(),
            vmin=0.0,
            vmax=1.0,
        )

# %% [markdown]
# ## Visualization


# %%
# TODO: Rewrite the visualization code to match the new model
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h : j * h + h, i * w : i * w + w, :] = image
    return img


def imsave(images, size, path):
    # getting the pixel values between [0, 1] to save it
    return plt.imsave(path, merge(images, size) * 0.5 + 0.5)


def save_images(images, size, image_path):
    return imsave(images, size, image_path)


def sample_generator(caption, batch_size):
    caption = np.asarray(caption)
    caption = caption.astype(np.int32)
    dataset = tf.data.Dataset.from_tensor_slices(caption)
    dataset = dataset.batch(batch_size)
    return dataset


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.chdir("./evaluation")
os.system("python inception_score.py ../output/inference ../output/score.csv 39")
os.chdir("..")

# %%
df_score = pd.read_csv("./output/score.csv")
print(f"Score: {np.mean(df_score['score']):.4f} ± {np.std(df_score['score']):.4f}")

# %%
