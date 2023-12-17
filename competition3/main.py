# %% [markdown]
# ## Prepare environment

# %%
import os
import shutil
from dataclasses import asdict, fields
from datetime import datetime, timedelta, timezone
from time import sleep

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from icecream import ic, install
from sklearn.model_selection import train_test_split
from src.callback import EMACallback, PBarCallback, SamplePlotCallback
from src.config import (
    RANDOM_STATE,
    DatasetConfig,
    DirPath,
    ModelConfig,
    TrainConfig,
    export_config,
)
from src.dataset import generate_dataset
from src.model2 import DiffusionModel
from src.preprocess import (
    generate_embedding_df,
    generate_resize_image,
    generate_sample_embeddings,
    generate_unconditional_embeddings,
)
from src.utils import check_gpu
from tensorflow import keras
from tqdm import tqdm, trange

check_gpu()
install()

# %%
for field in fields(DirPath):
    dir = getattr(DirPath, field.name)
    if not dir.exists():
        dir.mkdir(parents=True)


# %% [markdown]
# ## Preprocess data

# %%
if (DirPath.dataset / "embeddings_train.pkl").exists():
    df_train = pd.read_pickle(DirPath.dataset / "embeddings_train.pkl")
    df_test = pd.read_pickle(DirPath.dataset / "embeddings_test.pkl")
else:
    generate_embedding_df()
    df_train = pd.read_pickle(DirPath.dataset / "embeddings_train.pkl")
    df_test = pd.read_pickle(DirPath.dataset / "embeddings_test.pkl")

if len(list((DirPath.resize_image).glob("*.jpg"))) != len(
    list((DirPath.resize_image).glob("*.jpg"))
):
    shutil.rmtree(DirPath.resize_image)
    (DirPath.resize_image).mkdir(parents=True)
    generate_resize_image()

sample_embeddings = generate_sample_embeddings()
unconditional_sample_embeddings = generate_unconditional_embeddings(10)
unconditional_test_embeddings = generate_unconditional_embeddings(
    TrainConfig.batch_size
)

# %% [markdown]
# ## Dataset

# %%
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)
df_train.head(2)

# %%
# train, val: (image, embedding) | test: (id, embedding)
dataset_train = generate_dataset(df_train, "train", augment=True, method="all")
dataset_val = generate_dataset(df_val, "val", augment=False, method="all")
dataset_test = generate_dataset(df_test, "test")

dataset_test_size = len(df_test) // TrainConfig.batch_size + 1

print("Dataset size:")
print(
    f"Train: {len(dataset_train)}, Val: {len(dataset_val)}, Test: {dataset_test_size}"
)
# %% [markdown]
# ## Train

# %%
timestamp = datetime.now(timezone(timedelta(hours=-8))).strftime("%Y%m%d_%H%M%S")

ckpt_path = DirPath.checkpoint / "diffusion.ckpt"
ckpt_callback = keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    save_best_only=True,
    monitor="val_image_loss",
    verbose=0,
)
ema_callback = EMACallback(TrainConfig.ema)
plot_callback = SamplePlotCallback(
    sample_embeddings,
    unconditional_sample_embeddings,
    TrainConfig.plot_diffusion_steps,
    num_rows=2,
    num_cols=5,
    plot_frequency=5,
    cfg_scale=TrainConfig.cfg_scale,
)
pbar_callback = PBarCallback()

log_path = DirPath.log / f"{timestamp}_loss.csv"
params_path = DirPath.log / f"{timestamp}_params.toml"
csv_logger = keras.callbacks.CSVLogger(log_path, separator=",", append=False)
export_config(params_path)

print("Nomalizer adapting...")
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(dataset_train.map(lambda image, embedding: image))

model = DiffusionModel(**asdict(ModelConfig()))
model.compile(
    prediction_type="velocity",
    normalizer=normalizer,
    optimizer=keras.optimizers.Lion(
        learning_rate=TrainConfig.lr_init,
        weight_decay=TrainConfig.weight_decay,
    ),
    loss=keras.losses.mean_absolute_error,
)

if TrainConfig.transfer:
    model.load_weights(ckpt_path)

print("Start training...")
model.fit(
    dataset_train,
    validation_data=dataset_val,
    epochs=TrainConfig.epochs,
    verbose=0,
    callbacks=[
        ckpt_callback,
        csv_logger,
        pbar_callback,
        ema_callback,
        plot_callback,
    ],
)

# %%
log = pd.read_csv(log_path)
train_image_loss = log["image_loss"]
train_noise_loss = log["noise_loss"]
val_image_loss = log["val_image_loss"]
val_noise_loss = log["val_noise_loss"]

plt.plot(train_image_loss, label="train_image_loss", color="blue")
plt.plot(val_image_loss, label="val_image_loss", linestyle="--", color="blue")
plt.plot(train_noise_loss, label="train_noise_loss", color="orange")
plt.plot(val_noise_loss, label="val_noise_loss", linestyle="--", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
print("Load best model...")
model.load_weights(ckpt_path)

if (DirPath.output / "inference").exists():
    shutil.rmtree(DirPath.output / "inference")

(DirPath.output / "inference").mkdir(parents=True)

test_epoch = len(df_test) // TrainConfig.batch_size + 1
step = 0
for id, text_embeddings in tqdm(
    dataset_test, total=test_epoch, desc="Generate image:", colour="green"
):
    step += 1
    if step > test_epoch:
        break

    generated_images = model.generate(
        num_images=TrainConfig.batch_size,
        text_embs=text_embeddings,
        un_text_embs=unconditional_test_embeddings,
        diffusion_steps=TrainConfig.plot_diffusion_steps,
        cfg_scale=TrainConfig.cfg_scale,
    )

    for i, img in enumerate(generated_images):
        plt.imsave(
            DirPath.output / f"inference/inference_{id[i]:04d}.jpg",
            img.numpy(),
            vmin=0.0,
            vmax=1.0,
        )

# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.chdir("./evaluation")
os.system("python inception_score.py ../output/inference ../output/score.csv 39")
os.chdir("..")

# %%
df_score = pd.read_csv("./output/score.csv")
print(f"Score: {np.mean(df_score['score']):.4f} ± {np.std(df_score['score']):.4f}")

# # %%
