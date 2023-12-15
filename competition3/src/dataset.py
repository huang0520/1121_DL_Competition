# %%
import numpy as np
import pandas as pd
import tensorflow as tf

from .config import (
    AUTOTUNE,
    RANDOM_STATE,
    RNG_GENERATOR,
    DatasetConfig,
    ModelConfig,
    TrainConfig,
)


def load_image(path: tf.Tensor) -> tf.Tensor:
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (ModelConfig.image_size, ModelConfig.image_size))
    return image


def augment_image(image: tf.Tensor, augmenter) -> tf.Tensor:
    apply = tf.random.uniform((), seed=RANDOM_STATE) < DatasetConfig.aug_prob
    image = augmenter(image) if apply else image
    return image


def generate_dataset(
    df: pd.DataFrame, type: str, method: str = "random", augment: bool = True
) -> tf.data.Dataset:
    assert type in ["train", "val", "test"]
    assert method in ["random", "all"]

    augmenter = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomRotation(1, seed=RANDOM_STATE),
        tf.keras.layers.experimental.preprocessing.RandomFlip(
            "horizontal_and_vertical", seed=RANDOM_STATE
        ),
    ])

    def map_fn(path, embedding):
        image = load_image(path)
        image = augment_image(image, augmenter) if augment else image
        return image, embedding

    if method == "random":
        df_new = df
        embeddings = np.array([
            RNG_GENERATOR.choice(_embeddings, size=1).squeeze()
            for _embeddings in df["Embeddings"]
        ])
    elif method == "all":
        df_new = df.explode("Embeddings") if type != "test" else df
        embeddings = df_new["Embeddings"].to_numpy()
        embeddings = np.stack(embeddings)

    if type == "train" or type == "val":
        img_paths = df_new["ImagePath"].to_numpy().astype(str)

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, embeddings))
        dataset = dataset.map(map_fn, num_parallel_calls=AUTOTUNE)
        dataset = (
            dataset.shuffle(len(embeddings), seed=RANDOM_STATE)
            if type == "train"
            else dataset
        )
        dataset = dataset.batch(TrainConfig.batch_size, drop_remainder=True).prefetch(
            AUTOTUNE
        )

    elif type == "test":
        id = df["ID"].to_numpy()
        dataset = tf.data.Dataset.from_tensor_slices((id, embeddings))
        dataset = dataset.repeat().batch(TrainConfig.batch_size).prefetch(AUTOTUNE)

    return dataset
