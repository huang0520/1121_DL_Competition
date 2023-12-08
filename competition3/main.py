# %% [markdown]
# ## Prepare environment

# %%
import functools
import math
import os
import random
import re
import string
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm, trange
from transformers import CLIPTextModel, CLIPTokenizer

# %%
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %%
DICT_DIR: str = "./data/dictionary"
DATASET_DIR: str = "./data/dataset"
IMAGE_DIR: str = "./data/102flowers"
CHECKPOINT_DIR: str = "./checkpoints"
OUTPUT_DIR: str = "./output"

# Tokenizer parameters
MAX_SEQ_LENGTH: int = 20

# Image parameters
IMAGE_SIZE: int = 64
IMAGE_CHANNELS: int = 3

# Dataset parameters
BATCH_SIZE: int = 64

# Loss parameters
KID_IMAGE_SIZE: int = 75
KID_DIFFUSION_STEPS: int = 5
PLOT_DIFFUSION_STEPS: int = 20

# Model parameters
INPUT_EMBED_DIM: int = 512
EMBEDDIN_DIM: int = 32
UNET_WIDTHS: list[int] = [32, 64, 96, 128]
RES_BLOCK_DEPTH: int = 2
EMBEDDING_MAX_FREQ: float = 1000.0
MIN_SIGNAL_RATE: float = 0.02
MAX_SIGNAL_RATE: float = 0.95

# Training parameters
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EMA = 0.999
N_EPOCH: int = 10

# Other parameters
RANDOM_STATE: int = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%
for dir in [DICT_DIR, DATASET_DIR, IMAGE_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    if not os.path.exists(dir):
        os.makedirs(dir)


# %% [markdown]
# ## Preprocess text

# %%
vocab_path: str = os.path.join(DICT_DIR, "vocab.npy")
word2idx_path: str = os.path.join(DICT_DIR, "word2Id.npy")
idx2word_path: str = os.path.join(DICT_DIR, "id2Word.npy")

word2idx: dict = dict(np.load(word2idx_path))
idx2word: dict = dict(np.load(idx2word_path))


def seq_list2sent_list(seq_list: list[list]) -> list[str]:
    pad_id = word2idx["<PAD>"]

    sent_list = []
    for seq in seq_list:
        sent = [idx2word[idx] for idx in seq if idx != pad_id]
        sent_list.append(" ".join(sent))

    return sent_list


# %%


def generate_embed_df(df_train, df_test):
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    def embed_sent_list(sents: list[str]):
        tokens = tokenizer(
            sents,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
        )
        embeddings = encoder(**tokens).last_hidden_state.detach().numpy()
        return embeddings

    # Create embedding for training data
    tqdm.pandas(desc="Embedding training data")
    cap_seqs = df_train["Captions"]
    cap_sents = cap_seqs.apply(seq_list2sent_list)
    embeddings = cap_sents.progress_apply(embed_sent_list)

    # Change image path
    image_paths = (
        df_train["ImagePath"]
        .apply(lambda path: os.path.join(IMAGE_DIR, os.path.basename(path)))
        .values
    )
    df_train = pd.DataFrame({"Embeddings": embeddings, "ImagePath": image_paths})

    # Create embedding for testing data
    tqdm.pandas(desc="Embedding testing data")
    cap_seqs = df_test["Captions"]
    cap_sents = cap_seqs.apply(seq_list2sent_list)
    embeddings = cap_sents.progress_apply(embed_sent_list)
    df_test = pd.DataFrame({"Embeddings": embeddings, "ID": df_test["ID"]})

    return df_train, df_test


if os.path.exists(os.path.join(DATASET_DIR, "embeddings_train.pkl")):
    df_train = pd.read_pickle(os.path.join(DATASET_DIR, "embeddings_train.pkl"))
    df_test = pd.read_pickle(os.path.join(DATASET_DIR, "embeddings_test.pkl"))
    print("Load embeddings from pickle file")
else:
    df_train = pd.read_pickle(os.path.join(DATASET_DIR, "text2ImgData.pkl"))
    df_test = pd.read_pickle(os.path.join(DATASET_DIR, "testData.pkl"))
    df_train, df_test = generate_embed_df(df_train, df_test)
    df_train.to_pickle(os.path.join(DATASET_DIR, "embeddings_train.pkl"))
    df_test.to_pickle(os.path.join(DATASET_DIR, "embeddings_test.pkl"))
    print("Generate embeddings and save to pickle file")

# %%
df_train.head(5)

# %% [markdown]
# ## Dataset


# %%
def load_image(path: tf.Tensor) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img


def map_func(embedding, img_path):
    img = load_image(img_path)
    return img, embedding


def generate_train(df_train):
    datas = df_train.explode("Embeddings")
    embedding = np.stack(datas["Embeddings"].values)
    img_path = datas["ImagePath"].values

    dataset = tf.data.Dataset.from_tensor_slices((embedding, img_path))
    dataset = (
        dataset.map(map_func, num_parallel_calls=AUTOTUNE)
        .shuffle(len(datas))
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    return dataset


def generate_val(df_val):
    datas = df_val.explode("Embeddings")
    embedding = np.stack(datas["Embeddings"].values)
    img_path = datas["ImagePath"].values

    dataset = tf.data.Dataset.from_tensor_slices((embedding, img_path))
    dataset = (
        dataset.map(map_func, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )
    return dataset


def generate_test(df_test):
    embeddings = df_test["Embeddings"].values
    embeddings = np.stack(embeddings)
    idx = df_test["ID"].values

    dataset = tf.data.Dataset.from_tensor_slices((embeddings, idx))
    dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


# %%
_df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)
dataset_train = generate_train(_df_train)
dataset_val = generate_val(df_val)
dataset_test = generate_test(df_test)

# %% [markdown]
# ## Define model


# %% Loss function
# Kernel inception distance
class KID(keras.metrics.Metric):
    def __init__(self, name="KID", **kwargs):
        super(KID, self).__init__(name=name, **kwargs)
        self.kid_record = tf.keras.metrics.Mean(name="kid_record")
        self.encoder = keras.Sequential(
            [
                layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS]),
                layers.Rescaling(255.0),
                layers.Resizing(KID_IMAGE_SIZE, KID_IMAGE_SIZE),
                layers.Lambda(keras.applications.inception_v3.preprocess_input),
                keras.applications.InceptionV3(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(KID_IMAGE_SIZE, KID_IMAGE_SIZE, 3),
                ),
                layers.GlobalAveragePooling2D(),
            ],
            name="inception_encoder",
        )

    def polynomial_kernel(self, x, y, c=1, d=3):
        feature_dim = tf.cast(x.shape[1], tf.float32)
        return (tf.matmul(x, y, transpose_b=True) / feature_dim + c) ** d

    def update_state(self, img_real, img_pred, sample_weight=None):
        real_feature = self.encoder(img_real)
        pred_feature = self.encoder(img_pred)

        kernel_real = self.polynomial_kernel(real_feature, real_feature)
        kernel_pred = self.polynomial_kernel(pred_feature, pred_feature)
        kernel_cross = self.polynomial_kernel(real_feature, pred_feature)

        batch_size = img_real.shape[0]
        batch_size_f = tf.cast(batch_size, tf.float32)
        mask = 1.0 - tf.eye(batch_size)

        kid = (
            tf.reduce_sum(kernel_real * mask)
            + tf.reduce_sum(kernel_pred * mask)
            - 2 * tf.reduce_sum(kernel_cross)
        ) / (batch_size_f * (batch_size_f - 1))

        self.kid_record.update_state(kid)

    def result(self):
        return self.kid_record.result()

    def reset_state(self):
        self.kid_record.reset_state()


# %% Network architecture
@tf.function
def sinusoidal_embedding(x: tf.Tensor):
    min_freq = 1.0
    max_freq = EMBEDDING_MAX_FREQ
    freqs = tf.exp(
        tf.linspace(
            start=tf.math.log(min_freq),
            stop=tf.math.log(max_freq),
            num=EMBEDDIN_DIM // 2,
        )
    )
    angular_speeds = tf.cast(2 * math.pi * freqs, tf.float32)
    embeddings = tf.concat([tf.sin(x * angular_speeds), tf.cos(x * angular_speeds)], -1)

    return embeddings


def residual_block(width):
    def apply(x):
        input_width = x.shape[3]
        if input_width == width:
            residual = x
        else:
            residual = layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def down_block(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width)(x)
            skips.append(x)
        x = layers.AveragePooling2D(pool_size=2)(x)
        return x

    return apply


def up_block(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(width)(x)
        return x

    return apply


def get_network(image_size, widths, block_depth):
    noisy_images = keras.Input(shape=(image_size, image_size, 3))
    noise_variances = keras.Input(shape=(1, 1, 1))
    text_embedding = keras.Input(shape=(MAX_SEQ_LENGTH, INPUT_EMBED_DIM))

    e = layers.Lambda(sinusoidal_embedding, output_shape=(1, 1, 32))(noise_variances)
    e = layers.UpSampling2D(size=image_size, interpolation="nearest")(e)

    t = layers.Lambda(
        tf.transpose,
        arguments={"perm": [0, 2, 1]},
        output_shape=(-1, INPUT_EMBED_DIM, MAX_SEQ_LENGTH),
    )(text_embedding)
    t = layers.MaxPool1D(pool_size=(INPUT_EMBED_DIM))(t)
    t = layers.Flatten()(t)
    t = layers.Reshape((1, 1, -1))(t)
    t = layers.UpSampling2D(size=image_size, interpolation="nearest")(t)

    x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e, t])

    skips = []
    for width in widths[:-1]:
        x = down_block(width, block_depth)([x, skips])

    for _ in range(block_depth):
        x = residual_block(widths[-1])(x)

    for width in reversed(widths[:-1]):
        x = up_block(width, block_depth)([x, skips])

    noise = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model(
        [noisy_images, noise_variances, text_embedding], noise, name="noise_predictor"
    )


# %%
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(image_size, widths, block_depth)
        self.ema_network = get_network(image_size, widths, block_depth)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.kid = KID(name="kid")

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker, self.kid]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        images = self.normalizer.mean + images * self.normalizer.variance**0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.cast(tf.math.acos(MAX_SIGNAL_RATE), "float32")
        end_angle = tf.cast(tf.math.acos(MIN_SIGNAL_RATE), "float32")

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.math.cos(diffusion_angles)
        noise_rates = tf.math.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_rates, signal_rates

    def denoise(
        self, noisy_images, noise_rates, signal_rates, text_embeddings, training
    ):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network(
            [noisy_images, noise_rates**2, text_embeddings], training=training
        )
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, text_embeddings, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, text_embeddings, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, text_embeddings, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(
            shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
        )
        generated_images = self.reverse_diffusion(
            initial_noise, text_embeddings, diffusion_steps
        )
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, batch):
        images, embeddings = batch

        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(
            shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)
        )

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, embeddings, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, batch):
        images, embeddings = batch
        # normalize images to have standard deviation of 1, like the noises
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=(BATCH_SIZE, 1, 1, 1), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, embeddings, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=BATCH_SIZE,
            text_embeddings=embeddings,
            diffusion_steps=KID_DIFFUSION_STEPS,
        )
        self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=PLOT_DIFFUSION_STEPS,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0))
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index])
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()


# %%
model = DiffusionModel(
    image_size=IMAGE_SIZE,
    widths=UNET_WIDTHS,
    block_depth=RES_BLOCK_DEPTH,
)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    ),
    loss=keras.losses.mean_absolute_error,
)


checkpoint = tf.train.Checkpoint(diffusion_model=model)
ckpt_manager = tf.train.CheckpointManager(
    checkpoint, CHECKPOINT_DIR, max_to_keep=5, checkpoint_name="ckpt"
)

model.normalizer.adapt(dataset_train.map(lambda x, y: x))

model.fit(
    dataset_train,
    validation_data=dataset_val,
    epochs=N_EPOCH,
)


# %%
_, embedding = next(iter(dataset_train))
embedding = embedding[:8]

imgs = model.generate(8, embedding, 20)
imgs = imgs.numpy()

plt.figure(figsize=(16, 8))
for i, img in enumerate(imgs):
    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.axis("off")


# %% [markdown]
# ## Visualization


# %%
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
ni = int(np.ceil(np.sqrt(BATCH_SIZE)))
sample_size = BATCH_SIZE
sample_sentence = (
    ["the flower shown has yellow anther red pistil and bright red petals."]
    * int(sample_size / ni)
    + ["this flower has petals that are yellow, white and purple and has dark lines"]
    * int(sample_size / ni)
    + ["the petals on this flower are white with a yellow center"]
    * int(sample_size / ni)
    + ["this flower has a lot of small round pink petals."] * int(sample_size / ni)
    + ["this flower is orange in color, and has petals that are ruffled and rounded."]
    * int(sample_size / ni)
    + ["the flower has yellow petals and the center of it is brown."]
    * int(sample_size / ni)
    + ["this flower has petals that are blue and white."] * int(sample_size / ni)
    + [
        "these white flowers have petals that start off white in color and end in a white towards the tips."
    ]
    * int(sample_size / ni)
)

# for i, sent in enumerate(sample_sentence):
# sample_sentence[i] = sentence2sequence(sent)
sample_sentence = sample_generator(sample_sentence, BATCH_SIZE)


# %%
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %cd evaluation
# !python inception_score.py ../output/inference ../output/score.csv 39
# %cd ..


# %%
df_score = pd.read_csv("./output/score.csv")
print(f"Score: {np.mean(df_score['score']):.4f} ± {np.std(df_score['score']):.4f}")
