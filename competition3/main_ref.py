# %% [markdown]
# ## Prepare environment

# %%
import math
import os
import shutil
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import albumentations as alb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.api._v2.keras import layers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
from tqdm.keras import TqdmCallback
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
EMBEDDING_DIM: int = 512
UNET_WIDTHS: list[int] = [32, 64, 96, 128]
RES_BLOCK_DEPTH: int = 2
STEP_EMBED_MAX_FREQ: float = 1000.0
MIN_SIGNAL_RATE: float = 0.02
MAX_SIGNAL_RATE: float = 0.95

# Training parameters
LEARNING_RATE: float = 1e-3
WEIGHT_DECAY: float = 1e-4
EMA = 0.999
N_EPOCH: int = 30

# Other parameters
RANDOM_STATE: int = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

# %%
for dir in [DICT_DIR, DATASET_DIR, IMAGE_DIR, CHECKPOINT_DIR, OUTPUT_DIR]:
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True)

# %% [markdown]
# ## Preprocess text

# %%
vocab_path = Path(DICT_DIR) / "vocab.npy"
word2idx_path = Path(DICT_DIR) / "word2Id.npy"
idx2word_path = Path(DICT_DIR) / "id2Word.npy"

word2idx: dict = dict(np.load(word2idx_path))
idx2word: dict = dict(np.load(idx2word_path))


def seq2sent(seq: list[int]) -> str:
    pad_id = word2idx["<PAD>"]
    sent = [idx2word[idx] for idx in seq if idx != pad_id]
    return " ".join(sent)


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
    cap_seqs = df_train["Captions"].to_numpy()
    cap_sents = [[seq2sent(cap_seq) for cap_seq in _cap_seqs] for _cap_seqs in cap_seqs]
    embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()

    # Change image path
    image_paths = (
        df_train["ImagePath"].apply(lambda x: Path(IMAGE_DIR) / Path(x).name).to_numpy()
    )
    df_train = pd.DataFrame(
        {"Captions": cap_sents, "Embeddings": embeddings, "ImagePath": image_paths}
    )

    # Create embedding for testing data
    tqdm.pandas(desc="Embedding testing data")
    cap_seqs = df_test["Captions"]
    cap_sents = [seq2sent(cap_seq) for cap_seq in cap_seqs]
    embeddings = pd.Series(cap_sents).progress_apply(embed_sent_list).to_numpy()
    id = df_test["ID"].to_numpy()
    df_test = pd.DataFrame({"Captions": cap_sents, "Embeddings": embeddings, "ID": id})

    return df_train, df_test


if (Path(DATASET_DIR) / "embeddings_train.pkl").exists():
    df_train = pd.read_pickle(Path(DATASET_DIR) / "embeddings_train.pkl")
    df_test = pd.read_pickle(Path(DATASET_DIR) / "embeddings_test.pkl")
    print("Load embeddings from pickle file")
else:
    df_train = pd.read_pickle(Path(DATASET_DIR) / "text2ImgData.pkl")
    df_test = pd.read_pickle(Path(DATASET_DIR) / "testData.pkl")
    df_train, df_test = generate_embed_df(df_train, df_test)
    df_train.to_pickle(Path(DATASET_DIR) / "embeddings_train.pkl")
    df_test.to_pickle(Path(DATASET_DIR) / "embeddings_test.pkl")
    print("Generate embeddings and save to pickle file")

df_train.head()
# %% [markdown]
# ## Dataset


# %%
def load_image(path: tf.Tensor) -> np.ndarray:
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [IMAGE_SIZE, IMAGE_SIZE])
    return img


@tf.numpy_function(Tout=tf.float32)
def augment_image(img: tf.Tensor) -> tf.Tensor:
    aug = alb.Compose(
        [
            alb.HorizontalFlip(p=0.25),
            alb.VerticalFlip(p=0.25),
            alb.RandomRotate90(p=0.25),
        ]
    )
    img = aug(image=img)["image"]
    return img


def map_func(embedding, img_path):
    img = load_image(img_path)
    img = augment_image(img)
    img = tf.reshape(img, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
    return img, embedding


def generate_dataset(df: pd.DataFrame, type: str):
    expolde_df = df.explode("Embeddings")
    embedding = np.stack(expolde_df["Embeddings"].values)
    if type == "train" or type == "val":
        img_path = expolde_df["ImagePath"].to_numpy().astype(str)
        dataset = tf.data.Dataset.from_tensor_slices((embedding, img_path))
        dataset = dataset.map(map_func, num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(len(expolde_df)) if type == "train" else dataset
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    elif type == "test":
        id = expolde_df["ID"].to_numpy()
        dataset = tf.data.Dataset.from_tensor_slices((embedding, id))
        dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


# %%
_df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)
dataset_train = generate_dataset(_df_train, "train")
dataset_val = generate_dataset(df_val, "val")
dataset_test = generate_dataset(df_test, "test")

# %% [markdown]
# ## Define model


# %% Loss function
# Kernel inception distance
class KID(keras.metrics.Metric):
    def __init__(self, name="KID", **kwargs):
        super().__init__(name=name, **kwargs)
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

        batch_size = tf.shape(img_real)[0]
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


# %% Noise predictor
# @tf.function
# def sinusoidal_embedding(x: tf.Tensor):
#     """Generate sinusoidal embeddings for step (noise variance)"""
#     min_freq = 1.0
#     max_freq = STEP_EMBED_MAX_FREQ

#     # Calculate frequencies and angular speeds
#     freqs = tf.exp(
#         tf.linspace(
#             start=tf.math.log(min_freq),
#             stop=tf.math.log(max_freq),
#             num=EMBEDDING_DIM // 2,
#         )
#     )
#     angular_speeds = tf.cast(2 * math.pi * freqs, tf.float32)

#     # Generate embeddings by concatenating sine and cosine functions
#     embeddings = tf.concat([tf.sin(x * angular_speeds), tf.cos(x * angular_speeds)], -1)

#     return embeddings


# def residual_block(width):
#     def apply(x):
#         input_width = x.shape[3]
#         residual = x if input_width == width else layers.Conv2D(width, kernel_size=1)(x)
#         x = layers.BatchNormalization(center=False, scale=False)(x)
#         x = layers.Conv2D(width, kernel_size=3, padding="same", activation="swish")(x)
#         x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
#         x = layers.Add()([x, residual])
#         return x

#     return apply


# def get_network(image_size, widths, block_depth):
#     noisy_images = keras.Input(shape=(image_size, image_size, 3))
#     noise_variances = keras.Input(shape=(1, 1))
#     text_embedding = keras.Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM))

#     noise_embedding = layers.Lambda(
#         sinusoidal_embedding, output_shape=(1, EMBEDDING_DIM)
#     )(noise_variances)
#     embeddings = layers.Concatenate(axis=1)([text_embedding, noise_embedding])
#     embeddings = layers.Dense(widths[0])(embeddings)

#     x = layers.Conv2D(widths[0], kernel_size=1)(noisy_images)
#     embeddings = layers.Attention()([x, embeddings])
#     x = layers.concatenate([x, embeddings])

#     skips = []
#     # Downsampling blocks
#     for width in widths[:-1]:
#         for _ in range(block_depth):
#             x = residual_block(width)(x)
#             skips.append(x)
#         x = layers.AveragePooling2D(pool_size=2)(x)

#     # Bottleneck block
#     for _ in range(block_depth):
#         x = residual_block(widths[-1])(x)

#     # Upsampling blocks
#     for width in reversed(widths[:-1]):
#         x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
#         for _ in range(block_depth):
#             x = layers.Concatenate()([x, skips.pop()])
#             x = residual_block(width)(x)

#     noise = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

#     return keras.Model(
#         [noisy_images, noise_variances, text_embedding], noise, name="noise_predictor"
#     )


def get_network(
    image_size,
    noise_embedding_max_frequency,
    noise_embedding_dims,
    image_embedding_dims,
    block_depth,
    widths,
    attentions,
    patch_size,
):
    def EmbeddingLayer(embedding_max_frequency, embedding_dims):
        def sinusoidal_embedding(x):
            embedding_min_frequency = 1.0
            frequencies = tf.exp(
                tf.linspace(
                    tf.math.log(embedding_min_frequency),
                    tf.math.log(embedding_max_frequency),
                    embedding_dims // 2,
                )
            )
            angular_speeds = 2.0 * math.pi * frequencies
            embeddings = tf.concat(
                [
                    tf.sin(angular_speeds * x),
                    tf.cos(angular_speeds * x),
                ],
                axis=3,
            )
            return embeddings

        def forward(x):
            x = layers.Lambda(sinusoidal_embedding)(x)
            return x

        return forward

    def ResidualBlock(width, attention):
        def forward(x):
            x, n = x
            input_width = x.shape[3]
            if input_width == width:
                residual = x
            else:
                residual = layers.Conv2D(width, kernel_size=1)(x)

            n = layers.Dense(width)(n)

            x = layers.GroupNormalization(groups=8)(x)
            x = keras.activations.swish(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

            x = layers.Add()([x, n])

            x = layers.GroupNormalization(groups=8)(x)
            x = keras.activations.swish(x)
            x = layers.Conv2D(width, kernel_size=3, padding="same")(x)

            x = layers.Add()([residual, x])

            if attention:
                residual = x
                x = layers.GroupNormalization(groups=8, center=False, scale=False)(x)
                x = layers.MultiHeadAttention(
                    num_heads=4, key_dim=width, attention_axes=(1, 2)
                )(x, x)

                x = layers.Add()([residual, x])

            return x

        return forward

    def DownBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            for _ in range(block_depth):
                x = ResidualBlock(width, attention)([x, n])
                skips.append(x)
            x = layers.AveragePooling2D(pool_size=2)(x)
            return x

        return forward

    def UpBlock(block_depth, width, attention):
        def forward(x):
            x, n, skips = x
            x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
            for _ in range(block_depth):
                x = layers.Concatenate()([x, skips.pop()])
                x = ResidualBlock(width, attention)([x, n])
            return x

        return forward

    images = keras.Input(shape=(image_size, image_size, 3))
    noise_powers = keras.Input(shape=(1, 1, 1))

    x = layers.Conv2D(image_embedding_dims, kernel_size=patch_size, strides=patch_size)(
        images
    )

    n = EmbeddingLayer(noise_embedding_max_frequency, noise_embedding_dims)(
        noise_powers
    )
    n = layers.Dense(noise_embedding_dims, activation=keras.activations.swish)(n)
    n = layers.Dense(noise_embedding_dims, activation=keras.activations.swish)(n)
    n = layers.Reshape((1, noise_embedding_dims))(n)

    text_embedding = keras.Input(shape=(MAX_SEQ_LENGTH, EMBEDDING_DIM))

    n = layers.Concatenate(axis=1)([text_embedding, n])
    n = layers.Flatten()(n)
    n = layers.Dense(image_embedding_dims, activation=keras.activations.swish)(n)
    n = layers.Dense(image_embedding_dims, activation=keras.activations.swish)(n)

    skips = []
    for width, attention in zip(widths[:-1], attentions[:-1]):
        x = DownBlock(block_depth, width, attention)([x, n, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], attentions[-1])([x, n])

    for width, attention in zip(widths[-2::-1], attentions[-2::-1]):
        x = UpBlock(block_depth, width, attention)([x, n, skips])

    x = layers.Conv2DTranspose(
        3, kernel_size=patch_size, strides=patch_size, kernel_initializer="zeros"
    )(x)

    return keras.Model([images, noise_powers, text_embedding], x, name="residual_unet")


# %%
class EMACallback(keras.callbacks.Callback):
    def __init__(self, decay=0.999):
        super().__init__()
        self.decay = decay

    def on_train_begin(self, logs=None):
        self.ema_weights = self.model.get_weights()

    def on_train_batch_end(self, batch, logs=None):
        for i, weight in enumerate(self.model.get_weights()):
            self.ema_weights[i] = (
                self.decay * self.ema_weights[i] + (1 - self.decay) * weight
            )

    def on_train_end(self, logs=None):
        self.model.set_weights(self.ema_weights)

    def on_test_begin(self, logs=None):
        self.backup = self.model.get_weights()
        self.model.set_weights(self.ema_weights)

    def on_test_end(self, logs=None):
        self.model.set_weights(self.backup)


# %%
# TODO: Refactor the model
class DiffusionModel(keras.Model):
    def __init__(self, image_size, widths, block_depth):
        super().__init__()

        self.normalizer = layers.Normalization()
        self.network = get_network(
            image_size,
            STEP_EMBED_MAX_FREQ,
            512,
            UNET_WIDTHS[0],
            RES_BLOCK_DEPTH,
            UNET_WIDTHS,
            [False, False, True, True],
            1,
        )
        self.ema_weights = None

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="noise_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="image_loss")
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
        image_ratio = tf.math.cos(diffusion_angles)
        noise_ratio = tf.math.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1
        return noise_ratio, image_ratio

    def denoise(
        self, noisy_images, noise_ratio, image_ratio, text_embeddings, training
    ):
        # the exponential moving average weights are used at evaluation
        network = self.network

        # predict noise component and calculate the image component using it
        pred_noises = network(
            [noisy_images, noise_ratio**2, text_embeddings], training=training
        )
        pred_images = (noisy_images - noise_ratio * pred_noises) / image_ratio

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
            noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_ratio, image_ratio, text_embeddings, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_ratio, next_image_ratio = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                next_image_ratio * pred_images + next_noise_ratio * pred_noises
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
        noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = image_ratio * images + noise_ratio * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_ratio, image_ratio, embeddings, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # Increate the robustness of the model by using exponential moving average

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
        noise_ratio, image_ratio = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = image_ratio * images + noise_ratio * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_ratio, image_ratio, embeddings, training=False
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

    def plot_images(
        self, epoch=None, logs=None, num_rows=3, num_cols=6, text_embeddings=None
    ):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            text_embeddings=text_embeddings,
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


# %% [markdown]
# ## Train

# %%
ckpt_path = Path(CHECKPOINT_DIR) / "diffusion.ckpt"
ckpt_callback = keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    verbose=0,
)

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

model.normalizer.adapt(dataset_train.map(lambda x, y: x))

model.fit(
    dataset_train,
    validation_data=dataset_val,
    epochs=N_EPOCH,
    verbose=1,
    callbacks=[EMACallback(decay=EMA), ckpt_callback],
)

# %%
if (Path(OUTPUT_DIR) / "inference").exists():
    shutil.rmtree(Path(OUTPUT_DIR) / "inference")

(Path(OUTPUT_DIR) / "inference").mkdir(parents=True)

test_epoch = len(df_test) // BATCH_SIZE + 1
step = 0
for embedding, id in dataset_test:
    step += 1
    if step > test_epoch:
        break

    generated_images = model.generate(
        num_images=BATCH_SIZE,
        text_embeddings=embedding,
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
