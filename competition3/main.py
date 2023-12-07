# %% [markdown]
# ## Prepare environment

# %%
import functools
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
from tensorflow import keras
from tensorflow.python.keras import layers
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
MAX_SENTENCE_LENGTH: int = 20

# Image parameters
IMAGE_HEIGHT: int = 64
IMAGE_WIDTH: int = 64
IMAGE_CHANNELS: int = 3

# Dataset parameters
BATCH_SIZE: int = 64

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
            max_length=MAX_SENTENCE_LENGTH,
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
class DatasetGenerator:
    def __init__(self, df_train, df_test) -> None:
        self.df_train = df_train
        self.df_test = df_test

    def _load_image(self, path: tf.Tensor) -> np.ndarray:
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=IMAGE_CHANNELS)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [IMAGE_HEIGHT, IMAGE_WIDTH])
        return img

    def _process_pipeline(self, embedding, img_path):
        img = self._load_image(img_path)
        return img, embedding

    def generate_train(self) -> tf.data.Dataset:
        datas = self.df_train.explode("Embeddings")
        embedding = np.stack(datas["Embeddings"].values)
        img_path = datas["ImagePath"].values

        dataset = tf.data.Dataset.from_tensor_slices((embedding, img_path))
        dataset = (
            dataset.map(self._process_pipeline, num_parallel_calls=AUTOTUNE)
            .shuffle(len(datas))
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(AUTOTUNE)
        )
        return dataset

    def generate_test(self) -> tf.data.Dataset:
        embeddings = self.df_test["Embeddings"].values
        embeddings = np.stack(embeddings)
        idx = self.df_test["ID"].values

        dataset = tf.data.Dataset.from_tensor_slices((embeddings, idx))
        dataset = dataset.repeat().batch(BATCH_SIZE).prefetch(AUTOTUNE)
        return dataset


# %%
dataset_generator = DatasetGenerator(df_train, df_test)
dataset_train = dataset_generator.generate_train()
dataset_test = dataset_generator.generate_test()

del dataset_generator

# %%# %% [markdown]
# ## Define model


# %%
class TextEncoder(tf.keras.Model):
    """
    Encode text (a caption) into hidden representation
    input: text, which is a list of ids
    output: embedding, or hidden representation of input text in dimension of RNN_HIDDEN_SIZE
    """

    def __init__(self, hparas):
        super(TextEncoder, self).__init__()
        self.hparas = hparas
        self.batch_size = self.hparas["BATCH_SIZE"]

        # embedding with tensorflow API
        self.embedding = layers.Embedding(
            self.hparas["VOCAB_SIZE"], self.hparas["EMBED_DIM"]
        )
        # RNN, here we use GRU cell, another common RNN cell similar to LSTM
        self.gru = layers.GRU(
            self.hparas["RNN_HIDDEN_SIZE"],
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, text, hidden):
        text = self.embedding(text)
        output, state = self.gru(text, initial_state=hidden)
        return output[:, -1, :], state

    def initialize_hidden_state(self):
        return tf.zeros((self.hparas["BATCH_SIZE"], self.hparas["RNN_HIDDEN_SIZE"]))


class Generator(tf.keras.Model):
    """
    Generate fake image based on given text(hidden representation) and noise z
    input: text and noise
    output: fake image with size 64*64*3
    """

    def __init__(self, hparas):
        super(Generator, self).__init__()
        self.hparas = hparas
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(self.hparas["DENSE_DIM"])
        self.d2 = tf.keras.layers.Dense(64 * 64 * 3)

    def call(self, text, noise_z):
        text = self.flatten(text)
        text = self.d1(text)
        text = tf.nn.leaky_relu(text)

        # concatenate input text and random noise
        text_concat = tf.concat([noise_z, text], axis=1)
        text_concat = self.d2(text_concat)

        logits = tf.reshape(text_concat, [-1, 64, 64, 3])
        output = tf.nn.tanh(logits)

        return logits, output


class Discriminator(tf.keras.Model):
    """
    Differentiate the real and fake image
    input: image and corresponding text
    output: labels, the real image should be 1, while the fake should be 0
    """

    def __init__(self, hparas):
        super(Discriminator, self).__init__()
        self.hparas = hparas
        self.flatten = tf.keras.layers.Flatten()
        self.d_text = tf.keras.layers.Dense(self.hparas["DENSE_DIM"])
        self.d_img = tf.keras.layers.Dense(self.hparas["DENSE_DIM"])
        self.d = tf.keras.layers.Dense(1)

    def call(self, img, text):
        text = self.flatten(text)
        text = self.d_text(text)
        text = tf.nn.leaky_relu(text)

        img = self.flatten(img)
        img = self.d_img(img)
        img = tf.nn.leaky_relu(img)

        # concatenate image with paired text
        img_text = tf.concat([text, img], axis=1)

        logits = self.d(img_text)
        output = tf.nn.sigmoid(logits)

        return logits, output


# %%
hparas = {
    "MAX_SEQ_LENGTH": 20,  # maximum sequence length
    "EMBED_DIM": 256,  # word embedding dimension
    "VOCAB_SIZE": len(word2idx),  # size of dictionary of captions
    "RNN_HIDDEN_SIZE": 128,  # number of RNN neurons
    "Z_DIM": 512,  # random noise z dimension
    "DENSE_DIM": 128,  # number of neurons in dense layer
    "IMAGE_SIZE": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],  # render image size
    "BATCH_SIZE": BATCH_SIZE,
    "LR": 1e-4,
    "LR_DECAY": 0.5,
    "BETA_1": 0.5,
    "N_EPOCH": 600,
    "N_SAMPLE": len(dataset_train) * BATCH_SIZE,  # size of training data
    "PRINT_FREQ": 1,  # printing frequency of loss
}

text_encoder = TextEncoder(hparas)
generator = Generator(hparas)
discriminator = Discriminator(hparas)


# %% [markdown]
# ## Loss function and optimizer

# %%
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_logit, fake_logit):
    real_loss = cross_entropy(tf.ones_like(real_logit), real_logit)
    fake_loss = cross_entropy(tf.zeros_like(fake_logit), fake_logit)
    return real_loss + fake_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(hparas["LR"])
discriminator_optimizer = tf.keras.optimizers.Adam(hparas["LR"])


# %%
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    text_encoder=text_encoder,
    generator=generator,
    discriminator=discriminator,
)
ckpt_manager = tf.train.CheckpointManager(
    checkpoint, CHECKPOINT_DIR, max_to_keep=5, checkpoint_name="ckpt"
)


# %%
@tf.function
def train_step(real_image, caption, hidden):
    # random noise for generator
    noise = tf.random.normal(
        shape=[hparas["BATCH_SIZE"], hparas["Z_DIM"]], mean=0.0, stddev=1.0
    )

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        text_embed, hidden = text_encoder(caption, hidden)
        _, fake_image = generator(text_embed, noise)
        real_logits, real_output = discriminator(real_image, text_embed)
        fake_logits, fake_output = discriminator(fake_image, text_embed)

        g_loss = generator_loss(fake_logits)
        d_loss = discriminator_loss(real_logits, fake_logits)

    grad_g = gen_tape.gradient(g_loss, generator.trainable_variables)
    grad_d = disc_tape.gradient(d_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(grad_g, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(grad_d, discriminator.trainable_variables)
    )

    return g_loss, d_loss


@tf.function
def test_step(caption, noise, hidden):
    text_embed, hidden = text_encoder(caption, hidden)
    _, fake_image = generator(text_embed, noise)
    return fake_image


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
sample_seed = np.random.normal(
    loc=0.0, scale=1.0, size=(sample_size, hparas["Z_DIM"])
).astype(np.float32)
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


# %% [markdown]
# ## Training

# %%
output_dir = os.path.join(OUTPUT_DIR, "train")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%
def train(dataset, epochs):
    # hidden state of RNN
    hidden = text_encoder.initialize_hidden_state()
    steps_per_epoch = int(hparas["N_SAMPLE"] / BATCH_SIZE)
    pbar = trange(epochs, desc="Epoch", unit="epoch")

    for epoch in pbar:
        g_total_loss = 0
        d_total_loss = 0
        start = time.time()

        for image, caption in dataset:
            g_loss, d_loss = train_step(image, caption, hidden)
            g_total_loss += g_loss
            d_total_loss += d_loss

        pbar.set_postfix(
            {
                "gen_loss": g_total_loss.numpy() / steps_per_epoch,
                "disc_loss": d_total_loss.numpy() / steps_per_epoch,
                "time": time.time() - start,
            }
        )

        # save the model
        if (epoch + 1) % 50 == 0:
            ckpt_manager.save()

        # visualization
        if (epoch + 1) % hparas["PRINT_FREQ"] == 0:
            for caption in sample_sentence:
                fake_image = test_step(caption, sample_seed, hidden)
            save_images(fake_image, [ni, ni], f"{output_dir}/train_{epoch:02d}.jpg")


# %%
train(dataset_train, hparas["N_EPOCH"])


# %% [markdown]
# ## Evalutation

# %%
output_dir = os.path.join(OUTPUT_DIR, "inference")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# %%
def inference(dataset):
    hidden = text_encoder.initialize_hidden_state()
    sample_size = BATCH_SIZE
    sample_seed = np.random.normal(
        loc=0.0, scale=1.0, size=(sample_size, hparas["Z_DIM"])
    ).astype(np.float32)

    start = time.time()
    pbar = trange(819 // BATCH_SIZE + 1, desc="Inference", unit="batch")
    for _ in pbar:
        captions, idx = next(iter(dataset))
        fake_image = test_step(captions, sample_seed, hidden)
        for i in range(BATCH_SIZE):
            plt.imsave(
                f"{output_dir}/inference_{idx[i]:04d}.jpg",
                fake_image[i].numpy() * 0.5 + 0.5,
            )

    print("Time for inference is {:.4f} sec".format(time.time() - start))


# %%
ckpt_manager.restore_or_initialize()
inference(dataset_test)


# %%
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# %cd evaluation
# !python inception_score.py ../output/inference ../output/score.csv 39
# %cd ..


# %%
df_score = pd.read_csv("./output/score.csv")
print(f"Score: {np.mean(df_score['score']):.4f} ± {np.std(df_score['score']):.4f}")
