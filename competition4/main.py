# %%
import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ["VISIBLE_CUDA_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation.environment import TestingEnvironment, TrainingEnvironment
from src.recommender import FunkSVDRecommender
from tqdm import tqdm, trange
from transformers import BertModel, BertTokenizer

# %% Check GPU
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Select GPU number 1
        tf.config.experimental.set_visible_devices(gpus[0], "GPU")
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# %% Hyperparameters
@dataclass
class ConstParams:
    N_TRAIN_USERS: int = 1000
    N_TEST_USERS: int = 2000
    N_ITEMS: int = 209527
    HORIZON: int = 2000
    TEST_EPISODES: int = 5
    SLATE_SIZE: int = 5


@dataclass
class HParams:
    EMBED_SIZE: int = 256
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.004
    RANDOM_STATE: int = 42
    NUM_EPOCHS: int = 25


@dataclass
class Paths:
    USER_DATA: Path = Path("./dataset/user_data.json")
    ITEM_DATA: Path = Path("./dataset/item_data.json")
    OUTPUT: Path = Path("./output/output.csv")
    CHECKPOINT_DIR: Path = Path("./checkpoint")


random.seed(HParams.RANDOM_STATE)


# %% Load data
# user_id, history (last 3 clicked items)
df_user = pd.read_json(Paths.USER_DATA, lines=True)
# item_id, headline, short_description
df_item = pd.read_json(Paths.ITEM_DATA, lines=True)

# %% Training
# Init
model = FunkSVDRecommender(
    m_users=ConstParams.N_TRAIN_USERS,
    n_items=ConstParams.N_ITEMS,
    embedding_size=HParams.EMBED_SIZE,
    learning_rate=HParams.LEARNING_RATE,
)
train_env = TrainingEnvironment()

# Pre-training with history
history: np.ndarray = df_user.explode("history").reset_index(drop=True).to_numpy()
history: np.ndarray = np.hstack((history, np.ones((history.shape[0], 1)))).astype(float)
dataset: tf.data.Dataset = (
    tf.data.Dataset.from_tensor_slices(history)
    .shuffle(buffer_size=len(history))
    .batch(HParams.BATCH_SIZE, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

print("[Pre-training]")
for _ in trange(HParams.NUM_EPOCHS):
    _ = [model.train_step(data) for data in dataset]

# Initialize the training environment
print("[Training]")
for epoch in range(HParams.NUM_EPOCHS):
    train_env.reset()
    losses = []

    pbar = tqdm(desc=f"Epoch {epoch}")
    while train_env.has_next_state():
        user_id = train_env.get_state()
        slate = model.recommand_top5([user_id])
        clicked_id, in_environment = train_env.get_response(slate)

        data = tf.convert_to_tensor([(user_id, x, int(x == clicked_id)) for x in slate])

        loss = model.train_step(data)
        losses.append(loss)

        pbar.update(1)
    pbar.set_postfix({"loss": np.mean(losses), "score": np.mean(train_env.get_score())})
    pbar.close()


# %% Testing
# Initialize the testing environment
test_env = TestingEnvironment()
scores = []

# The item_ids here is for the random recommender
item_ids = list(range(ConstParams.N_ITEMS))

# Repeat the testing process for 5 times
for epoch in range(ConstParams.TEST_EPISODES):
    # [TODO] Load your model weights here (in the beginning of each testing episode)
    # [TODO] Code for loading your model weights...
    model = tf.keras.models.load_model(Paths.CHECKPOINT_DIR / "model")

    # Start the testing process
    with tqdm(desc="Testing") as pbar:
        # Run as long as there exist some active users
        while test_env.has_next_state():
            # Get the current user id
            cur_user = test_env.get_state()

            # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
            # [TODO] Code for generating the recommended slate...
            # Here we provide a simple random implementation
            slate = model.recommand_top5([user_id])

            # Get the response of the slate from the environment
            clicked_id, in_environment = test_env.get_response(slate)

            # [TODO] Update your model here (optional)
            # [TODO] You can update your model at each step, or perform a batched update after some interval
            # [TODO] Code for updating your model...

            # Update the progress indicator
            pbar.update(1)

    # Record the score of this testing episode
    scores.append(test_env.get_score())

    # Reset the testing environment
    test_env.reset()

    # [TODO] Delete or reset your model weights here (in the end of each testing episode)
    # [TODO] Code for deleting your model weights...
    del model

# Calculate the average scores
avg_scores = [np.average(score) for score in zip(*scores)]

# Generate a DataFrame to output the result in a .csv file
df_result = pd.DataFrame(
    [[user_id, avg_score] for user_id, avg_score in enumerate(avg_scores)],
    columns=["user_id", "avg_score"],
)
df_result.to_csv(Paths.OUTPUT, index=False)
df_result

# %% Encode items
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedder = BertModel.from_pretrained("bert-base-uncased")


# %%
def embed_pipeline(input_text: str) -> np.ndarray:
    token_seq = tokenizer(input_text, return_tensors="pt")
    embedding_seq = embedder(**token_seq)[0]
    return embedding_seq.detach().numpy()


input_text = df_item["headline"][:10]
embeddings = list(map(embed_pipeline, tqdm(input_text)))

# %% AutoRec
