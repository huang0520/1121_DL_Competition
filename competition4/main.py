# %%
import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation.environment import TestingEnvironment, TrainingEnvironment
from src.dataset import DatasetGenerator
from src.loss import PairwiseRankingLoss
from src.recommender import FunkSVDRecommender, NeuMF
from tensorflow import keras
from tqdm.auto import tqdm, trange
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
    EMBED_SIZE: int = 64
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.00005
    WEIGHT_DECAY: float = 0.003
    RANDOM_STATE: int = 42
    NUM_EPOCHS: int = 10
    NUM_EPOSIDES: int = 200
    NUMS_HIDDENS: tuple[int] = (32, 32, 32, 32)


@dataclass
class Paths:
    USER_DATA: Path = Path("./dataset/user_data.json")
    ITEM_DATA: Path = Path("./dataset/item_data.json")
    OUTPUT: Path = Path("./output/output.csv")
    CHECKPOINT_DIR: Path = Path("./checkpoint")


random.seed(HParams.RANDOM_STATE)


# %% Load data
# # user_id, history (last 3 clicked items)
# df_user = pd.read_json(Paths.USER_DATA, lines=True)
# # item_id, headline, short_description
# df_item = pd.read_json(Paths.ITEM_DATA, lines=True)


# %%
# Training pipeline
def train(model, dataset):
    loss_record = []

    pbar = trange(HParams.NUM_EPOCHS, desc="Training", ncols=60)
    for _ in pbar:
        losses = [model.train_step(data) for data in dataset]
        loss = tf.reduce_mean(losses).numpy()
        loss_record.append(loss)
        pbar.set_postfix({"loss": loss})
    pbar.set_postfix({"loss": np.mean(loss_record)}, refresh=True)

    return model, loss_record


# Explore pipeline
def explore(env, model: NeuMF, slate_size=5):
    hit_pairs = []

    pbar = tqdm(desc="Exploring")
    while env.has_next_state():
        user_id = env.get_state()
        slate = model.get_topn(user_id, slate_size)
        clicked_id, _ = env.get_response(slate)

        if clicked_id != -1:
            hit_pairs.append((user_id, clicked_id))

        pbar.update(1)
        pbar.set_postfix({"#click": len(hit_pairs)})

    return hit_pairs


# Simulate pipeline
def explore_and_train(env, model: NeuMF, dataset_generator: DatasetGenerator):
    for i in range(HParams.NUM_EPOSIDES):
        print(f"[Eposide {i + 1}/{HParams.NUM_EPOSIDES}]")

        # Explore
        env.reset()
        add_pairs = explore(env, model, ConstParams.SLATE_SIZE)

        # Add new items
        num_new_items = dataset_generator.add_items(*zip(*add_pairs))
        print(f"Add {num_new_items} new items to the dataset")

        # Train
        dataset = dataset_generator.generate(HParams.BATCH_SIZE)
        model, _ = train(model, dataset)

        print(f"Average Score: {np.mean(env.get_score()):.6f}")
    return model


# %% Training
# Init
model = NeuMF(
    HParams.EMBED_SIZE,
    ConstParams.N_TRAIN_USERS,
    ConstParams.N_ITEMS,
    HParams.NUMS_HIDDENS,
)
model.compile(
    optimizer=keras.optimizers.Lion(
        HParams.LEARNING_RATE, weight_decay=HParams.WEIGHT_DECAY
    ),
    loss=PairwiseRankingLoss(margin=0.2),
)
dataset_generator = DatasetGenerator(Paths.USER_DATA, Paths.ITEM_DATA)
train_env = TrainingEnvironment()

model = explore_and_train(train_env, model, dataset_generator)

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

    # Start the testing process
    with tqdm(desc="Testing") as pbar:
        # Run as long as there exist some active users
        while test_env.has_next_state():
            # Get the current user id
            cur_user = test_env.get_state()

            # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
            # [TODO] Code for generating the recommended slate...
            # Here we provide a simple random implementation
            slate = model.get_topn(cur_user, 5)

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
