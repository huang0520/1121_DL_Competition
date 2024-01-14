# %%
import itertools
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation.environment import TestingEnvironment, TrainingEnvironment
from icecream import ic
from sklearn.metrics.pairwise import cosine_similarity
from src.dataset import DataManager, History, LabelTokenDatasetGenerator
from src.recommender import FMEmbeding, SparseFM, SparseHotEncoder
from tensorflow import keras
from tqdm.auto import tqdm, trange

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
    NUM_TOKENS: int = 49408


@dataclass
class HParams:
    EMBED_SIZE: int = 20
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.00001
    WEIGHT_DECAY: float = 0.00012
    RANDOM_STATE: int = 42
    NUM_EPOCHS: int = 5
    NUM_EPISODES: int = 100
    N_NEGTIVES: int = 12


@dataclass
class Paths:
    USER_DATA: Path = Path("./dataset/user_data.json")
    ITEM_DATA: Path = Path("./dataset/item_data.json")
    TOKEN_PATH: Path = Path("./dataset/item_token.pkl")
    OUTPUT: Path = Path("./output/output.csv")
    CHECKPOINT_DIR: Path = Path("./checkpoint")
    USER_DATA_PLUS: Path = Path("./dataset/user_data_plus.pkl")


# %%
# Initialize
random.seed(HParams.RANDOM_STATE)

data_manager = DataManager(Paths.USER_DATA, Paths.ITEM_DATA, Paths.TOKEN_PATH)
data_manager.load(Paths.USER_DATA_PLUS) if Paths.USER_DATA_PLUS.exists() else None
history = History(Paths.USER_DATA)

model = SparseFM(HParams.EMBED_SIZE, 0.1, 0.1)
model.compile(
    optimizer=tf.keras.optimizers.Lion(
        learning_rate=HParams.LEARNING_RATE,
        weight_decay=HParams.WEIGHT_DECAY,
        use_ema=True,
    ),
    loss=tf.keras.losses.BinaryFocalCrossentropy(
        apply_class_balancing=True, from_logits=True, label_smoothing=0.1
    ),
)
encoder = SparseHotEncoder(
    ConstParams.N_TEST_USERS, ConstParams.N_ITEMS, ConstParams.NUM_TOKENS
)


# %%
# Functions
def train(model, encoder, dataset, n_neg=8):
    epoch_loss = []

    pbar = trange(HParams.NUM_EPOCHS, desc="Training", ncols=0)
    for _ in pbar:
        batch_loss = []

        for data in dataset:
            user_ids, item_ids, title, desc, labels = data
            batch_size = len(user_ids)
            losses = []

            # Positive samples
            features = encoder((user_ids, item_ids, title, desc))
            loss = model.train_step((features, labels))
            losses.append(loss)

            # Random negative samples
            neg_item_ids = tf.random.uniform(
                shape=(n_neg, batch_size),
                minval=0,
                maxval=ConstParams.N_ITEMS,
                dtype=tf.int32,
            )
            for _neg_item_ids in neg_item_ids:
                features = encoder((user_ids, _neg_item_ids, title, desc))
                loss = model.train_step((features, tf.zeros(batch_size)))
                losses.append(loss)

            batch_loss.append(tf.reduce_mean(losses).numpy())

        epoch_loss.append(np.mean(batch_loss))
        pbar.set_postfix({"loss": epoch_loss[-1]})
    pbar.set_postfix({"loss": np.mean(epoch_loss)}, refresh=True)

    return model, np.mean(epoch_loss)


class RecallSystem:
    def __init__(self, model, encoder, data_manager):
        fm_embedding = FMEmbeding(
            encoder,
            ConstParams.N_TEST_USERS,
            ConstParams.N_ITEMS,
            ConstParams.NUM_TOKENS,
            model,
        )

        df_seq = pd.DataFrame(
            data_manager.get_sequences(), columns=["user_id", "item_id"]
        )
        df_seq["title"] = df_seq["item_id"].map(data_manager.item_to_tokens["headline"])
        df_seq["desc"] = df_seq["item_id"].map(
            data_manager.item_to_tokens["short_description"]
        )

        self.user_embedding_dict = fm_embedding.get_user_embedding(df_seq)
        self.item_embedding_dict = fm_embedding.get_item_embedding(df_seq)

    def u2i_topk(self, user_id, k=5):
        user_embedding = self.user_embedding_dict[user_id]
        l2_dis = {
            key: np.linalg.norm(user_embedding - item_embedding)
            for key, item_embedding in self.item_embedding_dict.items()
        }
        return tuple(zip(*(sorted(l2_dis.items(), key=lambda x: x[1])[:k])))[0]

    def i2i_topk(self, item_id, k=5):
        item_embedding = self.item_embedding_dict[item_id]
        similarity = cosine_similarity(
            [item_embedding], list(self.item_embedding_dict.values())
        )[0]
        similarity = dict(zip(self.item_embedding_dict.keys(), similarity))

        return tuple(
            zip(*(sorted(similarity.items(), key=lambda x: x[1], reverse=True)[:k]))
        )[0]


def update(user_id, clicked_id, model, encoder, data_manager):
    # Update model
    # Postive samples
    title, desc = data_manager.item_to_tokens.iloc[clicked_id]
    features = encoder((user_id, clicked_id, title, desc))
    model.train_step((features, tf.ones(1)))

    # Negative samples
    neg_item_ids = tf.random.uniform(
        shape=(HParams.N_NEGTIVES,),
        minval=0,
        maxval=ConstParams.N_ITEMS,
        dtype=tf.int32,
    )
    neg_title = tf.ragged.constant([
        data_manager.item_to_tokens.iloc[item_id.numpy()].iloc[0]
        for item_id in neg_item_ids
    ])
    neg_desc = tf.ragged.constant([
        data_manager.item_to_tokens.iloc[item_id.numpy()].iloc[1]
        for item_id in neg_item_ids
    ])
    neg_features = encoder((
        [user_id] * HParams.N_NEGTIVES,
        neg_item_ids,
        neg_title,
        neg_desc,
    ))
    model.train_step((neg_features, tf.zeros(HParams.N_NEGTIVES)))

    # Update embedding dict
    recall_system = RecallSystem(model, encoder, data_manager)

    return model, recall_system


def explore(env, model, encoder, recall_system, history, data_manager, update_freq=1):
    hit_count = 0
    pbar = tqdm(desc="Explore")
    while env.has_next_state():
        user_id = env.get_state()
        item_id = random.choice(history.get(user_id))

        slate = recall_system.i2i_topk(item_id, ConstParams.SLATE_SIZE)
        clicked_id, _ = env.get_response(slate)

        if clicked_id != -1:
            hit_count += 1
            history.add(user_id, clicked_id)
            data_manager.add(user_id, clicked_id)

            if hit_count % update_freq == 0:
                model, recall_system = update(
                    user_id, clicked_id, model, encoder, data_manager
                )

        pbar.update(1)
        pbar.set_postfix({"hit": hit_count})

    return model, history, data_manager


# %%

# Explore and train
for episode in range(HParams.NUM_EPISODES):
    print("=" * 5, f"Episode {episode + 1}/{HParams.NUM_EPISODES}", "=" * 5)
    # Initialize
    env = TrainingEnvironment()
    history.reset()

    # Train
    dataset_generator = LabelTokenDatasetGenerator(
        data_manager.get_sequences(), data_manager.item_to_tokens
    )
    dataset = dataset_generator(HParams.BATCH_SIZE)
    model, loss = train(model, encoder, dataset, HParams.N_NEGTIVES)

    # Get embedding dict
    recall_system = RecallSystem(model, encoder, data_manager)

    # Explore
    model, history, data_manager = explore(
        env, model, encoder, recall_system, history, data_manager, update_freq=50
    )
    print(f"Avg. score: {np.mean(env.get_score())}")

    # Save
    data_manager.save(Paths.USER_DATA_PLUS)
    history.update_init(data_manager.get_sequences())
