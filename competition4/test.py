# %%
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation.environment import TestingEnvironment, TrainingEnvironment
from icecream import ic
from src.dataset import DataManager, History, LabelTokenDatasetGenerator
from src.layer import SparseFMLayer
from src.recommender import FMEmbeding, SparseFM, SparseHotEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
item_to_tokens = pd.read_pickle("./dataset/item_token.pkl")

# %%
a, b = item_to_tokens.loc[[0, 222, 333]].to_numpy().T
a
# %%
a = pd.read_pickle("./dataset/user_data_plus.pkl")
a["history"].map(len).value_counts()

# %%
data_manager = DataManager(
    "./dataset/user_data.json", "./dataset/item_data.json", "./dataset/item_token.pkl"
)
dataset_generator = LabelTokenDatasetGenerator(
    data_manager.get_sequences(), data_manager.item_to_tokens
)

encoder = SparseHotEncoder(2000, 209527, 49408)
model = SparseFM(8, 0.1, 0.1)
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
)

dataset = dataset_generator(8)
for data in dataset:
    user_ids, item_ids, title, desc, labels = data

    features = encoder((user_ids, item_ids, title, desc))
    loss = model.train_step((features, labels))
    break

fm_embedding = FMEmbeding(encoder, 2000, 209527, 49408, model)

df_seq = dataset_generator.df_seq
user_embedding_dict = fm_embedding.get_user_embedding(df_seq)
item_embedding_dict = fm_embedding.get_item_embedding(df_seq)


count = 5
for k, v in user_embedding_dict.items():
    ic(k, v)
    count -= 1
    if count == 0:
        break

# %%
# user_embedder = tf.keras.layers.Embedding(
#     len(df_user),
#     64,
#     embeddings_initializer="random_normal",
#     embeddings_regularizer=tf.keras.regularizers.L2(0.1),
#     sparse=True,
# )
# item_embedder = tf.keras.layers.Embedding(
#     len(item_to_tokens),
#     64,
#     embeddings_initializer="random_normal",
#     embeddings_regularizer=tf.keras.regularizers.L2(0.1),
#     sparse=True,
# )
# token_embedder = tf.keras.layers.Embedding(
#     30522,
#     64,
#     embeddings_initializer="random_normal",
#     embeddings_regularizer=tf.keras.regularizers.L2(0.1),
#     sparse=True,
# )
