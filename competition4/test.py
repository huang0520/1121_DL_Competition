# %%
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from src.layer import FMLayer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# %%
df_user = pd.read_json("./dataset/user_data.json", lines=True)
item_to_tokens = pd.read_pickle("./dataset/item_token.pkl")
seq = df_user.explode("history", ignore_index=True)
seq["token"] = seq["history"].map(item_to_tokens)

seq.head()

# %%
user_id = tf.keras.Input(shape=(1,), dtype=tf.int32)
item_id = tf.keras.Input(shape=(1,), dtype=tf.int32)
tokens = tf.keras.Input(shape=(None,), dtype=tf.int32, ragged=True)

user_encoder = tf.keras.layers.CategoryEncoding(
    len(df_user), output_mode="one_hot", sparse=True
)
item_encoder = tf.keras.layers.CategoryEncoding(
    len(item_to_tokens), output_mode="one_hot", sparse=True
)
token_encoder = tf.keras.layers.CategoryEncoding(
    30522, output_mode="multi_hot", sparse=True
)
user_embedder = tf.keras.layers.Embedding(
    len(df_user),
    64,
    embeddings_initializer="random_normal",
    embeddings_regularizer=tf.keras.regularizers.L2(0.1),
    sparse=True,
)
item_embedder = tf.keras.layers.Embedding(
    len(item_to_tokens),
    64,
    embeddings_initializer="random_normal",
    embeddings_regularizer=tf.keras.regularizers.L2(0.1),
    sparse=True,
)
token_embedder = tf.keras.layers.Embedding(
    30522,
    64,
    embeddings_initializer="random_normal",
    embeddings_regularizer=tf.keras.regularizers.L2(0.1),
    sparse=True,
)

user_field = user_encoder(user_id)
item_field = item_encoder(item_id)
token_field = token_encoder(tokens)

# user_embedding = user_embedder(user_field)
# item_embedding = item_embedder(item_field)
# token_embedding = token_embedder(token_field)

# output = tf.keras.layers.Concatenate()([
#     user_embedding,
#     item_embedding,
#     token_embedding,
# ])

output = tf.keras.layers.Concatenate()([user_field, item_field, token_field])

input_encoder = tf.keras.models.Model(inputs=(user_id, item_id, tokens), outputs=output)

# %%
dataset = (
    tf.data.Dataset.from_tensor_slices((
        user_encoder(seq["user_id"].to_numpy(dtype=int)),
        item_encoder(seq["history"].to_numpy(dtype=int)),
        token_encoder(tf.ragged.constant(seq["token"])),
    ))
    .batch(8)
    .prefetch(tf.data.AUTOTUNE)
)

# %%
# fm = FMLayer(64, 0.1, 0.1)

# for batch in dataset.take(1):
#     inputs = input_encoder(batch)
#     print(fm(inputs))

# %%
a = tf.constant([[1, 2]])
b = tf.constant([[1, 2]])
print(a.shape, b.shape)

tf.matmul(a, b, transpose_b=True)

# %%
tf.reduce_sum(a * b, axis=1)
