# %%
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from evaluation.environment import TestingEnvironment, TrainingEnvironment
from icecream import ic
from src.dataset import DataManager, History, LabelTokenDatasetGenerator
from src.layer import SparseFMLayer
from src.recommender import FMEmbeding, SparseFM, SparseHotEncoder
from tqdm.auto import tqdm, trange

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# %%
data_manager = DataManager(
    "./dataset/user_data.json",
    "./dataset/item_data.json",
    "./dataset/item_token.pkl",
    "./dataset/item_to_embedding.pkl",
)
data_manager.load("./dataset/user_plus.pkl", "./dataset/similarity.pkl")

# %%
df_seq = pd.DataFrame(data_manager.get_sequences(), columns=["user_id", "item_id"])
item_to_embeddings: pd.DataFrame = data_manager.item_to_embedding
item_to_embeddings = item_to_embeddings.apply(tuple, axis=1)
df_seq["embeddings"] = df_seq["item_id"].map(item_to_embeddings)
print(tf.convert_to_tensor(df_seq["embeddings"].to_list(), tf.float32))
