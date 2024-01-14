import random
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf


class History:
    def __init__(self, user_path):
        df_user = pd.read_json(user_path, lines=True)
        self.init_histories = df_user.set_index("user_id")["history"]
        self.curr_histories = self.init_histories.copy()

    def reset(self):
        self.curr_histories = self.init_histories.copy()

    def add(self, user_id, item_id):
        self.curr_histories.loc[user_id].append(item_id)

    def get(self, user_id):
        return self.curr_histories.loc[user_id]

    def update_init(self, sequence):
        self.init_histories = (
            pd.DataFrame(sequence, columns=["user_id", "history"])
            .groupby("user_id")["history"]
            .apply(list)
        )


class DatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.num_users = len(df_user)
        self.num_items = len(df_item)

        self.pairs = set(df_user.explode("history").itertuples(index=False, name=None))

        all_item_set = set(df_item["item_id"])
        pos_item_sets = df_user["history"].apply(set).to_list()
        neg_item_sets = [all_item_set - pos_item_set for pos_item_set in pos_item_sets]
        self.neg_item_tuples = [tuple(neg_item_set) for neg_item_set in neg_item_sets]

    def __get_curr_generator(self):
        """
        Returns a generator that yields elements from the current dataset.
        """
        users, pos_items = zip(*self.pairs)
        neg_items = (random.choice(self.neg_item_tuples[user_id]) for user_id in users)
        yield from zip(users, pos_items, neg_items)

    def generate(self, batch_size):
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            self.__get_curr_generator,
            output_types=(tf.int32, tf.int32, tf.int32),
        )
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class DataManager:
    def __init__(self, user_path, item_path, token_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.num_users = len(df_user)
        self.num_items = len(df_item)

        self.pairs = set(df_user.explode("history").itertuples(index=False, name=None))
        self.item_to_tokens = pd.read_pickle(token_path)

        self.all_item_set = set(df_item["item_id"])
        self.pos_item_sets = df_user["history"].apply(set).to_list()
        self.neg_item_sets = [
            self.all_item_set - pos_item_set for pos_item_set in self.pos_item_sets
        ]

    def add(self, user_id, item_id):
        self.pairs.add((user_id, item_id))
        self.pos_item_sets[user_id].add(item_id)
        self.neg_item_sets[user_id].discard(item_id)

    def remove(self, user_id, item_id):
        self.pairs.remove((user_id, item_id))
        self.pos_item_sets[user_id].discard(item_id)
        self.neg_item_sets[user_id].add(item_id)

    def get_sequences(self):
        return list(self.pairs)

    def save(self, save_path):
        df_user = pd.DataFrame({
            "user_id": range(self.num_users),
            "history": self.pos_item_sets,
        })
        df_user.to_pickle(save_path)

    def load(self, load_path):
        df_user = pd.read_pickle(load_path)
        self.pairs = set(df_user.explode("history").itertuples(index=False, name=None))
        self.pos_item_sets = df_user["history"].to_list()
        self.neg_item_sets = [
            self.all_item_set - pos_item_set for pos_item_set in self.pos_item_sets
        ]


class LabelTokenDatasetGenerator:
    def __init__(self, user_item_pairs, item_to_tokens):
        self.df_seq = pd.DataFrame(user_item_pairs, columns=["user_id", "item_id"])
        self.df_seq["title"] = self.df_seq["item_id"].map(item_to_tokens["headline"])
        self.df_seq["desc"] = self.df_seq["item_id"].map(
            item_to_tokens["short_description"]
        )
        self.df_seq["label"] = 1

    def __call__(self, batch_size):
        dataset: tf.data.Dataset = tf.data.Dataset.from_tensor_slices((
            tf.convert_to_tensor(self.df_seq["user_id"].to_numpy(dtype=int)),
            tf.convert_to_tensor(self.df_seq["item_id"].to_numpy(dtype=int)),
            tf.ragged.constant(self.df_seq["title"].to_numpy()),
            tf.ragged.constant(self.df_seq["desc"].to_numpy()),
            tf.convert_to_tensor(self.df_seq["label"].to_numpy(dtype=int)),
        ))
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
