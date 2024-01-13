import random
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf


class History:
    def __init__(self, user_path):
        df_user = pd.read_json(user_path, lines=True)
        df_user["history"] = df_user["history"].apply(set)
        self.init_histories = df_user
        self.curr_histories = self.init_histories.copy()

    def reset(self):
        self.curr_histories = self.init_histories.copy()

    def add(self, user_id, item_id):
        self.curr_histories.loc[user_id, "history"].add(item_id)

    def get(self, user_id):
        return self.curr_histories.loc[user_id, "history"]

    def get_sequences(self):
        return self.curr_histories.explode("history").itertuples(index=False, name=None)


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


class LabeldDatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.num_users = len(df_user)
        self.num_items = len(df_item)

        self.df_seq = df_user.explode("history")
        self.df_seq["label"] = 1

    def generate(self, batch_size):
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            lambda: tuple(self.df_seq.itertuples(index=False, name=None)),
            output_types=(tf.int32, tf.int32, tf.int32),
        )
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


class SparseDatasetGenerator:
    def __init__(self, user_path, token_path):
        df_user = pd.read_json(user_path, lines=True)
        df_token = pd.read_pickle(token_path)

        self.user_item_pairs = tuple(
            df_user.explode("history").itertuples(index=False, name=None)
        )

        self.item_token_pairs = df_token

        self.user_encoder = tf.keras.layers.CategoryEncoding(
            len(df_user), output_mode="one_hot"
        )
        self.item_encoder = tf.keras.layers.CategoryEncoding(
            len(df_token), output_mode="one_hot"
        )
        self.token_encoder = tf.keras.layers.CategoryEncoding(30522)

    def encode(self, user_id, item_id, label=1):
        user_field = self.user_encoder(user_id)
        item_field = self.item_encoder(item_id)
        token_field = self.token_encoder(self.item_token_pairs[item_id])

        return tf.concat([user_field, item_field, token_field], axis=0), tf.constant(
            label, dtype=tf.float32
        )

    def __generator(self):
        yield from (self.encode(*pair) for pair in self.user_item_pairs)

    def generate(self, batch_size):
        dataset = tf.data.Dataset.from_generator(
            self.__generator,
            output_types=(tf.float32, tf.float32),
        )
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(
            batch_size, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


if __name__ == "__main__":
    dataset_generator = SparseDatasetGenerator(
        "./dataset/user_data.json", "./dataset/item_token.pkl"
    )

    dataset = dataset_generator.generate(16)

    for x, y in dataset.take(1):
        print(x.shape)
        print(y.shape)
