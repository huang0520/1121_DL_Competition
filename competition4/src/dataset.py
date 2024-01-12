import random
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.num_users = len(df_user)
        self.num_items = len(df_item)

        self.pairs = set(df_user.explode("history").itertuples(index=False, name=None))
        self.all_item_set = set(df_item["item_id"])
        self.pos_item_sets = df_user["history"].apply(set).to_list()
        self.neg_item_sets = [
            self.all_item_set - pos_item_list for pos_item_list in self.pos_item_sets
        ]

        self.neg_item_tuples = [
            tuple(neg_item_set) for neg_item_set in self.neg_item_sets
        ]
        self.modified = False

    def __len__(self):
        return len(self.pairs)

    def add_item(self, user_id, item_id):
        if (user_id, item_id) not in self.pairs:
            self.pairs.add((user_id, item_id))
            self.pos_item_sets[user_id].add(item_id)
            self.neg_item_sets[user_id].discard(item_id)
            self.modified = True

    def __get_curr_generator(self):
        """
        Returns a generator that yields elements from the current dataset.
        """
        users, pos_items = zip(*self.pairs)
        if self.modified:
            self.neg_item_tuples = [
                tuple(neg_item_set) for neg_item_set in self.neg_item_sets
            ]
            self.modified = False

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


class MultiHotDatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.num_users = len(df_user)
        self.num_items = len(df_item)

        self.pos_item_sets = df_user["history"].apply(set).to_list()

    def test(self):
        print(
            tf.keras.layers.CategoryEncoding(self.num_items)(
                tuple(self.pos_item_sets[0])
            )
        )


class OneHotDatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.all_item_set = set(df_item["item_id"])
        self.pos_item_sets = df_user["history"].apply(set).to_list()
        self.neg_item_sets = [set() for _ in range(len(df_user))]

    def __add_item(self, user_id, item_id, label):
        if label == 1:
            self.pos_item_sets[user_id].add(item_id)
            self.neg_item_sets[user_id].discard(item_id)
        elif label == 0 and item_id not in self.pos_item_sets[user_id]:
            self.neg_item_sets[user_id].add(item_id)

    def add_items(self, user_ids, item_ids, labels):
        for user_id, item_id, label in zip(user_ids, item_ids, labels):
            self.__add_item(user_id, item_id, label)

    def __generate_one_hot(self):
        data = []
        for user_id, (pos_item_set, neg_item_set) in enumerate(
            zip(
                self.pos_item_sets,
                self.neg_item_sets,
            )
        ):
            data.extend((user_id, item_id, 1) for item_id in pos_item_set)
            data.extend((user_id, item_id, 0) for item_id in neg_item_set)

        data = pd.DataFrame(data, columns=["user_id", "item_id", "label"])
        return pd.get_dummies(data, columns=["user_id", "item_id"], dtype=float)

    def generate(self, batch_size):
        data = self.__generate_one_hot()
        x = data.drop(columns=["label"])
        y = data["label"]

        return (
            tf.data.Dataset.from_tensor_slices((x, y))
            .shuffle(buffer_size=batch_size * 10)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )


if __name__ == "__main__":
    dataset_generator = MultiHotDatasetGenerator(
        "./dataset/user_data.json", "./dataset/item_data.json"
    )
    dataset_generator.test()
