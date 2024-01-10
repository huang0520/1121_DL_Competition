import random

import pandas as pd
import tensorflow as tf


class DatasetGenerator:
    def __init__(self, user_path, item_path):
        df_user = pd.read_json(user_path, lines=True)
        df_item = pd.read_json(item_path, lines=True)

        self.users, self.items = df_user.explode("history").to_numpy().T
        self.users = self.users.tolist()
        self.items = self.items.tolist()
        self.pos_items = df_user["history"].to_list()
        self.all_items = df_item["item_id"].to_list()

    def __get_neg_item(self, user_id):
        neg_items = list(set(self.all_items) - set(self.pos_items[user_id]))
        return random.choice(neg_items)

    def add_item(self, user_id, item_id):
        self.users.append(user_id)
        self.items.append(item_id)
        self.pos_items[user_id].append(item_id)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.__get_neg_item(self.users[idx])

    def __get_curr_generator(self):
        """
        Returns a generator that yields elements from the current dataset.
        """
        yield from (self[i] for i in range(len(self)))

    def generate(self, batch_size):
        dataset: tf.data.Dataset = tf.data.Dataset.from_generator(
            self.__get_curr_generator,
            output_types=(tf.int32, tf.int32, tf.int32),
        )
        dataset = dataset.shuffle(buffer_size=batch_size * 10)
        dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


if __name__ == "__main__":
    dataset = DatasetGenerator("./dataset/user_data.json", "./dataset/item_data.json")

    for data in dataset.generate(32).take(1):
        print(data)
