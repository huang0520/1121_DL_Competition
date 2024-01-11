import random
from time import time

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
        self.pos_item_lists = df_user["history"].to_list()
        self.neg_item_lists = [
            list(self.all_item_set - set(pos_item_list))
            for pos_item_list in self.pos_item_lists
        ]

    def __len__(self):
        return len(self.pairs)

    def __unzip_pairs(self, pairs):
        return tuple(zip(*pairs))

    def __add_item(self, user_id, item_id):
        if (user_id, item_id) not in self.pairs:
            self.pairs.add((user_id, item_id))
            self.pos_item_lists[user_id].append(item_id)
            self.neg_item_lists[user_id].remove(item_id)
            return True
        return False

    def add_items(self, user_ids, item_ids):
        add_check_list = [
            self.__add_item(user_id, item_id)
            for user_id, item_id in zip(user_ids, item_ids)
        ]

        return sum(add_check_list)

    def __get_curr_generator(self):
        """
        Returns a generator that yields elements from the current dataset.
        """
        users, items = self.__unzip_pairs(self.pairs)
        neg_items = [random.choice(self.neg_item_lists[user_id]) for user_id in users]
        yield from zip(users, items, neg_items)

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


if __name__ == "__main__":
    curr_time = time()
    dataset_generator = DatasetGenerator(
        "./dataset/user_data.json", "./dataset/item_data.json"
    )
    print(f"Create generator: {time() - curr_time:.2f}s")
    curr_time = time()

    dataset = dataset_generator.generate(16)
    print(f"Create dataset: {time() - curr_time:.2f}s")
    curr_time = time()

    for data in dataset.take(1):
        print((data[0][0].numpy(), data[1][0].numpy()) in dataset_generator.pairs)
    print(f"Take 1 from dataset: {time() - curr_time:.2f}s")

    print(f"#New items: {dataset_generator.add_items((0, 0, 1), (1, 1, 2))}")

    print(len(dataset_generator))
