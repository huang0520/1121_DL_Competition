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
from src.dataset import DatasetGenerator, History
from src.loss import PairwiseRankingLoss
from src.recommender import NeuMF
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


@dataclass
class HParams:
    EMBED_SIZE: int = 128
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.00005
    WEIGHT_DECAY: float = 0.0004
    RANDOM_STATE: int = 42
    NUM_EPOCHS: int = 10
    NUM_EPOSIDES: int = 100
    NUMS_HIDDENS: tuple[int] = (64, 32, 16, 8)


@dataclass
class Paths:
    USER_DATA: Path = Path("./dataset/user_data.json")
    ITEM_DATA: Path = Path("./dataset/item_data.json")
    OUTPUT: Path = Path("./output/output.csv")
    CHECKPOINT_DIR: Path = Path("./checkpoint")


random.seed(HParams.RANDOM_STATE)


# %%
# Training pipeline
def train(model, dataset):
    loss_record = []

    pbar = trange(HParams.NUM_EPOCHS, desc="Training", ncols=0)
    for _ in pbar:
        losses = [model.train_step(data) for data in dataset]
        loss_record.append(tf.reduce_mean(losses).numpy())
        pbar.set_postfix({"loss": loss_record[-1]})
    pbar.set_postfix({"loss": np.mean(loss_record)}, refresh=True)

    return model, loss_record


def update(model, history, user_id, slate, clicked_id):
    neg_item_ids = slate[slate != clicked_id]

    if clicked_id != -1:
        history.add(user_id, clicked_id)
        pos_item_ids = tf.convert_to_tensor([clicked_id] * len(neg_item_ids))

    elif clicked_id == -1:
        pos_item_tuple = tuple(history.get(user_id))
        if len(pos_item_tuple) < len(neg_item_ids):
            pos_item_ids = pos_item_tuple + tuple(
                random.sample(pos_item_tuple, len(neg_item_ids) - len(pos_item_tuple))
            )

        elif len(pos_item_tuple) >= len(neg_item_ids):
            pos_item_ids = random.sample(pos_item_tuple, len(neg_item_ids))

    user_ids = tf.convert_to_tensor([user_id] * len(neg_item_ids))
    pos_item_ids = tf.convert_to_tensor(pos_item_ids)
    neg_item_ids = tf.convert_to_tensor(neg_item_ids)
    loss = model.train_step((user_ids, pos_item_ids, neg_item_ids))

    return model, loss.numpy()


# Explore pipeline
def explore_with_update(env, model, history, slate_size=5):
    hit_count = 0
    losses = []

    pbar = tqdm(desc="Explore & Update")
    while env.has_next_state():
        user_id = env.get_state()
        slate = model.get_topk(user_id, slate_size)
        clicked_id, _ = env.get_response(slate)

        model, loss = update(model, history, user_id, slate, clicked_id)
        hit_count += 1 if clicked_id != -1 else 0
        losses.append(loss)

        pbar.update(1)
        pbar.set_postfix({"#click": hit_count, "loss": loss})
    pbar.set_postfix({"#click": hit_count, "loss": np.mean(losses)}, refresh=True)

    return model, hit_count, np.mean(losses)


# Simulate pipeline
def simulate_train(model, checkpoint_dir, transfer=False):
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=1)
    if transfer:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"Model restored from {ckpt_manager.latest_checkpoint}.")

    env = TrainingEnvironment()
    history = History(Paths.USER_DATA)

    # Pre-train
    if not transfer:
        print("=" * 5 + " Pre-train " + "=" * 5)
        dataset = DatasetGenerator(Paths.USER_DATA, Paths.ITEM_DATA).generate(
            HParams.BATCH_SIZE
        )
        model, _ = train(model, dataset)
        del dataset

    for i in range(HParams.NUM_EPOSIDES):
        print("=" * 5 + f" Eposide {i + 1}/{HParams.NUM_EPOSIDES} " + "=" * 5)

        # Explore and update
        env.reset()
        history.reset()
        model, _, _ = explore_with_update(env, model, history, ConstParams.SLATE_SIZE)
        print(f"Average Score: {np.mean(env.get_score()):.6f}")

        # Save checkpoint
        ckpt_manager.save()
    return model


def test(model, dataset_generator, checkpoint_dir):
    test_env = TestingEnvironment()
    scores = []

    # The item_ids here is for the random recommender
    list(range(ConstParams.N_ITEMS))

    # Repeat the testing process for 5 times
    for epoch in range(ConstParams.TEST_EPISODES):
        # [TODO] Load your model weights here (in the beginning of each testing episode)
        # [TODO] Code for loading your model weights...
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        # Start the testing process
        with tqdm(desc="Testing") as pbar:
            # Run as long as there exist some active users
            while test_env.has_next_state():
                # Get the current user id
                cur_user = test_env.get_state()

                # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
                # [TODO] Code for generating the recommended slate...
                # Here we provide a simple random implementation
                slate = model.get_topk(cur_user, 5)

                # Get the response of the slate from the environment
                clicked_id, _in_environment = test_env.get_response(slate)

                # [TODO] Update your model here (optional)
                # [TODO] You can update your model at each step, or perform a batched update after some interval
                # [TODO] Code for updating your model...
                model = update(model, dataset_generator, cur_user, slate, clicked_id)

                # Update the progress indicator
                pbar.update(1)

        # Record the score of this testing episode
        scores.append(test_env.get_score())

        # Reset the testing environment
        test_env.reset()

        # [TODO] Delete or reset your model weights here (in the end of each testing episode)
        # [TODO] Code for deleting your model weights...
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Calculate the average scores
    avg_scores = [np.average(score) for score in zip(*scores)]

    # Generate a DataFrame to output the result in a .csv file
    df_result = pd.DataFrame(
        [[user_id, avg_score] for user_id, avg_score in enumerate(avg_scores)],
        columns=["user_id", "avg_score"],
    )
    df_result.to_csv(Paths.OUTPUT, index=False)

    print(f"Average Score: {np.mean(avg_scores):.6f}")


def main():
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
        loss=PairwiseRankingLoss(margin=1),
    )

    model = simulate_train(model, Paths.CHECKPOINT_DIR / "NeuMF", transfer=False)

    # test(model, dataset_generator, Paths.CHECKPOINT_DIR / "NeuMF")


# # %% Testing
# # Initialize the testing environment
# test_env = TestingEnvironment()
# scores = []

# # The item_ids here is for the random recommender
# item_ids = list(range(ConstParams.N_ITEMS))

# # Repeat the testing process for 5 times
# for epoch in range(ConstParams.TEST_EPISODES):
#     # [TODO] Load your model weights here (in the beginning of each testing episode)
#     # [TODO] Code for loading your model weights...

#     # Start the testing process
#     with tqdm(desc="Testing") as pbar:
#         # Run as long as there exist some active users
#         while test_env.has_next_state():
#             # Get the current user id
#             cur_user = test_env.get_state()

#             # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
#             # [TODO] Code for generating the recommended slate...
#             # Here we provide a simple random implementation
#             slate = model.get_topn(cur_user, 5)

#             # Get the response of the slate from the environment
#             clicked_id, in_environment = test_env.get_response(slate)

#             # [TODO] Update your model here (optional)
#             # [TODO] You can update your model at each step, or perform a batched update after some interval
#             # [TODO] Code for updating your model...

#             # Update the progress indicator
#             pbar.update(1)

#     # Record the score of this testing episode
#     scores.append(test_env.get_score())

#     # Reset the testing environment
#     test_env.reset()

#     # [TODO] Delete or reset your model weights here (in the end of each testing episode)
#     # [TODO] Code for deleting your model weights...

# # Calculate the average scores
# avg_scores = [np.average(score) for score in zip(*scores)]

# # Generate a DataFrame to output the result in a .csv file
# df_result = pd.DataFrame(
#     [[user_id, avg_score] for user_id, avg_score in enumerate(avg_scores)],
#     columns=["user_id", "avg_score"],
# )
# df_result.to_csv(Paths.OUTPUT, index=False)
# df_result

# %%
if __name__ == "__main__":
    main()
