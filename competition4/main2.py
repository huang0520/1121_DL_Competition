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
from src.dataset import DataManager, History, LabelTokenDatasetGenerator
from src.loss import PairwiseRankingLoss
from src.recommender import FunkSVD, NeuMF
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
    EMBED_SIZE: int = 384
    BATCH_SIZE: int = 128
    LEARNING_RATE: float = 0.00005
    WEIGHT_DECAY: float = 0.0004
    RANDOM_STATE: int = 42
    NUM_EPOCHS: int = 5
    NUM_EPOSIDES: int = 100
    N_NEGTIVES: int = 8


@dataclass
class Paths:
    USER_DATA: Path = Path("./dataset/user_data.json")
    ITEM_DATA: Path = Path("./dataset/item_data.json")
    OUTPUT: Path = Path("./output/output.csv")
    CHECKPOINT_DIR: Path = Path("./checkpoint")
    TOKEN_PATH: Path = Path("./dataset/item_token.pkl")
    EMBEDDING_PATH: Path = Path("./dataset/item_to_embedding.pkl")
    USER_DATA_PLUS: Path = Path("./dataset/user_data_plus.pkl")
    SIMILARITY_PATH: Path = Path("./dataset/similarity_items.pkl")


random.seed(HParams.RANDOM_STATE)


# %%
# Training pipeline
def train(model, dataset, n_neg=14):
    epoch_loss = []

    pbar = trange(HParams.NUM_EPOCHS, desc="Training", ncols=0)
    for _ in pbar:
        batch_loss = []

        for user_ids, pos_item_ids, text_embeddings, labels in dataset:
            losses = []
            batch_size = len(user_ids)

            # Train positive samples
            loss = model.train_step((
                user_ids,
                pos_item_ids,
                labels,
            ))
            losses.append(loss)

            # Train negative samples
            neg_item_ids = tf.random.uniform(
                shape=(n_neg, batch_size),
                minval=0,
                maxval=ConstParams.N_ITEMS,
                dtype=tf.int32,
            )
            for _neg_item_id in neg_item_ids:
                loss = model.train_step((
                    tf.constant(user_ids),
                    tf.constant(_neg_item_id),
                    tf.zeros(batch_size),
                ))
                losses.append(loss)

            batch_loss.append(tf.reduce_mean(losses).numpy())
        epoch_loss.append(np.mean(batch_loss))
        pbar.set_postfix({"loss": epoch_loss[-1]})
    pbar.set_postfix({"loss": np.mean(epoch_loss)}, refresh=True)

    return model, np.mean(epoch_loss)


def update(model, data_manager, user_id, clicked_id):
    # Positive samples
    model.train_step((
        tf.convert_to_tensor([[user_id]]),
        tf.convert_to_tensor([[clicked_id]]),
        tf.convert_to_tensor([data_manager.item_to_embeddings[clicked_id]]),
        tf.ones(1),
    ))

    # Negative samples
    neg_item_ids = tf.random.uniform(
        shape=(HParams.N_NEGTIVES,),
        minval=0,
        maxval=ConstParams.N_ITEMS,
        dtype=tf.int32,
    )
    neg_embeddings = tf.convert_to_tensor(
        data_manager.item_to_embeddings[neg_item_ids].to_list()
    )
    model.train_step((
        tf.repeat(user_id, HParams.N_NEGTIVES),
        neg_item_ids,
        neg_embeddings,
        tf.zeros(HParams.N_NEGTIVES),
    ))

    return model


def get_content_topk(data_manager, clicked_id, k=2, choose_self=True):
    n = 0 if choose_self else 1
    if clicked_id in data_manager.similarity_items:
        return data_manager.similarity_items[clicked_id][n : n + k]

    item_to_embeddings = data_manager.item_to_embeddings
    scores = tf.losses.CosineSimilarity(reduction="none")(
        tf.repeat(
            tf.constant(item_to_embeddings.iloc[clicked_id], shape=(1, 384)),
            len(item_to_embeddings),
            axis=0,
        ),
        tf.constant(item_to_embeddings.to_list()),
    )

    sort_items = tf.argsort(scores).numpy().tolist()

    data_manager.add_top100_items(clicked_id, sort_items)
    return sort_items[n : n + k]


# Explore pipeline
def explore(env, model, data_manager, slate_size=5):
    hit_count = 0
    pbar = tqdm(desc="Explore")
    while env.has_next_state():
        user_id = env.get_state()
        random_pos_item_id = random.choice(tuple(data_manager.pos_item_sets[user_id]))
        coll_slate = model.get_topk(user_id, data_manager, 3)
        cont_slate = get_content_topk(data_manager, random_pos_item_id, 2, False)
        slate = np.unique(coll_slate + cont_slate).tolist()
        while len(slate) < slate_size:
            slate = np.unique(
                slate
                + random.sample(model.get_topk(user_id, 10), slate_size - len(slate))
            ).tolist()
        clicked_id, _ = env.get_response(slate)

        if clicked_id != -1:
            hit_count += 1
            data_manager.add(user_id, clicked_id)
            model = update(model, data_manager, user_id, clicked_id)

        pbar.update(1)
        pbar.set_postfix({"#click": hit_count})

    return model, hit_count


# Simulate pipeline
def simulate_train(model, data_manager, checkpoint_dir, transfer=False):
    checkpoint = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    best_manager = tf.train.CheckpointManager(
        checkpoint, checkpoint_dir / "best", max_to_keep=1
    )
    best_score = 0

    if transfer:
        checkpoint.restore(ckpt_manager.latest_checkpoint)
        print(f"Model restored from {ckpt_manager.latest_checkpoint}.")

    for i in range(HParams.NUM_EPOSIDES):
        print("=" * 5 + f" Eposide {i + 1}/{HParams.NUM_EPOSIDES} " + "=" * 5)

        # Initialize
        env = TrainingEnvironment()
        dataset_generator = LabelTokenDatasetGenerator(
            data_manager.get_sequences(), data_manager.item_to_embeddings
        )

        # Train
        dataset = dataset_generator(HParams.BATCH_SIZE)
        model, _ = train(model, dataset, data_manager, HParams.N_NEGTIVES)

        # Explore and update
        model, _ = explore(env, model, data_manager, ConstParams.SLATE_SIZE)
        score = np.mean(env.get_score())
        print(f"Avg. Score: {score:.6f}")

        # Save
        ckpt_manager.save()
        data_manager.save(Paths.USER_DATA_PLUS, Paths.SIMILARITY_PATH)

        # Save best model
        if score > best_score:
            best_score = score
            best_manager.save()
            print(f"Best model saved at {best_manager.latest_checkpoint}.")

    return model


def test(model, checkpoint_dir, data_manager):
    test_env = TestingEnvironment()
    scores = []

    # Repeat the testing process for 5 times
    for epoch in range(ConstParams.TEST_EPISODES):
        # [TODO] Load your model weights here (in the beginning of each testing episode)
        # [TODO] Code for loading your model weights...
        print(f"Model restored from {tf.train.latest_checkpoint(checkpoint_dir)}.")
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        history = History(Paths.USER_DATA)
        clicked_count = 0

        # Start the testing process
        with tqdm(desc="Testing") as pbar:
            # Run as long as there exist some active users
            while test_env.has_next_state():
                # Get the current user id
                cur_user = test_env.get_state()

                # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
                # [TODO] Code for generating the recommended slate...
                # latest_item_id = history.get(cur_user)[-1]
                random_pos_item_id = random.choice(
                    np.unique(history.get(cur_user)).tolist()
                )
                coll_slate = model.get_topk(cur_user, 3)
                cont_slate = get_content_topk(
                    data_manager, random_pos_item_id, 2, False
                )
                slate = np.unique(coll_slate + cont_slate).tolist()
                while len(slate) < ConstParams.SLATE_SIZE:
                    slate = np.unique(
                        slate
                        + random.sample(
                            model.get_topk(cur_user, 10),
                            ConstParams.SLATE_SIZE - len(slate),
                        )
                    ).tolist()

                # Get the response of the slate from the environment
                clicked_id, _in_environment = test_env.get_response(slate)

                # [TODO] Update your model here (optional)
                # [TODO] You can update your model at each step, or perform a batched update after some interval
                # [TODO] Code for updating your model...
                if clicked_id != -1:
                    clicked_count += 1
                    history.add(cur_user, clicked_id)
                    model = update(model, data_manager, cur_user, clicked_id)
                    pbar.set_postfix({"#click": clicked_count})

                # Update the progress indicator
                pbar.update(1)

        # Record the score of this testing episode
        scores.append(test_env.get_score())

        # Reset the testing environment
        test_env.reset()

        # [TODO] Delete or reset your model weights here (in the end of each testing episode)
        # [TODO] Code for deleting your model weights...
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        history.reset()

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
    data_manager = DataManager(
        Paths.USER_DATA, Paths.ITEM_DATA, Paths.TOKEN_PATH, Paths.EMBEDDING_PATH
    )
    data_manager.load(Paths.USER_DATA_PLUS, Paths.SIMILARITY_PATH)

    model = FunkSVD(
        HParams.EMBED_SIZE,
        ConstParams.N_TRAIN_USERS,
        ConstParams.N_ITEMS,
        l2_lambda=0.005,
    )
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=0.0001, weight_decay=0.0004, use_ema=True
        ),
        loss=tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True, from_logits=True, label_smoothing=0.15
        ),
    )

    model = simulate_train(
        model, data_manager, Paths.CHECKPOINT_DIR / "FunkSVD", transfer=False
    )

    test(model, Paths.CHECKPOINT_DIR / "FunkSVD" / "best", data_manager)

    data_manager.save(Paths.USER_DATA_PLUS, Paths.SIMILARITY_PATH)


if __name__ == "__main__":
    main()
