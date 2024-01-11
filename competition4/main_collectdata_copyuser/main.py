# %%
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from evaluation.environment import TrainingEnvironment, TestingEnvironment

# %%
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

# %%
# Official hyperparameters for this competition (do not modify)
N_TRAIN_USERS = 1000
N_TEST_USERS = 2000
N_ITEMS = 209527
HORIZON = 2000
TEST_EPISODES = 5
SLATE_SIZE = 5

# %%
LEARNING_RATE = 2e-5
EMBEDDING_SIZE = 512
N_EPOCHS = 2500
TRAIN_EPISODES = 100
COLLABRATIVE_SLATE_SIZE = 5
CONTENT_SLATE_SIZE = 0


# %%
# Dataset paths
USER_DATA = os.path.join("../dataset", "user_data.json")
ITEM_DATA = os.path.join("../dataset", "item_data.json")

# Output file path
OUTPUT_PATH = os.path.join("output", "output_main.csv")


# %%
df_user = pd.read_json(USER_DATA, lines=True)
df_user

# %%
df_item = pd.read_json(ITEM_DATA, lines=True)
df_item.head()

# %%
# Process Data
uids = []
histories = []
ratings = []

for uid in df_user.user_id:
    for his in df_user.at[uid, "history"]:
        uids.append(uid)
        histories.append(his)
        ratings.append(1)

print(len(uids))
print(uids)

# %%
# Collabrative Model: funk-svd
class FunkSVDRecommender(tf.keras.Model):
    """
    Simplified Funk-SVD recommender model
    """

    def __init__(self, m_users: int, n_items: int, embedding_size: int, learning_rate: float):
        """
        Constructor of the model
        """
        super().__init__()
        self.m = m_users
        self.n = n_items
        self.k = embedding_size
        self.lr = learning_rate

        # user embeddings P
        self.P = tf.Variable(tf.keras.initializers.RandomNormal()(shape=(self.m, self.k)))

        # item embeddings Q
        self.Q = tf.Variable(tf.keras.initializers.RandomNormal()(shape=(self.n, self.k)))

        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_update = tf.optimizers.Adam(learning_rate=self.lr)

    @tf.function
    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        """
        Forward pass used in training and validating
        """
        # dot product the user and item embeddings corresponding to the observed interaction pairs to produce predictions
        y_pred = tf.reduce_sum(
            tf.gather(self.P, indices=user_ids) * tf.gather(self.Q, indices=item_ids), axis=1
        )

        return y_pred

    @tf.function
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute the MSE loss of the model
        """
        loss = tf.losses.mean_squared_error(y_true, y_pred)

        return loss

    @tf.function
    def train_step(self, uids: tf.Tensor, histories: tf.Tensor, ratings: tf.Tensor) -> tf.Tensor:
        """
        Train the model with one batch
        data: batched user-item interactions
        each record in data is in the format [UserID, MovieID, Rating, Timestamp]
        """
        print("train")

        print(f"uid:{uids} items{histories} y: {ratings}")

        # compute loss
        with tf.GradientTape() as tape:
            y_pred = self(uids, histories)
            loss = self.compute_loss(ratings, y_pred)

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def val_step(self, uids: tf.Tensor, histories: tf.Tensor, ratings: tf.Tensor) -> tf.Tensor:
        """
        Validate the model with one batch
        data: batched user-item interactions
        each record in data is in the format [UserID, MovieID, Rating, Timestamp]
        """
        user_ids = tf.cast(uids, dtype=tf.int32)
        item_ids = tf.cast(histories, dtype=tf.int32)
        y_true = tf.cast(ratings, dtype=tf.float32)

        # compute loss
        y_pred = self(user_ids, item_ids)
        loss = self.compute_loss(y_true, y_pred)

        return loss

    @tf.function
    def eval_predict_onestep(self, query: tf.Tensor) -> tf.Tensor:
        """
        Retrieve and return the MovieIDs of the 10 recommended movies given a query
        You should return a tf.Tensor with shape=(10,)
        query will be a tf.Tensor with shape=(2,) and dtype=tf.int64
        query[0] is the UserID of the query
        #### query[1] is the Timestamp of the query
        """
        # dot product the selected user and all item embeddings to produce predictions
        user_id = tf.cast(query, tf.int32)
        y_pred = tf.reduce_sum(tf.gather(self.P, user_id) * self.Q, axis=1)

        # select the top 10 items with highest scores in y_pred
        y_recommends = tf.math.top_k(y_pred, k=COLLABRATIVE_SLATE_SIZE).indices

        return y_recommends

    @tf.function
    def eval_update_onestep(
        self, uids: tf.Tensor, histories: tf.Tensor, ratings: tf.Tensor
    ) -> None:

        # compute loss
        y_pred = self(uids, histories)
        loss = self.compute_loss(ratings, y_pred)

        # compute loss
        with tf.GradientTape() as tape:
            y_pred = self(uids, histories)
            loss = self.compute_loss(ratings, y_pred)

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # update weights
        self.optimizer_update.apply_gradients(zip(gradients, self.trainable_variables))

model = FunkSVDRecommender(
    m_users=N_TEST_USERS,
    n_items=N_ITEMS,
    embedding_size=EMBEDDING_SIZE,
    learning_rate=LEARNING_RATE,
)
for epoch in range(1, N_EPOCHS + 1):
    train_loss = []
    val_loss = []
    print(f"Epoch {epoch}:")

    # training
    loss = model.train_step(uids, histories, ratings)
    train_loss.append(loss.numpy())

    # record losses
    avg_train_loss = np.mean(train_loss)

    # print losses
    print(f"Epoch {epoch} train_loss: {avg_train_loss:.4f}\n")
model.save("model_funk_svd")

# %%
# Initialize the training environment
train_env = TrainingEnvironment()

for _ in range(TRAIN_EPISODES):
    # Reset the training environment (this can be useful when you have finished one episode of simulation and do not want to re-initialize a new environment)
    train_env.reset()

    # Check if there exist any active users in the environment
    while train_env.has_next_state():
        # print(f'There is {"still some" if env_has_next_state else "no"} active users in the training environment.')

        # Get the current user ID
        user_id = train_env.get_state()
        print(f'The current user is user {user_id}.')

        # Get the response of recommending the slate to the current user
        slate = model.eval_predict_onestep([user_id])

        clicked_id, in_environment = train_env.get_response(slate)
        uids_ = []
        uids_test = []
        histories_ = []
        ratings_ = []
        for his in df_user.at[user_id, "history"]:
            uids_.append(user_id)
            histories_.append(his)
            ratings_.append(1)
            uids_.append(user_id + 1000)
            histories_.append(his)
            ratings_.append(1)

        for s in slate:
            uids_.append(user_id)
            histories_.append(s)
            uids_.append(user_id + 1000)
            histories_.append(s)
            if clicked_id == s:
                ratings_.append(1)
                ratings_.append(1)
                df_user.at[user_id, "history"].append(s)
                df_user.at[user_id + 1000, "history"].append(s)
            else:
                ratings_.append(-1)
                ratings_.append(-1)
        model.eval_update_onestep(uids_, histories_, ratings_)
        if clicked_id != -1:
            print(
                f'The click result of recommending {slate} to user {user_id} is {f"item {clicked_id}" if clicked_id != -1 else f"{clicked_id} (no click)"}.'
            )
            print(
                f'User {user_id} {"is still in" if in_environment else "leaves"} the environment.'
            )
            print(ratings_)

model.save("model_funk_svd")

# Get the normalized session length score of all users
train_score = train_env.get_score()
df_train_score = pd.DataFrame(
    [[user_id, score] for user_id, score in enumerate(train_score)],
    columns=["user_id", "avg_score"],
)
df_train_score

# %%
# Initialize the testing environment
test_env = TestingEnvironment()
scores = []

# The item_ids here is for the random recommender
# item_ids = [i for i in range(N_ITEMS)]

# Repeat the testing process for 5 times
for _ in range(TEST_EPISODES):
    # [TODO] Load your model weights here (in the beginning of each testing episode)
    # [TODO] Code for loading your model weights...
    # model.save('model_funk_svd_update')
    # model = tf.keras.models.load_model('model_funk_svd', compile=False)

    # Start the testing process
    with tqdm(desc="Testing") as pbar:
        # Run as long as there exist some active users
        while test_env.has_next_state():
            # Get the current user id
            cur_user = test_env.get_state()

            # [TODO] Employ your recommendation policy to generate a slate of 5 distinct items
            # [TODO] Code for generating the recommended slate...
            # Here we provide a simple random implementation
            # slate = random.sample(item_ids, k=SLATE_SIZE)
            # model = tf.keras.models.load_model('model_funk_svd_update')
            # print(cur_user)
            collab_slate = model.eval_predict_onestep([cur_user])
            # content_slate = get_recommendations(df_user[0])

            # Get the response of the slate from the environment
            clicked_id, in_environment = test_env.get_response(slate)

            # [TODO] Update your model here (optional)
            # [TODO] You can update your model at each step, or perform a batched update after some interval
            # [TODO] Code for updating your model...

            uids_ = []
            histories_ = []
            ratings_ = []
            for his in df_user.at[cur_user, "history"]:
                uids_.append(cur_user)
                histories_.append(his)
                ratings_.append(1)
            for s in slate:
                uids_.append(cur_user)
                histories_.append(s)
                if clicked_id == s:
                    ratings_.append(1)
                    df_user.at[uid, "history"].append(s)
                else:
                    ratings_.append(-1)

            model.eval_update_onestep(uids_, histories_, ratings_)
            # model.save('model_funk_svd_update')

            # Update the progress indicator
            pbar.update(1)

    # Record the score of this testing episode
    scores.append(test_env.get_score())

    # Reset the testing environment
    test_env.reset()

    # [TODO] Delete or reset your model weights here (in the end of each testing episode)
    # [TODO] Code for deleting your model weights...

# %%
# Calculate the average scores
avg_scores = [np.average(score) for score in zip(*scores)]

# Generate a DataFrame to output the result in a .csv file
df_result = pd.DataFrame(
    [[user_id, avg_score] for user_id, avg_score in enumerate(avg_scores)],
    columns=["user_id", "avg_score"],
)
df_result.to_csv(OUTPUT_PATH, index=False)
df_result


# %%
