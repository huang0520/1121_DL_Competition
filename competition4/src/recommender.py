import tensorflow as tf
from dataset import DatasetGenerator
from loss import PairwiseRankingLoss
from tensorflow import keras


class FunkSVDRecommender(tf.keras.Model):
    """
    Simplified Funk-SVD recommender model
    """

    def __init__(
        self, m_users: int, n_items: int, embedding_size: int, learning_rate: float
    ):
        """
        Constructor of the model
        """
        super().__init__()
        self.m = m_users
        self.n = n_items
        self.k = embedding_size
        self.lr = learning_rate

        # user embeddings P
        self.P = tf.Variable(
            tf.keras.initializers.RandomNormal()(shape=(self.m, self.k))
        )

        # item embeddings Q
        self.Q = tf.Variable(
            tf.keras.initializers.RandomNormal()(shape=(self.n, self.k))
        )

        # optimizer
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)

    @tf.function
    def call(self, user_ids: tf.Tensor, item_ids: tf.Tensor) -> tf.Tensor:
        """
        Forward pass used in training and validating
        """
        # dot product the user and item embeddings corresponding to the observed interaction pairs to produce predictions
        y_pred = tf.reduce_sum(
            tf.gather(self.P, indices=user_ids) * tf.gather(self.Q, indices=item_ids),
            axis=1,
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
    def train_step(self, data: tf.Tensor) -> tf.Tensor:
        # data: user_id, item_id, rating
        user_ids = tf.cast(data[:, 0], dtype=tf.int32)
        item_ids = tf.cast(data[:, 1], dtype=tf.int32)
        y_true = tf.cast(data[:, 2], dtype=tf.float32)

        # compute loss
        with tf.GradientTape() as tape:
            y_pred = self(user_ids, item_ids)
            loss = self.compute_loss(y_true, y_pred)

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    @tf.function
    def val_step(self, data: tf.Tensor) -> tf.Tensor:
        """
        Validate the model with one batch
        data: batched user-item interactions
        each record in data is in the format [UserID, MovieID, Rating, Timestamp]
        """
        user_ids = tf.cast(data[:, 0], dtype=tf.int32)
        item_ids = tf.cast(data[:, 1], dtype=tf.int32)
        y_true = tf.cast(data[:, 2], dtype=tf.float32)

        # compute loss
        y_pred = self(user_ids, item_ids)
        loss = self.compute_loss(y_true, y_pred)

        return loss

    def recommand_top5(self, query: tf.Tensor) -> tf.Tensor:
        """
        Predict the top 5 item_ids for each user_id in query
        query: user_id
        """
        # dot product the selected user and all item embeddings to produce predictions
        user_id = tf.cast(query[0], tf.int32)
        y_pred = tf.reduce_sum(tf.gather(self.P, user_id) * self.Q, axis=1)

        # select the top 10 items with highest scores in y_pred
        y_top_5 = tf.math.top_k(y_pred, k=5).indices

        return y_top_5


class NeuMF(tf.keras.Model):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens, **kwargs):
        super().__init__(self, **kwargs)
        self.P = keras.layers.Embedding(num_users, num_factors)
        self.Q = keras.layers.Embedding(num_items, num_factors)
        self.U = keras.layers.Embedding(num_users, num_factors)
        self.V = keras.layers.Embedding(num_items, num_factors)

        self.mlp = keras.Sequential([
            keras.layers.Dense(num_hiddens, activation="relu")
            for num_hiddens in nums_hiddens
        ])

        self.output_layer = keras.layers.Dense(1, activation="sigmoid", use_bias=False)

    def compute(self, user_ids, item_ids):
        p_mf = self.P(user_ids)
        q_mf = self.Q(item_ids)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_ids)
        q_mlp = self.V(item_ids)
        mlp = self.mlp(tf.concat([p_mlp, q_mlp], axis=1))

        return self.output_layer(tf.concat([gmf, mlp], axis=1))

    def compile(self, optimizer, loss):
        super().compile(optimizer, loss)

    def _get_hit(self, ranklist, predict_items, k=5):
        pass

    def train_step(self, inputs):
        user_ids, item_ids, neg_item_ids = inputs

        with tf.GradientTape() as tape:
            p_pos = self.compute(user_ids, item_ids)
            p_neg = self.compute(user_ids, neg_item_ids)
            loss = self.loss(p_pos, p_neg)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


if __name__ == "__main__":
    num_users = 10
    num_items = 2000000

    model = NeuMF(64, num_users, num_items, [10, 10, 10])
    model.compile(optimizer="adam", loss=PairwiseRankingLoss())

    dataset_generator = DatasetGenerator(
        "./dataset/user_data.json", "./dataset/item_data.json"
    )
    dataset = dataset_generator.generate(16)

    losses = []
    for data in dataset.take(10):
        # loss = model.train_step(data)
        loss = model.train_step(data)
        losses.append(loss.numpy())

    print(losses)
