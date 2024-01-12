import tensorflow as tf
from tensorflow import keras

from .dataset import DatasetGenerator, OneHotDatasetGenerator
from .layer import FMLayer
from .loss import PairwiseRankingLoss


class FunkSVD(tf.keras.Model):
    """
    Simplified Funk-SVD recommender model
    """

    def __init__(self, m_users: int, n_items: int, embedding_size: int):
        """
        Constructor of the model
        """
        super().__init__()
        self.m = m_users
        self.n = n_items
        self.k = embedding_size

        self.P = keras.layers.Embedding(self.m, self.k)
        self.Q = keras.layers.Embedding(self.n, self.k)

        self.B_user = keras.layers.Embedding(self.m, 1)
        self.B_item = keras.layers.Embedding(self.n, 1)

    @tf.function
    def call(self, inputs) -> tf.Tensor:
        """
        Forward pass used in training and validating
        """
        user_ids, item_ids = inputs
        p = self.P(user_ids)
        q = self.Q(item_ids)
        b_user = self.B_user(user_ids)
        b_item = self.B_item(item_ids)

        return tf.matmul(p, q, transpose_b=True) + b_user + b_item

    @tf.function
    def train_step(self, inputs: tf.Tensor) -> tf.Tensor:
        # data: user_id, item_id, rating
        user_ids, pos_item_ids, neg_item_ids = inputs

        # compute loss
        with tf.GradientTape() as tape:
            p_pos = self((user_ids, pos_item_ids))
            n_pos = self((user_ids, neg_item_ids))
            loss = self.loss(p_pos, n_pos)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def update(self, user_id, item_ids, clicked_id):
        item_ids = tf.convert_to_tensor(
            tuple(filter(lambda x: x != clicked_id, item_ids))
        )
        user_ids = tf.repeat(user_id, len(item_ids))
        clicked_ids = tf.repeat(clicked_id, len(item_ids))
        self.train_step((user_ids, item_ids, clicked_ids))

    @tf.function
    def __get_rank_list(self, user_id):
        p = self.P(user_id)
        q = self.Q(tf.range(self.n))
        return tf.matmul(p, q, transpose_b=True)

    def get_topk(self, user_id, k=5) -> tf.Tensor:
        rank_list = self.__get_rank_list(tf.constant(user_id))
        return tf.math.top_k(rank_list, k=k).indices.numpy()


class NeuMF(tf.keras.Model):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens, **kwargs):
        super().__init__(self, **kwargs)
        self.num_users = num_users
        self.num_items = num_items

        self.P = keras.layers.Embedding(num_users, num_factors)
        self.Q = keras.layers.Embedding(num_items, num_factors)
        self.U = keras.layers.Embedding(num_users, num_factors)
        self.V = keras.layers.Embedding(num_items, num_factors)

        self.mlp = keras.Sequential([
            keras.layers.Dense(num_hiddens, activation=keras.activations.relu)
            for num_hiddens in nums_hiddens
        ])

        self.output_layer = keras.layers.Dense(1, activation="sigmoid", use_bias=False)

    def call(self, inputs):
        user_ids, item_ids = inputs

        p_mf = self.P(user_ids)
        q_mf = self.Q(item_ids)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_ids)
        q_mlp = self.V(item_ids)
        mlp = self.mlp(tf.concat([p_mlp, q_mlp], axis=1))

        return self.output_layer(tf.concat([gmf, mlp], axis=1))

    @tf.function
    def train_step(self, inputs):
        user_ids, pos_item_ids, neg_item_ids = inputs

        with tf.GradientTape() as tape:
            p_pos = self((user_ids, pos_item_ids))
            p_neg = self((user_ids, neg_item_ids))
            loss = self.loss(p_pos, p_neg)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def get_topk(self, user_id, k=5):
        item_probs = self((
            tf.repeat(tf.constant(user_id), self.num_items),
            tf.range(self.num_items),
        ))
        return tf.math.top_k(tf.squeeze(item_probs), k=k).indices.numpy()


class FM(keras.Model):
    def __init__(self, k=8, w_reg=1e-4, v_reg=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.fm = FMLayer(k, w_reg, v_reg)

    @tf.function
    def call(self, inputs):
        fm_output = self.fm(inputs)
        output = tf.nn.sigmoid(fm_output)
        return output

    @tf.function
    def train_step(self, inputs):
        x, y_true = inputs

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    # @tf.function
    def __get_rank_list(self, user_id, num_users, num_items):
        pass

    def get_topn(self, user_id, n=5, num_users=2000, num_items=209527):
        pass


if __name__ == "__main__":
    num_users = 10
    num_items = 100

    model = FM()
    model.compile(optimizer="adam", loss=PairwiseRankingLoss())

    # dataset_generator

    # dataset_generator = DatasetGenerator(
    #     "./dataset/user_data.json", "./dataset/item_data.json"
    # )
    # dataset = dataset_generator.generate(16)

    # losses = []
    # for data in dataset.take(10):
    #     # loss = model.train_step(data)
    #     loss = model.train_step(data)
    #     losses.append(loss.numpy())

    # print(losses)
