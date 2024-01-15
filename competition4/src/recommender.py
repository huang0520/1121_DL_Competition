from itertools import combinations

import pandas as pd
import tensorflow as tf
from icecream import ic
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Add, Dot, Embedding, Flatten, Input
from tensorflow.keras.regularizers import L2

from .layer import SparseFMLayer


class FunkSVD(tf.keras.Model):
    """
    Simplified Funk-SVD recommender model
    """

    def __init__(self, num_factors, num_users, num_items, l2_lambda=0.1, **kwargs):
        """
        Constructor of the model
        """
        super().__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.continue_hit_count = 0

        # Input
        user_id = Input(shape=(1,), dtype=tf.int32)
        item_id = Input(shape=(1,), dtype=tf.int32)
        text_embeddings = Input(shape=(num_factors,), dtype=tf.float32)

        # Embedding
        vec_user = Embedding(
            num_users,
            num_factors,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(user_id)
        vec_item = Embedding(
            num_items,
            num_factors,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(item_id)
        embeddings = Add()([
            Dot(axes=2)([vec_user, vec_item]),
            Dot(axes=2)([vec_user, tf.expand_dims(text_embeddings, axis=1)]),
        ])

        # Bias
        b_user = Embedding(
            num_users,
            1,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(user_id)
        b_item = Embedding(
            num_items,
            1,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(item_id)
        biases = Add()([
            b_user,
            b_item,
        ])

        # Output
        output = Add()([embeddings, biases])
        output = Flatten()(output)

        self.model = keras.Model(
            inputs=(user_id, item_id, text_embeddings),
            outputs=output,
        )

    @tf.function
    def call(self, inputs) -> tf.Tensor:
        user_id, item_id, text_embedding = inputs
        return self.model((user_id, item_id, text_embedding))

    @tf.function
    def train_step(self, inputs: tf.Tensor) -> tf.Tensor:
        # user_ids, item_ids, title_tokens, desc_tokens, y_trues = inputs
        user_ids, item_ids, text_embeddings, y_trues = inputs

        # compute loss
        with tf.GradientTape() as tape:
            y_preds = self.call((user_ids, item_ids, text_embeddings))
            loss = self.loss(y_trues, y_preds)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # @tf.function
    # def update(self, user_id, previous_id, clicked_id, discount=0.9):
    #     if clicked_id == -1:
    #         reward = 0
    #         self.continue_hit_count = 0
    #     else:
    #         reward = 0.1
    #         self.continue_hit_count += 1

    #     if reward == 0:

    #     gradients = tape.gradient(loss, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    def get_topk(self, user_id, data_manager, k=5) -> tf.Tensor:
        if not hasattr(self, "all_item_ids"):
            self.all_item_ids = tf.range(self.num_items)
        if not hasattr(self, "all_text_embeddings"):
            self.all_text_embeddings = tf.convert_to_tensor(
                data_manager.item_to_embeddings.to_list(), dtype=tf.float32
            )

        user_ids = tf.repeat(tf.constant(user_id), self.num_items)
        rank_list = tf.squeeze(
            self.call((user_ids, self.all_item_ids, self.all_text_embeddings))
        )
        return tf.math.top_k(rank_list, k=k).indices.numpy().tolist()


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
        return tf.math.top_k(tf.squeeze(item_probs), k=k).indices.numpy().tolist()


class SparseHotEncoder:
    def __init__(self, num_users, num_items, num_tokens):
        self.user_encoder = keras.layers.experimental.preprocessing.CategoryEncoding(
            num_users, output_mode="one_hot", sparse=True
        )
        self.item_encoder = keras.layers.experimental.preprocessing.CategoryEncoding(
            num_items, output_mode="one_hot", sparse=True
        )
        self.token_encoder = keras.layers.experimental.preprocessing.CategoryEncoding(
            num_tokens, output_mode="multi_hot", sparse=True
        )

    def __call__(self, inputs):
        if type(inputs) is tuple:
            user_ids, item_ids, title_token, desc_token = inputs
        elif type(inputs) is pd.DataFrame:
            user_ids = tf.convert_to_tensor(inputs["user_id"].to_numpy(dtype=int))
            item_ids = tf.convert_to_tensor(inputs["item_id"].to_numpy(dtype=int))
            title_token = tf.ragged.constant(inputs["title"], dtype=tf.int32)
            desc_token = tf.ragged.constant(inputs["desc"], dtype=tf.int32)
        else:
            raise ValueError("Invalid inputs type")

        user_fields: tf.sparse.SparseTensor = self.user_encoder(user_ids)
        item_fields: tf.sparse.SparseTensor = self.item_encoder(item_ids)
        title_fields: tf.sparse.SparseTensor = self.token_encoder(title_token)
        desc_fields: tf.sparse.SparseTensor = self.token_encoder(desc_token)

        if len(title_fields.shape) == 1:
            title_fields = tf.sparse.expand_dims(title_fields, axis=0)
        if len(user_fields.shape) == 1:
            user_fields = tf.sparse.expand_dims(user_fields, axis=0)
        if len(item_fields.shape) == 1:
            item_fields = tf.sparse.expand_dims(item_fields, axis=0)
        if len(desc_fields.shape) == 1:
            desc_fields = tf.sparse.expand_dims(desc_fields, axis=0)

        return tf.sparse.concat(
            axis=-1, sp_inputs=[user_fields, item_fields, title_fields, desc_fields]
        )


class SparseFM(tf.keras.Model):
    def __init__(self, embed_dim, w_reg, v_reg, **kwargs):
        super().__init__(**kwargs)
        self.fm = SparseFMLayer(embed_dim, w_reg, v_reg)

    @tf.function
    def call(self, inputs):
        return self.fm(inputs)

    @tf.function
    def train_step(self, inputs):
        features, y_trues = inputs

        with tf.GradientTape() as tape:
            y_preds = self.call(features)
            loss = self.loss(y_trues, y_preds)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


class FMEmbeding:
    def __init__(
        self,
        encoder: SparseHotEncoder,
        num_users,
        num_items,
        num_tokens,
        model: SparseFM,
    ) -> None:
        self.encoder = encoder
        self.user_field_indices = num_users
        self.item_field_indices = num_items + 2 * num_tokens

        self.field_vector = model.fm.embedding
        self.field_weight = model.fm.weight
        self.bias = model.fm.bias

    def get_user_embedding(self, primitive_features: pd.DataFrame):
        features = self.encoder(primitive_features)
        user_fields = tf.sparse.slice(
            features,
            start=[0, 0],
            size=[features.shape[0], self.user_field_indices],
        )

        # field vector
        user_fields_vector = tf.sparse.sparse_dense_matmul(
            user_fields, self.field_vector[: self.user_field_indices, :]
        )  # (batch_size, embed_dim)
        user_fields_cross = self._cross_vector(user_fields_vector)  # (batch_size,)

        # field weight
        user_fields_weight = tf.sparse.sparse_dense_matmul(
            user_fields, self.field_weight[: self.user_field_indices, :]
        )
        user_fields_weight = tf.reduce_sum(user_fields_weight, axis=1)  # (batch_size,)

        # Embedding composition
        ones = tf.ones([user_fields_cross.shape[0], 1], dtype=tf.float32)
        embeddings = tf.concat(
            [
                ones,
                tf.expand_dims(user_fields_cross + user_fields_weight, axis=1),
                user_fields_vector,
            ],
            axis=1,
        ).numpy()

        return dict(zip(primitive_features["user_id"], embeddings))

    def get_item_embedding(self, primitive_features):
        features = self.encoder(primitive_features)
        item_fields = tf.sparse.slice(
            features,
            start=[0, self.user_field_indices],
            size=[features.shape[0], self.item_field_indices],
        )

        # field vector
        item_fields_vector = tf.sparse.sparse_dense_matmul(
            item_fields, self.field_vector[self.user_field_indices :, :]
        )
        item_fields_cross = self._cross_vector(item_fields_vector)

        # field weight
        item_fields_weight = tf.sparse.sparse_dense_matmul(
            item_fields, self.field_vector[self.user_field_indices :, :]
        )
        item_fields_weight = tf.reduce_sum(item_fields_weight, axis=1)

        # Embedding composition
        ones = tf.ones([item_fields_cross.shape[0], 1], dtype=tf.float32)
        embeddings = tf.concat(
            [
                tf.expand_dims(item_fields_cross + item_fields_weight, axis=1),
                ones,
                item_fields_vector,
            ],
            axis=1,
        ).numpy()

        return dict(zip(primitive_features["item_id"], embeddings))

    def _cross_vector(self, vector):
        square_sum = tf.pow(tf.reduce_sum(vector, axis=1, keepdims=True), 2)
        sum_square = tf.reduce_sum(tf.pow(vector, 2), axis=1, keepdims=True)
        return 0.5 * tf.reduce_sum(square_sum - sum_square, axis=1)
