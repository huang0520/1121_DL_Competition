from itertools import combinations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Add, Dot, Embedding, Flatten, Input
from tensorflow.keras.regularizers import L2

from .layer import FMLayer
from .loss import PairwiseRankingLoss
from .utils import compose


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

        user_id = Input(shape=(1,), dtype=tf.int32)
        item_id = Input(shape=(1,), dtype=tf.int32)
        p = Embedding(
            num_users,
            num_factors,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(user_id)
        q = Embedding(
            num_items,
            num_factors,
            embeddings_initializer=RandomNormal(),
            embeddings_regularizer=L2(l2_lambda),
        )(item_id)
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
        output = Dot(axes=2)([p, q])
        output = Add()([output, b_user, b_item])
        output = Flatten()(output)

        self.model = keras.Model(inputs=(user_id, item_id), outputs=output)

    @tf.function
    def call(self, inputs) -> tf.Tensor:
        """
        Forward pass used in training and validating
        """
        return self.model(inputs)

    @tf.function
    def train_step(self, inputs: tf.Tensor) -> tf.Tensor:
        user_ids, items, y_trues = inputs

        # compute loss
        with tf.GradientTape() as tape:
            y_preds = self.call((user_ids, items))
            loss = self.loss(y_trues, y_preds)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    @tf.function
    def update_step(self, inputs):
        user_id, pos_item_id, neg_item_ids = inputs
        user_id = tf.expand_dims(user_id, axis=0)
        pos_item_id = tf.expand_dims(pos_item_id, axis=0)

        with tf.GradientTape() as tape:
            logit_pos = self.call((user_id, pos_item_id))
            logits_neg = self.call((
                tf.repeat(user_id, len(neg_item_ids)),
                neg_item_ids,
            ))
            logits_neg = tf.transpose(logits_neg)

            # InfoNCE loss
            logits = tf.concat([logit_pos, logits_neg], axis=1)
            labels = tf.concat(
                [tf.ones_like(logit_pos), tf.zeros_like(logits_neg)], axis=1
            )

            loss = keras.losses.CategoricalCrossentropy(from_logits=True)(
                labels, logits / logits.shape[-1]
            )
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def get_topk(self, user_id, k=5) -> tf.Tensor:
        user_ids = tf.repeat(tf.constant(user_id), self.num_items)
        item_ids = tf.range(self.num_items)
        rank_list = tf.squeeze(self.call((user_ids, item_ids)))
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
