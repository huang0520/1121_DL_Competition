import tensorflow as tf
from tensorflow import keras


class SparseFMLayer(keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.weight = self.add_weight(
            name="weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            regularizer=keras.regularizers.l2(self.w_reg),
            trainable=True,
        )
        self.embedding = self.add_weight(
            name="embedding",
            shape=(input_shape[-1], self.k),
            initializer="random_normal",
            regularizer=keras.regularizers.l2(self.v_reg),
            trainable=True,
        )

    def call(self, inputs):
        linear = self.bias + tf.sparse.sparse_dense_matmul(inputs, self.weight)
        square_sum = tf.pow(tf.sparse.sparse_dense_matmul(inputs, self.embedding), 2)
        sum_square = tf.sparse.sparse_dense_matmul(
            tf.sparse.map_values(tf.math.pow, inputs, 2),
            tf.math.pow(self.embedding, 2),
        )
        second_order = 0.5 * tf.reduce_sum(
            square_sum - sum_square, axis=1, keepdims=True
        )
        return linear + second_order
