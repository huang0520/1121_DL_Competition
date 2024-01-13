import tensorflow as tf
from tensorflow import keras


class FMLayer(keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.w_reg = w_reg
        self.v_reg = v_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(
            name="w0",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        self.w = self.add_weight(
            name="w",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            regularizer=keras.regularizers.l2(self.w_reg),
            trainable=True,
        )
        self.v = self.add_weight(
            name="v",
            shape=(input_shape[-1], self.k),
            initializer="random_normal",
            regularizer=keras.regularizers.l2(self.v_reg),
            trainable=True,
        )

    def call(self, inputs):
        linear = self.w0 + tf.sparse.sparse_dense_matmul(inputs, self.w)

        square_sum = tf.pow(tf.matmul(inputs, self.v), 2)
        sum_square = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))
        second_order = 0.5 * tf.reduce_sum(
            square_sum - sum_square, axis=1, keepdims=True
        )
        return linear + second_order
