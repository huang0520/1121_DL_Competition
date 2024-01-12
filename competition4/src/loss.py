import tensorflow as tf
from tensorflow import keras


class PairwiseRankingLoss(keras.losses.Loss):
    def __init__(self, margin=0.3, name="pairwise_ranking_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, x1, x2):
        """
        Compute the pairwise ranking loss.
        x1 should be the positive item.

        Parameters:
        x1 (Tensor): The first input tensor.
        x2 (Tensor): The second input tensor.

        Returns:
        Tensor: The calculated loss.

        """

        return tf.math.maximum(self.margin - (x1 - x2), 0.0)

    def get_config(self):
        return super().get_config()
