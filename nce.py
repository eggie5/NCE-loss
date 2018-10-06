import keras
from keras.layers import Input, Dense, Embedding, Flatten
import tensorflow as tf
import keras.backend as K


class NCE(keras.layers.Layer):
    def __init__(self, num_classes, neg_samples=100, **kwargs):

        self.num_classes = num_classes
        self.neg_samples = neg_samples

        super(NCE, self).__init__(**kwargs)

    # keras Layer interface
    def build(self, input_shape):

        self.W = self.add_weight(
            name="approx_softmax_weights",
            shape=(self.num_classes, input_shape[0][1]),
            initializer="glorot_normal",
        )

        self.b = self.add_weight(
            name="approx_softmax_biases", shape=(self.num_classes,), initializer="zeros"
        )

        # keras
        super(NCE, self).build(input_shape)

    # keras Layer interface
    def call(self, x):
        predictions, targets = x

        # tensorflow
        loss = tf.nn.nce_loss(
            self.W, self.b, targets, predictions, self.neg_samples, self.num_classes
        )

        # keras
        self.add_loss(loss)

        logits = K.dot(predictions, K.transpose(self.W))

        return logits

    # keras Layer interface
    def compute_output_shape(self, input_shape):
        return 1