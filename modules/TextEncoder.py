from Hyperparams import Hyperparams as hp
from KerasModules import *
from tensorflow.keras.layers import Embedding, Conv1D, LayerNormalization
import tensorflow as tf


class TextEncoder(tf.keras.Model):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.char_embedding = Embedding(
            input_dim=len(hp.vocab),
            output_dim=hp.e
        )
        self.conv_1 = Conv1D(
            filters=2*hp.d,
            kernel_size=1
        )
        self.conv_2 = Conv1D(
            filters=2*hp.d,
            kernel_size=1
        )
        self.HC_3 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_4 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_5 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=9
        )
        self.HC_6 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=27
        )
        self.HC_7 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_8 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_9 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=9
        )
        self.HC_10 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=27
        )
        self.HC_11 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_12 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_13 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=1,
            dilation_rate=1
        )
        self.HC_14 = HighwayCNet(
            filters=2*hp.d,
            kernel_size=1,
            dilation_rate=1
        )

    def call(self, inputs, is_training):
        x = self.char_embedding(inputs)
        x = self.conv_1(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_2(x)
        # Normalization?
        x = self.HC_3(x, is_training)
        x = self.HC_4(x, is_training)
        x = self.HC_5(x, is_training)
        x = self.HC_6(x, is_training)
        x = self.HC_7(x, is_training)
        x = self.HC_8(x, is_training)
        x = self.HC_9(x, is_training)
        x = self.HC_10(x, is_training)
        x = self.HC_11(x, is_training)
        x = self.HC_12(x, is_training)
        x = self.HC_13(x, is_training)
        x = self.HC_14(x, is_training)
        K, V = tf.split(x, 2, -1)
        return K, V
