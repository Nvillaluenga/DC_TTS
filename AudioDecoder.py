import tensorflow as tf
from Hyperparams import Hyperparams as hp
from tensorflow.keras.layers import Conv1D
from KerasModules import HighwayCNet


class AudioDecoder(tf.keras.Model):
    def __init__(self):
        super(AudioDecoder, self).__init__()
        self.conv_1 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.HC_2 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_3 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_4 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=9
        )
        self.HC_5 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=27
        )
        self.HC_6 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_7 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.conv_8 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.conv_9 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.conv_10 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.conv_11 = Conv1D(
            filters=hp.n_mels,
            kernel_size=1
        )

    def call(self, inputs, is_training):
        x = self.conv_1(inputs)
        x = self.HC_2(x, is_training)
        x = self.HC_3(x, is_training)
        x = self.HC_4(x, is_training)
        x = self.HC_5(x, is_training)
        x = self.HC_6(x, is_training)
        x = self.HC_7(x, is_training)
        x = self.conv_8(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_9(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_10(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_11(x)
        x = tf.keras.activations.sigmoid(x)

        return x
