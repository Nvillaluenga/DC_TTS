import tensorflow as tf
from Hyperparams import Hyperparams as hp
from tensorflow.keras.layers import Conv1D
from KerasModules import HighwayCNet


class AudioEncoder(tf.keras.Model):
    def __init__(self):
        super(AudioEncoder, self).__init__()
        self.conv_1 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.conv_2 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.conv_3 = Conv1D(
            filters=hp.d,
            kernel_size=1
        )
        self.HC_4 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_5 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_6 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=9
        )
        self.HC_7 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=27
        )
        self.HC_8 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_9 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_10 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=9
        )
        self.HC_11 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=27
        )
        self.HC_12 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=3
        )
        self.HC_13 = HighwayCNet(
            filters=hp.d,
            kernel_size=3,
            dilation_rate=3
        )

    def call(self, inputs, is_training):
        x = self.conv_1(inputs)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_2(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_3(x)
        # Normalization?
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
        return x
