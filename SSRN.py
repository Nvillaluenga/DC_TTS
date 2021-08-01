import tensorflow as tf
from Hyperparams import Hyperparams as hp
from tensorflow.keras.layers import Conv1D, Conv1DTranspose
from KerasModules import HighwayCNet


class SSRN(tf.keras.Model):
    def __init__(self):
        super(SSRN, self).__init__()
        self.conv_1 = Conv1D(
            filters=hp.c,
            kernel_size=1
        )
        self.HC_2 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_3 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=3
        )
        self.deconv_4 = Conv1DTranspose(
            filters=hp.c,
            kernel_size=2,
            dilation_rate=1,
            strides=2
        )
        self.HC_5 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_6 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=3
        )
        self.deconv_7 = Conv1DTranspose(
            filters=hp.c,
            kernel_size=2,
            dilation_rate=1,
            strides=2
        )
        self.HC_8 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_9 = HighwayCNet(
            filters=hp.c,
            kernel_size=3,
            dilation_rate=3
        )
        self.conv_10 = Conv1D(
            filters=2*hp.c,
            kernel_size=1
        )
        self.HC_11 = HighwayCNet(
            filters=2*hp.c,
            kernel_size=3,
            dilation_rate=1
        )
        self.HC_12 = HighwayCNet(
            filters=2*hp.c,
            kernel_size=3,
            dilation_rate=1
        )
        self.conv_13 = Conv1D(
            filters=1+hp.n_fft//2,
            kernel_size=1
        )
        self.conv_14 = Conv1D(
            filters=1+hp.n_fft//2,
            kernel_size=1
        )
        self.conv_15 = Conv1D(
            filters=1+hp.n_fft//2,
            kernel_size=1
        )
        self.conv_16 = Conv1D(
            filters=1+hp.n_fft//2,
            kernel_size=1
        )

    def call(self, inputs, is_training):
        x = self.conv_1(inputs)
        x = self.HC_2(x, is_training)
        x = self.HC_3(x, is_training)
        x = self.deconv_4(x)
        x = self.HC_5(x, is_training)
        x = self.HC_6(x, is_training)
        x = self.deconv_7(x)
        x = self.HC_8(x, is_training)
        x = self.HC_9(x, is_training)
        x = self.conv_10(x)
        x = self.HC_11(x, is_training)
        x = self.HC_12(x, is_training)
        x = self.conv_13(x)
        x = self.conv_14(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_15(x)
        x = tf.keras.activations.relu(x)
        # Normalization?
        x = self.conv_16(x)
        x = tf.keras.activations.sigmoid(x)

        return x
