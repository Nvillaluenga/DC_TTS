import tensorflow as tf


class HighwayNet(tf.keras.layers.Layer):
    def __init__(self, num_units):
        super(HighwayNet, self).__init__()
        self.H = tf.keras.layers.Dense(units=num_units, activation="relu")
        self.T = tf.keras.layers.Dense(
            units=num_units,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-2.0)
        )

    def call(self, inputs):
        h = self.H(inputs)
        t = self.T(inputs)
        outputs = h * t + inputs * (1. - t)
        return outputs
