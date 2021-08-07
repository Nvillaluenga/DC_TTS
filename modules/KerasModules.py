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


class HighwayCNet(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate=1, dropout_rate=0):
        super(HighwayCNet, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters=2*filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=tf.keras.initializers.variance_scaling,
            use_bias=True,
            padding="SAME")
        self.normalize = tf.keras.layers.LayerNormalization()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, is_training):
        _inputs = inputs
        tensor = self.conv1d(inputs)
        H1, H2 = tf.split(tensor, 2, axis=-1)
        H1 = self.normalize(H1)
        H2 = self.normalize(H2)
        H1 = tf.nn.sigmoid(H1, "gate")
        tensor = H1*H2 + (1.-H1)*_inputs
        tensor = self.dropout(tensor, training=is_training)

        return tensor


class HighwayCNet_experimental(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate):
        super(HighwayCNet_experimental, self).__init__()
        self.conv1d = tf.keras.layers.Conv1D(
            filters=2*filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            kernel_initializer=tf.keras.initializers.variance_scaling,
            use_bias=True)
        self.normalize = tf.keras.layers.LayerNormalization()
        self.T = tf.keras.layers.Dense(
            units=filters,
            activation="sigmoid",
            bias_initializer=tf.keras.initializers.Constant(-3.0)
        )
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, is_training):
        _inputs = inputs
        tensor = self.conv1d(inputs)
        H1, H2 = tf.split(tensor, 2, axis=-1)
        H1 = self.normalize(H1)
        H1 = self.T(H1)
        H1 = self.normalize(H1)
        H2 = self.normalize(H2)
        tensor = H1*H2 + (1.-H1)*_inputs
        tensor = self.dropout(tensor, training=is_training)

        return tensor
