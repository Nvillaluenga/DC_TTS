import tensorflow as tf
from Hyperparams import Hyperparams as hp
import unittest
from Attention import Attention


class AttentionTest(unittest.TestCase):
    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testAudioEncoder(self):
        # Arrange
        Q = tf.random.normal(shape=(hp.B, int(hp.max_T/hp.r), hp.d))
        K = tf.random.normal(shape=(hp.B, hp.max_N, hp.d))
        V = tf.random.normal(shape=(hp.B, hp.max_N, hp.d))
        # Act
        attention = Attention()
        output, _, _ = attention.calculateAttention(Q, K, V)
        # Assert
        print(output.shape)
        assert hp.B == output.shape[0]
        assert int(hp.max_T/hp.r) == output.shape[1]
        assert 2*hp.d == output.shape[2]


if __name__ == "__main__":
    unittest.main()
