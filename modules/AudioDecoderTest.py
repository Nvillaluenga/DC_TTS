import unittest
import tensorflow as tf
from Hyperparams import Hyperparams as hp
from AudioDecoder import AudioDecoder


class AudioDecoderTest(unittest.TestCase):
    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testAudioDecoder(self):
        # Arrange
        input = tf.random.normal(shape=(hp.B, int(hp.max_T/hp.r), 2*hp.d))
        audioDecoder = AudioDecoder()
        # Act
        output = audioDecoder(input, True)
        # Assert
        assert hp.B == output.shape[0]
        assert int(hp.max_T/hp.r) == output.shape[1]
        assert hp.n_mels == output.shape[2]


if __name__ == "__main__":
    unittest.main()
