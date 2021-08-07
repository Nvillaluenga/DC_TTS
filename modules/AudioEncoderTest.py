import unittest
import numpy as np
import tensorflow as tf
from Hyperparams import Hyperparams as h
from AudioEncoder import AudioEncoder
from os import path


class AudioEncoderTest(unittest.TestCase):
    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testAudioEncoder(self):
        # Arrange
        input = np.load(
            path.join(path.pardir, "tests_resources", "mel_test.npy"))
        input = tf.expand_dims(input, axis=0)
        audioEncoder = AudioEncoder()
        # Act
        output = audioEncoder(input, True)
        # Assert
        assert input.shape[1] == output.shape[1]
        assert h.d == output.shape[2]


if __name__ == "__main__":
    unittest.main()
