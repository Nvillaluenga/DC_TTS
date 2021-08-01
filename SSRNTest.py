import unittest
import tensorflow as tf
from Hyperparams import Hyperparams as hp
from SSRN import SSRN


class SSRNTest(unittest.TestCase):
    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testAudioEncoder(self):
        # Arrange
        input = tf.random.normal(shape=(hp.B, int(hp.max_T/hp.r), hp.n_mels))
        sSRN = SSRN()
        # Act
        output = sSRN(input, True)
        # Assert
        print(output)
        assert hp.B == output.shape[0]
        assert int(hp.max_T/hp.r)*4 == output.shape[1]
        assert int(1+hp.n_fft//2) == output.shape[2]


if __name__ == "__main__":
    unittest.main()
