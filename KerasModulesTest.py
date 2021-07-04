import unittest
import tensorflow as tf
from KerasModules import HighwayNet


class KerasModuleTest(unittest.TestCase):

    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testHighwayNet(self):
        hwNetLayer = HighwayNet(10)
        input = tf.constant(1., shape=(2, 10))
        output = hwNetLayer(input)
        assert input.shape == output.shape


if __name__ == "__main__":
    unittest.main()
