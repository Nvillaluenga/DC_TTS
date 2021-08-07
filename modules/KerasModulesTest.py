import unittest
import tensorflow as tf
from KerasModules import HighwayNet, HighwayCNet, HighwayCNet_experimental


class KerasModuleTest(unittest.TestCase):

    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testHighwayNet(self):
        hwNetLayer = HighwayNet(10)
        input = tf.constant(1., shape=(2, 10))
        output = hwNetLayer(input)
        # print(output)
        assert input.shape == output.shape

    def testHighwayCNet(self):
        hwCNetLayer = HighwayCNet(10, 1, 1, 0)
        input = tf.constant(1., shape=(1, 2, 10))
        output = hwCNetLayer(input, False)
        assert input.shape == output.shape

    def testHighwayCNet_exp(self):
        hwCNetLayer = HighwayCNet_experimental(10, 1, 1, 0)
        input = tf.constant(1., shape=(1, 2, 10))
        output = hwCNetLayer(input, False)
        print(output)
        assert input.shape == output.shape


if __name__ == "__main__":
    unittest.main()
