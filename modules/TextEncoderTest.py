import unittest
import tensorflow as tf
from TextEncoder import TextEncoder
from DataLoad import text_normalize, load_vocab


class TextEncoderTest(unittest.TestCase):

    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testTextEncoder(self):
        # Arrange
        input = preprocess_text("Hello World")
        input = tf.expand_dims(input, axis=0)
        textEncoder = TextEncoder()
        # Act
        K, V = textEncoder(input, True)
        # Assert
        assert [1, 11, 256] == K.shape
        assert [1, 11, 256] == V.shape


def preprocess_text(text):
    char2idx, _ = load_vocab()
    text = text_normalize(text)
    text_normalized = [char2idx[char] for char in text]
    return text_normalized


if __name__ == "__main__":
    unittest.main()
