import unittest
import numpy as np
from TextEncoder import TextEncoder
from DataLoad import text_normalize, load_vocab


class TextEncoderTest(unittest.TestCase):

    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testTextEncoder(self):
        # Arrange
        text1 = preprocess_text("Hello World")
        text2 = preprocess_text("Hola Mundo!")
        inputs = np.append([text1], [text2], axis=0)
        textEncoder = TextEncoder()
        # Act
        output = textEncoder(inputs, True)
        # Assert
        assert [2, 11, 512] == output.shape


def preprocess_text(text):
    char2idx, _ = load_vocab()
    text = text_normalize(text)
    text_normalized = [char2idx[char] for char in text]
    return text_normalized


if __name__ == "__main__":
    unittest.main()
