import unittest
import unicodedata
import re
import numpy as np
from Hyperparams import Hyperparams as hp
from TextEncoder import TextEncoder


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


# TODO delete all this function and put them in a pre processor / feeder (Still need to do that)
def preprocess_text(text):
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    text = text_normalize(text)
    text_normalized = [char2idx[char] for char in text]
    return text_normalized


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text


if __name__ == "__main__":
    unittest.main()
