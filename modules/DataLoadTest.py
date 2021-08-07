import unittest
from DataLoad import load_data


class DataLoadTest(unittest.TestCase):
    def setUp(self):
        print("running setup")

    def tearDown(self):
        print("running teardown")

    def testLoadData(self):
        # Arrange
        expectedfpath = "..\datasets\EN\LJSpeech-1.1\wavs\LJ001-0001.wav"
        expectedTextLength = 150
        # Act
        fpaths, textLengths, texts = load_data()
        # Assert
        assert expectedfpath == fpaths[0]
        assert expectedTextLength == textLengths[0]
        assert texts[0]


if __name__ == "__main__":
    unittest.main()
