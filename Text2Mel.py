import tensorflow as tf
from modules.TextEncoder import TextEncoder
from modules.AudioEncoder import AudioEncoder
from modules.Attention import Attention
from modules.AudioDecoder import AudioDecoder


class Text2Mel(tf.keras.models.Model):
    def __init__(self):
        super(Text2Mel, self).__init__()
        self.textEncoder = TextEncoder()
        self.audioEncoder = AudioEncoder()
        self.attention = Attention
        self.audioDecoder = AudioDecoder()

    def call(self, text_input, audio_input, is_training):
        K, V = self.textEncoder(text_input, is_training)
        Q = self.audioEncoder(audio_input, is_training)
