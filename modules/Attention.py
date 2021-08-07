import tensorflow as tf
import numpy as np
from Hyperparams import Hyperparams as hp


class Attention(tf.keras.models.Model):
    def __init__(self):
        super(Attention, self).__init__()

        self.prev_max_attentions = tf.ones(shape=(hp.B,), dtype=tf.int32)
        self.gts = tf.convert_to_tensor(self.guided_attention())

        self.alignments = None
        self.max_attentions = None
        self.A = None
        self.R = None

    def call(self):
        raise NotImplementedError()

    def calculateAttention(self, Q, K, V, mononotic_attention=False, prev_max_attentions=None):
        '''
        Args:
        Q: Queries. (B, T/r, d)
        K: Keys. (B, N, d)
        V: Values. (B, N, d)
        mononotic_attention: A boolean. At training, it is False.
        prev_max_attentions: (B,). At training, it is set to None.

        Returns:
        R: [Context Vectors; Q]. (B, T/r, 2d)
        alignments: (B, N, T/r)
        max_attentions: (B, T/r)
        '''
        A = tf.matmul(Q, K, transpose_b=True) * \
            tf.math.rsqrt(tf.cast(hp.d, tf.float32))
        if mononotic_attention:  # for inference
            key_masks = tf.sequence_mask(prev_max_attentions, hp.max_N)
            reverse_masks = tf.sequence_mask(
                hp.max_N - hp.attention_win_size - prev_max_attentions, hp.max_N)[:, ::-1]
            masks = tf.logical_or(key_masks, reverse_masks)
            masks = tf.tile(tf.expand_dims(masks, 1), [1, hp.max_T, 1])
            paddings = tf.ones_like(A) * (-2 ** 32 + 1)  # (B, T/r, N)
            A = tf.where(tf.equal(masks, False), A, paddings)
        A = tf.nn.softmax(A)  # (B, T/r, N)
        max_attentions = tf.argmax(A, -1)  # (B, T/r)
        R = tf.matmul(A, V)
        R = tf.concat((R, Q), -1)

        alignments = tf.transpose(A, [0, 2, 1])  # (B, N, T/r)

        return R, alignments, max_attentions

    def guided_attention(self, g=0.2):
        '''Guided attention. Refer to page 3 on the paper.'''
        W = np.zeros((hp.max_N, hp.max_T), dtype=np.float32)
        for n_pos in range(W.shape[0]):
            for t_pos in range(W.shape[1]):
                W[n_pos, t_pos] = 1 - \
                    np.exp(-(t_pos / float(hp.max_T) - n_pos /
                           float(hp.max_N)) ** 2 / (2 * g ** 2))
        return W
