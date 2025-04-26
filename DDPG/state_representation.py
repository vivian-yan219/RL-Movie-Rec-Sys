import tensorflow as tf
import numpy as np

class DRRAveStateRepresentation(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(DRRAveStateRepresentation, self).__init__()
        self.embedding_dim = embedding_dim
        self.wav = tf.keras.layers.Conv1D(1, 1, 1)
        self.concat = tf.keras.layers.Concatenate()
        self.flatten = tf.keras.layers.Flatten()
        
    def call(self, x):
        user_eb, items_eb = x  # unpack for readability
        
        if tf.shape(items_eb)[1] == 0:  # if no items
            # fallback: return a zero vector or just user embedding flattened
            fallback = self.flatten(self.concat([user_eb, user_eb, user_eb]))
            return fallback

        items_eb = tf.transpose(items_eb, perm=(0, 2, 1)) / self.embedding_dim
        wav = self.wav(items_eb)
        wav = tf.transpose(wav, perm=(0, 2, 1))
        wav = tf.squeeze(wav, axis=1)
        user_wav = tf.keras.layers.multiply([user_eb, wav])
        concat = self.concat([user_eb, user_wav, wav])
        return self.flatten(concat)