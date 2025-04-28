import tensorflow as tf
import numpy as np

class MovieGenreEmbedding(tf.keras.Model):
    def __init__(self, len_movies, len_genres, embedding_size):
        super(MovieGenreEmbedding, self).__init__()
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_size)
        self.g_embedding = tf.keras.layers.Embedding(name='genre_embedding', input_dim=len_genres, output_dim=embedding_size)
        self.m_g_merge = tf.keras.layers.Dot(name='movie_genre_dot', normalize=True, axes=1)
        self.m_g_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        movie_input, genre_input = inputs
        memb = self.m_embedding(movie_input)
        gemb = self.g_embedding(genre_input)
        m_g = self.m_g_merge([memb, gemb])
        return self.m_g_fc(m_g)


class UserMovieEmbedding(tf.keras.Model):
    def __init__(self, len_users, len_movies, embedding_size):
        super(UserMovieEmbedding, self).__init__()
        self.u_embedding = tf.keras.layers.Embedding(name='user_embedding', input_dim=len_users, output_dim=embedding_size)
        self.m_embedding = tf.keras.layers.Embedding(name='movie_embedding', input_dim=len_movies, output_dim=embedding_size)
        self.m_u_merge = tf.keras.layers.Dot(name='movie_user_dot', normalize=False, axes=1)
        self.m_u_fc = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs):
        user_input, movie_input = inputs
        uemb = self.u_embedding(user_input)
        memb = self.m_embedding(movie_input)
        m_u = self.m_u_merge([memb, uemb])
        return self.m_u_fc(m_u)