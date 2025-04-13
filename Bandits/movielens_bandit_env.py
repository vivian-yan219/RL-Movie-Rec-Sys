# movielens_bandit_env.py
import numpy as np
import tensorflow_datasets as tfds
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, Sequential
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class RealMovieLensEmbeddingEnv(py_environment.PyEnvironment):
    def __init__(self, num_users=100, num_movies=100, embedding_dim=16):
        super().__init__()
        self._num_users = num_users
        self._num_movies = num_movies
        self._embedding_dim = embedding_dim

        # Load raw MovieLens data
        raw = tfds.as_numpy(tfds.load('movielens/100k-ratings', split='train', batch_size=-1))
        user_ids, movie_ids = raw['user_id'], raw['movie_id']
        ratings = raw['user_rating']
        movie_genres = raw['movie_genres']
        user_age = raw['raw_user_age']
        user_gender = raw['user_gender']
        user_occupation = raw['user_occupation_label']

        # Select subset of users and movies
        unique_users = np.unique(user_ids)[:num_users]
        unique_movies = np.unique(movie_ids)[:num_movies]
        self._user2idx = {u: i for i, u in enumerate(unique_users)}
        self._movie2idx = {m: i for i, m in enumerate(unique_movies)}

        # Keep only samples in selected subset
        mask = np.isin(user_ids, unique_users) & np.isin(movie_ids, unique_movies)
        self._user_ids = user_ids[mask]
        self._movie_ids = movie_ids[mask]
        self._ratings = ratings[mask]
        self._movie_genres = movie_genres[mask]
        self._user_age = user_age[mask]
        self._user_gender = np.array(user_gender)[mask]
        self._user_occupation = user_occupation[mask]
        self._num_samples = len(self._ratings)

        # Movie genre → dense embedding
        mlb = MultiLabelBinarizer()
        genre_multi_hot = mlb.fit_transform(movie_genres)
        genre_embed_net = Sequential([
            layers.InputLayer(input_shape=(genre_multi_hot.shape[1],)),
            layers.Dense(embedding_dim)
        ])
        self._movie_embeddings = genre_embed_net(genre_multi_hot).numpy()

        # User features → dense embedding
        age_norm = ((self._user_age - 18) / 50).reshape(-1, 1)

        # One-hot encoding: M = 1.0, F = 0.0
        user_gender = np.array(user_gender, dtype='S')  # 'S' = byte string
        self._user_gender = user_gender[mask]
        decoded_gender = np.array([g.decode('utf-8') for g in self._user_gender])
        gender_onehot = (decoded_gender == 'M').astype(np.float32).reshape(-1, 1)

        # One-hot encode masked occupation
        num_occupations = np.max(self._user_occupation) + 1
        occupation_onehot = np.eye(num_occupations)[self._user_occupation]

        # Final user features
        user_features = np.concatenate([age_norm, gender_onehot, occupation_onehot], axis=1)

        user_embed_net = Sequential([
            layers.InputLayer(input_shape=(user_features.shape[1],)),
            layers.Dense(embedding_dim)
        ])
        self._user_embeddings = user_embed_net(user_features).numpy()

        # Environment spec
        self._context_dim = embedding_dim * 2
        self._index = 0
        self._done = False

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._context_dim,), dtype=np.float32, minimum=-10.0, maximum=10.0
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1  # Recommend or not
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._index = (self._index + 1) % self._num_samples
        self._done = False
        return ts.restart(self._get_context())

    def _step(self, action):
        if self._done:
            return self.reset()

        action = int(action)
        rating = self._ratings[self._index]
        reward = 1.0 if (action == 1 and rating >= 4.0) else 0.0

        self._done = True
        return ts.termination(self._get_context(), reward)

    def _get_context(self):
        user_idx = self._user2idx[self._user_ids[self._index]]
        movie_idx = self._movie2idx[self._movie_ids[self._index]]
        return np.concatenate([
            self._user_embeddings[user_idx],
            self._movie_embeddings[movie_idx]
        ])
