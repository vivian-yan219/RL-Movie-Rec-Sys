import numpy as np
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tensorflow.keras import layers, Sequential
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class RealMovieLensEmbeddingEnv(py_environment.PyEnvironment):
    def __init__(self, num_users=100, num_movies=100, embedding_dim=16, split='train'):
        super().__init__()
        self._num_users = num_users
        self._num_movies = num_movies
        self._embedding_dim = embedding_dim

        self._steps_per_episode = 5  # Number of steps before done
        self._current_steps = 0  # Current step count in an episode

        raw = tfds.as_numpy(tfds.load('movielens/100k-ratings', split='train', batch_size=-1))
        user_ids, movie_ids = raw['user_id'], raw['movie_id']
        ratings = raw['user_rating']
        movie_genres = raw['movie_genres']
        user_age = raw['raw_user_age']
        user_gender = raw['user_gender']
        user_occupation = raw['user_occupation_label']

        all_indices = np.arange(len(user_ids))
        train_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
        if split == 'train':
            indices = train_indices
        elif split == 'test':
            indices = test_indices
        else:
            raise ValueError(f"Invalid split: {split}")

        user_ids = user_ids[indices]
        movie_ids = movie_ids[indices]
        ratings = ratings[indices]
        movie_genres = movie_genres[indices]
        user_age = user_age[indices]
        user_gender = np.array(user_gender)[indices]
        user_occupation = user_occupation[indices]

        unique_users = np.unique(user_ids)[:num_users]
        unique_movies = np.unique(movie_ids)[:num_movies]
        self._user2idx = {u: i for i, u in enumerate(unique_users)}
        self._movie2idx = {m: i for i, m in enumerate(unique_movies)}
        self._idx2movie = {i: m for m, i in self._movie2idx.items()}

        mask = np.isin(user_ids, unique_users) & np.isin(movie_ids, unique_movies)
        self._user_ids = user_ids[mask]
        self._movie_ids = movie_ids[mask]
        self._ratings = ratings[mask]
        self._movie_genres = movie_genres[mask]
        self._user_age = user_age[mask]
        self._user_gender = user_gender[mask]
        self._user_occupation = user_occupation[mask]
        self._num_samples = len(self._ratings)

        mlb = MultiLabelBinarizer()
        genre_multi_hot = mlb.fit_transform(self._movie_genres)
        genre_embed_net = Sequential([
            layers.InputLayer(input_shape=(genre_multi_hot.shape[1],)),
            layers.Dense(embedding_dim)
        ])
        self._movie_embeddings = genre_embed_net(genre_multi_hot).numpy()

        age_norm = ((self._user_age - 18) / 50).reshape(-1, 1)
        gender_onehot = self._user_gender.astype(np.float32).reshape(-1, 1)
        num_occupations = np.max(self._user_occupation) + 1
        occupation_onehot = np.eye(num_occupations)[self._user_occupation]

        user_features = np.concatenate([age_norm, gender_onehot, occupation_onehot], axis=1)
        user_embed_net = Sequential([
            layers.InputLayer(input_shape=(user_features.shape[1],)),
            layers.Dense(embedding_dim)
        ])
        self._user_embeddings = user_embed_net(user_features).numpy()

        self._context_dim = embedding_dim
        self._index = 0
        self._done = False

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._context_dim,), dtype=np.float32, minimum=-10.0, maximum=10.0
        )
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._num_movies - 1
        )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def _reset(self):
        self._index = (self._index + 1) % self._num_samples
        self._done = False
        self._current_steps = 0  # Reset the step counter
        return ts.restart(self._get_context())

    def _step(self, action):
        if self._done:
            return self.reset()

        action = int(action)
        user_idx = self._user2idx[self._user_ids[self._index]]

        rating = self.get_user_rating(user_idx, action)
        if rating >= 4.0:
            reward = 1.0
        elif rating == 3.0:
            reward = 0.5
        else:
            reward = 0.0

        self._current_steps += 1

        if self._current_steps >= self._steps_per_episode:
            self._done = True
            return ts.termination(self._get_context(), reward)
        else:
            return ts.transition(self._get_context(), reward=reward)

    def _get_context(self):
        user_idx = self._user2idx[self._user_ids[self._index]]
        return self._user_embeddings[user_idx]

    def get_user_liked_items(self, user_id, threshold=4.0):
        original_user_id = list(self._user2idx.keys())[user_id]
        liked_movies = []
        for u, m, r in zip(self._user_ids, self._movie_ids, self._ratings):
            if u == original_user_id and r >= threshold:
                liked_movies.append(self._movie2idx[m])
        return liked_movies

    def get_user_rating(self, user_id, movie_id):
        """
        Given a user_id (internal index) and movie_id (internal index),
        return the user's rating for the specified movie.
        If the rating does not exist, return 0.0.
        """
        original_user_id = list(self._user2idx.keys())[user_id]
        original_movie_id = self._idx2movie.get(movie_id, None)

        if original_movie_id is None:
            return 0.0

        for u, m, r in zip(self._user_ids, self._movie_ids, self._ratings):
            if u == original_user_id and m == original_movie_id:
                return float(r)

        return 0.0  # Return 0.0 if no rating is found



