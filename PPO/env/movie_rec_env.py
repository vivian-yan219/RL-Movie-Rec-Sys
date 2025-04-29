import numpy as np
import pandas as pd
from gymnasium import Env, spaces
from typing import Optional, Tuple, Dict

class MovieRecEnv(Env):
    """
    Gymnasium environment for MovieLens recommendation with PPO.
    Observations: vector of user+item features
    Actions: discrete movie indices
    Rewards: based on exact rating match or negative error
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        ratings: pd.DataFrame,
        users: pd.DataFrame,
        movies: pd.DataFrame,
        max_steps: int = 1000,
        seed: int = 42
    ):
        ratings['user_id']  = ratings['user_id'].astype(int)
        ratings['movie_id'] = ratings['movie_id'].astype(int)
        users['user_id']    = users['user_id'].astype(int)
        movies['movie_id']  = movies['movie_id'].astype(int)

        # preprocess dataframes
        self.ratings = ratings.reset_index(drop=True)
        self.users   = users.set_index('user_id')

        # coerce age to numeric, fill any non‑numeric with the mean, and cast to int
        self.users['age'] = pd.to_numeric(self.users['age'], errors='coerce')
        avg_age = self.users['age'].mean()
        self.users['age'].fillna(avg_age, inplace=True)
        self.users['age'] = self.users['age'].astype(int)
        
        self.user_info = self.users.to_dict(orient='index')
        self.movies  = movies.set_index('movie_id')

        # compute means
        self.user_mean  = self.ratings.groupby('user_id')['rating'].mean().to_dict()
        self.movie_mean = self.ratings.groupby('movie_id')['rating'].mean().to_dict()

        # genre columns
        self.genre_cols = [c for c in self.movies.columns if c != 'title']
        self.num_genres = len(self.genre_cols)

        # occupation info
        self.occupations = self.users['occupation'].unique().tolist()
        self.num_occ     = len(self.occupations)

        # observation dimension
        # [u_mean, m_mean] + genres + age_bucket(7) + occ_bucket + gender_bucket(2)
        self.obs_dim = 1 + 1 + self.num_genres + 7 + self.num_occ + 2

        # action & observation spaces
        self.movie_ids       = list(self.movies.index)
        self.action_space    = spaces.Discrete(len(self.movie_ids))
        self.observation_space = spaces.Box(
            low= -1.0,
            high=  1.0,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

        # episode control
        self.max_steps     = max_steps
        self.current_step  = 0
        self.rng           = np.random.RandomState(seed)
        self.order         = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset episode. Returns (obs, info).
        """
        super().reset(seed=seed)

        # shuffle a ordering of all rating‐rows
        n = len(self.ratings)
        self.order = self.rng.permutation(n).tolist()

        # reset step counter
        self.current_step = 0

        # initial observation
        first_idx = self.order[self.current_step]
        obs = self._get_obs(first_idx)
        return obs, {}

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one recommendation (action = predicted rating − 1).
        """
        idx = self.order[self.current_step]
        row = self.ratings.iloc[idx]
        user_id, movie_id, true_rating = row['user_id'], row['movie_id'], row['rating']

        pred_rating = action + 1
        error = abs(pred_rating - true_rating)
        reward = np.exp( - (error**2) / 2 )

        # advance
        self.current_step += 1
        done = self.current_step >= self.max_steps or self.current_step >= len(self.ratings)

        # next observation (or zeros if done)
        if not done:
            next_idx = self.order[self.current_step]
            obs = self._get_obs(next_idx)
        else:
            obs = np.zeros(self.obs_dim, dtype=np.float32)

        terminated = done
        truncated  = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self, idx: int) -> np.ndarray:
        """
        Build feature vector for the rating‐row at self.ratings.iloc[idx].
        """
        row = self.ratings.iloc[idx]
        user_id, movie_id = row['user_id'], row['movie_id']

        # user & movie mean (scaled)
        umean = self.user_mean.get(user_id, 3.0) / 5.0
        mmean = self.movie_mean.get(movie_id, 3.0) / 5.0

        # genre one‐hot
        genre_vec = self.movies.loc[movie_id, self.genre_cols].values.astype(np.float32)

        # age bucket (mean if missing)
        info = self.user_info.get(user_id, {})
        age = info.get('age', self.users['age'].mean())
        age_bucket = np.zeros(7, dtype=np.float32)
        age_bucket[min(age // 10, 6)] = 1.0

        # occupation bucket (fallback to first occupation)
        occ = info.get('occupation', self.occupations[0])
        occ_bucket = np.zeros(self.num_occ, dtype=np.float32)
        occ_bucket[self.occupations.index(occ)] = 1.0

        # gender bucket (fallback to 'M') (M=[1,0], F=[0,1])
        gender = info.get('gender', 'M')
        gen_bucket = np.array([1.0, 0.0], dtype=np.float32) if gender == 'M' else np.array([0.0, 1.0], dtype=np.float32)

        return np.concatenate(
            [[umean], [mmean], genre_vec, age_bucket, occ_bucket, gen_bucket],
            axis=0
        )

    def render(self, mode='human'):
        pass

    def close(self):
        pass
