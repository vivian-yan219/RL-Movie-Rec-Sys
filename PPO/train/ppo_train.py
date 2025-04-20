import gymnasium
from stable_baselines3 import PPO
from utils.data_loader import load_data
from env.movie_rec_env import MovieRecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def train():
    # Load data
    ratings, users, movies = load_data()

    # Create env
    env = MovieRecEnv(ratings=ratings, users=users, movies=movies, max_steps=500)

    # Train PPO
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=200_000)
    import os
    # print("Saving model in:", os.getcwd())
    model.save('ppo_reco_model')
    # print('Files here:', os.listdir(os.getcwd()))