import numpy as np
from stable_baselines3 import PPO
from utils.data_loader import load_data
from env.movie_rec_env import MovieRecEnv


def evaluate():
    # Load data and env
    ratings, users, movies = load_data()

    env = MovieRecEnv(ratings=ratings, users=users, movies=movies, max_steps=500)

    # Load trained model
    model = PPO.load('ppo_reco_model', env=env)

    # Evaluate
    episodes = 10
    results = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        results.append(total_reward)

    avg_reward = float(np.mean(results))
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

    return avg_reward
