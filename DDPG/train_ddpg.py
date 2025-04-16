import pandas as pd
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import time

from envs import OfflineEnv
from ddpg import DDPGAgent

import os

ROOT_DIR = '/Users/vivianyan/Desktop/Reinforcement-Learning/Project/RL-Movie-Rec-Sys'
DATA_DIR = os.path.join(ROOT_DIR, 'ml-100k')
STATE_SIZE = 10
MAX_EPISODE_NUM = 10

if __name__ == "__main__":
    # Load data
    print('Data loading...')

    ratings_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'mod_ratings.csv'), 'r').readlines()]
    ratings_df = pd.DataFrame(ratings_list[1:], columns = ['userId', 'movieId', 'rating', 'timestamp'])
    ratings_df['userId'] = ratings_df['userId'].apply(pd.to_numeric)
    ratings_df['movieId'] = ratings_df['movieId'].apply(pd.to_numeric)
    ratings_df['rating'] = ratings_df['rating'].astype(float)

    movies_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'mod_movies.csv'),encoding='latin-1').readlines()]
    movies_df = pd.DataFrame(movies_list[1:], columns = ['movieId', 'title', 'genres'])
    movies_df['movieId'] = movies_df['movieId'].apply(pd.to_numeric)

    users_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'users.csv'), 'r').readlines()]
    users_df = pd.DataFrame(users_list[1:], columns=['userId', 'gender', 'age', 'occupation', 'zip'])

    tags_list = [i.strip().split("\t") for i in open(os.path.join(DATA_DIR,'mod_tags.csv'), 'r').readlines()]
    tags_df = pd.DataFrame(tags_list[1:], columns=['userId', 'movieId', 'tag', 'timestamp'])
    tags_df['userId'] = tags_df['userId'].apply(pd.to_numeric)
    tags_df['movieId'] = tags_df['movieId'].apply(pd.to_numeric)

    print("Data loading complete!")
    print("Data preprocessing...")

    # Training preparation
    movies_id_to_movies = {movie[0]: movie[1:] for movie in movies_list[1:]}
    users_dict = np.load(os.path.join(os.getcwd(), "data/user_dict.npy"), allow_pickle=True)
    users_history_lens = np.load(os.path.join(os.getcwd(), "data/users_histroy_len.npy"))

    users_num = max(ratings_df['userId']) + 1
    movies_num = max(ratings_df['movieId']) + 1
    train_users_num = int(users_num * 0.8)
    train_movies_num = movies_num
    train_users_dict = {k: users_dict[k] for k in range(1, train_users_num+1)}
    train_users_history_lens = users_history_lens[:train_users_num]
    
    print("Training prep done!")
    time.sleep(0.2)

    # Training
    env = OfflineEnv(train_users_dict, train_users_history_lens, movies_id_to_movies, STATE_SIZE)
    
    ddpg = DDPGAgent(env, users_num, items_num, STATE_SIZE, use_wandb=False)
    ddpg.actor.build_networks()
    ddpg.critic.build_networks()
    ddpg.train(MAX_EPISODE_NUM, load_model=False)