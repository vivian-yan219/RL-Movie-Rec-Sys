import numpy as np
import pandas as pd
from collections import defaultdict
from stable_baselines3 import PPO
from utils.data_loader import load_data
from env.movie_rec_env import MovieRecEnv
from tqdm import tqdm


# ─── Metric functions ─────────────────────────────────────────────────────────

def precision_at_k(ranked, gt, k):
    hits = sum(1 for m in ranked[:k] if m in gt)
    return hits / k

def recall_at_k(ranked, gt, k):
    return precision_at_k(ranked, gt, k) * k / len(gt) if gt else 0.0

def dcg_at_k(ranked, gt, k):
    return sum((1.0 / np.log2(i+2)) for i,m in enumerate(ranked[:k]) if m in gt)

def idcg_at_k(gt, k):
    return sum(1.0/np.log2(i+2) for i in range(min(len(gt), k)))

def ndcg_at_k(ranked, gt, k):
    idcg = idcg_at_k(gt, k)
    return (dcg_at_k(ranked, gt, k)/idcg) if idcg>0 else 0.0

def average_precision(ranked, gt, k):
    hits = 0
    sum_prec = 0.0
    for i, m in enumerate(ranked[:k], start=1):
        if m in gt:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(gt) if gt else 0.0


# ─── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
        episodes: int=10,
        max_steps: int=500,
        k: int=10
):
    # Load data and env
    ratings, users, movies = load_data()

    # Split off 20% of interactions for test
    test_df = ratings.groupby('user_id', group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=0))

    gt = defaultdict(list)
    for _, r in test_df.iterrows():
        gt[r.user_id].append(r.movie_id)

    env = MovieRecEnv(ratings=ratings, users=users, movies=movies, max_steps=max_steps)

    # Load trained model
    model = PPO.load('ppo_reco_model', env=env)

    # Evaluate
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        rewards.append(total_reward)

    avg_reward = float(np.mean(rewards))
    print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")

    # Compute metrics
    precisions, recalls, ndcgs, aps = [], [], [], []
    movie_list = list(movies.index)

    for user_id, true_items in tqdm(
        gt.items(),
        total=len(gt),
        desc=f"Eval @K={k}"
    ):
        # score every candidate movie
        scores = {}
        for m in movie_list:
            obs_vec = env._get_obs(env.order[0])
            idx = ratings[(ratings.user_id==user_id)&(ratings.movie_id==m)].index
            if len(idx)==0:
                continue
            obs_vec = env._get_obs(idx[0])

            pred, _ = model.predict(obs_vec, deterministic=True)
            scores[m] = pred + 1

        ranked = sorted(scores, key=lambda x: scores[x], reverse=True)

        precisions.append(precision_at_k(ranked, true_items, k))
        recalls.append(   recall_at_k(ranked,    true_items, k))
        ndcgs.append(     ndcg_at_k(ranked,      true_items, k))
        aps.append(       average_precision(ranked, true_items, k))

    print(f"Precision@{k}: {np.mean(precisions):.4f}")
    print(f"Recall@{k}:    {np.mean(recalls):.4f}")
    print(f"NDCG@{k}:      {np.mean(ndcgs):.4f}")
    print(f"MAP@{k}:       {np.mean(aps):.4f}")

    return {
        'avg_reward': np.mean(rewards),
        'precision':  np.mean(precisions),
        'recall':     np.mean(recalls),
        'ndcg':       np.mean(ndcgs),
        'map':        np.mean(aps),
    }
