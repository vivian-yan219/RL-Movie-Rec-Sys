# RL-Movie-Rec-Sys
DS-GA 3001 Reinforcement Learning Project Spring 2025

## Introduction
The objective of this project is to explore the application of RL in personalized recommendation systems. In this GitHub, we developed a custom RL environment based on the Movielens-100k dataset that simulates realistic user interactions. Building on this environment, we implemented and experiment with four different reinforcement learning algorithms, including bandits, and actor-critic methods. The performance of these models is then systematically evaluated using offline metrics, such as precision, recall, ndcg, and MAP.

## Methodology
**Bandit**

**DDPG**

We built a DDPG agent from scratch, using the actor and critic network with pretrained embeddings, reducing overestimated Q-values, and applying priorized experience replay (PER).

The saved model of actor and critic are generated after the training is done: `python train.py`. During evaluation, we experimented with the saved actor and critic models and recommended related movies with respect to user's watch history.

**PPO**

## Results
||Precision@10|Recall@10|NDCG@10|MAP@10|
|-|-|-|-|-|
|Bandit|||||
|DDPG|0.0102|0.0016|0.0132|0.0451|
|PPO|||||
|..|||||
